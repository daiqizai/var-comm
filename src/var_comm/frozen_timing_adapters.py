"""Read-only archive access and CPU-endpoint wrappers for existing frozen systems."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from .next_scale_prior import load_models, state_sha256
from .progressive import complete_image, receive_whole, transmit_whole
from .online_timing import single_waveform
from .study import seeded_noise, sha256


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
DIGITAL = ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-001"
R2 = ROOT / "outputs/WETOK-JOINT-SUFFICIENCY-R2-EVALUATION/quality_0010000"
DEEP = ROOT / "outputs/VAR-PREFIX-JSCC-EVAL-001"
NATIVE = ROOT / "outputs/WETOK-NATIVE-CACHE-20260912"
WETOK_DIGITAL = ROOT / "outputs/WETOK-DIGITAL-8PSK-20260912-PAIRED"


def install_frozen_paths():
    for experiment in ("wetok-joint-sufficiency-r2", "wetok-joint-grid-controls-r1", "wetok-joint-sender-r1",
                       "wetok-innovation-r1", "wetok-comm-v2-20260912"):
        for kind in ("src", "scripts"):
            path = str(EXPERIMENTS / experiment / kind)
            if path not in sys.path:
                sys.path.append(path)


def rows_from(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def keyed(rows):
    return {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"]), row["arm"]): row for row in rows}


class ArchiveInputs:
    def __init__(self):
        self.hashes, self.receipts = {}, {}
        for directory in (DIGITAL, R2, DEEP, NATIVE, WETOK_DIGITAL):
            path = directory / "completion.json"
            self.hashes[str(path)] = sha256(path)
            self.receipts[directory] = json.loads(path.read_text())
        self.targets = json.loads(self.bound(DIGITAL, "populations.json").read_text())["target"]
        self.digital_rows = keyed(rows_from(self.bound(DIGITAL, "per_frame.csv")))
        self.r2_rows = keyed(rows_from(self.bound(R2, "per_frame.csv")))
        self.deep_rows = keyed(rows_from(self.bound(DEEP, "per_frame.csv")))
        self.wetok_rows = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"])): row
                           for row in rows_from(self.bound(WETOK_DIGITAL, "per_frame.csv"))}
        self.codes = np.load(self.bound(NATIVE, "development_codes.npy"), mmap_mode="r")
        identifiers = json.loads(self.bound(NATIVE, "development_ids.json").read_text())
        if identifiers != [target["image_id"] for target in self.targets] or len(self.targets) != 100:
            raise RuntimeError("frozen systems do not use the same original development sources")
        self.loaded_images = {}

    def bound(self, directory, relative):
        directory = Path(directory)
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise RuntimeError("input escaped its archive")
        expected = self.receipts[directory]["output_hashes"][relative]
        if str(path) not in self.hashes:
            if sha256(path) != expected:
                raise RuntimeError(f"frozen archive changed: {path}")
            self.hashes[str(path)] = expected
        return path

    def quality_image(self, row):
        path = Path(row["image_archive"])
        if str(path) not in self.loaded_images:
            with np.load(path, allow_pickle=False) as archive:
                self.loaded_images[str(path)] = archive["images"].copy()
            self.hashes[str(path)] = sha256(path)
        image = self.loaded_images[str(path)][int(row["image_ref"])]
        from joint_sender.evaluation_io import digest
        if digest(image) != row["image_sha256"]:
            raise RuntimeError("referenced quality image differs from the frozen selected output")
        return image

    def source(self, index, modem):
        self.loaded_images.clear()
        target = self.targets[index]
        if target["role"] != "target_development" or sha256(target["path"]) != target["file_sha256"]:
            raise RuntimeError("source is not the original development image")
        self.hashes[target["path"]] = target["file_sha256"]
        relative = f"images/{index:03d}"
        with np.load(self.bound(DIGITAL, relative + "/reconstructions.npz"), allow_pickle=False) as archive:
            digital_images, source = archive["images"].copy(), archive["source"].copy()
        pixels = np.rint(source * 255).astype(np.uint8)
        from joint_sender.evaluation_io import digest
        if digest(pixels) != target["preprocessed_rgb_sha256"]:
            raise RuntimeError("canonical cropped source pixels changed")
        with np.load(self.bound(DIGITAL, relative + "/waveforms.npz"), allow_pickle=False) as archive:
            digital_waves = {name: archive[name].copy() for name in archive.files if name.startswith("whole_")}
        with np.load(self.bound(R2, relative + "/waveforms.npz"), allow_pickle=False) as archive:
            r2_waves = {name: archive[name].copy() for name in archive.files if name.startswith("r2__full_grid_innovation")}
        with np.load(self.bound(DEEP, relative + "/waveforms.npz"), allow_pickle=False) as archive:
            deep_waves = {name: archive[name].copy() for name in archive.files if "deep_" in name}
        deep_snrs = sorted({float(row["snr_db"]) for row in self.deep_rows.values()})
        wetok_signal = modem.transmit(np.array(self.codes[index, 0], copy=True))
        expected = {}
        for snr_index, snr in enumerate((1., 4., 7., 13., 19.)):
            noise = seeded_noise(target["image_id"], 2001, (3060, 2))
            for arm in ("whole_m7", "whole_m8", "whole_m9", "whole_adaptive"):
                mode = ({1.: 7, 4.: 8, 7.: 9, 13.: 9, 19.: 9}[snr] if arm == "whole_adaptive" else int(arm[-1]))
                signal = digital_waves[f"whole_m{mode}_{snr_index}"]
                received = signal + noise / np.sqrt(10 ** (snr / 10))
                row = self.digital_rows[index, snr, 2001, arm]
                if row["image_id"] != target["image_id"] or digest(received) != row["received_sha256"]:
                    raise RuntimeError("digital observation is not the original waveform")
                expected[snr, arm] = {"signal": signal, "received": received,
                    "image": digital_images[int(row["reconstruction_index"])], "row": row,
                    "header_accepted": bool(int(row["header_accepted"])),
                    "body_crc_accepted": bool(int(row["last_accepted_scale"])) if int(row["header_accepted"]) else None}
            arm = "r2__full_grid_innovation"
            row = self.r2_rows[index, snr, 2001, arm]
            signal = single_waveform(r2_waves[f"{arm}__snr{snr}"])
            received = single_waveform(r2_waves[f"{arm}__snr{snr}__seed2001"])
            if digest(signal) != row["transmitted_sha256"] or digest(received) != row["received_sha256"]:
                raise RuntimeError("selected R2 waveform changed")
            expected[snr, arm] = {"signal": signal, "received": received, "image": self.quality_image(row), "row": row,
                                 "header_accepted": None, "body_crc_accepted": None}
            arm = "perceptual_deepjscc"
            row = self.r2_rows[index, snr, 2001, arm]
            deep_row = self.deep_rows[index, snr, 2001, arm]
            frame = f"frame_{deep_snrs.index(snr)}_0_deep"
            if row["image_id"] != deep_row["image_id"] or deep_row["noise_sha256"] != digest(noise):
                raise RuntimeError("Deep reference source/noise changed")
            expected[snr, arm] = {"signal": deep_waves[frame + "_tx"], "received": deep_waves[frame + "_rx"],
                                 "image": self.quality_image(row), "row": row, "header_accepted": None, "body_crc_accepted": None}
            arm = "wetok_8PSK_FEC"
            row = self.r2_rows[index, snr, 2001, arm]
            received = wetok_signal + noise / np.sqrt(10 ** (snr / 10))
            phy_row = self.wetok_rows[index, snr, 2001]
            if digest(received) != phy_row["received_sha256"]:
                raise RuntimeError("WeTok digital observation changed")
            expected[snr, arm] = {"signal": wetok_signal, "received": received, "image": self.quality_image(row), "row": row,
                                 "header_accepted": None, "body_crc_accepted": phy_row["crc_accepted"] == "True"}
        return pixels, target, expected


class FrozenSystems:
    def __init__(self, inputs, device):
        install_frozen_paths()
        from grid_controls.common import initial_system
        from sufficiency.evaluation import load_evaluation
        from sufficiency.common import load_parent
        from wetok_comm.native import FrozenWeTok
        from wetok_comm.deep_support import FrozenDeepSupport, load_deep_support
        from wetok_comm.digital import WeTokDigital, library
        from wetok_comm.common import configure_torch
        from .scale_channel import load_native

        configure_torch()
        self.device = torch.device(device)
        model_config = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())
        for name in ("vae_checkpoint", "var_checkpoint"):
            path = Path(model_config["paths"][name])
            expected = model_config["paths"][name + "_sha256"]
            if sha256(path) != expected:
                raise RuntimeError("official VAR visual checkpoint changed")
            inputs.hashes[str(path)] = expected
        self.vae, self.var = load_models(model_config["paths"], self.device)
        evaluation, config, original, grid, reference, base, parent_record = load_evaluation()
        milestone_path = ROOT / "outputs/WETOK-JOINT-SUFFICIENCY-R2-TRAINING/milestones/step_0010000.json"
        if sha256(milestone_path) != inputs.receipts[R2]["r2_milestone_sha256"]:
            raise RuntimeError("R2 quality used a different selected milestone")
        inputs.hashes[str(milestone_path)] = sha256(milestone_path)
        milestone = json.loads(milestone_path.read_text())
        self.choice = milestone["selected"]["full_grid_innovation"]
        path = Path(self.choice["checkpoint"])
        if self.choice["step"] != 10000 or sha256(path) != self.choice["checkpoint_sha256"]:
            raise RuntimeError("R2 residual selected checkpoint changed")
        inputs.hashes[str(path)] = self.choice["checkpoint_sha256"]
        parent = load_parent(reference, base, self.device)
        self.r2 = initial_system(parent, "full_grid_innovation", reference, self.device).eval().requires_grad_(False)
        stored = torch.load(path, map_location="cpu", weights_only=True)
        self.r2.load_state_dict(stored["model"], strict=True)
        del parent, stored
        self.native = FrozenWeTok(self.device, "both")
        self.deep = FrozenDeepSupport(load_deep_support(), self.device)
        self.modem = WeTokDigital()
        load_native()
        library()
        self.models = {"vae": self.vae, "var": self.var, "native": self.native.codec,
                       "r2__full_grid_innovation": self.r2, "deep": self.deep.model}
        if any(model.training or any(parameter.requires_grad for parameter in model.parameters()) for model in self.models.values()):
            raise RuntimeError("timing model is not frozen evaluation-only")
        self.before = self.model_hashes()
        for name in ("native", "r2__full_grid_innovation"):
            if self.before[name] != inputs.receipts[R2]["frozen_models_before"][name]:
                raise RuntimeError(f"loaded {name} differs from the frozen quality run")
        for name in ("vae", "var"):
            if self.before[name] != inputs.receipts[DIGITAL]["frozen_before"][name]:
                raise RuntimeError(f"loaded {name} differs from digital quality")

    def model_hashes(self):
        from wetok_comm.training import module_sha256
        return {name: (module_sha256(model) if name in ("native", "r2__full_grid_innovation") else state_sha256(model))
                for name, model in self.models.items()}

    @torch.no_grad()
    def check_native_source(self, pixels, expected):
        reference = self.native.encode(torch.from_numpy(pixels[None]).to(self.device).float().div(255))
        online = self.native_indices(pixels)
        if not torch.equal(reference, online) or not np.array_equal(online[0].cpu().numpy(), expected):
            raise RuntimeError("online native tokenization differs from its original checked wrapper/cache")

    def native_indices(self, pixels):
        images = torch.from_numpy(pixels[None]).to(self.device).float().div(255)
        encoded = self.native.codec.encoder(images.mul(2).sub(1))
        quantized, auxiliary, indices = self.native.quantizer(encoded)
        return indices.reshape(1, 16, 16, 4).to(torch.uint8)

    @torch.no_grad()
    def transmit(self, pixels, snr, arm, label):
        if arm.startswith("whole_"):
            mode = ({1.: 7, 4.: 8, 7.: 9, 13.: 9, 19.: 9}[snr] if arm == "whole_adaptive" else int(arm[-1]))
            images = torch.from_numpy(pixels[None]).float().div(127.5).sub(1).to(self.device)
            indices = self.vae.img_to_idxBl(images)
            source = [tokens[0].cpu().numpy() for tokens in indices[:mode]]
            return transmit_whole(source, label, mode)
        from wetok_comm.native import indices_to_features
        if arm == "perceptual_deepjscc":
            images = torch.from_numpy(pixels[None]).float().div(127.5).sub(1).add(1).mul(.5).to(self.device)
            condition = torch.full((1,), float(snr), device=self.device)
            encoded = self.deep.model.encode(images, condition)
            normalized, unused_power = self.deep.model.normalize_channel_input(encoded)
            signal = normalized.flatten(1).index_select(1, self.deep.model.active_real_indices).reshape(1, 3060, 2)
        else:
            indices = self.native_indices(pixels)
            if arm == "wetok_8PSK_FEC":
                return self.modem.transmit(indices[0].cpu().numpy())
            if arm != "r2__full_grid_innovation":
                raise ValueError("unregistered frozen arm")
            signal = self.r2.transmit(indices_to_features(indices), torch.full((1,), float(snr), device=self.device))
        return signal[0].cpu().numpy()

    @torch.no_grad()
    def receive(self, received, snr, arm):
        if arm.startswith("whole_"):
            result = receive_whole(received, snr)
            image = complete_image(self.vae, self.var, result["prefix"], result["label"], self.device)
            return image, {"header_accepted": bool(result["header"]["accepted"]),
                           "body_crc_accepted": bool(result["events"][0]["accepted"]) if result["events"] else None}
        from wetok_comm.native import indices_to_features
        from innovation_comm.inference import receive_for_image
        flags = {"header_accepted": None, "body_crc_accepted": None}
        if arm == "wetok_8PSK_FEC":
            result = self.modem.receive(received, snr)
            features = indices_to_features(torch.from_numpy(result["indices"][None]).to(self.device))
            image = self.native.decode(features)
            flags["body_crc_accepted"] = bool(result["crc_accepted"])
        else:
            waveform = torch.from_numpy(received[None]).to(self.device)
            condition = torch.full((1,), float(snr), device=self.device)
            if arm == "perceptual_deepjscc":
                flat = waveform.new_zeros((1, self.deep.model.native_real_symbols))
                latent = flat.index_copy(1, self.deep.model.active_real_indices, waveform.reshape(1, 6120))
                image = self.deep.model.decode(latent.reshape(1, *self.deep.layout), condition).clamp(0, 1)
            elif arm == "r2__full_grid_innovation":
                result = receive_for_image(self.r2, waveform, condition)
                image = self.native.decode(result["receiver_features"])
            else:
                raise ValueError("unregistered frozen arm")
        return image[0].cpu().numpy(), flags


def loaded_local_sources():
    paths = set()
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_file() and path.is_relative_to(ROOT.parent) and ".venv" not in path.parts:
                paths.add(path)
    return sorted(paths)
