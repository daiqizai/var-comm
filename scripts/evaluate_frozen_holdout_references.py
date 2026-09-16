#!/usr/bin/env python3
"""Evaluate only existing R2/R3/Deep/WeTok-digital systems after the holdout protocol is frozen."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from benchmark_frozen_systems import assert_gpu_available, digest, gpu_state
from var_comm.frozen_timing_adapters import ArchiveInputs, FrozenSystems, install_frozen_paths, loaded_local_sources, rows_from
from var_comm.next_scale_prior import preprocess, state_sha256
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.study import artifact_hashes, create_output, seeded_noise, sha256, write_csv, write_json

R2_NAME = "r2__full_grid_innovation"
R3_NAME = "r3__full_grid_prediction_features"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development_smoke", "holdout"), default="holdout")
    parser.add_argument("--frozen-method", type=Path)
    parser.add_argument("--holdout-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        print("PLAN ONLY: frozen holdout references; no training, timing or model selection")
        return
    config = json.loads((ROOT / "configs/communication_decision_study.json").read_text())
    if arguments.mode == "holdout":
        if arguments.frozen_method is None or arguments.holdout_manifest is None:
            raise RuntimeError("holdout requires the frozen method and manifest")
        method = json.loads(arguments.frozen_method.read_text())
        manifest = json.loads(arguments.holdout_manifest.read_text())
        if (method["status"] != "FINAL_METHOD_AND_HOLDOUT_PROTOCOL_FROZEN" or
            method["holdout_manifest_sha256"] != sha256(arguments.holdout_manifest) or
            method["config_sha256"] != sha256(ROOT / "configs/communication_decision_study.json") or
            manifest["role"] != "independent_holdout_after_method_freeze" or len(manifest["images"]) != 1000):
            raise RuntimeError("holdout method/data must be frozen before image access")
        source_indices, seeds = list(range(1000)), config["holdout_seeds"]
        method_hash, manifest_hash = sha256(arguments.frozen_method), sha256(arguments.holdout_manifest)
    else:
        population_path = ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-001/populations.json"
        manifest = {"images": json.loads(population_path.read_text())["target"]}
        source_indices, seeds = [0, 99], config["development_seeds"]
        method_hash, manifest_hash = None, sha256(population_path)
    output = arguments.output_dir.resolve()
    if not output.is_relative_to(ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915"):
        raise ValueError("holdout reference output escaped the study")
    if (output / "completion.json").exists():
        raise RuntimeError("holdout references are already complete")
    if not arguments.resume:
        create_output(output)
    (output / "images").mkdir(exist_ok=True)
    started = time.perf_counter()
    rows, completed = [], 0
    try:
        assert_gpu_available()
        install_frozen_paths()
        sys.path.insert(0, str(ROOT / "experiments/wetok-reencoding-vector-control-r3/src"))
        from vector_control.evaluation import load_evaluation, model_registry
        from wetok_comm.native import indices_to_features
        from wetok_comm.training import module_sha256
        systems = FrozenSystems(ArchiveInputs(), "cuda:0")
        evaluation, r3_config, r2, original, grid, reference, base, parent_record = load_evaluation()
        milestone_path = ROOT / "outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-TRAINING/milestones/step_0010000.json"
        milestone = json.loads(milestone_path.read_text())
        if milestone["selected"]["checkpoint_sha256"] != "667abc43639c09b0f0c465bc3bf669117ff727095ffdb7e563f6dad71c11ab3e":
            raise RuntimeError("R3 was reselected after development")
        parent, models, choices = model_registry(r3_config, reference, base, milestone, torch.device("cuda:0"))
        model3 = models[R3_NAME].eval().requires_grad_(False)
        del parent
        quality_config = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
        perceptual, dino, linear_weights = load_quality_models(quality_config, "cuda:0")
        before = {**systems.model_hashes(), R3_NAME: module_sha256(model3),
                  "lpips": state_sha256(perceptual), "dino": state_sha256(dino)}
        r3_receipt = json.loads((ROOT / "outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-EVALUATION/quality_0010000/completion.json").read_text())
        if before[R3_NAME] != r3_receipt["frozen_models_before"][R3_NAME]:
            raise RuntimeError("R3 model differs from its sealed development checkpoint")
        sources = {str(path): sha256(path) for path in loaded_local_sources()}
        if arguments.resume:
            metadata = json.loads((output / "metadata.json").read_text())
            if (metadata["method_sha256"] != method_hash or metadata["manifest_sha256"] != manifest_hash or
                    metadata["frozen_models"] != before or metadata["source_hashes"] != sources):
                raise RuntimeError("holdout resume changed its frozen method or sources")
        else:
            metadata = {"started_local": datetime.now().astimezone().isoformat(), "method_sha256": method_hash,
                "manifest_sha256": manifest_hash, "frozen_models": before, "source_hashes": sources, "mode": arguments.mode,
                "quality_only_no_new_timing": True, "new_training": False, "GPU": gpu_state(),
                "Deep_support_mapping": {"5.0": 4.0, "6.0": 7.0}, "Deep_support_mapping_preexists_holdout": True}
            write_json(output / "metadata.json", metadata)
            write_json(output / "population.json", manifest["images"])
        old_frames = {}
        if arguments.mode == "development_smoke":
            original_root = ROOT / "outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-EVALUATION/quality_0010000"
            old_frames = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"]), row["arm"]): row for row in rows_from(original_root / "per_frame.csv")}
            for row in rows_from(original_root / "deep_support_supplement.csv"):
                old_frames[int(row["image_index"]), float(row["snr_db"]), int(row["seed"]), "perceptual_deepjscc"] = row
        for index in source_indices:
            target = manifest["images"][index]
            directory = output / "images" / f"{index:04d}"
            receipt_path = directory / "receipt.json"
            if receipt_path.exists():
                saved = json.loads(receipt_path.read_text())
                if saved["image_id"] != target["image_id"]:
                    raise RuntimeError("resumed holdout source identity changed")
                for relative, expected in saved["output_hashes"].items():
                    if sha256(directory / relative) != expected:
                        raise RuntimeError("committed holdout result changed")
                rows.extend(rows_from(directory / "per_frame.csv"))
                completed += 1
                continue
            assert_gpu_available()
            directory.mkdir(exist_ok=True)
            path = Path(target["path"])
            if sha256(path) != target["file_sha256"]:
                raise RuntimeError("frozen holdout source bytes changed")
            normalized, pixel_hash = preprocess(path)
            if pixel_hash != target["preprocessed_rgb_sha256"]:
                raise RuntimeError("frozen holdout preprocessing changed")
            pixels = normalized.add(1).mul(127.5).round().to(torch.uint8).numpy()
            metric_source = torch.from_numpy(pixels).float().div(127.5).sub(1).add(1).mul(.5).numpy()
            images01 = torch.from_numpy(pixels[None]).to("cuda:0").float().div(255)
            indices = systems.native.encode(images01)
            truth = indices_to_features(indices)
            digital_signal = systems.modem.transmit(indices[0].cpu().numpy())
            reconstructed, image_indices, local_rows, waveforms, digital_cache = [], {}, [], {}, {}

            def add_image(image):
                key = digest(image)
                if key not in image_indices:
                    image_indices[key] = len(reconstructed)
                    reconstructed.append(image)
                return image_indices[key], key

            for snr in config["snrs_db"]:
                condition = torch.tensor([snr], device="cuda:0")
                deep_condition_value = {5.: 4., 6.: 7.}.get(float(snr), float(snr))
                deep_condition = torch.tensor([deep_condition_value], device="cuda:0")
                deep_source = torch.from_numpy(metric_source[None]).to("cuda:0")
                encoded = systems.deep.model.encode(deep_source, deep_condition)
                normalized_signal, unused_power = systems.deep.model.normalize_channel_input(encoded)
                deep_signal = normalized_signal.flatten(1).index_select(1, systems.deep.model.active_real_indices).reshape(1, 3060, 2)
                signals = {R2_NAME: systems.r2.transmit(truth, condition), R3_NAME: model3.transmit(truth, condition), "perceptual_deepjscc": deep_signal}
                for name, signal in signals.items():
                    if signal.shape != (1, 3060, 2) or abs(float(signal.square().sum()) - 6120) > .02:
                        raise RuntimeError("holdout learned reference violated the fixed physical budget")
                    waveforms[f"{name}_snr{snr}_tx"] = signal[0].cpu().numpy()
                waveforms[f"wetok_8PSK_FEC_snr{snr}_tx"] = digital_signal
                for seed in seeds:
                    noise64 = seeded_noise(target["image_id"], seed, (3060, 2))
                    noise = torch.tensor(noise64[None], dtype=torch.float32, device="cuda:0")
                    for name, signal in signals.items():
                        received = signal + noise * torch.pow(10., condition / 10).rsqrt()[:, None, None]
                        if name == "perceptual_deepjscc":
                            flat = received.new_zeros((1, systems.deep.model.native_real_symbols))
                            latent = flat.index_copy(1, systems.deep.model.active_real_indices, received.reshape(1, 6120))
                            image = systems.deep.model.decode(latent.reshape(1, *systems.deep.layout), deep_condition).clamp(0, 1)
                        else:
                            result = (systems.r2 if name == R2_NAME else model3).receive(received, condition)
                            image = systems.native.decode(result["receiver_features"])
                        image_index, image_hash = add_image(image[0].cpu().numpy())
                        local_rows.append({"population": arguments.mode, "image_index": index, "image_id": target["image_id"], "snr_db": snr,
                            "seed": seed, "arm": name, "total_complex_uses": 3060, "header_uses": 0, "data_uses": 3060,
                            "total_energy": float(signal.square().sum()), "condition_snr": deep_condition_value if name == "perceptual_deepjscc" else snr,
                            "crc_accepted": "", "image_index_in_archive": image_index, "image_sha256": image_hash,
                            "received_sha256": digest(received[0].cpu().numpy()), "transmitted_sha256": digest(signal[0].cpu().numpy())})
                        waveforms[f"{name}_snr{snr}_seed{seed}_rx"] = received[0].cpu().numpy()
                    observed = digital_signal + noise64 / np.sqrt(10 ** (snr / 10))
                    decoded = systems.modem.receive(observed, snr)
                    key = digest(decoded["indices"])
                    if key not in digital_cache:
                        features = indices_to_features(torch.from_numpy(decoded["indices"][None]).to("cuda:0"))
                        digital_cache[key] = systems.native.decode(features)[0].cpu().numpy()
                    image_index, image_hash = add_image(digital_cache[key])
                    local_rows.append({"population": arguments.mode, "image_index": index, "image_id": target["image_id"], "snr_db": snr,
                        "seed": seed, "arm": "wetok_8PSK_FEC", "total_complex_uses": 3060, "header_uses": 0, "data_uses": 3060,
                        "total_energy": float(np.square(digital_signal).sum()), "condition_snr": snr,
                        "crc_accepted": int(decoded["crc_accepted"]), "image_index_in_archive": image_index, "image_sha256": image_hash,
                        "received_sha256": digest(observed), "transmitted_sha256": digest(digital_signal)})
                    waveforms[f"wetok_8PSK_FEC_snr{snr}_seed{seed}_rx"] = observed
            quality, source_features, image_features = quality_metrics(metric_source, reconstructed, perceptual, dino, "cuda:0")
            for row in local_rows:
                metric = quality[row["image_index_in_archive"]]
                row.update(psnr_db=metric["psnr_db"], lpips=metric["lpips_alex"], dino=metric["dino_cosine"])
                if old_frames:
                    old = old_frames[index, float(row["snr_db"]), int(row["seed"]), row["arm"]]
                    with np.load(old["image_archive"], allow_pickle=False) as old_archive:
                        old_image = old_archive["images"][int(old["image_ref"])].copy()
                    error = float(np.max(np.abs(reconstructed[row["image_index_in_archive"]] - old_image)))
                    if error > 1e-6:
                        raise RuntimeError("existing frozen reference implementation failed the development replay")
                    row["development_replay_max_pixel_error"] = error
            assert_gpu_available()
            np.savez(directory / "reconstructions.npz", images=np.stack(reconstructed), source_dino=source_features, reconstruction_dino=image_features)
            np.savez(directory / "waveforms.npz", **waveforms)
            write_csv(directory / "per_frame.csv", local_rows)
            write_json(receipt_path, {"image_id": target["image_id"], "rows": len(local_rows), "source_pixels_sha256": pixel_hash,
                                     "output_hashes": artifact_hashes(directory)})
            rows.extend(local_rows)
            completed += 1
            write_json(output / "status.json", {"status": "EVALUATING_FROZEN_HOLDOUT_REFERENCES", "pid": __import__("os").getpid(),
                "completed_sources": completed, "source_count": len(source_indices), "rows": len(rows), "updated_local": datetime.now().astimezone().isoformat()})
            print(f"{arguments.mode} frozen references {completed}/{len(source_indices)} rows={len(rows)}", flush=True)
        after = {**systems.model_hashes(), R3_NAME: module_sha256(model3), "lpips": state_sha256(perceptual), "dino": state_sha256(dino)}
        if before != after or len(rows) != len(source_indices) * len(config["snrs_db"]) * len(seeds) * 4:
            raise RuntimeError("holdout reference models changed or the required matrix is incomplete")
        for path, expected in sources.items():
            if sha256(path) != expected:
                raise RuntimeError("holdout reference source changed during execution")
        write_csv(output / "per_frame.csv", rows)
        write_json(output / "completion.json", {"status": "FROZEN_HOLDOUT_REFERENCES_COMPLETE" if arguments.mode == "holdout" else "FROZEN_REFERENCE_DEVELOPMENT_REPLAY_PASS", "completed_local": datetime.now().astimezone().isoformat(),
            "source_images": len(source_indices), "rows": len(rows), "method_sha256": method_hash,
            "manifest_sha256": manifest_hash, "frozen_before": before, "frozen_after": after, "mode": arguments.mode,
            "source_hashes": sources, "Deep_support_mapping": {"5.0": 4.0, "6.0": 7.0}, "no_new_timing": True,
            "all_CRC_failures_retained_for_image": True, "new_training": False, "research_goal_complete": False,
            "elapsed_seconds": time.perf_counter() - started, "output_hashes": artifact_hashes(output)})
    except BaseException as error:
        write_json(output / f"failure_{time.time_ns()}.json", {"error": repr(error), "traceback": traceback.format_exc(), "completed_sources": completed})
        raise


if __name__ == "__main__":
    main()
