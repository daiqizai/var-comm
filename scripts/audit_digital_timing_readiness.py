#!/usr/bin/env python3
"""Verify existing digital timing inputs on CPU; never load models or run a benchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from var_comm.study import artifact_hashes, create_output, seeded_noise, sha256, write_csv, write_json


SOURCE_INDICES = tuple(np.rint(np.linspace(0, 99, 32)).astype(int).tolist())
SNRS = (1.0, 4.0, 7.0, 13.0, 19.0)
SEED = 2001
ARMS = ("whole_m7", "whole_m8", "whole_m9", "whole_adaptive")
MODES = {1.0: 7, 4.0: 8, 7.0: 9, 13.0: 9, 19.0: 9}
PATCH_SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
DIGITAL = ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-001"
R2_TIMING = ROOT / "outputs/WETOK-JOINT-SUFFICIENCY-R2-EVALUATION/timing_0010000"


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def frame_key(row):
    return int(row["image_index"]), float(row["snr_db"]), int(row["seed"]), row["arm"]


def index_rows(rows):
    result = {}
    for row in rows:
        key = frame_key(row)
        if key in result:
            raise ValueError(f"duplicate timing-reference key: {key}")
        result[key] = row
    return result


def require_matrix(rows, indices, snrs, seed, arms):
    lookup = index_rows(rows)
    expected = {(index, snr, seed, arm) for index in indices for snr in snrs for arm in arms}
    if set(lookup) != expected:
        raise ValueError("timing-reference matrix is incomplete or contains unexpected rows")
    return lookup


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bound_file(directory, receipt, relative, inspected):
    directory = Path(directory).resolve()
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("input artifact escaped its frozen directory")
    expected = receipt["output_hashes"][relative]
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"frozen input hash changed: {path}")
    inspected[str(path)] = actual
    return path


def check_source_image(image, expected_digest):
    image = np.asarray(image)
    if (image.shape != (3, 256, 256) or not np.isfinite(image).all() or
            np.min(image) < 0 or np.max(image) > 1):
        raise ValueError("invalid archived source RGB")
    pixels = np.rint(image * 255).astype(np.uint8)
    if array_digest(pixels) != expected_digest:
        raise ValueError("archived RGB differs from the declared preprocessed source")


def check_frame(row, receiver, signal, image):
    mode = MODES[float(row["snr_db"])] if row["arm"] == "whole_adaptive" else int(row["arm"][-1])
    signal = np.asarray(signal)
    if (signal.shape != (3060, 2) or signal.dtype != np.float64 or
            not np.isfinite(signal).all() or not np.all(np.abs(signal) == 1)):
        raise ValueError("digital QPSK waveform or original dtype changed")
    if int(row["complex_uses"]) != 3060 or float(np.sum(signal ** 2)) != 6120.0:
        raise ValueError("digital N/E ledger changed")
    noise = seeded_noise(row["image_id"], int(row["seed"]), signal.shape)
    received = signal + noise / np.sqrt(10 ** (float(row["snr_db"]) / 10))
    if array_digest(received) != row["received_sha256"] or row["received_sha256"] != receiver["received_sha256"]:
        raise ValueError("waveform/noise replay differs from the original observation")
    if (receiver["arm"] != row["arm"] or float(receiver["snr_db"]) != float(row["snr_db"]) or
            int(receiver["seed"]) != int(row["seed"]) or
            int(receiver["reconstruction_index"]) != int(row["reconstruction_index"])):
        raise ValueError("receiver output reference changed")
    image = np.asarray(image)
    if (image.shape != (3, 256, 256) or not np.isfinite(image).all() or
            np.min(image) < 0 or np.max(image) > 1):
        raise ValueError("invalid archived receiver RGB")
    header_ok = bool(receiver["header"]["accepted"])
    if header_ok != bool(int(row["header_accepted"])):
        raise ValueError("header acceptance differs across archived records")
    if not header_ok:
        if receiver["label"] is not None or receiver["output_prefix"] or not np.all(image == 0.5):
            raise ValueError("header failure acquired free information or changed its gray output")
        failure = "header"
    else:
        received_mode = int(receiver["header"]["mode"])
        prefix = receiver["output_prefix"]
        if received_mode not in (7, 8, 9) or len(prefix) != received_mode:
            raise ValueError("invalid receiver prefix length")
        if receiver["label"] != receiver["header"]["label"]:
            raise ValueError("receiver class did not come from its header")
        for scale, tokens in enumerate(prefix):
            tokens = np.asarray(tokens)
            if (tokens.shape != (PATCH_SIZES[scale] ** 2,) or
                    not np.issubdtype(tokens.dtype, np.integer) or np.min(tokens) < 0 or np.max(tokens) >= 4096):
                raise ValueError("invalid receiver token candidate")
        if len(receiver["events"]) != 1:
            raise ValueError("whole-frame receiver no longer has one body decision")
        accepted = bool(receiver["events"][0]["accepted"])
        failure = "accepted" if accepted else "body_crc"
        expected_trusted = received_mode if accepted else 0
        if int(receiver["last_accepted_scale"]) != expected_trusted:
            raise ValueError("retained CRC-failed prefix was relabeled reliable")
    if int(row["last_accepted_scale"]) != int(receiver["last_accepted_scale"]):
        raise ValueError("trusted-state count differs across archived records")
    return {"image_index": int(row["image_index"]), "image_id": row["image_id"],
            "snr_db": float(row["snr_db"]), "seed": int(row["seed"]), "arm": row["arm"],
            "transmitted_mode": mode, "total_complex_uses": 3060, "total_energy": 6120.0,
            "transmitted_sha256": array_digest(signal), "received_sha256": array_digest(received),
            "archived_image_sha256": array_digest(image), "failure_stratum": failure,
            "header_false_acceptance": int(row["header_false_acceptance"]),
            "source_crc_false_acceptance_count": int(row["source_crc_false_acceptance_count"])}


def source_evidence(path, marker):
    source = Path(path)
    matches = [index for index, line in enumerate(source.read_text().splitlines(), 1) if marker in line]
    if len(matches) != 1:
        raise ValueError(f"timing-boundary evidence is ambiguous: {source}: {marker}")
    return f"{source.relative_to(ROOT)}:{matches[0]}"


def boundary_rows():
    digital_eval = ROOT / "scripts/evaluate_progressive_channel.py"
    renderer = ROOT / "src/var_comm/progressive.py"
    r2_eval = ROOT / "experiments/wetok-joint-sufficiency-r2/scripts/evaluate_timing.py"
    return [
        {"entry": "digital_archived_receiver_seconds", "start": "CPU received waveform", "end": "CPU decoded prefix",
         "omitted": "VAR completion and visual Decoder", "direct_cross_system_rank_allowed": False,
         "evidence": source_evidence(digital_eval, "receiver_times[arm] = time.perf_counter() - tick")},
        {"entry": "digital_complete_image", "start": "CPU decoded prefix", "end": "CPU RGB numpy array",
         "omitted": "PHY decoding; transfer is included by cpu().numpy()", "direct_cross_system_rank_allowed": False,
         "evidence": source_evidence(renderer, "return vae.fhat_to_img(latent).clamp")},
        {"entry": "R2_receiver_seconds", "start": "GPU received waveform", "end": "GPU decoded RGB",
         "omitted": "host-to-device and device-to-host transfers", "direct_cross_system_rank_allowed": False,
         "evidence": source_evidence(r2_eval, "receiver_seconds = time.perf_counter() - tick") + ";" +
                     source_evidence(r2_eval, "image_error = float(np.max")},
        {"entry": "R2_online_TX_seconds", "start": "GPU source RGB; two separate timers", "end": "GPU channel symbols",
         "omitted": "indices_to_features between timers; host/device transfers; contiguous end-to-end overhead",
         "direct_cross_system_rank_allowed": False,
         "evidence": source_evidence(r2_eval, "truth = indices_to_features(encoded)") + ";" +
                     source_evidence(r2_eval, "'online_TX_seconds': visual_seconds + transmitter_seconds")},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/DIGITAL-ONLINE-TIMING-READINESS-20260915")
    arguments = parser.parse_args()
    if arguments.output_dir.exists():
        raise FileExistsError("refusing to overwrite a readiness audit")
    inspected = {}
    digital_receipt = json.loads((DIGITAL / "completion.json").read_text())
    timing_receipt = json.loads((R2_TIMING / "completion.json").read_text())
    if (digital_receipt["status"] != "MATCHED_MECHANISM_ONLY_STRONG_CONTROLS_NOT_BEATEN" or
            timing_receipt["status"] != "R2_TIMING_COMPLETE"):
        raise ValueError("expected frozen complete reference runs")
    for directory in (DIGITAL, R2_TIMING):
        inspected[str(directory / "completion.json")] = sha256(directory / "completion.json")
    for root, receipt in ((ROOT, digital_receipt), (ROOT.parent, timing_receipt)):
        for relative, expected in receipt["source_hashes"].items():
            path = root / relative
            if sha256(path) != expected:
                raise ValueError(f"frozen reference source changed: {path}")
            inspected[str(path)] = expected
    population_path = bound_file(DIGITAL, digital_receipt, "populations.json", inspected)
    targets = json.loads(population_path.read_text())["target"]
    if len(targets) != 100 or len({target["image_id"] for target in targets}) != 100:
        raise ValueError("original development population changed")
    digital_rows = read_rows(bound_file(DIGITAL, digital_receipt, "per_frame.csv", inspected))
    selected = [row for row in digital_rows if int(row["image_index"]) in SOURCE_INDICES and
                row["arm"] in ARMS and int(row["seed"]) == SEED]
    digital_lookup = require_matrix(selected, SOURCE_INDICES, SNRS, SEED, ARMS)
    timing_rows = read_rows(bound_file(R2_TIMING, timing_receipt, "per_frame.csv", inspected))
    timing_arms = sorted({row["arm"] for row in timing_rows})
    if len(timing_arms) != 15 or "r2__full_grid_innovation" not in timing_arms:
        raise ValueError("R2 qualified timing references changed")
    require_matrix(timing_rows, SOURCE_INDICES, SNRS, SEED, timing_arms)
    for row in timing_rows:
        if row["image_id"] != targets[int(row["image_index"])]["image_id"]:
            raise ValueError("R2 and digital timing-source identities are not paired")
    boundaries = boundary_rows()
    rows = []
    for image_index in SOURCE_INDICES:
        target = targets[image_index]
        if target["role"] != "target_development" or sha256(target["path"]) != target["file_sha256"]:
            raise ValueError("source is not the unchanged existing development image")
        inspected[target["path"]] = target["file_sha256"]
        relative = f"images/{image_index:03d}"
        receiver_path = bound_file(DIGITAL, digital_receipt, f"{relative}/receivers.json", inspected)
        receivers = json.loads(receiver_path.read_text())
        receiver_lookup = index_rows([{"image_index": image_index, **receiver} for receiver in receivers])
        waveform_path = bound_file(DIGITAL, digital_receipt, f"{relative}/waveforms.npz", inspected)
        image_path = bound_file(DIGITAL, digital_receipt, f"{relative}/reconstructions.npz", inspected)
        with np.load(waveform_path, allow_pickle=False) as waveforms, np.load(image_path, allow_pickle=False) as archive:
            check_source_image(archive["source"], target["preprocessed_rgb_sha256"])
            images = archive["images"]
            for snr_index, snr in enumerate(SNRS):
                for arm in ARMS:
                    key = image_index, snr, SEED, arm
                    row = digital_lookup[key]
                    if row["image_id"] != target["image_id"]:
                        raise ValueError("digital timing-source identity changed")
                    mode = MODES[snr] if arm == "whole_adaptive" else int(arm[-1])
                    image = images[int(row["reconstruction_index"])]
                    rows.append(check_frame(row, receiver_lookup[key], waveforms[f"whole_m{mode}_{snr_index}"], image))
    if "torch" in sys.modules:
        raise RuntimeError("CPU-only audit unexpectedly imported a model framework")
    output = create_output(arguments.output_dir)
    write_csv(output / "archived_frame_inputs.csv", rows)
    write_csv(output / "timing_boundaries.csv", boundaries)
    strata = [{"arm": arm, "failure_stratum": stratum,
               "count": sum(row["arm"] == arm and row["failure_stratum"] == stratum for row in rows)}
              for arm in ARMS for stratum in ("header", "body_crc", "accepted")]
    write_csv(output / "retained_failure_counts.csv", strata)
    receipt = {"status": "OFFLINE_INPUTS_VERIFIED_TIMING_UNMEASURED", "completed_local": datetime.now().astimezone().isoformat(),
               "source_images": len(SOURCE_INDICES), "source_indices": SOURCE_INDICES, "snrs_db": SNRS, "noise_seed": SEED,
               "digital_reference_rows": len(rows), "R2_reference_rows": len(timing_rows), "source_role": "existing_development_only",
               "GPU_model_inference_started": False, "new_training_or_search": False, "new_holdout_accessed": False,
               "new_latency_values_measured": False, "new_system_quality_measured": False, "automatic_GPU_queue_started": False,
               "direct_comparison_of_existing_timers_allowed": False, "R2_pixel_identity_recomputed": False,
               "R2_pairing_scope": "source identifiers and original SHA-bound timing records, not a fresh tensor replay",
               "archived_output_scope": "SHA-bound stored RGB and failure rule; no new rendering or metric computation",
               "recommended_new_TX_scope": "resident CPU source RGB to resident CPU channel waveform, one contiguous timer",
               "recommended_new_RX_scope": "resident CPU received waveform to resident CPU RGB, one contiguous timer",
               "recommended_exclusions": "file IO, model loading, warmup, AWGN construction, quality metrics",
               "required_followup": "authorization plus a paired full-timing implementation and exact output replay; no new training",
               "all_failures_included": True, "input_sha256": inspected,
               "script_sha256": sha256(Path(__file__)), "output_hashes": artifact_hashes(output)}
    write_json(output / "completion.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "source_images", "digital_reference_rows", "R2_reference_rows",
                     "GPU_model_inference_started", "new_latency_values_measured")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
