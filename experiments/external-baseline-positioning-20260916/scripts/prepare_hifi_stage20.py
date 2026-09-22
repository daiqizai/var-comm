#!/usr/bin/env python3
"""Freeze a quality-independent exploratory cohort and audit committed diffusion work."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(PROJECT / "src")]

import numpy as np

from var_comm.study import sha256, write_csv, write_json


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value in ("True", "true", 1):
        return True
    if value in ("False", "false", 0, "", None):
        return False
    raise ValueError("invalid boolean record")


def account_frame(record):
    rows = record["rows"]
    if len(rows) != 3:
        raise ValueError("each completed HiFi frame must contain three record views")
    common = next(row for row in rows if row["method"] == "hifi_diffcom_c2" and row["protocol"] == "common_paid_information")
    native = next(row for row in rows if row["method"] == "hifi_diffcom_c2" and row["protocol"] == "author_assumed_information")
    base = next(row for row in rows if row["method"] == "adjscc_c2")
    common_calls = int(as_bool(common["metadata_usable"]))
    extra_calls = int(not as_bool(native["same_receive_inputs_reused"]))
    common_steps = int(common["NFE"])
    native_steps = int(native["NFE"]) if extra_calls else 0
    if common_calls != int(common_steps > 0):
        raise ValueError("common sampling and actual fallback disagree")
    return {"frame_key": record["frame_key"], "image_index": common["image_index"], "snr_db": common["snr_db"],
            "seed": common["seed"], "record_views": len(rows), "common_sampler_calls": common_calls,
            "metadata_extra_author_sampler_calls": extra_calls, "actual_sampler_calls": common_calls + extra_calls,
            "common_NFE_reverse_steps": common_steps, "extra_author_reverse_steps": native_steps,
            "total_actual_reverse_steps": common_steps + native_steps,
            "full_common_RX_seconds": float(common["RX_seconds"]), "TX_seconds": float(common["TX_seconds"]),
            "companion_ADJSCC_seconds": float(base["RX_seconds"]),
            "extra_author_RX_seconds": "not_recorded" if extra_calls else 0.,
            "metadata_crc_accepted": common["metadata_crc_accepted"], "metadata_usable": common["metadata_usable"],
            "metadata_payload_exact_offline": common["offline_metadata_exact"], "fallback": common["fallback"],
            "count_basis": "actual_committed_views_and_frozen_evaluator_branch_logic_not_view_count"}


def summarize_cost(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[float(row["snr_db"])].append(row)
    result = []
    for snr in (1., 4., 7., 13., 19.):
        chosen = groups[snr]
        elapsed = np.array([row["full_common_RX_seconds"] for row in chosen])
        steps = np.array([row["common_NFE_reverse_steps"] for row in chosen])
        result.append({"snr_db": snr, "completed_transmissions": len(chosen), "record_views": len(chosen) * 3,
                       "actual_sampler_calls": sum(row["actual_sampler_calls"] for row in chosen),
                       "common_sampler_calls": sum(row["common_sampler_calls"] for row in chosen),
                       "metadata_extra_author_sampler_calls": sum(row["metadata_extra_author_sampler_calls"] for row in chosen),
                       "total_actual_reverse_steps": sum(row["total_actual_reverse_steps"] for row in chosen),
                       "common_NFE_mean": float(steps.mean()) if len(chosen) else "",
                       "common_NFE_min": int(steps.min()) if len(chosen) else "",
                       "common_NFE_max": int(steps.max()) if len(chosen) else "",
                       "RX_mean_seconds": float(elapsed.mean()) if len(chosen) else "",
                       "RX_median_seconds": float(np.median(elapsed)) if len(chosen) else "",
                       "RX_p95_seconds": float(np.quantile(elapsed, .95)) if len(chosen) else "",
                       "RX_sum_seconds": float(elapsed.sum()),
                       "extra_author_time_available": not any(row["metadata_extra_author_sampler_calls"] for row in chosen),
                       "quality_selection": False})
    return result


def run(arguments):
    root, output = arguments.root, arguments.output
    output.mkdir(parents=True, exist_ok=False)
    original_manifest = root / "inputs_001/development_inputs.json"
    population = json.loads(original_manifest.read_text())
    indices = np.rint(np.linspace(0, 99, 20)).astype(int).tolist()
    selected = [row for row in population if row["image_index"] in indices]
    if len(selected) != 20 or sorted(row["image_index"] for row in selected) != indices:
        raise RuntimeError("original development source order changed")
    protocol = {"study": "HIFI_EXPLORATORY_STAGE20_20260916", "scope": "exploratory20_sources_5SNRs_one_original_noise",
                "source_indices": indices, "source_selection": "rint(linspace(0,99,20)); independent of images/quality",
                "snrs_db": [1., 4., 7., 13., 19.], "noise_seeds": [2001], "transmissions_per_method": 100,
                "sampling_seed": 23, "sampling": "unchanged_author_N1_DDPM_adaptive_start_full_schedule",
                "data_uses": 4096, "HiFi_total_uses": 4204, "HiFi_energy": 8408,
                "failure_rule": "unchanged_actual_metadata_and_same_observation_ADJSCC_fallback",
                "stage_inference_variant": "original_forced_attention_checkpoint",
                "old_1500_plan_retained_not_declared_complete": True, "new_training_or_holdout": False,
                "bootstrap_resamples": 10000, "bootstrap_seed": 20260916,
                "DINO_for_selection": False, "incomplete_original_population_not_used_as_completed_stage": True,
                "calibration_profile_cases": [{"image_index": 0, "snr_db": 1.}, {"image_index": 500, "snr_db": 7.}, {"image_index": 999, "snr_db": 19.}],
                "calibration_noise_seed": 4101, "profile_timed_passes": 1, "profile_instrumented_passes": 1,
                "profile_order": "alternate_original_first_and_direct_first_by_case; no model parameter updates",
                "profile_difference": "only process_local_AttentionBlock.forward_calls_its_original__forward_without_checkpoint",
                "full_RGB_max_abs_tolerance": 2e-4, "input_gradient_rtol": 1e-4, "input_gradient_atol": 1e-6,
                "input_gradient_relative_L2_tolerance": 1e-4,
                "old_inference_and_vendor_files_modified": False, "no_automatic_switch_of_original_queue": True,
                "original_manifest_sha256": sha256(original_manifest), "registered_at": datetime.now().astimezone().isoformat()}
    write_json(output / "protocol.json", protocol)
    write_json(output / "development_inputs.json", selected)
    write_json(output / "original_plan_snapshot.json", json.loads((EXPERIMENT / "configs/protocol.json").read_text()))
    receipts = sorted((root / "development_hifi_001/frames").glob("*/frame.json"))
    records = [account_frame(json.loads(path.read_text())) for path in receipts]
    if len({row["frame_key"] for row in records}) != len(records):
        raise RuntimeError("duplicate committed transmissions")
    write_csv(output / "committed_frame_work.csv", records)
    write_csv(output / "committed_cost_by_snr.csv", summarize_cost(records))
    targets = {(index, snr, 2001) for index in indices for snr in protocol["snrs_db"]}
    reused = [row for row in records if (int(row["image_index"]), float(row["snr_db"]), int(row["seed"])) in targets]
    write_csv(output / "initial_reusable_frames.csv", reused)
    write_json(output / "registration.json", {"status": "STAGE20_AND_PROFILE_REGISTERED_NO_QUALITY_SELECTION",
               "original_completed_frames_at_snapshot": len(records), "reusable_stage_frames_at_snapshot": len(reused),
               "missing_stage_frames_at_snapshot": 100 - len(reused),
               "actual_sampler_calls_at_snapshot": sum(row["actual_sampler_calls"] for row in records),
               "extra_metadata_recoveries_at_snapshot": sum(row["metadata_extra_author_sampler_calls"] for row in records),
               "snapshot_scope": "committed_transmissions_only; excludes_inflight/warmup/interrupted_calls",
               "protocol_sha256": sha256(output / "protocol.json"), "GPU_used": False})
    print(json.dumps(protocol, indent=2))
    print(json.dumps(json.loads((output / "registration.json").read_text()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
