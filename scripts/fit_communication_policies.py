#!/usr/bin/env python3
"""Freeze the preregistered lookup rules once, using the complete calibration population only."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from var_comm.mode_policies import fit_actions, summarize_candidates
from var_comm.study import artifact_hashes, create_output, sha256, snapshot, verify_snapshot, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        print("PLAN ONLY: complete calibration, fixed three rules; no development or holdout fitting")
        return
    config_path = ROOT / "configs/communication_decision_study.json"
    config = json.loads(config_path.read_text())
    run = arguments.calibration_run.resolve()
    receipt = json.loads((run / "completion.json").read_text())
    if (receipt["status"] != "FIXED_MODE_MATRIX_COMPLETE" or receipt["population"] != "calibration" or
            receipt["mode"] != "full" or receipt["sources"] != 1000 or receipt["rows"] != 126000 or
            receipt["config_sha256"] != sha256(config_path)):
        raise RuntimeError("policy fitting requires the complete original1000 calibration grid, never smoke/development")
    verify_snapshot(receipt["source_hashes"])
    with (run / "per_frame.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if sha256(run / "per_frame.csv") != receipt["output_hashes"]["per_frame.csv"]:
        raise RuntimeError("completed calibration rows changed")
    population = json.loads((run / "population.json").read_text())
    expected = {(row["image_id"], family, mode, snr, seed) for row in population for family in config["coding_families"]
                for mode in config["prefix_modes"] for snr in config["snrs_db"] for seed in config["calibration_seeds"]}
    actual = {(row["image_id"], row["family"], int(row["mode"]), float(row["snr_db"]), int(row["seed"])) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise RuntimeError("calibration omitted/duplicated sources, failures, noise or modes")
    for row in rows:
        fields_and_payload = int(row["header_fields_correct"]) and int(row["payload_bits_correct"])
        if fields_and_payload and not (int(row["source_correct"]) and int(row["source_complete"])):
            raise RuntimeError("an exact source payload failed the lossless inverse; repair implementation rather than fit a weakened control")
        correct_acceptance = bool(fields_and_payload and int(row["header_accepted"]) and int(row["body_crc_accepted"]) and int(row["source_correct"]))
        if int(row["accepted_correct"]) != int(correct_acceptance):
            raise RuntimeError("calibration reliable-success accounting is inconsistent")
    summary = summarize_candidates(rows)
    actions, selected_family, primary = fit_actions(summary, config)
    output = create_output(arguments.output_dir)
    write_csv(output / "calibration_candidates.csv", summary)
    selections = [{"family": family, "rule": rule, "snr_db": float(snr), "mode": mode}
                  for family, rules in actions.items() for rule, choices in rules.items() for snr, mode in choices.items()]
    write_csv(output / "selected_modes.csv", selections)
    source_hashes = snapshot(output, [Path(__file__), ROOT / "src/var_comm/mode_policies.py", config_path,
        ROOT / "reports/whole_entropy_and_mode_policy_protocol_20260915.md"])
    policies = {"status": "CALIBRATION_POLICIES_FROZEN", "frozen_local": datetime.now().astimezone().isoformat(),
        "config_sha256": sha256(config_path), "calibration_receipt_sha256": sha256(run / "completion.json"),
        "calibration_rows_sha256": sha256(run / "per_frame.csv"), "calibration_sources": 1000, "calibration_frames": len(rows),
        "actions": actions, "global_selected_family": selected_family, "global_selected_rule": "quality",
        "quality_primary_calibration_lpips": primary, "reliability_BLER_target": .10, "PSNR_guard_db": .25,
        "receiver_observables_for_mode_selection": "shared_nominal_SNR_only",
        "source_packet_success": "header_fields_payload_bits_and_source_prefix_correct_with_CRC_acceptance",
        "development_used_for_fitting": False, "holdout_accessed": False, "DINO_used_for_new_selection": False,
        "DINO_used_for_historical_digital_development": True, "source_hashes": source_hashes}
    write_json(output / "policies.json", policies)
    write_json(output / "completion.json", {"status": "CALIBRATION_POLICY_FIT_COMPLETE", "frozen_local": policies["frozen_local"],
        "policies_sha256": sha256(output / "policies.json"), "source_hashes": source_hashes,
        "research_goal_complete": False, "output_hashes": artifact_hashes(output)})
    print(json.dumps({"actions": actions, "selected_family": selected_family, "calibration_primary_lpips": primary}, indent=2))


if __name__ == "__main__":
    main()
