#!/usr/bin/env python3
"""Plot frozen quality curves and diagnose the finite-prefix recovery ceiling."""

import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import numpy as np

from var_comm.study import artifact_hashes, create_output, sha256, snapshot, verify_artifacts, write_csv, write_json


def main():
    run = ROOT / "outputs/VAR-WHOLE-FRAME-PRIOR-001"
    receipt = verify_artifacts(run, "completion.json")
    output = create_output(ROOT / "outputs/VAR-WHOLE-FRAME-SUMMARY-001")
    sources = snapshot(output, [Path(__file__)])
    with (run / "per_frame.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"]), row["arm"]): row for row in rows}
    with np.load(ROOT / "outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001/source_tokens.npz", allow_pickle=False) as cache:
        tokens = dict(zip(cache["image_ids"].tolist(), cache["tokens"].copy()))
    diagnostics = {snr: {"snr_db": snr, "transmissions": 300, "ML_correct_accepted": 0, "triggered": 0,
                         "true_prefix_in_list": 0, "true_branch_uniform_correct": 0, "true_branch_VAR_correct": 0,
                         "VAR_correct_accepted": 0} for snr in (4.0, 5.0, 6.0, 7.0)}
    for index in range(100):
        directory = run / "images" / f"{index:03d}"
        records = json.loads((directory / "receivers.json").read_text())
        with np.load(directory / "candidates.npz", allow_pickle=False) as arrays:
            for record in records:
                snr, seed = record["snr_db"], record["seed"]
                row = lookup[index, snr, seed, "whole_m9_var"]
                original = lookup[index, snr, seed, "whole_m9_ml"]
                entry = diagnostics[snr]
                entry["ML_correct_accepted"] += int(original["correct_accepted"])
                entry["VAR_correct_accepted"] += int(row["correct_accepted"])
                entry["triggered"] += int(record["trigger"])
                if not record["trigger"] or not int(row["true_prefix_covered"]):
                    continue
                require_label = record["receivers"]["whole_m9_ml"]["label"]
                assert int(row["header_false_acceptance"]) == 0 and require_label is not None
                rank = int(row["true_prefix_rank"])
                entry["true_prefix_in_list"] += 1
                values = tokens[row["image_id"]][255:424]
                truth = ((values[:, None] >> np.arange(11, -1, -1)) & 1).astype(np.uint8).ravel()
                for family in ("uniform", "VAR"):
                    listed = record["details"]["assisted"][family]["branches"][rank]["list"]
                    if any(listed["accepted"]):
                        suffix = np.unpackbits(arrays[listed["array_key"]][-1])[:2028]
                        entry["true_branch_" + family + "_correct"] += int(np.array_equal(suffix, truth))
    for entry in diagnostics.values():
        entry["fixed_list_correct_frame_ceiling"] = entry["ML_correct_accepted"] + entry["true_prefix_in_list"]
        entry["true_prefix_coverage_given_trigger"] = entry["true_prefix_in_list"] / entry["triggered"] if entry["triggered"] else None
        assert entry["VAR_correct_accepted"] <= entry["fixed_list_correct_frame_ceiling"]
    write_csv(output / "conditional_coverage.csv", list(diagnostics.values()))
    with (run / "summary.csv").open() as handle:
        summary = {(row["arm"], float(row["snr_db"])): row for row in csv.DictReader(handle)}
    figure, axes = pyplot.subplots(1, 3, figsize=(15, 4.4))
    methods = [("whole_m8", "Fixed m8", "#777777"), ("whole_m9_ml", "m9 ML", "#377eb8"),
               ("whole_m9_list65", "m9 CRC list65", "#4daf4a"), ("whole_m9_list_time", "m9 large CRC list", "#984ea3"),
               ("whole_m9_var", "m9 VAR hypotheses", "#e41a1c")]
    for axis, metric, title, target in zip(axes, ("psnr_db", "lpips_alex", "dino_cosine"),
                                         ("PSNR (higher is better)", "LPIPS (lower is better)", "DINO (higher is better)"), (20.5, 0.15, 0.90)):
        for method, label, color in methods:
            axis.plot([4, 5, 6, 7], [float(summary[method, snr][metric]) for snr in (4, 5, 6, 7)], marker="o", label=label, color=color, linewidth=1.7)
        axis.axhline(target, color="black", linestyle=":", linewidth=1, label="Registered quality threshold")
        axis.set_title(title)
        axis.set_xlabel("SNR (dB)")
        axis.set_xticks([4, 5, 6, 7])
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Frozen whole-frame prototype: 100 development images × 3 noise seeds; 3060 complex uses")
    figure.tight_layout(rect=(0, 0.16, 1, 0.93))
    figure.savefig(output / "quality_curves.png", dpi=180)
    pyplot.close(figure)
    write_json(output / "completion.json", {"status": "FROZEN_SUMMARY_COMPLETE", "input_completion_sha256": sha256(run / "completion.json"),
                                            "source_hashes": sources, "primary_status": receipt["status"], "output_hashes": artifact_hashes(output)})
    print(list(diagnostics.values()), flush=True)


if __name__ == "__main__":
    main()
