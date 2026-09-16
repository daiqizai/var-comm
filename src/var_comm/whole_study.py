"""Registered image-paired summaries for the fixed whole-frame receiver prototype."""

from collections import defaultdict

import numpy as np

from .study import paired_interval

METRICS = ("psnr_db", "lpips_alex", "dino_cosine")
PROFILE_FIELDS = ("receiver_seconds", "auxiliary_seconds", "prior_seconds", "prior_batch_calls", "prior_prefix_count",
                  "candidates_checked", "channel_products", "backward_edges", "deviation_edges", "heap_comparisons",
                  "traceback_layers", "native_owned_peak_bytes", "gpu_peak_allocated_bytes")


def aggregate(rows, config):
    expected = {(index, snr, seed, arm) for index in range(config["target_count"]) for snr in config["snr_db"]
                for seed in config["noise_seeds"] for arm in config["arms"]}
    lookup = {(row["image_index"], row["snr_db"], row["seed"], row["arm"]): row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise RuntimeError("incomplete paired whole-frame grid")
    summary = []
    for snr in config["snr_db"]:
        for arm in config["arms"]:
            selected = [row for row in rows if row["snr_db"] == snr and row["arm"] == arm]
            record = {"snr_db": snr, "arm": arm, "transmissions": len(selected), "source_images": config["target_count"]}
            for metric in (*METRICS, "source_bler", "correct_accepted", "correct_m9_accepted", "aux_triggered", "new_candidate_accepted", "image_condition_changed"):
                record[metric] = float(np.mean([row[metric] for row in selected]))
            for metric in ("undetected_error", "new_false_acceptance", "header_false_acceptance", "VAR_above_time_allowance", "time_control_capacity_limited"):
                record[metric + "_count"] = int(sum(row[metric] for row in selected))
            covered = [row["true_prefix_covered"] for row in selected if row["true_prefix_covered"] != ""]
            record["true_prefix_covered_given_trigger"] = float(np.mean(covered)) if covered else None
            for metric in PROFILE_FIELDS:
                values = np.asarray([row[metric] for row in selected])
                record["mean_" + metric] = float(values.mean())
                record["p95_" + metric] = float(np.percentile(values, 95))
                record["max_" + metric] = float(values.max())
            summary.append(record)
    comparisons = []
    controls = ["whole_m9_ml", "whole_m9_list65", "whole_m9_list_time", "whole_m9_hyp_uniform", "whole_m8", "whole_adaptive"]
    pairs = [("whole_m9_var", control) for control in controls]
    pairs += [("whole_adaptive_var", "whole_adaptive"), ("whole_m9_list65", "whole_m9_ml"), ("whole_m9_list_time", "whole_m9_ml")]
    intervals = [config["gate"]["primary_snr_db"], *[[snr] for snr in config["snr_db"]]]
    for interval in intervals:
        for method, control in pairs:
            for metric in METRICS:
                differences = [np.mean([lookup[index, snr, seed, method][metric] - lookup[index, snr, seed, control][metric]
                                        for snr in interval for seed in config["noise_seeds"]]) for index in range(config["target_count"])]
                result = paired_interval(differences, config["bootstrap"]["seed"], config["bootstrap"]["resamples"])
                comparisons.append({"snr_db": "+".join(str(value) for value in interval), "method": method, "control": control,
                                    "metric": metric, "delta": result["gain"], "ci_low": result["ci_low"], "ci_high": result["ci_high"]})
    primary_name = "+".join(str(value) for value in config["gate"]["primary_snr_db"])
    paired = {(row["method"], row["control"], row["metric"]): row for row in comparisons if row["snr_db"] == primary_name}
    gates = {}
    for method, control in [("whole_m9_var", name) for name in config["gate"]["mechanism_controls"]] + [("whole_adaptive_var", "whole_adaptive")]:
        baseline = np.mean([row["lpips_alex"] for row in rows if row["arm"] == control and row["snr_db"] in config["gate"]["primary_snr_db"]])
        perceptual = paired[method, control, "lpips_alex"]
        checks = {"lpips_magnitude": -perceptual["delta"] >= config["gate"]["minimum_relative_lpips_gain"] * baseline,
                  "lpips_interval": perceptual["ci_high"] < 0,
                  "psnr_guard": paired[method, control, "psnr_db"]["delta"] >= -config["gate"]["maximum_psnr_drop_db"],
                  "dino_guard": paired[method, control, "dino_cosine"]["delta"] >= -config["gate"]["maximum_dino_drop"]}
        gates[method + "__" + control] = {"passed": bool(all(checks.values())), "checks": {name: bool(value) for name, value in checks.items()}}
    new_false = sum(row["new_false_acceptance"] for row in rows if row["arm"] == "whole_m9_var")
    fairness = not any(row["VAR_above_time_allowance"] or row["time_control_capacity_limited"] for row in rows if row["arm"] == "whole_m9_var")
    mechanism = new_false <= config["gate"]["maximum_new_VAR_false_acceptances"] and all(gates["whole_m9_var__" + control]["passed"] for control in config["gate"]["mechanism_controls"])
    adaptive = gates["whole_adaptive_var__whole_adaptive"]["passed"]
    status = ("STOP_THIS_FINITE_PREFIX_VAR_RECEIVER" if not mechanism else
              "MECHANISM_GAIN_COMPUTE_COMPARISON_INCOMPLETE" if not fairness else
              "DEVELOPMENT_GAIN_ENTROPY_COMPARISON_PENDING" if adaptive else "MECHANISM_ONLY_NO_ADAPTIVE_SYSTEM_GAIN")
    targets = {}
    for arm in config["arms"]:
        passed = [row["snr_db"] for row in summary if row["arm"] == arm and row["lpips_alex"] <= config["quality_target"]["maximum_lpips"]
                  and row["psnr_db"] >= config["quality_target"]["minimum_psnr_db"] and row["dino_cosine"] >= config["quality_target"]["minimum_dino_cosine"]]
        targets[arm] = min(passed) if passed else None
    changes = []
    for snr in config["snr_db"]:
        for arm in ("whole_m9_list65", "whole_m9_list_time", "whole_m9_hyp_uniform", "whole_m9_var"):
            selected = [row for row in rows if row["snr_db"] == snr and row["arm"] == arm]
            counts = defaultdict(int)
            for row in selected:
                if row["image_condition_changed"]:
                    before = lookup[row["image_index"], snr, row["seed"], "whole_m9_ml"]
                    difference = row["lpips_alex"] - before["lpips_alex"]
                    counts["lpips_improved" if difference < 0 else "lpips_worsened" if difference > 0 else "lpips_equal"] += 1
            changes.append({"snr_db": snr, "arm": arm, "transmissions": len(selected),
                            "new_acceptances": sum(row["new_candidate_accepted"] for row in selected),
                            "new_false_acceptances": sum(row["new_false_acceptance"] for row in selected),
                            "changed_images": sum(row["image_condition_changed"] for row in selected),
                            **{name: counts[name] for name in ("lpips_improved", "lpips_worsened", "lpips_equal")}})
    return summary, comparisons, changes, {"status": status, "gates": gates, "new_VAR_false_acceptances": new_false,
                                           "time_control_has_no_recorded_budget_disadvantage": bool(fairness), "quality_target_first_SNR": targets,
                                           "scope": "development_prototype_no_strong_whole_entropy_or_unseen_test_claim"}
