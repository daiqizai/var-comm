"""Image-only retention of frozen receiver candidates, with paired diagnostics."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import numpy as np

from .study import paired_interval

SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
METRICS = ("psnr_db", "lpips_alex", "dino_cosine")
STAGES = ("header_failure", "B0_failure", "B7_failure", "B8_failure", "B9_failure", "m9_accepted")


def legal_tokens(values, count):
    tokens = np.asarray(values)
    return (tokens.shape == (count,) and np.issubdtype(tokens.dtype, np.integer)
            and bool(np.all((tokens >= 0) & (tokens < 4096))))


def image_only_state(record):
    trusted = deepcopy(record["output_prefix"])
    if len(trusted) != record["last_accepted_scale"]:
        raise ValueError("grouped output prefix is not the frozen trusted state")
    if any(not legal_tokens(tokens, SIZES[index] ** 2) for index, tokens in enumerate(trusted)):
        raise ValueError("malformed trusted prefix")
    result = {"label": record["label"], "trusted_prefix": trusted, "render_prefix": deepcopy(trusted),
              "trusted_scales": len(trusted), "render_scales": len(trusted), "candidate_available": False,
              "retention_applied": False, "failure_group": None, "reason": "all_groups_accepted"}
    header_ok = record["header"]["accepted"] and record["header"]["mode"] == 9
    if not header_ok:
        if record["label"] is not None or trusted or record["events"]:
            raise ValueError("failed header has downstream receiver information")
        result.update(failure_stage="header_failure", reason="unchanged_header_failure")
        return result
    if record["label"] != record["header"]["label"] or not 0 <= record["label"] < 1000:
        raise ValueError("output branch did not use the received header label")
    accepted, failure = [], None
    for position, event in enumerate(record["events"]):
        if event["group"] != position or failure is not None:
            raise ValueError("receiver continued after the first failure or skipped a block")
        if not event["accepted"]:
            failure = event
            continue
        group = event["group"]
        tokens = event["recovered_tokens"]
        if group == 0:
            if not legal_tokens(tokens, 91):
                raise ValueError("invalid accepted base candidate")
            accepted = [part.tolist() for part in np.split(np.asarray(tokens), np.cumsum([size ** 2 for size in SIZES[:6]])[:-1])]
        else:
            if not legal_tokens(tokens, SIZES[group + 5] ** 2):
                raise ValueError("invalid accepted fine candidate")
            accepted.append(list(tokens))
    if accepted != trusted:
        raise ValueError("trusted state differs from accepted received source blocks")
    if failure is None:
        if len(trusted) != 9 or len(record["events"]) != 4:
            raise ValueError("incomplete trace without a failure")
        result["failure_stage"] = "m9_accepted"
        return result
    group = failure["group"]
    stage = "B0_failure" if group == 0 else f"B{group + 6}_failure"
    result.update(failure_stage=stage, failure_group=group, reason="no_complete_legal_candidate")
    expected = 91 if group == 0 else SIZES[group + 5] ** 2
    tokens = failure.get("recovered_tokens", [])
    if not legal_tokens(tokens, expected):
        return result
    if group == 0:
        rendered = [part.tolist() for part in np.split(np.asarray(tokens), np.cumsum([size ** 2 for size in SIZES[:6]])[:-1])]
    else:
        rendered = deepcopy(trusted) + [list(tokens)]
    result.update(render_prefix=rendered, render_scales=len(rendered), candidate_available=True,
                  retention_applied=True, reason="first_failed_hard_candidate_for_image_only")
    return result


def assert_empty_attention(var):
    if any(block.attn.caching or block.attn.cached_k is not None or block.attn.cached_v is not None for block in var.blocks):
        raise RuntimeError("render-only model has a live attention cache")


def render_from_scratch(vae, var, prefix, label, device):
    from .progressive import complete_image

    assert_empty_attention(var)
    pixels = complete_image(vae, var, deepcopy(prefix), label, device)
    assert_empty_attention(var)
    return pixels


def choose_oracle(discard, retain):
    chosen = retain if retain["lpips_alex"] < discard["lpips_alex"] else discard
    result = dict(chosen)
    result.update(arm=discard["family"] + "_oracle_lpips", policy="oracle_lpips", deployable=0,
                  oracle_selected_policy=chosen["policy"])
    return result


def summarize(rows, config):
    lookup = {(row["image_index"], row["seed"], row["arm"]): row for row in rows}
    arms = [family + "_" + policy for family in config["grouped_arms"] for policy in config["output_policies"]] + ["whole_m9"]
    target_count, seeds = config["target_count"], config["noise_seeds"]
    expected = {(index, seed, arm) for index in range(target_count) for seed in seeds for arm in arms}
    if set(lookup) != expected or len(lookup) != len(rows):
        raise ValueError("incomplete or duplicated paired replay grid")
    summary, means = [], {}
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        entry = {"arm": arm, "deployable": selected[0]["deployable"], "transmissions": len(selected),
                 "source_images": target_count, "complex_uses": config["complex_uses"]}
        for metric in METRICS:
            means[arm, metric] = np.array([np.mean([lookup[index, seed, arm][metric] for seed in seeds]) for index in range(target_count)])
            entry[metric] = float(means[arm, metric].mean())
        entry["correct_m9_accepted"] = float(np.mean([row["correct_m9_accepted"] for row in selected]))
        summary.append(entry)
    contrasts = [(arm, control) for arm in arms for control in ("whole_m9", "group_entropy_discard", "group_entropy_retain") if arm != control]
    contrasts.extend((family + "_retain", family + "_discard") for family in config["grouped_arms"])
    contrasts.extend((family + "_oracle_lpips", family + "_discard") for family in config["grouped_arms"])
    contrasts.extend(("group_var_" + policy, "group_ml_" + policy) for policy in ("discard", "retain"))
    comparisons = []
    boot = config["bootstrap"]
    for method, control in dict.fromkeys(contrasts):
        for metric in METRICS:
            interval = paired_interval(means[method, metric] - means[control, metric], boot["seed"], boot["resamples"])
            comparisons.append({"method": method, "control": control, "metric": metric, "source_images": target_count,
                                "delta": interval["gain"], "ci_low": interval["ci_low"], "ci_high": interval["ci_high"]})
    for metric in METRICS:
        differences = ((means["group_var_retain", metric] - means["group_var_discard", metric])
                       - (means["group_ml_retain", metric] - means["group_ml_discard", metric]))
        interval = paired_interval(differences, boot["seed"], boot["resamples"])
        comparisons.append({"method": "VAR_retention_change", "control": "ML_retention_change", "metric": metric,
                            "source_images": target_count, "delta": interval["gain"], "ci_low": interval["ci_low"], "ci_high": interval["ci_high"]})
    strata = []
    for family in config["grouped_arms"]:
        for stage in STAGES:
            selected = [row for row in rows if row["family"] == family and row["policy"] == "discard" and row["failure_stage"] == stage]
            entry = {"family": family, "failure_stage": stage, "transmissions": len(selected),
                     "source_images": len({row["image_index"] for row in selected}),
                     "available_candidates": sum(row["candidate_available"] for row in selected),
                     "lpips_improved": 0, "lpips_worsened": 0, "lpips_unchanged": 0}
            bers = [row["failed_candidate_true_ber"] for row in selected if row["failed_candidate_true_ber"] != ""]
            entry["mean_failed_candidate_true_ber_diagnostic"] = float(np.mean(bers)) if bers else None
            for metric in METRICS:
                contributions = np.zeros(target_count)
                discarded, retained = [], []
                for row in selected:
                    after = lookup[row["image_index"], row["seed"], family + "_retain"]
                    discarded.append(row[metric])
                    retained.append(after[metric])
                    contributions[row["image_index"]] += (after[metric] - row[metric]) / len(seeds)
                    if metric == "lpips_alex":
                        category = "improved" if after[metric] < row[metric] else ("worsened" if after[metric] > row[metric] else "unchanged")
                        entry["lpips_" + category] += 1
                interval = paired_interval(contributions, boot["seed"], boot["resamples"])
                entry["discard_" + metric] = float(np.mean(discarded)) if discarded else None
                entry["retain_" + metric] = float(np.mean(retained)) if retained else None
                entry["delta_" + metric] = float(np.mean(np.asarray(retained) - discarded)) if discarded else None
                entry["overall_contribution_" + metric] = interval["gain"]
                entry["contribution_ci_low_" + metric] = interval["ci_low"]
                entry["contribution_ci_high_" + metric] = interval["ci_high"]
            strata.append(entry)
    table = {row["arm"]: row for row in summary}
    initial = table["group_var_discard"]["lpips_alex"] - table["whole_m9"]["lpips_alex"]
    retained_gap = table["group_var_retain"]["lpips_alex"] - table["whole_m9"]["lpips_alex"]
    oracle_gap = table["group_var_oracle_lpips"]["lpips_alex"] - table["whole_m9"]["lpips_alex"]
    status = ("STOP_TWO_CANDIDATE_SELECTOR_FIXED_M9" if oracle_gap > 0 else
              "ORACLE_HEADROOM_ONLY_NO_DEPLOYABLE_GAIN" if retained_gap > 0 else
              "RETAIN_MEAN_GAP_CLOSED_CHECK_FAIR_CONTROLS")
    failed = [row for row in rows if row["arm"] == "group_var_discard" and not row["correct_m9_accepted"]]
    available = [row for row in failed if row["candidate_available"]]
    decision = {"status": status, "original_VAR_minus_whole_lpips": initial,
                "retained_VAR_minus_whole_lpips": retained_gap, "oracle_VAR_minus_whole_lpips": oracle_gap,
                "fraction_original_mean_gap_closed_by_retain": (initial - retained_gap) / initial if initial else None,
                "fraction_original_mean_gap_closed_by_oracle": (initial - oracle_gap) / initial if initial else None,
                "VAR_not_correct_m9_transmissions": len(failed), "VAR_available_failure_candidates": len(available),
                "required_mean_failure_LPIPS_reduction": initial * target_count * len(seeds) / len(available) if available else None,
                "inference_scope": "development_only_two_frozen_outputs_no_equivalence_claim"}
    return summary, comparisons, strata, decision
