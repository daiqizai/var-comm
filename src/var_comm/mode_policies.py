"""Closed-set calibration decisions; deployment observes only the shared nominal SNR."""

from collections import defaultdict

import numpy as np


RULES = ("reliability", "quality", "goodput")
CONTEXT_FIELDS = ("budget", "renderer", "decoder_sha", "protocol_id")


def _context(row, explicit=None):
    """Return the experiment identity used for one policy row.

    Policy candidates from different budgets, renderers, decoders, or
    protocols must never share a group.  An explicit context is useful for
    callers that keep the identity beside (rather than inside) a row, but it
    is still checked against any fields present on the row.
    """
    if explicit is not None:
        if not isinstance(explicit, dict) or any(field not in explicit for field in CONTEXT_FIELDS):
            raise ValueError(f"context must provide {', '.join(CONTEXT_FIELDS)}")
        expected = tuple(explicit[field] for field in CONTEXT_FIELDS)
    else:
        missing = [field for field in CONTEXT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"policy row missing experiment context: {', '.join(missing)}")
        expected = tuple(row[field] for field in CONTEXT_FIELDS)
    for field, value in zip(CONTEXT_FIELDS, expected):
        if field in row and row[field] != value:
            raise ValueError(f"row context differs from requested {field}")
    return expected


def summarize_candidates(rows, *, context=None):
    groups = defaultdict(list)
    for row in rows:
        if row["population"] != "calibration":
            raise ValueError("mode fitting may only use calibration rows")
        identity = _context(row, context)
        groups[identity, row["family"], float(row["snr_db"]), int(row["mode"])].append(row)
    result = []
    for (identity, family, snr, mode), selected in sorted(
            groups.items(), key=lambda item: (tuple(str(value) for value in item[0][0]), item[0][1], item[0][2], item[0][3])):
        payloads = {int(row["raw_payload_bits"]) for row in selected}
        if len(payloads) != 1:
            raise ValueError("a mode changed its original source payload")
        success = np.array([int(row["accepted_correct"]) for row in selected], dtype=bool)
        lpips = np.array([float(row["lpips"]) for row in selected])
        psnr = np.array([float(row["psnr_db"]) for row in selected])
        accepted_header = np.array([bool(int(row["header_accepted"])) for row in selected])
        accepted_body = np.array([bool(int(row["body_crc_accepted"])) for row in selected])
        raw_bits = next(iter(payloads))
        result.append({**dict(zip(CONTEXT_FIELDS, identity)), "family": family, "snr_db": snr, "mode": mode, "frames": len(selected),
            "source_images": len({row["image_id"] for row in selected}), "raw_payload_bits": raw_bits,
            "accepted_correct_probability": float(success.mean()), "source_packet_BLER": float(1 - success.mean()),
            "source_index_goodput": float(raw_bits * success.mean()), "psnr_db": float(psnr.mean()), "lpips": float(lpips.mean()),
            "dino_report_only": float(np.mean([float(row["dino"]) for row in selected])),
            "actual_payload_bits_mean": float(np.mean([int(row["actual_payload_bits"]) for row in selected])),
            "header_failure_probability": float(np.mean([not int(row["header_accepted"]) for row in selected])),
            "no_accepted_body_probability": float((~accepted_body).mean()),
            "body_CRC_failure_given_header_accept": float((~accepted_body[accepted_header]).mean()) if accepted_header.any() else "",
            "success_lpips_mean": float(lpips[success].mean()) if success.any() else "",
            "failure_lpips_mean": float(lpips[~success].mean()) if (~success).any() else "",
            "failure_contribution_to_mean_lpips": float(np.where(success, 0, lpips).mean()),
            "failure_frames": int((~success).sum())})
    return result


def choose_modes(candidates, bler_target=.10, psnr_drop=.25, tie_tolerance=1e-12):
    if len(candidates) != 3 or {int(row["mode"]) for row in candidates} != {7, 8, 9}:
        raise ValueError("exactly the fixed m7/m8/m9 candidate set is required")
    eligible = [row for row in candidates if row["source_packet_BLER"] <= bler_target + 1e-12]
    reliable = max(eligible, key=lambda row: row["mode"]) if eligible else max(candidates,
        key=lambda row: (row["accepted_correct_probability"], -row["mode"]))
    admissible = [row for row in candidates if row["psnr_db"] >= reliable["psnr_db"] - psnr_drop - 1e-12]
    best_lpips = min(row["lpips"] for row in admissible)
    quality = min((row for row in admissible if row["lpips"] <= best_lpips + tie_tolerance), key=lambda row: row["mode"])
    best_goodput = max(row["source_index_goodput"] for row in candidates)
    goodput = min((row for row in candidates if row["source_index_goodput"] >= best_goodput - tie_tolerance), key=lambda row: row["mode"])
    return {"reliability": int(reliable["mode"]), "quality": int(quality["mode"]), "goodput": int(goodput["mode"])}


def fit_actions(summary, config, *, context=None):
    if not summary:
        raise ValueError("cannot fit policy actions from an empty summary")
    identities = {_context(row, context) for row in summary}
    if len(identities) != 1:
        raise ValueError("mode fitting cannot combine multiple experiment contexts")
    lookup = {}
    for row in summary:
        key = (row["family"], float(row["snr_db"]), int(row["mode"]))
        if key in lookup:
            raise ValueError(f"duplicate policy candidate for {key}")
        lookup[key] = row
    actions = {family: {rule: {} for rule in RULES} for family in config["coding_families"]}
    for family in config["coding_families"]:
        for snr in config["snrs_db"]:
            candidates = [lookup[family, snr, mode] for mode in config["prefix_modes"]]
            selected = choose_modes(candidates, config["reliability_BLER_target"], config["quality_max_PSNR_drop_db"], config["quality_tie_tolerance"])
            for rule, mode in selected.items():
                actions[family][rule][str(float(snr))] = mode
    primary = {family: float(np.mean([lookup[family, snr, actions[family]["quality"][str(float(snr))]]["lpips"]
        for snr in config["primary_snrs_db"]])) for family in config["coding_families"]}
    best = min(primary.values())
    family = next(name for name in config["coding_families"] if primary[name] <= best + config["quality_tie_tolerance"])
    return actions, family, primary


def selected_mode(actions, family, rule, snr):
    support = sorted(float(value) for value in actions[family][rule])
    closest = min(support, key=lambda value: (abs(float(snr) - value), value))
    return int(actions[family][rule][str(closest)])
