"""CPU-only, source-paired summaries for complete frozen comparison points."""

from collections import defaultdict

import numpy as np


METRICS = ("psnr_db", "ssim", "lpips", "dino")
SNRS = (1., 4., 7., 13., 19.)
SEEDS = (2001, 2002, 2003)
SCOPES = [(f"snr_{snr:g}", (snr,)) for snr in SNRS] + [
    ("primary_1_4_7", SNRS[:3]), ("high_13_19", SNRS[3:])]


def group_key(row):
    return row["method"], row["protocol"], int(row["complex_uses"])


def validate_groups(rows, sources):
    expected = {(index, snr, seed) for index in sources for snr in SNRS for seed in SEEDS}
    grouped = defaultdict(list)
    for row in rows:
        index = int(row["image_index"])
        source = sources[index]
        if row["image_id"] != source["image_id"] or row["source_pixels_sha256"] != source["source_pixels_sha256"]:
            raise ValueError("different source pixels or identities cannot enter a paired comparison")
        if not np.isfinite([float(row[name]) for name in METRICS]).all():
            raise ValueError("nonfinite quality metric")
        if abs(float(row["total_energy"]) - 2 * int(row["complex_uses"])) > .02:
            raise ValueError("transmitted energy disagrees with the charged physical budget")
        grouped[group_key(row)].append(row)
    for key, selected in grouped.items():
        actual = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"])) for row in selected}
        if actual != expected or len(selected) != len(expected):
            raise ValueError(f"incomplete or duplicated development population: {key}")
    return grouped


def source_values(rows, support, indices):
    grouped = defaultdict(list)
    for row in rows:
        if float(row["snr_db"]) in support:
            grouped[int(row["image_index"])].append([float(row[name]) for name in METRICS])
    if any(len(grouped[index]) != len(support) * len(SEEDS) for index in indices):
        raise ValueError("every source needs the same SNR/noise support")
    return np.array([np.mean(grouped[index], axis=0) for index in indices])


def summarize(rows, sources, resamples=10000, seed=20260916):
    grouped = validate_groups(rows, sources)
    indices = sorted(sources)
    draws = np.random.default_rng(seed).integers(len(indices), size=(resamples, len(indices)))
    summary, paired, per_source = [], [], []
    values = {}
    for key, selected in sorted(grouped.items()):
        for scope, support in SCOPES:
            matrix = source_values(selected, support, indices)
            values[key, scope] = matrix
            intervals = np.quantile(matrix[draws].mean(axis=1), (.025, .975), axis=0)
            record = {"method": key[0], "protocol": key[1], "complex_uses": key[2],
                      "total_energy": 2 * key[2], "scope": scope, "source_images": len(indices),
                      "transmissions": len(indices) * len(support) * len(SEEDS)}
            for metric_index, name in enumerate(METRICS):
                record[name] = float(matrix[:, metric_index].mean())
                record[name + "_ci_low"] = float(intervals[0, metric_index])
                record[name + "_ci_high"] = float(intervals[1, metric_index])
            summary.append(record)
            for source_index, index in enumerate(indices):
                per_source.append({"method": key[0], "protocol": key[1], "complex_uses": key[2],
                                   "scope": scope, "image_index": index, "image_id": sources[index]["image_id"],
                                   **dict(zip(METRICS, matrix[source_index].tolist()))})
    paid = sorted(key for key in grouped if key[1] == "common_paid_information")
    contrasts = []
    for method in paid:
        for control in paid:
            if method == control:
                continue
            if control[0] in ("raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc", "wetok_r3"):
                contrasts.append((method, control))
            elif method[0].startswith(("raw_adaptive_", "arithmetic_adaptive_")) and method[2] == control[2]:
                contrasts.append((method, control))
    for method, control in contrasts:
        for scope, _support in SCOPES:
            difference = values[method, scope] - values[control, scope]
            intervals = np.quantile(difference[draws].mean(axis=1), (.025, .975), axis=0)
            for metric_index, name in enumerate(METRICS):
                paired.append({"method": method[0], "method_complex_uses": method[2], "control": control[0],
                               "control_complex_uses": control[2], "scope": scope, "metric": name,
                               "equal_resource": method[2] == control[2], "source_images": len(indices),
                               "mean_method_minus_control": float(difference[:, metric_index].mean()),
                               "ci_low": float(intervals[0, metric_index]), "ci_high": float(intervals[1, metric_index]),
                               "interpretation": "system_comparison_not_architecture_causality"})
    return summary, paired, per_source
