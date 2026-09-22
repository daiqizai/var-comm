"""Common endpoints and source-paired statistics for frozen-system measurements."""

import time

import numpy as np

from .study import paired_interval


ARMS = ("whole_m7", "whole_m8", "whole_m9", "whole_adaptive", "r2__full_grid_innovation",
        "perceptual_deepjscc", "wetok_8PSK_FEC")
SNRS = (1.0, 4.0, 7.0, 13.0, 19.0)
SOURCE_INDICES = tuple(np.rint(np.linspace(0, 99, 32)).astype(int).tolist())


def validate_config(config):
    if (tuple(config["arms"]) != ARMS or tuple(config["snrs_db"]) != SNRS or
            config["source_count"] != 32 or config["noise_seed"] != 2001 or config["repeats"] != 3 or
            config["smoke_source_indices"] != [0, 99] or config["smoke_repeats"] != 1 or
            config["total_complex_uses"] != 3060 or config["total_energy"] != 6120 or
            config["batch_size"] != 1 or config["GPU_index"] != 0 or
            any(config[key] for key in ("new_training", "new_architecture", "new_parameter_search", "new_holdout"))):
        raise ValueError("frozen timing contract changed")


def measurement_order(frame, repeat):
    if frame < 0 or repeat < 0:
        raise ValueError("negative measurement position")
    shift = (frame * 3 + repeat) % len(ARMS)
    return ARMS[shift:] + ARMS[:shift]


def timed_call(callback, synchronize, clock=time.perf_counter):
    synchronize()
    started = clock()
    value = callback()
    synchronize()
    elapsed = clock() - started
    if not np.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("invalid synchronized elapsed time")
    return value, elapsed


def compare_arrays(actual, expected, tolerance, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.isfinite(actual).all():
        raise RuntimeError(f"{name} changed shape, dtype or finiteness")
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    if error > tolerance:
        raise RuntimeError(f"{name} changed: max error {error} exceeds {tolerance}")
    return error, bool(np.array_equal(actual, expected))


def single_waveform(value):
    value = np.asarray(value)
    if value.shape == (1, 3060, 2):
        return value[0]
    if value.shape != (3060, 2):
        raise ValueError("archive is not one paid complex waveform")
    return value


def validate_rows(rows, indices, repeats):
    expected = {(index, snr, arm, repeat) for index in indices for snr in SNRS for arm in ARMS for repeat in range(repeats)}
    lookup = {(int(row["image_index"]), float(row["snr_db"]), row["arm"], int(row["repeat"])): row for row in rows}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("timing matrix is incomplete, duplicated or changed")
    for position, index in enumerate(indices):
        identities = {row["image_id"] for key, row in lookup.items() if key[0] == index}
        if len(identities) != 1:
            raise ValueError("timing source identity changed")
        for snr_index, snr in enumerate(SNRS):
            for repeat in range(repeats):
                order = measurement_order(position * len(SNRS) + snr_index, repeat)
                for arm in ARMS:
                    row = lookup[index, snr, arm, repeat]
                    if (int(row["order_index"]) != order.index(arm) or int(row["noise_seed"]) != 2001 or
                            int(row["total_complex_uses"]) != 3060 or
                            abs(float(row["total_energy"]) - 6120) > .02 or
                            row["timing_scope"] != "CPU_to_CPU_contiguous_TX_and_RX"):
                        raise ValueError("timing order, endpoint or physical resource changed")
                    if any(not np.isfinite(float(row[key])) or float(row[key]) <= 0 for key in ("TX_seconds", "RX_seconds")):
                        raise ValueError("invalid latency record")
    return lookup


def summarize(rows, config, indices, repeats):
    lookup = validate_rows(rows, indices, repeats)
    scopes = [(str(int(snr)), (snr,)) for snr in SNRS] + [("primary_1_4_7", SNRS[:3]), ("all_5_snrs", SNRS)]
    metrics = ("TX_seconds", "RX_seconds", "processing_sum_seconds")
    summary, paired = [], []
    contrasts = [("whole_adaptive", arm) for arm in ARMS if arm != "whole_adaptive"]
    contrasts += [("r2__full_grid_innovation", "perceptual_deepjscc"),
                  ("r2__full_grid_innovation", "wetok_8PSK_FEC")]
    for scope, snrs in scopes:
        values = {}
        for arm in ARMS:
            selected = [lookup[index, snr, arm, repeat] for index in indices for snr in snrs for repeat in range(repeats)]
            record = {"scope": scope, "arm": arm, "source_images": len(indices), "timed_calls": len(selected)}
            for metric in metrics:
                per_source = np.array([np.mean([float(lookup[index, snr, arm, repeat][metric])
                    for snr in snrs for repeat in range(repeats)]) for index in indices])
                values[arm, metric] = per_source
                record[metric.replace("seconds", "mean_ms")] = float(per_source.mean() * 1000)
                record[metric.replace("seconds", "raw_call_p95_ms")] = float(np.percentile([float(row[metric]) for row in selected], 95) * 1000)
            summary.append(record)
        for method, control in contrasts:
            for metric in metrics:
                interval = paired_interval((values[method, metric] - values[control, metric]) * 1000,
                    config["bootstrap_seed"], config["bootstrap_resamples"])
                paired.append({"scope": scope, "method": method, "control": control, "metric": metric,
                               "source_images": len(indices), "difference_unit": "ms_method_minus_control", **interval})
    return summary, paired
