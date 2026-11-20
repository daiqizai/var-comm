"""Source-paired nonparametric bootstrap 95% confidence intervals and tail percentiles."""
from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    values_a: list[float] | np.ndarray,
    values_b: list[float] | np.ndarray | None = None,
    *,
    num_resamples: int = 10000,
    seed: int = 20261120,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute mean, 10th-percentile (worst-10% for higher-is-better, 90th for lower-is-better),
    and 10,000-draw source-level bootstrap 95% CI. If values_b is given, computes stats on (a - b).
    """
    a = np.asarray(values_a, dtype=np.float64)
    if values_b is not None:
        b = np.asarray(values_b, dtype=np.float64)
        assert a.shape == b.shape
        arr = a - b
    else:
        arr = a

    n = int(arr.shape[0])
    assert n >= 2
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(num_resamples, n))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return {
        "n": n,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "ci95_lo": lo,
        "ci95_hi": hi,
    }
