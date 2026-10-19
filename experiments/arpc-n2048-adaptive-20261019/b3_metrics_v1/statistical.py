"""Fixed source denominator and direct paired source bootstrap intervals."""
import math
import statistics
import numpy as np

METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine',
           'convnext_top1_source_prediction', 'dists', 'dreamsim', 'ms_ssim')
METHODS = ('NEST_ADAPTIVE_N2048', 'ARPC_ADAPTIVE_N2048')
SNRS = (4, 10)
SEEDS = (6201, 6202, 6203)
DIRECTIONS = {k: ('lower' if k in ('lpips_alex', 'dists', 'dreamsim') else 'higher') for k in METRICS}


def source_means(rows, ids):
    assert len(ids) == len(set(ids))
    seen, groups = set(), {}
    for r in rows:
        key = (r['method'], r['source_id'], int(r['snr_db']), int(r['noise_seed']))
        assert key not in seen and key[0] in METHODS and key[1] in ids and key[2] in SNRS and key[3] in SEEDS
        seen.add(key)
        groups.setdefault(key[:3], []).append(r)
    assert len(seen) == len(ids) * 12
    out = []
    for m in METHODS:
        for sid in ids:
            for snr in SNRS:
                a = groups[m, sid, snr]
                assert len(a) == 3
                for metric in METRICS:
                    values = [float(r[metric]) for r in a]
                    assert all(math.isfinite(v) for v in values)
                    out.append(dict(method=m, source_id=sid, snr_db=snr, metric=metric,
                                    source_mean=statistics.mean(values), noise_count=3))
    return out


def summarize(rows, ids, draws):
    assert draws.dtype == np.int64 and draws.ndim == 2 and draws.shape[1] == len(ids)
    assert draws.min() >= 0 and draws.max() < len(ids)
    v = {(r['method'], r['source_id'], int(r['snr_db']), r['metric']): float(r['source_mean']) for r in rows}
    assert len(v) == len(rows) == len(ids) * 2 * 2 * 7
    def interval(x):
        assert np.isfinite(x).all()
        lo, hi = np.quantile(x[draws].mean(axis=1), [.025, .975], method='linear')
        return dict(mean=float(x.mean()), ci_low=float(lo), ci_high=float(hi))
    summaries, paired = [], []
    for snr in SNRS:
        for metric in METRICS:
            vectors = {m: np.asarray([v[m, s, snr, metric] for s in ids], np.float64) for m in METHODS}
            for method, x in vectors.items():
                summaries.append(dict(method=method, snr_db=snr, metric=metric,
                    direction=DIRECTIONS[metric], source_count=len(ids), **interval(x)))
            ci = interval(vectors[METHODS[0]] - vectors[METHODS[1]])
            paired.append(dict(comparison='NeST_adaptive_minus_ARPC_adaptive', snr_db=snr,
                metric=metric, direction=DIRECTIONS[metric], source_count=len(ids), **ci,
                interval_crosses_or_touches_zero=ci['ci_low'] <= 0 <= ci['ci_high']))
    return summaries, paired
