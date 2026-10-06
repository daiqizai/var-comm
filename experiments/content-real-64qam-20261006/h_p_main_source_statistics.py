"""Pure source-paired H/P/MAIN comparison without relabelling noise streams.

The caller admits original completion receipts and supplies explicit normalized
rows in a separate view. Original metric records and policy files stay intact.
No policy, metric, source or comparison is selected using these results.
"""
from __future__ import annotations
import math
import numpy as np

SOURCES = 100
REPLICATES = 10000
BOOTSTRAP_SEED = 2026100605
NOISE = {'H': (6201, 6202, 6203), 'MAIN': (6201, 6202, 6203), 'P': (2001, 2002, 2003)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def original100(ids):
    require(type(ids) is list and len(ids) == SOURCES and len(set(ids)) == SOURCES
            and all(type(x) is str and x for x in ids), 'Exactly original100 source IDs required')


def point_contract(points, metrics):
    require(type(points) is dict and points and type(metrics) is list and metrics
            and len(metrics) == len(set(metrics)), 'Explicit points and metrics required')
    for point, spec in points.items():
        require(type(point) is str and point and set(spec) == {
            'branch', 'snr_db', 'noise_seeds', 'policy_sha256', 'metric_identity'}, 'Explicit point identity required')
        require(spec['branch'] in NOISE and spec['snr_db'] in (13, 19)
                and tuple(spec['noise_seeds']) == NOISE[spec['branch']], 'Original per-branch noises required')
        require(type(spec['policy_sha256']) is str and len(spec['policy_sha256']) == 64
                and all(c in '0123456789abcdef' for c in spec['policy_sha256']), 'Frozen policy SHA required')
        require(set(spec['metric_identity']) == set(metrics)
                and all(type(v) is str and v for v in spec['metric_identity'].values()),
                'Every metric needs an admitted definition/model identity')


def source_means(rows, ids, points, metrics):
    """Only a declared normalized view enters; no implicit CSV or seed rewriting.

    metric_identity is per metric, so genuinely different definitions cannot be
    silently pooled. An externally admitted common definition must explicitly
    describe any numerical-precision provenance difference.
    """
    original100(ids); point_contract(points, metrics)
    groups = {point: {} for point in points}
    source_truth = {}
    for row in rows:
        point = row['point_id']; require(point in points, 'Undeclared point')
        spec = points[point]; i = row['source_index']; n = row['noise_seed']
        require(type(i) is int and 0 <= i < SOURCES and type(n) is int
                and n in spec['noise_seeds'] and (i, n) not in groups[point], 'Unexpected or duplicate source/noise')
        require(row['source_id'] == ids[i] and row['N'] == 1024
                and row['snr_db'] == spec['snr_db'] and row['branch'] == spec['branch']
                and row['policy_sha256'] == spec['policy_sha256']
                and row['metric_identity'] == spec['metric_identity'], 'Point/source/metric identity changed')
        require(row['used_for_selection'] is False and row['holdout_used'] is False,
                'Only fixed-policy independent development rows')
        for m in metrics:
            value = row[m]
            require(isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_))
                    and math.isfinite(float(value)), 'Missing or nonnumeric metric: ' + m)
        # Every row must carry the original target and independent classification
        # reference. There is no silent intersection of mismatched populations.
        truth = tuple(row[k] for k in ('reference_sha256', 'true_class',
                    'resnet50_source_prediction', 'convnext_source_prediction'))
        require(type(truth[0]) is str and len(truth[0]) == 64
                and all(type(v) is int and 0 <= v < 1000 for v in truth[1:]), 'Original target evidence required')
        require(i not in source_truth or source_truth[i] == truth, 'Target pixels/label/source predictions differ')
        source_truth[i] = truth
        groups[point][i, n] = row
    for point, group in groups.items():
        expected = {(i, n) for i in range(SOURCES) for n in points[point]['noise_seeds']}
        require(set(group) == expected, 'Incomplete original100 x original3 noises: ' + point)
    return {point: {m: np.asarray([np.mean([group[i, n][m] for n in points[point]['noise_seeds']],
                         dtype=np.float64) for i in range(SOURCES)], dtype=np.float64) for m in metrics}
            for point, group in groups.items()}


def draws():
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, SOURCES, (REPLICATES, SOURCES))


def interval(a, samples):
    require(a.shape == (100,) and np.isfinite(a).all() and samples.shape == (10000, 100),
            'Original finite source bootstrap required')
    lo, hi = np.quantile(a[samples].mean(axis=1), [.025, .975])
    return dict(mean=float(a.mean()), ci_low=float(lo), ci_high=float(hi), source_count=100,
                noise_count=3, frame_count=300, bootstrap_seed=BOOTSTRAP_SEED,
                bootstrap_replicates=REPLICATES, bootstrap_unit='same source after mean of each branch original3 noises')


def summarize(rows, ids, points, metrics, comparisons):
    """All pairs are frozen before rows are read. There is no winner selection."""
    values = source_means(rows, ids, points, metrics)
    require(type(comparisons) is list and len({tuple(c) for c in comparisons}) == len(comparisons),
            'Explicit unique pairs required')
    samples = draws()
    summary = [dict(point_id=p, branch=points[p]['branch'], snr_db=points[p]['snr_db'],
                    noise_seeds=points[p]['noise_seeds'], metric=m, metric_identity=points[p]['metric_identity'][m],
                    **interval(a, samples)) for p, metrics_by_name in values.items() for m, a in metrics_by_name.items()]
    paired = []
    for method, reference in comparisons:
        require(method in points and reference in points and method != reference
                and points[method]['snr_db'] == points[reference]['snr_db'], 'Distinct same-SNR fixed pair required')
        for m in metrics:
            require(points[method]['metric_identity'][m] == points[reference]['metric_identity'][m],
                    'Metric definition/model mismatch; comparison not permitted')
            paired.append(dict(method=method, reference=reference, snr_db=points[method]['snr_db'], metric=m,
                metric_identity=points[method]['metric_identity'][m], delta_definition='method minus reference',
                method_noise_seeds=points[method]['noise_seeds'], reference_noise_seeds=points[reference]['noise_seeds'],
                frame_level_noise_pairing_claimed=False,
                **interval(values[method][m]-values[reference][m], samples)))
    means = [dict(point_id=p, metric=m, source_index=i, source_id=ids[i], mean=float(a[i]))
             for p, data in values.items() for m, a in data.items() for i in range(SOURCES)]
    return dict(summary=summary, paired=paired, source_means=means, policy_selection=False,
                noise_seeds_modified=False, missing_points_imputed=False, source_intersection_used=False,
                holdout_used=False, multiple_comparisons='Predeclared descriptive pointwise95% intervals; no multiplicity-adjusted global winner claim')
