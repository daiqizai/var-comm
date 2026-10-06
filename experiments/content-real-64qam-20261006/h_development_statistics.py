"""Source-paired statistics for the frozen H development grid; no selection.

The caller must verify the measured RGB/metric receipts and declared policies.
This pure module cannot load models, add rows, choose methods, or launch jobs.
"""
import math
import numpy as np

SOURCES = 100
SEEDS = (6201, 6202, 6203)
BOOTSTRAP_SEED = 2026100605
REPLICATES = 10000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(rows, source_ids, points, metrics):
    """Each declared point requires all100 originals and three original noises.

    points maps an explicit point_id to {snr_db, role, candidate_id}. It comes
    from frozen receipts, never a development quality ranking. Additional MAIN
    or P points require separately admitted input receipts outside this module.
    """
    require(len(source_ids) == SOURCES and len(set(source_ids)) == SOURCES
            and all(isinstance(s, str) and s for s in source_ids), 'Original100 unique sources required')
    require(points and metrics and len(set(metrics)) == len(metrics), 'Explicit points/metrics required')
    expected = {(i, n) for i in range(SOURCES) for n in SEEDS}
    groups = {p: {} for p in points}
    source_predictions = {}
    for point, spec in points.items():
        require(isinstance(point, str) and point and set(spec) == {'snr_db', 'role', 'candidate_id'}
                and spec['snr_db'] in (13, 19) and spec['role'] and spec['candidate_id'],
                'Explicit frozen method identity required')
    for row in rows:
        point = row['point_id']
        require(point in points, 'Undeclared method cannot enter statistics')
        spec = points[point]
        i, seed = row['source_index'], row['noise_seed']
        require(type(i) is int and type(seed) is int and (i, seed) in expected
                and (i, seed) not in groups[point], 'Unexpected or duplicate source/noise')
        require(row['source_id'] == source_ids[i] and row['N'] == 1024
                and all(row[k] == spec[k] for k in spec), 'Method/source identity changed')
        require(row.get('used_for_selection') is False and row.get('holdout_used') is False,
                'Only independent fixed-policy development scores allowed')
        for name in metrics:
            require(name in row and isinstance(row[name], (int, float, bool, np.integer, np.floating, np.bool_))
                    and math.isfinite(float(row[name])), 'Missing/nonfinite metric: ' + name)
        for name in ('resnet50_source_prediction', 'convnext_source_prediction', 'true_class'):
            if name in row:
                key = (i, name)
                require(key not in source_predictions or source_predictions[key] == row[name],
                        'Original-source identity/prediction changed across methods')
                source_predictions[key] = row[name]
        groups[point][i, seed] = row
    for point, group in groups.items():
        require(set(group) == expected, 'Incomplete source/noise grid: ' + point)
    return {p: {m: np.asarray([np.mean([g[i, n][m] for n in SEEDS], dtype=np.float64)
                              for i in range(SOURCES)], dtype=np.float64)
                for m in metrics} for p, g in groups.items()}


def draws():
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, SOURCES, (REPLICATES, SOURCES))


def interval(values, resamples):
    a = np.asarray(values, dtype=np.float64)
    require(a.shape == (SOURCES,) and np.isfinite(a).all(), 'Exactly100 finite source means required')
    require(resamples.shape == (REPLICATES, SOURCES), 'Original bootstrap population required')
    lo, hi = np.quantile(a[resamples].mean(axis=1), [.025, .975])
    return dict(mean=float(a.mean()), ci_low=float(lo), ci_high=float(hi),
                source_count=SOURCES, noise_count=3, frame_count=300,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=REPLICATES,
                bootstrap_unit='source after mean of three registered noises')


def summarize(rows, source_ids, points, metrics, comparisons):
    """comparisons explicitly lists (method, reference); raw delta is method-ref.

    References must have the same SNR. No winner, best arm, or cross-SNR pool is
    computed. All declared points appear even when a comparison is unfavorable.
    """
    source_means = validate(rows, source_ids, points, metrics)
    require(len(comparisons) == len(set(tuple(v) for v in comparisons)), 'Duplicate comparison')
    boot = draws()
    summaries = [dict(point_id=p, metric=m, **points[p], **interval(a, boot))
                 for p, values in source_means.items() for m, a in values.items()]
    paired = []
    for method, reference in comparisons:
        require(method in points and reference in points and method != reference
                and points[method]['snr_db'] == points[reference]['snr_db'],
                'Pair must be explicit distinct methods at the same SNR')
        for metric in metrics:
            paired.append(dict(method=method, reference=reference, metric=metric,
                snr_db=points[method]['snr_db'], delta_definition='method minus reference',
                **interval(source_means[method][metric] - source_means[reference][metric], boot)))
    means = [dict(point_id=p, source_index=i, source_id=source_ids[i], metric=m, mean=float(a[i]))
             for p, values in source_means.items() for m, a in values.items() for i in range(SOURCES)]
    return dict(summary=summaries, paired=paired, source_means=means, policy_selection=False,
                holdout_used=False, missing_points_imputed=False)


def independent_success(paired_rows):
    """Evaluate one explicit H-versus-MAIN pair per SNR, after input admission.

    Missing metrics return INCOMPLETE. Confidence intervals crossing zero return
    UNRESOLVED. This function never interprets a DINO-only gain as content gain.
    """
    required = {'psnr_db': 'strict_positive', 'lpips_alex': 'nonpositive',
                'clip_image_cosine': 'nonnegative',
                'convnext_source_prediction_agreement': 'nonnegative'}
    keys = {(r['snr_db'], r['metric']) for r in paired_rows}
    require(len(keys) == len(paired_rows), 'Exactly one declared pair per SNR/metric required')
    for snr in (13, 19):
        pairs = {(r['method'], r['reference']) for r in paired_rows if r['snr_db'] == snr}
        require(len(pairs) <= 1, 'Cannot choose among development arms')
    result = {}
    for snr in (13, 19):
        by = {r['metric']: r for r in paired_rows if r['snr_db'] == snr}
        if not set(required) <= set(by):
            result[str(snr)] = 'INCOMPLETE'
            continue
        passed = []
        for metric, rule in required.items():
            row = by[metric]
            require(math.isfinite(row['ci_low']) and math.isfinite(row['ci_high'])
                    and row['ci_low'] <= row['ci_high'], 'Invalid paired interval')
            passed.append(row['ci_low'] > 0 if rule == 'strict_positive' else
                          row['ci_high'] <= 0 if rule == 'nonpositive' else row['ci_low'] >= 0)
        result[str(snr)] = 'PASS' if all(passed) else 'UNRESOLVED_OR_FAILED_NONINFERIORITY'
    return dict(per_snr=result, overall='PASS' if all(v == 'PASS' for v in result.values()) else
                ('INCOMPLETE' if 'INCOMPLETE' in result.values() else 'NOT_ESTABLISHED'),
                noninferiority_margin=0, independent_metrics_used_for_selection=False)
