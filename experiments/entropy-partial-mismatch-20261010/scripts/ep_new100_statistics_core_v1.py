"""Finite new100 source-paired statistics; no CLI, data reads or calls on import.

The execution owner must authenticate the closed actual metric population before
calling summarize with the unchanged published draws/interval functions. Tests
inject fake functions only. This module never selects policies or sources.
"""
from __future__ import annotations
import hashlib
import numpy as np
import ep_new100_confirmation_core_v1 as confirmation

SCHEMA = 'EP_NEW100_FOUR_ARM_STATISTICS_CORE_V1'
REPLICATES = 10000
SEED = 2026100701
CAPS = dict(draw_matrix=1, summary_intervals=48, paired_intervals=24)
require = confirmation.require


def definitions():
    summaries = [dict(N=1024, method=m, snr_db=s, metric=k)
                 for m in confirmation.METHODS for s in confirmation.SNRS
                 for k in confirmation.METRICS]
    pairs = [dict(N=1024, method=m, reference=r, snr_db=s, metric=k,
                  delta_definition='method minus reference')
             for m, r in confirmation.CONTRASTS for s in confirmation.SNRS
             for k in confirmation.METRICS]
    return summaries, pairs


def source_vectors(rows, expected):
    confirmation.complete_grid(rows, expected, scored=True)
    ids = {r['source_index']: r['source_id'] for r in expected}
    keyed = {(r['method'], r['snr_db'], r['source_index'], r['noise_seed']): r
             for r in rows}
    vectors, table = {}, []
    for d in definitions()[0]:
        m, s, k = d['method'], d['snr_db'], d['metric']
        # Retain the original common500/T6 source-mean operation and precision.
        values = np.asarray([np.mean([keyed[m, s, i, n][k]
                                     for n in confirmation.SEEDS], dtype=np.float64)
                             for i in range(100)], dtype=np.float64)
        require(values.shape == (100,) and np.isfinite(values).all(),
                'All100 finite three-noise source means required')
        vectors[m, s, k] = values
        table.extend(dict(d, source_index=i, source_id=ids[i], mean=float(v),
                          source_count=100, noise_count=3)
                     for i, v in enumerate(values))
    return vectors, table, ids


def decorate(metric, result, paired):
    agreement = metric == 'convnext_top1_source_prediction'
    scale = 100. if agreement else 1.
    unit = ('absolute_agreement_fraction' if paired else 'agreement_fraction') if agreement else {
        'psnr_db': 'dB', 'lpips_alex': 'LPIPS',
        'dinov2_vitl14_cosine': 'cosine_similarity'}[metric]
    return dict(result, unit=unit, display_scale=scale,
                display_unit=('percentage_points' if paired else 'percent') if agreement else unit,
                display_mean=scale * result['mean'],
                display_ci_low=scale * result['ci_low'],
                display_ci_high=scale * result['ci_high'],
                improvement_direction=('negative' if paired else 'lower') if metric == 'lpips_alex'
                else ('positive' if paired else 'higher'))


def summarize(rows, expected, stat, event):
    """Execute once only under a separately registered, finite statistics owner."""
    vectors, means, ids = source_vectors(rows, expected)
    event('reserved', 'draw_matrix', 0, {})
    samples = stat.draws()
    require(samples.shape == (10000, 100) and np.issubdtype(samples.dtype, np.integer)
            and samples.min() >= 0 and samples.max() < 100,
            'Exact registered source-resample matrix required')
    event('completed', 'draw_matrix', 0,
          dict(shape=list(samples.shape), sha256=hashlib.sha256(samples.tobytes()).hexdigest()))
    cache, counts = {}, dict(draw_matrix=1, summary_intervals=0, paired_intervals=0)

    def interval(metric, vector, kind, index):
        require(vector.shape == (100,) and vector.dtype == np.float64 and np.isfinite(vector).all(),
                'Finite exact100 float64 source vector required')
        vector_sha = hashlib.sha256(vector.tobytes()).hexdigest()
        key = metric, vector.tobytes()
        if key in cache:
            return dict(cache[key], source_vector_sha256=vector_sha,
                        interval_provenance='EXACT_SAME_SOURCE_VECTOR_REUSE_WITHIN_NEW100')
        if np.all(vector == 0):
            result = dict(mean=0., ci_low=0., ci_high=0., source_count=100, noise_count=3,
                          frame_count=300, bootstrap_seed=SEED, bootstrap_replicates=REPLICATES,
                          interval_provenance='EXACT_ZERO_SOURCE_VECTOR_NO_INTERVAL_CALL')
        else:
            require(counts[kind] < CAPS[kind], 'Registered interval cap exceeded')
            event('reserved', kind, index, dict(metric=metric, source_vector_sha256=vector_sha))
            original = stat.interval(vector, samples)
            require(original['source_count'] == 100 and original['noise_count'] == 3
                    and original['frame_count'] == 1500 and original['bootstrap_seed'] == SEED
                    and original['bootstrap_replicates'] == REPLICATES,
                    'Published bootstrap result metadata changed')
            result = dict(original, frame_count=300,
                          interval_provenance='PUBLISHED_BOOTSTRAP_FUNCTION_POPULATION100')
            require(all(np.isfinite(result[k]) for k in ('mean', 'ci_low', 'ci_high'))
                    and result['ci_low'] <= result['mean'] <= result['ci_high'],
                    'Finite ordered reported mean and interval required')
            counts[kind] += 1
            event('completed', kind, index, dict(metric=metric, source_vector_sha256=vector_sha,
                                                result=result))
        result.update(bootstrap_unit='source after registered3-noise mean', source_vector_sha256=vector_sha)
        cache[key] = dict(result)
        return result

    summary, paired, deltas = [], [], []
    for index, d in enumerate(definitions()[0]):
        m, s, k = d['method'], d['snr_db'], d['metric']
        result = interval(k, vectors[m, s, k], 'summary_intervals', index)
        summary.append(dict(d, **decorate(k, result, False)))
    for index, d in enumerate(definitions()[1]):
        m, r, s, k = d['method'], d['reference'], d['snr_db'], d['metric']
        vector = vectors[m, s, k] - vectors[r, s, k]
        result = interval(k, vector, 'paired_intervals', index)
        paired.append(dict(d, **decorate(k, result, True),
                           source_paired=True, identical_received_observations_claimed=False))
        deltas.extend(dict(d, source_index=i, source_id=ids[i], mean=float(v),
                           noise_count=3) for i, v in enumerate(vector))
    require(len(summary) == 48 and len(paired) == 24 and len(means) == 4800
            and len(deltas) == 2400 and all(counts[k] <= CAPS[k] for k in CAPS),
            'Every declared source and within-representation comparison must remain')
    return dict(summary=summary, paired=paired, source_means=means,
                source_paired_differences=deltas, actual_calls=counts,
                policy_selection=False, old500_mean_subtraction=False,
                LPIPS_sign_unchanged=True)
