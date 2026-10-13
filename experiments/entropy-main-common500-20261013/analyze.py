"""EMAIN-v1: once-only new common500 intervals and source-bit accounting.

No model/channel calls. Run only after the parent owner has admitted completed
scoring and the E0 reference bundle. All original intervals remain immutable.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np

NEW = 'EC_VAR_PARTIAL'
REFS = ('EC_VAR_WHOLE', 'RAW_PARTIAL', 'P100000', 'Swin200000', 'BPG_ADAPTIVE')
ALL_REFS = REFS + ('RAW_WHOLE',)
SNRS = (4, 10, 19)
NOISES = (6201, 6202, 6203)
METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine', 'convnext_top1_source_prediction')
STAT_SHA = 'e401a77ac216368dc968ce023853ff7cdb2e0a2c94edbd3dd33407c0ba90054a'
TARGETS = {4: (7, 25, 1245), 10: (8, 126, 2520), 19: (9, 192, 4751)}


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    with Path(path).open(newline='', encoding='utf8') as handle:
        return list(csv.DictReader(handle))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def csv_write(path, data):
    require(bool(data), 'Empty table: ' + str(path))
    fields = list(dict.fromkeys(k for row in data for k in row))
    with Path(path).open('x', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def bound(entry):
    path = Path(entry['path'])
    require(sha(path) == entry['sha256'], 'Input SHA mismatch: ' + str(path))
    return path


def value(row, metric):
    x = row[metric]
    if metric == METRICS[-1] and str(x).lower() in ('true', 'false'):
        return float(str(x).lower() == 'true')
    return float(x)


def normalize_frame_grid(data, ids):
    expected = {(i, s, n) for i in range(len(ids)) for s in SNRS for n in NOISES}
    keyed = {}
    for row in data:
        key = tuple(int(row[k]) for k in ('source_index', 'snr_db', 'noise_seed'))
        require(key in expected and key not in keyed, 'Invalid/duplicate new frame ' + str(key))
        require(row['source_id'] == ids[key[0]], 'Source identity/order mismatch')
        require(row.get('method', row.get('family')) == NEW and int(row['N']) == 1024, 'Wrong new method/bandwidth')
        require(all(np.isfinite(value(row, metric)) for metric in METRICS), 'Missing/nonfinite metric; do not drop failures')
        require(value(row, METRICS[-1]) in (0., 1.), 'Agreement must be a binary source-prediction event')
        require(bool(row.get('physical_status')) and bool(row.get('source_status')), 'Missing actual failure/status evidence')
        keyed[key] = row
    require(set(keyed) == expected, 'Incomplete 500 x 3 SNR x 3 noise grid')
    means = []
    vectors = {}
    for s in SNRS:
        for metric in METRICS:
            a = np.asarray([np.mean([value(keyed[i, s, n], metric) for n in NOISES], dtype=np.float64) for i in range(len(ids))])
            vectors[NEW, s, metric] = a
            means.extend(dict(method=NEW, snr_db=s, metric=metric, source_index=i,
                              source_id=sid, mean=float(a[i]), noise_count=3) for i, sid in enumerate(ids))
    return keyed, vectors, means


def reference_vectors(data, summaries, ids):
    expected = {(m, s, metric) for m in ALL_REFS for s in SNRS for metric in METRICS}
    vectors = {key: np.full(len(ids), np.nan) for key in expected}
    for row in data:
        key = (row['method'], int(row['snr_db']), row['metric'])
        i = int(row['source_index'])
        require(key in expected and 0 <= i < len(ids), 'Unexpected reference/source')
        require(row['source_id'] == ids[i] and np.isnan(vectors[key][i]), 'Duplicate/mismatched old source mean')
        vectors[key][i] = float(row['mean'])
    require(all(np.isfinite(a).all() for a in vectors.values()), 'Missing old source means')
    seen = set()
    for row in summaries:
        key = (row['method'], int(row['snr_db']), row['metric'])
        require(key in expected and key not in seen, 'Missing/duplicate original summary key')
        require(int(row['source_count']) == len(ids) and int(row['noise_count']) == 3, 'Reference denominator differs')
        require(abs(float(row['mean']) - vectors[key].mean()) < 1e-11, 'Old summary is not this source vector')
        require(float(row['ci_low']) <= float(row['mean']) <= float(row['ci_high']), 'Invalid old interval')
        seen.add(key)
    require(seen == expected, 'Missing original reference intervals')
    return vectors


def _int(row, *keys, default=None):
    for key in keys:
        if row.get(key) not in ('', None):
            return int(row[key])
    return default


def _attempts(row):
    x = row.get('fallback_attempts', row.get('fallback_attempts_json', []))
    return json.loads(x) if isinstance(x, str) else x


def source_accounting(data, ids, methods=(NEW,)):
    """Deduplicate identical TX records across noise; never deduplicate RX events."""
    required = {(method, s, i) for method in methods for s in SNRS for i in range(len(ids))}
    keyed = {}
    for row in data:
        method = row.get('method', row.get('family', NEW))
        s, i = int(row['snr_db']), int(row['source_index'])
        key = method, s, i
        require(key in required and row['source_id'] == ids[i], 'Unknown source-length record')
        attempts = _attempts(row)
        m, k = _int(row, 'actual_m'), _int(row, 'actual_K', 'K', default=0)
        t = _int(row, 'actual_T', 'transmitted_token_count', 'token_count', default=0)
        bits = _int(row, 'arithmetic_bits')
        capacity = _int(row, 'capacity_bits', 'source_capacity_bits')
        target_m = _int(row, 'target_m')
        target_k = _int(row, 'target_K', default=0)
        require(capacity == TARGETS[s][2], 'Wrong paid source capacity')
        if method == NEW:
            require((target_m, target_k) == TARGETS[s][:2], 'Frozen target changed')
        else:
            require((target_m, target_k) == ((7, 0) if s == 4 else (9, 0)), 'Whole-scale target changed')
        require(isinstance(attempts, list) and bool(attempts), 'Full real length-attempt record required')
        for attempt in attempts:
            require(int(attempt['bits']) >= 0 and isinstance(attempt['fits'], bool), 'Invalid actual attempt')
            require(attempt['fits'] == (int(attempt['bits']) <= capacity), 'Attempt fit differs from actual terminated length')
        legal = t > 0 and bits is not None and 0 <= bits <= capacity
        if legal:
            require(attempts[-1]['fits'] and int(attempts[-1]['bits']) == bits, 'Selected stream is not final fitting attempt')
        fallback_steps = len(attempts) - 1
        item = dict(method=method, snr_db=s, source_index=i, source_id=ids[i],
                    target_m=target_m, target_K=target_k, actual_m=m, actual_K=k, actual_T=t,
                    arithmetic_bits=bits, raw12_bits=12*t, capacity_bits=capacity,
                    information_margin_bits=(capacity-bits) if legal else '',
                    source_saving_fraction=(1-bits/(12*t)) if legal else '',
                    source_capacity_utilization=(bits/capacity) if legal else '',
                    target_did_not_fit=not attempts[0]['fits'], fallback_steps=fallback_steps,
                    final_stream_legal=legal, length_status='ACTUAL_TERMINATED_STREAM' if legal else 'NO_LEGAL_STREAM',
                    fallback_attempts=json.dumps(attempts, sort_keys=True, separators=(',', ':')))
        if key in keyed:
            require(keyed[key] == item, 'Noise replicates disagree on source coding; cannot collapse')
        else:
            keyed[key] = item
    require(set(keyed) == required, 'Incomplete source-level length grid')
    return [keyed[key] for key in sorted(keyed)]


def old_whole_lengths(data, ids):
    """Use only T1's actual chosen prefix; absent unchosen lengths stay absent."""
    result = []
    for row in data:
        if row['population'] != 'holdout' or row['family'] != 'EC_VAR_WHOLE':
            continue
        for s in SNRS:
            chosen = _int(row, 'actual_selected_m_SNR' + str(s))
            if chosen != int(row['complete_scale_m']):
                continue
            require(row['length_status'] == 'ACTUAL_ENCODED_LENGTH', 'Chosen old whole length is unavailable')
            result.append(dict(method='EC_VAR_WHOLE', source_id=row['source_id'], source_index=row['source_index'],
                               snr_db=s, target_m=row['target_m_SNR'+str(s)], target_K=0,
                               actual_m=chosen, actual_K=0, actual_T=row['token_count'],
                               arithmetic_bits=row['arithmetic_bits'], capacity_bits=row['source_capacity_SNR'+str(s)],
                               fallback_attempts=row['fallback_attempts_SNR'+str(s)]))
    return source_accounting(result, ids, methods=('EC_VAR_WHOLE',))


def length_summaries(data):
    prefixes = []
    summaries = []
    groups = defaultdict(list)
    for row in data:
        groups[row['method'], row['snr_db']].append(row)
    for (method, snr), group in sorted(groups.items()):
        counts = Counter((r['actual_m'], r['actual_K'], r['actual_T'], r['length_status']) for r in group)
        for (m, k, t, status), count in sorted(counts.items(), key=lambda x: (x[0][2], str(x[0]))):
            prefixes.append(dict(method=method, snr_db=snr, actual_m=m, actual_K=k, actual_T=t,
                                 status=status, source_count=count, source_fraction=count/len(group)))
        item = dict(method=method, snr_db=snr, source_count=len(group), noise_replicates_counted_as_sources=False,
                    target_not_fit_count=sum(r['target_did_not_fit'] for r in group),
                    target_not_fit_fraction=np.mean([r['target_did_not_fit'] for r in group]),
                    no_legal_stream_count=sum(not r['final_stream_legal'] for r in group),
                    mean_fallback_steps=np.mean([r['fallback_steps'] for r in group]),
                    max_fallback_steps=max(r['fallback_steps'] for r in group))
        for field in ('actual_T', 'arithmetic_bits', 'raw12_bits', 'source_saving_fraction',
                      'information_margin_bits', 'source_capacity_utilization'):
            values = [float(r[field]) for r in group if r[field] not in ('', None)]
            item[field+'_valid_sources'] = len(values)
            for name, fun in (('mean', np.mean), ('median', np.median), ('min', np.min), ('max', np.max)):
                item[field+'_'+name] = float(fun(values)) if values else ''
        summaries.append(item)
    return prefixes, summaries


def failure_table(data):
    counts = Counter((r.get('method', r.get('family', NEW)), int(r['snr_db']),
                      r.get('physical_status', 'NOT_AVAILABLE'), r.get('source_status', 'NOT_AVAILABLE'),
                      r.get('source_parse_reason', '')) for r in data)
    totals = Counter((r.get('method', r.get('family', NEW)), int(r['snr_db'])) for r in data)
    return [dict(method=m, snr_db=s, physical_status=p, source_status=q, source_parse_reason=reason,
                 frame_count=n, denominator=totals[m, s], fraction=n/totals[m, s])
            for (m, s, p, q, reason), n in sorted(counts.items())]


def display(row, paired=False):
    return dict(row, display_scale=100 if row['metric'] == METRICS[-1] else 1,
                display_unit=('percentage_points' if paired else 'percent') if row['metric'] == METRICS[-1]
                else ('dB' if row['metric'] == METRICS[0] else 'raw_metric_units'),
                improvement_direction='negative' if row['metric'] == METRICS[1] else 'positive')


def run(manifest_path, out):
    manifest_path, out = Path(manifest_path), Path(out)
    manifest = read(manifest_path)
    require(manifest['schema'] == 'EMAIN_ANALYSIS_INPUTS_V1', 'Wrong analysis manifest')
    ids = manifest['source_ids']
    require(len(ids) == len(set(ids)) == 500, 'Original 500 ordered sources required')
    inputs = {name: bound(entry) for name, entry in manifest['inputs'].items()}
    for flag in ('frozen_policy_verified', 'source_pixels_verified', 'metric_identity_verified', 'reference_reuse_verified'):
        require(manifest[flag] is True, 'E0 prerequisite missing: ' + flag)
    scoring = read(inputs['scoring_completion'])
    require(scoring['status'] == manifest['scoring_expected_status'], 'Scoring is not complete')
    wait = read(inputs['actual_wait'])
    require(wait['exit_code'] == 0 and wait['actual_child_waited'] is True, 'Actual successful scoring wait required')
    require(sha(inputs['statistics_module']) == STAT_SHA, 'Original bootstrap source differs')
    newrows = rows(inputs['new_frame_rows'])
    keyed, vectors, newmeans = normalize_frame_grid(newrows, ids)
    refs, oldsummary = rows(inputs['reference_source_means']), rows(inputs['reference_summary'])
    vectors.update(reference_vectors(refs, oldsummary, ids))
    lengths = source_accounting(rows(inputs['source_streams']), ids)
    lengths += old_whole_lengths(rows(inputs['reference_whole_lengths']), ids)
    lengths_by_key = {(r['snr_db'], r['source_index']): r for r in lengths if r['method'] == NEW}
    for (i, s, n), row in keyed.items():
        coded = lengths_by_key[s, i]
        for field, aliases in [('actual_m', ('actual_m',)), ('actual_K', ('actual_K', 'K')),
                               ('actual_T', ('actual_T', 'transmitted_token_count')), ('arithmetic_bits', ('arithmetic_bits',))]:
            require(_int(row, *aliases) == coded[field], 'Frame/source coding disagreement: ' + field)
    prefixes, lengthsummary = length_summaries(lengths)
    failures = failure_table(newrows)
    if 'reference_frame_rows' in inputs:
        failures += failure_table(rows(inputs['reference_frame_rows']))
    require(not out.exists(), 'Once-only analysis output already exists; audit recovery before retry')
    out.mkdir(parents=True)
    write(out/'launch.json', dict(started_unix=time.time(), manifest_sha256=sha(manifest_path),
                                   all_input_admissions_passed=True, automatic_retries=False))
    try:
        spec = importlib.util.spec_from_file_location('_emain_original_common500_statistics', inputs['statistics_module'])
        stat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stat)
        require((stat.SOURCE_COUNT, stat.REPLICATES, stat.BOOTSTRAP_SEED) == (500, 10000, 2026100701), 'Bootstrap contract differs')
        samples = stat.draws()
        require(samples.shape == (10000, 500), 'Wrong original bootstrap draws')
        count = 0
        def estimate(vector):
            nonlocal count
            count += 1
            result = stat.interval(vector, samples)
            require(result['ci_low'] <= result['mean'] <= result['ci_high'], 'Invalid new interval')
            return result
        summary, paired, differences = [], [], []
        for s in SNRS:
            for metric in METRICS:
                summary.append(display(dict(method=NEW, N=1024, snr_db=s, metric=metric, **estimate(vectors[NEW, s, metric]))))
        for reference in REFS:
            for s in SNRS:
                for metric in METRICS:
                    delta = vectors[NEW, s, metric] - vectors[reference, s, metric]
                    paired.append(display(dict(method=NEW, reference=reference, N=1024, snr_db=s, metric=metric,
                                               delta_definition='EC_VAR_PARTIAL minus reference', **estimate(delta)), paired=True))
                    differences.extend(dict(method=NEW, reference=reference, snr_db=s, metric=metric,
                                            source_index=i, source_id=sid, delta=float(delta[i])) for i, sid in enumerate(ids))
        require((len(summary), len(paired), count) == (12, 60, 72), 'Unexpected new interval budget')
        allsummary = [display(dict(r, interval_provenance='ORIGINAL_INTERVAL_REUSED_UNCHANGED')) for r in oldsummary] + summary
        mechanism = [r for r in allsummary if r['method'] in ('RAW_WHOLE', 'RAW_PARTIAL', 'EC_VAR_WHOLE', NEW)]
        for name, table in [('summary.csv', summary), ('summary_all_methods.csv', allsummary),
                            ('source_means.csv', newmeans+refs), ('paired.csv', paired),
                            ('source_paired_differences.csv', differences), ('mechanism_four_methods.csv', mechanism),
                            ('failure_breakdown.csv', failures), ('length_utilization.csv', lengths),
                            ('length_summary.csv', lengthsummary), ('prefix_distribution.csv', prefixes)]:
            csv_write(out/name, table)
        shutil.copyfile(inputs['reference_summary_unchanged'], out/'reference_summary_unchanged.json')
        report = ['# Frozen entropy partial-scale common500 extension', '',
                  'All 4,500 new logical frames and their failures are retained. Three registered noises are averaged within each source before source-level statistics. The frozen original common500 bootstrap (10,000 draws, seed 2026100701) computes only 12 new marginal intervals and 60 direct paired intervals. Original intervals are copied unchanged.', '',
                  'This is a subsequent evaluation on the already used common500 population, not a newly blind test. Pairing is by source, not a claim that distinct systems share received observations. Confidence intervals are pointwise. LPIPS retains its original sign; agreement is source-prediction consistency, not classification accuracy.', '',
                  'Raw and entropy coding differ in framing and failure rules, so the four matched-prior systems are not a pure two-factor ablation. Source lengths count 500 unique sources per SNR/method, not 1,500 noise realizations. Arithmetic lengths include termination. Compression saving concerns the same token sequence and does not imply equal wireless-symbol savings.', '',
                  '| SNR | Reference | Metric | Entropy partial minus reference | 95% CI |', '|---|---|---|---:|---|']
        for r in paired:
            scale = r['display_scale']
            report.append(f"| {r['snr_db']} | {r['reference']} | {r['metric']} | {scale*r['mean']:.6g} | [{scale*r['ci_low']:.6g}, {scale*r['ci_high']:.6g}] |")
        report += ['', 'Agreement differences above use percentage points. No failure was dropped, no policy was reselected, no old interval was recalculated, and no training, new bandwidth, Kodak, HiFi or additional SNR extension was started. Unified timing is reported by its separate registered owner; this statistical delivery does not assert that timing is complete.']
        (out/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf8')
        write(out/'completion.json', dict(status='EMAIN_COMMON500_E1_E2_STATISTICS_COMPLETE', source_count=500, noise_count=3,
              logical_new_frames=4500, new_summary_rows=12, new_paired_rows=60, old_summary_rows_reused=len(oldsummary),
              new_interval_calls=count, bootstrap_draw_calls=1, bootstrap_seed=2026100701, bootstrap_replicates=10000,
              draw_matrix_sha256=hashlib.sha256(samples.tobytes()).hexdigest(), old_intervals_recomputed=0,
              all_failure_rows_retained=True, source_length_noise_deduplication=True, policy_selection=False,
              new_model_calls=0, second_stage_started=False, input_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)),
              outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    except BaseException:
        write(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.manifest, args.out)
