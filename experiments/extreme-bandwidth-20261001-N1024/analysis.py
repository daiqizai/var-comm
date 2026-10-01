"""Frozen source-paired N1024 analysis. No inference or policy selection here."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260930
RESAMPLES = 10000
ZERO = 1e-12
SNRS = (1, 4, 7, 13, 19)
NOISE_SEEDS = {2001, 2002, 2003}
POLICIES = ('P1024', 'A1_policy', 'A2_policy', 'V_policy')
DIGITAL = ('D_U_QPSK', 'D_C_QPSK', 'D_U_16QAM', 'D_C_16QAM')
METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine', 'dino_mismatched',
           'latent_sq_err_final', 'dino_lt_0_6', 'lpips_gt_0_35',
           'dino_match_specificity', 'energy', 'header_failure', 'body_crc_failure')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty measured table: ' + str(path))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def interval(values, indices):
    values = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(values[indices].mean(1), [2.5, 97.5])
    return dict(mean=float(values.mean()), ci_low=float(lo), ci_high=float(hi))


def conditional_interval(values, indices):
    """Resample the same source indices, retaining only defined source means."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return dict(mean=None, ci_low=None, ci_high=None, valid_sources=0)
    samples = values[indices]
    counts = np.isfinite(samples).sum(1)
    means = np.divide(np.nansum(samples, axis=1), counts,
                      out=np.full(len(counts), np.nan), where=counts > 0)
    lo, hi = np.percentile(means[np.isfinite(means)], [2.5, 97.5])
    return dict(mean=float(values[valid].mean()), ci_low=float(lo), ci_high=float(hi),
                valid_sources=int(valid.sum()))


def is_failure(value):
    return float(str(value).lower() in ('false', '0'))


def normalize_rows(rows, expected_budget=1024):
    """Unify saved schemas without substituting missing scientific measurements."""
    required = ('source_id', 'source_index', 'preprocessing_id', 'snr_db', 'noise_seed', 'method',
                'psnr_db', 'lpips_alex', 'dino_cosine', 'dino_mismatched', 'decoder_id',
                'class_condition', 'phy_family', 'energy_constraint', 'N', 'E')
    normalized = []
    for original in rows:
        row = dict(original)
        if any(row.get(key) in (None, '') for key in required):
            raise ValueError('Missing registered frame identity/measurement: ' + str({k: row.get(k) for k in required}))
        row.setdefault('latent_sq_err_final', row.get('latent_sq_error', ''))
        latent = row['latent_sq_err_final']
        latent_valid = str(row.get('latent_valid', 'true')).lower() not in ('false', '0')
        if latent_valid and (latent in ('', None) or not np.isfinite(float(latent))):
            raise ValueError('A declared valid latent lacks its original-coordinate squared error')
        if not latent_valid:
            row['latent_sq_err_final'] = ''
        row['latent_valid'] = latent_valid
        row['latent_sq_error'] = row['latent_sq_err_final']
        row['energy'] = row['E']
        row['lambda'] = row.get('lambda', row.get('lambda_', ''))
        row['model_id'] = row.get('model_id') or row.get('base_model_id') or row['method']
        for key in ('N_header', 'N_body', 'source_bits', 'coded_bits', 'effective_code_rate', 'action_m',
                    'modulation', 'header_ok', 'body_crc_ok', 'rx_ms', 'training_seed'):
            row.setdefault(key, '')
        if int(row['N']) != expected_budget:
            raise ValueError('Analysis received another channel budget')
        energy = float(row['E'])
        if not np.isfinite(energy) or energy < 0:
            raise ValueError('Invalid actual energy')
        if row['energy_constraint'] == 'per_frame_2N' and not np.isclose(energy, 2 * expected_budget, rtol=1e-5, atol=.02):
            raise ValueError('Registered per-frame energy constraint violated')
        normalized.append(row)
    return normalized


def source_means(rows, baseline='P1024'):
    grouped = defaultdict(list)
    identities = {}
    for row in rows:
        sid = row['source_id']
        identity = (int(row['source_index']), row['preprocessing_id'])
        if sid in identities and identities[sid] != identity:
            raise ValueError('Source preprocessing or ordering changed across systems')
        identities[sid] = identity
        key = (float(row['snr_db']), row['method'], sid)
        values = {k: float(row[k]) for k in METRICS[:4]}
        values['latent_sq_err_final'] = float(row['latent_sq_err_final']) if row['latent_valid'] else float('nan')
        values.update(dino_lt_0_6=float(values['dino_cosine'] < .6),
                      lpips_gt_0_35=float(values['lpips_alex'] > .35),
                      dino_match_specificity=values['dino_cosine'] - values['dino_mismatched'],
                      energy=float(row['E']), header_failure=is_failure(row['header_ok']),
                      body_crc_failure=is_failure(row['body_crc_ok']))
        if not np.isfinite([v for k, v in values.items() if k != 'latent_sq_err_final']).all():
            raise ValueError('Nonfinite measured metric')
        grouped[key].append((int(row['noise_seed']), values, row))
    ids = sorted(identities, key=lambda sid: identities[sid][0])
    if len(ids) != 100 or {identities[sid][0] for sid in ids} != set(range(100)):
        raise ValueError('Expected the frozen 100 source images')
    snrs = sorted({k[0] for k in grouped})
    methods = sorted({k[1] for k in grouped})
    if snrs != list(map(float, SNRS)) or any(m not in methods for m in (baseline, 'A1_policy', 'A2_policy', 'V_policy', *DIGITAL)):
        raise ValueError('Missing registered SNR/method scope')
    matrix, means = {}, []
    for snr in snrs:
        for method in methods:
            vectors = []
            actions = set()
            for sid in ids:
                entries = grouped.get((snr, method, sid), [])
                if len(entries) != 3 or {seed for seed, _, _ in entries} != NOISE_SEEDS:
                    raise ValueError('Duplicate or incomplete source/noise grid: ' + str((snr, method, sid)))
                action = {(r.get('action_m', ''), r.get('lambda', ''), r.get('policy_action', '')) for _, _, r in entries}
                if len(action) != 1:
                    raise ValueError('Frozen SNR-level action changed across noise')
                actions.update(action)
                noise_values = np.asarray([[v[k] for k in METRICS] for _, v, _ in entries], dtype=np.float64)
                values = noise_values.mean(0)
                latent_noise = noise_values[:, METRICS.index('latent_sq_err_final')]
                values[METRICS.index('latent_sq_err_final')] = float(latent_noise[np.isfinite(latent_noise)].mean()) if np.isfinite(latent_noise).any() else float('nan')
                vectors.append(values)
                exemplar = entries[0][2]
                means.append(dict(source_id=sid, source_index=identities[sid][0], snr_db=snr,
                    method=method, noise_repeats=3, action_m=exemplar.get('action_m', ''),
                    lambda_=exemplar.get('lambda', ''), policy_action=exemplar.get('policy_action', ''),
                    decoder_id=exemplar['decoder_id'], class_condition=exemplar['class_condition'],
                    phy_family=exemplar['phy_family'], energy_constraint=exemplar['energy_constraint'],
                    latent_valid_noise_count=int(np.isfinite(latent_noise).sum()),
                    **{k: float(v) if np.isfinite(v) else None for k, v in zip(METRICS, values)}))
            if len(actions) != 1:
                raise ValueError('Policy changed across development source images')
            matrix[(snr, method)] = np.asarray(vectors)
    return matrix, means, ids, snrs, methods


def comparisons(methods):
    pairs = [('V_policy', c, 'H_R_primary') for c in ('P1024', 'A1_policy', 'A2_policy')]
    pairs += [(m, 'P1024', 'H_D_primary') for m in DIGITAL]
    pairs += [(m, 'P1024', 'control_vs_base') for m in ('A1_policy', 'A2_policy')]
    for kind in ('common', 'tok'):
        pairs += [('V_' + kind, p + '_' + kind, 'common_content' if kind == 'common' else 'token_diagnostic')
                  for p in ('A1', 'A2')]
    for family in ('QPSK', '16QAM'):
        pairs.append(('D_C_' + family, 'D_U_' + family, 'different_selected_actions_class_and_action'))
        pairs.append(('D_C_' + family + '_at_U_action', 'D_U_' + family, 'same_action_class_effect'))
        pairs.append(('D_C_' + family, 'D_U_' + family + '_at_C_action', 'same_action_class_effect'))
        for condition in ('U', 'C'):
            main = 'D_' + condition + '_' + family
            pairs += [(main, 'D_prefix_' + condition + '_' + family, 'same_observation_generation_effect'),
                      (main + '_D0', main, 'same_latent_decoder_effect')]
    pairs += [('D_' + condition + '_16QAM', 'D_' + condition + '_QPSK', 'separately_calibrated_phy_system_tradeoff')
              for condition in ('U', 'C')]
    primary = set(POLICIES) | set(DIGITAL)
    for method in methods:
        if method not in primary:
            pairs.append((method, 'P1024', 'diagnostic_vs_base'))
    missing = [(a, b) for a, b, _ in pairs if a not in methods or b not in methods]
    if missing:
        raise ValueError('Required diagnostic methods missing: ' + str(missing))
    return list(dict.fromkeys(pairs))


def verify_shared_receptions(rows, pairs):
    keyed = {(float(r['snr_db']), r['method'], r['source_id'], int(r['noise_seed'])): r for r in rows}
    receipts = []
    for method, control, role in pairs:
        if role not in ('same_action_class_effect', 'same_observation_generation_effect', 'same_latent_decoder_effect'):
            continue
        count = 0
        for (snr, candidate, sid, seed), row in keyed.items():
            if candidate != method:
                continue
            other = keyed[(snr, control, sid, seed)]
            for field in ('waveform_sha256', 'observation_sha256', 'action_m', 'N', 'E'):
                if row.get(field) in (None, '') or row[field] != other.get(field):
                    raise ValueError('Shared-reception diagnostic changed ' + field + ': ' + str((method, control, snr, sid, seed)))
            if role == 'same_latent_decoder_effect':
                if row['latent_valid'] != other['latent_valid']:
                    raise ValueError('D0/Dc diagnostic changed latent validity')
                if row['latent_valid'] and float(row['latent_sq_err_final']) != float(other['latent_sq_err_final']):
                    raise ValueError('D0/Dc diagnostic changed its received latent')
            count += 1
        receipts.append(dict(method=method, control=control, role=role, frames=count,
                             waveform_observation_action_energy_identical=True))
    return receipts


def analyze(matrix, snrs, methods, indices, means, raw, source_order):
    metadata = {(float(r['snr_db']), r['method']): r for r in means}
    summary = []
    for snr in snrs:
        for method in methods:
            exemplar = metadata[(snr, method)]
            row = dict(snr_db=snr, method=method, N=1024, sources=100, noise_repeats=3,
                       inference_unit='source_image', in_training_range=True,
                       **{k: exemplar[k] for k in ('decoder_id', 'class_condition', 'phy_family', 'energy_constraint',
                                                  'action_m', 'lambda_', 'policy_action')})
            for j, metric in enumerate(METRICS):
                ci = conditional_interval(matrix[(snr, method)][:, j], indices) if metric == 'latent_sq_err_final' else interval(matrix[(snr, method)][:, j], indices)
                row.update({metric: ci['mean'], metric + '_ci_low': ci['ci_low'], metric + '_ci_high': ci['ci_high']})
                if metric == 'latent_sq_err_final':
                    row.update(latent_valid_sources=ci['valid_sources'],
                        latent_valid_frames=sum(r['latent_valid_noise_count'] for r in means if float(r['snr_db']) == snr and r['method'] == method),
                        latent_statistic_scope='conditional on valid received latent; source mean over valid noise repeats')
            summary.append(row)
    paired, lookup = [], {}
    frames = {(float(r['snr_db']), r['method'], r['source_id'], int(r['noise_seed'])): r for r in raw}
    for snr in snrs:
        for method, control, role in comparisons(methods):
            diff = matrix[(snr, method)] - matrix[(snr, control)]
            for j, metric in enumerate(METRICS):
                raw = diff[:, j]
                valid_frames = 300
                if metric == 'latent_sq_err_final':
                    raw_values = []
                    valid_frames = 0
                    for sid in source_order:
                        repeated = []
                        for seed in sorted(NOISE_SEEDS):
                            a, b = frames[(snr, method, sid, seed)], frames[(snr, control, sid, seed)]
                            if a['latent_valid'] and b['latent_valid']:
                                repeated.append(float(a['latent_sq_err_final']) - float(b['latent_sq_err_final']))
                        valid_frames += len(repeated)
                        raw_values.append(float(np.mean(repeated)) if repeated else float('nan'))
                    raw = np.asarray(raw_values)
                    ci = conditional_interval(np.where(np.abs(raw) <= ZERO, 0., raw), indices)
                else:
                    ci = interval(np.where(np.abs(raw) <= ZERO, 0., raw), indices)
                name = 'delta_specific' if metric == 'dino_match_specificity' else metric
                row = dict(snr_db=snr, N=1024, method=method, control=control, comparison_role=role,
                           metric=name, delta=ci['mean'], ci_low=ci['ci_low'], ci_high=ci['ci_high'],
                           raw_delta_mean=float(raw[np.isfinite(raw)].mean()) if np.isfinite(raw).any() else None, statistical_zero_tolerance=ZERO,
                           sources=100, noise_repeats=3, bootstrap_resamples=RESAMPLES,
                           bootstrap_seed=SEED, multiple_comparison_adjustment='none',
                           observation_pairing='same P1024 observation for plugin; source/noise repetition between complete systems')
                if metric == 'latent_sq_err_final':
                    row.update(valid_paired_frames=valid_frames, valid_paired_sources=ci['valid_sources'],
                               statistic_scope='conditional intersection of frames with valid latent in both methods')
                paired.append(row)
                lookup[(snr, method, control, name)] = row
    return summary, paired, lookup


def gain_branches(lookup, snr, method, control):
    lp = lookup[(snr, method, control, 'lpips_alex')]
    di = lookup[(snr, method, control, 'dino_cosine')]
    specific = lookup[(snr, method, control, 'delta_specific')]
    supported, clear = [], []
    if lp['delta'] < 0 and lp['ci_high'] < 0:
        supported.append('LPIPS')
        if lp['delta'] <= -.02:
            clear.append('LPIPS')
    if di['delta'] > 0 and di['ci_low'] > 0 and specific['ci_low'] > 0:
        supported.append('DINO_WITH_SPECIFICITY')
        if di['delta'] >= .02:
            clear.append('DINO_WITH_SPECIFICITY')
    return dict(supported_branches=supported, clear_gain_branches=clear,
                delta_lpips=lp, delta_dino=di, delta_specific=specific)


def decisions(snrs, lookup, policy):
    records = []
    for snr in snrs:
        level = policy['levels'][str(int(snr))]
        selected = level['methods']['V']
        controls = {c: gain_branches(lookup, snr, 'V_policy', c) for c in ('P1024', 'A1_policy', 'A2_policy')}
        common_clear = set.intersection(*(set(x['clear_gain_branches']) for x in controls.values()))
        common_supported = set.intersection(*(set(x['supported_branches']) for x in controls.values()))
        cap = .2 if snr == 13 else .3
        psnr = lookup[(snr, 'V_policy', 'P1024', 'psnr_db')]
        if level['psnr_drop_cap_db'] != cap or selected['calibration_psnr_drop_cap_db'] != cap:
            raise ValueError('Plugin calibration PSNR cap differs from the registered protocol')
        actual_calibration_delta = selected['calibration']['psnr_db'] - level['P1024']['psnr_db']
        if abs(actual_calibration_delta - selected['calibration_psnr_delta_db']) > ZERO:
            raise ValueError('Plugin calibration PSNR receipt is inconsistent')
        if bool(selected['raw_feasible']) != (actual_calibration_delta >= -cap):
            raise ValueError('Plugin calibration feasibility receipt is inconsistent')
        feasible = bool(selected['raw_feasible'])
        if selected['policy_action'] == 'BYPASS':
            label = 'BYPASS_SELECTED'
        elif feasible and common_clear:
            label = 'CLEAR_GAIN'
        elif feasible and common_supported:
            label = 'SMALL_SUPPORTED_GAIN'
        elif feasible and any(x['supported_branches'] for x in controls.values()):
            label = 'PARTIAL_GAIN'
        else:
            label = 'NO_GAIN_ESTABLISHED'
        records.append(dict(hypothesis='H_R', snr_db=snr, N=1024, method='V_policy', label=label,
            policy_action=selected['policy_action'], same_branch_clear=sorted(common_clear),
            same_branch_supported=sorted(common_supported), controls=controls,
            calibration_psnr_feasible=bool(selected['raw_feasible']), observed_psnr_constraint_pass=psnr['delta'] >= -cap,
            psnr_drop_cap_db=cap, delta_psnr=psnr, energy_constraint='per_frame_2N', class_condition='unconditional',
            calibration_psnr_delta_db=actual_calibration_delta, observed_psnr_is_diagnostic_only=True,
            interpretation='same frozen branch must exceed all three controls; calibration PSNR selects policy, observed PSNR reports cost; bypass is not correction success'))
        for method in DIGITAL:
            branches = gain_branches(lookup, snr, method, 'P1024')
            label = 'CLEAR_GAIN' if branches['clear_gain_branches'] else 'SMALL_SUPPORTED_GAIN' if branches['supported_branches'] else 'NO_GAIN_ESTABLISHED'
            condition = 'unconditional' if method.startswith('D_U_') else 'paid_received_class'
            family = 'QPSK' if method.endswith('_QPSK') else '16QAM'
            records.append(dict(hypothesis='H_D', snr_db=snr, N=1024, method=method, label=label,
                class_condition=condition, phy_family=family,
                energy_constraint='per_frame_2N' if family == 'QPSK' else 'original_constellation_actual_frame_energy',
                strict_same_information_frame_energy=condition == 'unconditional' and family == 'QPSK',
                branches=branches, delta_psnr=lookup[(snr, method, 'P1024', 'psnr_db')],
                delta_latent_sq_error=lookup[(snr, method, 'P1024', 'latent_sq_err_final')],
                interpretation='perception/semantic tradeoff; no PSNR-win requirement; resources and class condition retained'))
    return records


def energy_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(float(row['snr_db']), row['method'])].append(row)
    result = []
    for (snr, method), frames in sorted(groups.items()):
        energy = np.asarray([float(r['E']) for r in frames])
        result.append(dict(snr_db=snr, method=method, frames=len(frames), N=1024,
            energy_constraint=frames[0]['energy_constraint'], mean=float(energy.mean()), min=float(energy.min()),
            p05=float(np.percentile(energy, 5)), p95=float(np.percentile(energy, 95)), max=float(energy.max()),
            frame_energy_2N=bool(np.all(np.isclose(energy, 2048., atol=.02, rtol=1e-5)))))
    return result


def timing_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if str(row.get('warmup', '')).lower() in ('true', '1') or row.get('phase') == 'warmup':
            continue
        groups[(float(row['snr_db']), row['method'])].append(row)
    result = []
    for (snr, method), calls in sorted(groups.items()):
        if len(calls) != 20 or len({r['source_id'] for r in calls}) != 10:
            raise ValueError('Timing must cover ten fixed sources and two repeats per method/SNR')
        row = dict(snr_db=snr, method=method, calls=len(calls), sources=10,
                   endpoints='CPU received waveform -> CPU float image; channel draw excluded',
                   ran_VAR=';'.join(sorted({str(r.get('ran_VAR', '')) for r in calls})),
                   policy_action=';'.join(sorted({r.get('policy_action', '') for r in calls})))
        for key in ('rx_ms', 'P_receive_ms', 'correction_ms', 'decode_ms', 'source_encode_ms', 'tx_ms', 'var_ms', 'phy_decode_ms'):
            values = [float(r[key]) for r in calls if r.get(key) not in ('', None)]
            if values:
                if not np.isfinite(values).all() or min(values) < 0:
                    raise ValueError('Invalid actual timing')
                row.update({key + '_mean': float(np.mean(values)), key + '_median': float(np.median(values)),
                            key + '_p95': float(np.percentile(values, 95))})
        result.append(row)
    return result


def token_summary(results, ids, indices):
    rows = csv_rows(results / 'token_accuracy.csv')
    groups = defaultdict(list)
    for row in rows:
        key = (float(row['snr_db']), row['method'], row['mode'], row['source_id'], int(row['scale']))
        groups[key].append(row)
    summary = []
    for snr, prior, mode in sorted({k[:3] for k in groups}):
        for metric in ('acc', 'path_accuracy', 'entropy', 'logp_true'):
            equal, weighted = [], []
            for sid in ids:
                means, weights = [], []
                for scale, pn in enumerate((1, 2, 3, 4, 5, 6, 8, 10, 13, 16), 1):
                    entries = groups[(snr, prior, mode, sid, scale)]
                    if len(entries) != 3 or {int(r['noise_seed']) for r in entries} != NOISE_SEEDS:
                        raise ValueError('Incomplete token diagnostic grid')
                    values = [float(r[metric]) for r in entries if r.get(metric) not in ('', None)]
                    if not values:
                        continue
                    if len(values) != 3 or {int(r['token_count']) for r in entries} != {pn * pn}:
                        raise ValueError('Token scale or metric coverage changed')
                    means.append(float(np.mean(values))); weights.append(pn * pn)
                if not means:
                    continue
                if len(means) != 10:
                    raise ValueError('Token diagnostic requires all ten scales')
                equal.append(float(np.mean(means))); weighted.append(float(np.average(means, weights=weights)))
            for weighting, values in (('scale_equal', equal), ('token_count_weighted', weighted)):
                if values:
                    ci = interval(values, indices)
                    summary.append(dict(snr_db=snr, method=prior, mode=mode, metric=metric, weighting=weighting,
                        mean=ci['mean'], ci_low=ci['ci_low'], ci_high=ci['ci_high'], sources=100, noise_repeats=3,
                        bootstrap_seed=SEED, bootstrap_resamples=RESAMPLES, total_tokens=680,
                        target='receiver_path_conditioned' if metric == 'path_accuracy' else 'original_encoding_tokens' if metric == 'acc' else 'score_diagnostic'))
    write_csv(results / 'token_summary.csv', summary)
    return summary


def delta_text(row, digits=5):
    if row['delta'] is None:
        return '不适用（没有双方有效 latent）'
    text = f"{row['delta']:.{digits}f} [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"
    if 'valid_paired_frames' in row:
        text += f" (有效{row['valid_paired_frames']}/300)"
    return text


def value_text(value, digits=3):
    return '不适用' if value is None else f'{value:.{digits}f}'


def compatible_budget_references(results, out):
    path = results / 'compatible_budget_references.csv'
    if not path.exists():
        return [], dict(status='no_compatible_old_budget_points', plotted_points=0)
    identity_path = results / 'compatible_budget_reference_identity.json'
    identity = read_json(identity_path)
    receipt = read_json(out / 'reference_import.json')
    if sha(path) != identity['reference_csv_sha256'] or sha(path) != receipt['reference_csv_sha256']:
        raise ValueError('Historical reference CSV changed after compatibility verification')
    if sha(identity_path) != receipt['identity_sha256']:
        raise ValueError('Historical reference identity changed after evidence import')
    if identity['synthetic'] or identity['status'] != 'READ_ONLY_HISTORICAL_CURVE_COMPATIBILITY_VERIFIED':
        raise ValueError('Historical reference is not a measured verified archive')
    if not identity['validation']['full_unique_source_snr_noise_grids'] or identity['missing']['fill_missing_metrics']:
        raise ValueError('Historical reference coverage or missing-metric boundary changed')
    measured = []
    for row in csv_rows(path):
        if str(row.get('compatibility_verified', '')).lower() not in ('true', '1'):
            continue
        if str(row.get('plot_only', '')).lower() not in ('true', '1') or str(row.get('new_H_R_or_CI_eligible', '')).lower() not in ('false', '0'):
            raise ValueError('Historical points must remain plot references, outside new gates and intervals')
        if int(row['N']) not in (2048, 3060, 4084) or int(float(row['snr_db'])) not in SNRS:
            raise ValueError('Unexpected old budget reference scope')
        if row['decoder_id'] != 'Dc' or row['energy_constraint'] != 'per_frame_2N' or row['class_condition'] != 'unconditional':
            raise ValueError('Old reference has different mechanism conditions')
        if not row.get('reference_path') or not row.get('model_id', row.get('checkpoint_id', '')):
            raise ValueError('Old compatible point lacks provenance')
        for metric in ('psnr_db', 'lpips_alex', 'dino_cosine'):
            if not np.isfinite(float(row[metric])):
                raise ValueError('Old point metric is nonfinite')
        measured.append(row)
    keys = [(int(r['N']), float(r['snr_db']), r['method']) for r in measured]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate old compatible point')
    return measured, dict(status='audited_compatible_old_budget_points', plotted_points=len(measured),
        source_file=str(path), source_sha256=sha(path),
        identity_file=str(identity_path), identity_sha256=sha(identity_path),
        evidence_import_receipt=str(out / 'reference_import.json'),
        source_ids=identity['source_ids'], preprocessing_ids=identity['preprocessing_ids'],
        decoder_state_sha256=identity['decoder_state_sha256'],
        models=identity['models'], caveats=identity['caveats'],
        interpretation='independently trained historical pure continuous models; source/decoder/metric/resource compatibility audited separately')


def training_budget(done, selected, budget):
    if not done['state']['finished'] or done['development_read'] or done['convergence_claimed']:
        raise ValueError('Training completion/selection boundary changed')
    if done['selected']['P' + str(budget)] != selected:
        raise ValueError('Training and evaluated selected checkpoint differ')
    decision = done['state']['decisions'][-1]
    return dict(N=budget, training_seed=selected['training_seed'], selected_updates=selected['step'],
        completed_updates=done['state']['step'], checkpoint_sha256=selected['checkpoint_sha256'],
        budget_truncated=done['budget_truncated'], stop_reason=decision['reason'],
        final_relative_improvements=decision['relative_improvements'], convergence_claimed=False)


def prior_budget_data(results, out, current_config, current_raw, source_order):
    receipt = read_json(out / 'prior_budget_import.json')
    if receipt['status'] != 'IMMUTABLE_N512_COMPARISON_IMPORTED' or receipt['N'] != 512:
        raise ValueError('Expected the executed N512 comparison receipt')
    for path, expected in receipt['source_bindings'].items():
        if sha(path) != expected:
            raise ValueError('Bound N512 comparison evidence changed: ' + path)
    folder = Path(receipt['original_result'])
    previous_config = read_json(folder / 'config.json')
    previous_completed = read_json(folder / 'analysis_completion.json')
    previous_raw = normalize_rows(csv_rows(folder / 'per_frame.csv'), expected_budget=512)
    if sha(folder / 'per_frame.csv') != previous_completed['files']['per_frame.csv']:
        raise ValueError('N512 measured table differs from its scientific receipt')
    for model in ('vae', 'var', 'decoder', 'lpips', 'dino'):
        if previous_config['plugin']['identity']['models'][model] != current_config['identity']['models'][model]:
            raise ValueError('Cross-budget visual/quality model differs: ' + model)
    if previous_config['plugin']['identity']['quality'] != current_config['identity']['quality']:
        raise ValueError('Cross-budget quality preprocessing/version differs')
    if previous_config['plugin']['identity']['stats_sha256'] != current_config['identity']['stats_sha256']:
        raise ValueError('Cross-budget common F normalization differs')
    if previous_config['plugin']['mismatch_permutation'] != current_config['mismatch_permutation']:
        raise ValueError('Cross-budget mismatch source permutation differs')
    matrix, means, ids, snrs, methods = source_means(previous_raw, baseline='P512')
    if ids != source_order or receipt['source_ids'] != ids or previous_completed['source_order'] != ids:
        raise ValueError('Cross-budget source order differs')
    current_pre = {r['source_id']: r['preprocessing_id'] for r in current_raw}
    previous_pre = {r['source_id']: r['preprocessing_id'] for r in previous_raw}
    if current_pre != previous_pre or receipt['preprocessing_ids'] != [current_pre[sid] for sid in ids]:
        raise ValueError('Cross-budget preprocessing differs')
    reconstructed = {(s, m): values for (s, m), values in matrix.items()}
    for row in csv_rows(folder / 'summary.csv'):
        values = reconstructed[(float(row['snr_db']), row['method'])]
        for metric in ('psnr_db', 'lpips_alex', 'dino_cosine'):
            mean = float(values[:, METRICS.index(metric)].mean())
            if abs(mean - float(row[metric])) > 1e-10:
                raise ValueError('N512 raw table does not reproduce its published means')
    write_csv(results / 'N512_source_means.csv', means)
    previous_decisions = read_json(folder / 'decisions.json')
    identity = dict(receipt=receipt, receipt_sha256=sha(out / 'prior_budget_import.json'),
        original_config_sha256=sha(folder / 'config.json'), original_per_frame_sha256=sha(folder / 'per_frame.csv'),
        original_policy_sha256=sha(folder / 'selected_policy.json'), original_analysis_sha256=sha(folder / 'analysis_completion.json'),
        original_decisions=previous_decisions, budget=receipt['budget'], sources=100, noise_repeats=3,
        pairing='source/preprocessing/SNR/noise-repeat index; separately transmitted N512 and N1024 observations',
        source_reuse_correlated=True, training_seed_variance_estimated=False)
    write_json(results / 'budget_comparison_identity.json', identity)
    return previous_raw, matrix, means, methods, identity


def latent_term_vector(terms, source_order, snr):
    """Only the common set of valid noise frames can define a latent contrast."""
    vector = []
    valid_frames = 0
    for sid in source_order:
        repeated = []
        for seed in sorted(NOISE_SEEDS):
            rows = [(frames[(float(snr), method, sid, seed)], coefficient) for frames, method, coefficient in terms]
            if all(row['latent_valid'] for row, _ in rows):
                repeated.append(sum(coefficient * float(row['latent_sq_err_final']) for row, coefficient in rows))
        valid_frames += len(repeated)
        vector.append(float(np.mean(repeated)) if repeated else float('nan'))
    return np.asarray(vector), valid_frames


def cross_budget_analysis(current_raw, current_matrix, previous_raw, previous_matrix, methods, ids, indices):
    frame_key = lambda r: (float(r['snr_db']), r['method'], r['source_id'], int(r['noise_seed']))
    current_frames = {frame_key(r): r for r in current_raw}
    previous_frames = {frame_key(r): r for r in previous_raw}
    paired, changed, source_rows, resource_rows = [], [], [], []
    bootstrap_hash = hashlib.sha256(indices.tobytes()).hexdigest()

    def row_for(snr, method, control, role, metric, vector, valid_frames=300):
        stable = np.where(np.abs(vector) <= ZERO, 0., vector)
        ci = conditional_interval(stable, indices) if metric == 'latent_sq_err_final' else interval(stable, indices)
        name = 'delta_specific' if metric == 'dino_match_specificity' else metric
        row = dict(N=1024, reference_N=512, snr_db=snr, method=method, control=control,
            comparison_role=role, metric=name, delta=ci['mean'], ci_low=ci['ci_low'], ci_high=ci['ci_high'],
            raw_delta_mean=float(vector[np.isfinite(vector)].mean()) if np.isfinite(vector).any() else None,
            statistical_zero_tolerance=ZERO, sources=100, noise_repeats=3,
            bootstrap_seed=SEED, bootstrap_resamples=RESAMPLES, bootstrap_indices_sha256=bootstrap_hash,
            multiple_comparison_adjustment='none', source_reuse_correlated=True,
            training_seed_variance_estimated=False,
            pairing='same source/preprocessing/SNR/noise-repeat index; distinct N/E and observed waveforms')
        if metric == 'latent_sq_err_final':
            row.update(valid_paired_frames=valid_frames, valid_paired_sources=ci['valid_sources'],
                statistic_scope='conditional intersection of valid latent frames in all contrast terms')
        return row

    for snr in SNRS:
        for method in methods:
            old_method = 'P512' if method == 'P1024' else method
            if (float(snr), old_method) not in previous_matrix:
                raise ValueError('Executed N512 counterpart missing: ' + old_method)
            for sid in ids:
                for seed in sorted(NOISE_SEEDS):
                    a, b = current_frames[(float(snr), method, sid, seed)], previous_frames[(float(snr), old_method, sid, seed)]
                    for field in ('preprocessing_id', 'source_index', 'decoder_id', 'class_condition', 'phy_family', 'energy_constraint'):
                        if a[field] != b[field]:
                            raise ValueError('Cross-budget matched-system conditions differ: ' + field)
                    resource_rows.append(dict(source_id=sid, snr_db=snr, noise_seed=seed,
                        current_method=method, reference_method=old_method, N_current=1024, N_reference=512,
                        E_current=float(a['E']), E_reference=float(b['E']),
                        waveform_sha_current=a['waveform_sha256'], waveform_sha_reference=b['waveform_sha256'],
                        observation_sha_current=a['observation_sha256'], observation_sha_reference=b['observation_sha256'],
                        same_waveform=a['waveform_sha256'] == b['waveform_sha256'],
                        same_observation=a['observation_sha256'] == b['observation_sha256'],
                        action_m_current=a['action_m'], action_m_reference=b['action_m'],
                        energy_constraint=a['energy_constraint'],
                        pairing='source/noise repeat only; N and total energy differ'))
            difference = current_matrix[(float(snr), method)] - previous_matrix[(float(snr), old_method)]
            for j, metric in enumerate(METRICS):
                vector, count = (latent_term_vector([(current_frames, method, 1), (previous_frames, old_method, -1)], ids, snr)
                                 if metric == 'latent_sq_err_final' else (difference[:, j], 300))
                row = row_for(snr, 'N1024/' + method, 'N512/' + old_method, 'cross_budget_same_method', metric, vector, count)
                paired.append(row)
        # Digital-vs-matched-P tradeoff change: (D1024-P1024)-(D512-P512).
        for method in DIGITAL:
            new_delta = current_matrix[(float(snr), method)] - current_matrix[(float(snr), 'P1024')]
            old_delta = previous_matrix[(float(snr), method)] - previous_matrix[(float(snr), 'P512')]
            for j, metric in enumerate(METRICS):
                vector, count = (latent_term_vector([(current_frames, method, 1), (current_frames, 'P1024', -1),
                    (previous_frames, method, -1), (previous_frames, 'P512', 1)], ids, snr)
                    if metric == 'latent_sq_err_final' else (new_delta[:, j] - old_delta[:, j], 300))
                row = row_for(snr, method, 'matched_P_at_each_N', 'digital_vs_P_tradeoff_change_N1024_minus_N512', metric, vector, count)
                row['contrast'] = '(D_N1024-P1024)-(D_N512-P512)'
                changed.append(row)
                for source_index, sid in enumerate(ids):
                    source_rows.append(dict(source_id=sid, source_index=source_index, snr_db=snr, method=method,
                        metric=row['metric'], contrast=row['contrast'],
                        change=float(vector[source_index]) if np.isfinite(vector[source_index]) else '', noise_repeats=3,
                        latent_scope='four-way valid frame intersection' if metric == 'latent_sq_err_final' else 'all measured frames'))
    return paired, changed, source_rows, resource_rows


def make_figures(results, summary, paired, plugin_policy, digital_policy, references, prior_matrix, tradeoff_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'figure.dpi': 140, 'savefig.bbox': 'tight'})
    figures = results / 'figures'; figures.mkdir(exist_ok=True)
    indexed = {(r['snr_db'], r['method']): r for r in summary}
    colors = {'P1024': '#555555', 'A1_policy': '#d18a15', 'A2_policy': '#49852e', 'V_policy': '#3769c5',
              'D_U_QPSK': '#bb4282', 'D_C_QPSK': '#7f3591', 'D_U_16QAM': '#b76a48', 'D_C_16QAM': '#c53735'}
    def save(fig, name):
        for suffix in ('png', 'svg'): fig.savefig(figures / (name + '.' + suffix))
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, metric in zip(axes, ('psnr_db', 'lpips_alex', 'dino_cosine')):
        for method in (*POLICIES, *DIGITAL):
            ax.plot(SNRS, [indexed[(float(s), method)][metric] for s in SNRS], 'o-', label=method, color=colors[method])
        ax.set(xlabel='SNR (dB)', ylabel=metric); ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    save(fig, 'quality_snr_conditions')
    for role, methods, name in [('H_D_primary', DIGITAL, 'digital_paired_deltas'), ('H_R_primary', ('V_policy',), 'plugin_paired_deltas')]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        for ax, metric in zip(axes, ('psnr_db', 'lpips_alex', 'dino_cosine')):
            groups = sorted({(r['method'], r['control']) for r in paired if r['comparison_role'] == role})
            for offset, (method, control) in zip(np.linspace(-.14, .14, len(groups)), groups):
                cells = sorted((r for r in paired if r['method'] == method and r['control'] == control and r['metric'] == metric), key=lambda r:r['snr_db'])
                values = np.asarray([r['delta'] for r in cells])
                ax.errorbar(np.asarray([r['snr_db'] for r in cells]) + offset, values,
                    yerr=[values - np.asarray([r['ci_low'] for r in cells]), np.asarray([r['ci_high'] for r in cells]) - values],
                    fmt='o-', capsize=2, label=method + ' - ' + control)
            ax.axhline(0, color='#888888', linewidth=.8); ax.set(xlabel='SNR (dB)', ylabel='Paired ' + metric); ax.grid(alpha=.2)
        axes[0].legend(fontsize=6); save(fig, name)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for row, kind in enumerate(('common', 'tok')):
        for ax, metric in zip(axes[row], ('psnr_db', 'lpips_alex', 'dino_cosine')):
            for prior in ('A1', 'A2', 'V'):
                ax.plot(SNRS, [indexed[(float(s), prior + '_' + kind)][metric] for s in SNRS], 'o-', label=prior + '_' + kind)
            ax.set(xlabel='SNR (dB)', ylabel=metric, title='A1 common alpha' if kind == 'common' else 'Before fusion'); ax.grid(alpha=.2)
    axes[0, 0].legend(); axes[1, 0].legend(); save(fig, 'common_content_and_token')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for prior in ('A1', 'A2', 'V'):
        cells = [plugin_policy['levels'][str(s)]['methods'][prior] for s in SNRS]
        axes[0].plot(SNRS, [float(c['raw_selected_lambda']) for c in cells], 'o-', label=prior)
        axes[1].plot(SNRS, [float(c['policy_action'] == 'BYPASS') for c in cells], 'o-', label=prior)
        axes[2].plot(SNRS, [float(np.mean(c['alpha'])) for c in cells], 'o-', label=prior)
    for ax, name in zip(axes, ('Selected raw lambda', 'BYPASS selected', 'Own alpha channel mean')):
        ax.set(xlabel='SNR (dB)', ylabel=name); ax.grid(alpha=.2)
    axes[0].legend(); save(fig, 'plugin_policy')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, family in zip(axes, ('QPSK', '16QAM')):
        for condition in ('U', 'C'):
            ax.plot(SNRS, [int(digital_policy['levels'][str(s)][family][condition]['action_m']) for s in SNRS],
                    'o-', label=condition + ' frozen action')
        ax.set(xlabel='SNR (dB)', ylabel='raw prefix scale m', title=family, yticks=[4, 5, 6, 7]); ax.grid(alpha=.2); ax.legend()
    save(fig, 'digital_m_policy_known_phy')
    fig, axes = plt.subplots(3, len(SNRS), figsize=(15, 8), constrained_layout=True)
    # Each N512/N1024 point uses that budget's already frozen selected policy.
    for col, snr in enumerate(SNRS):
        for ax, metric in zip(axes[:, col], ('psnr_db', 'lpips_alex', 'dino_cosine')):
            for method in (*POLICIES, *DIGITAL):
                old_method = 'P512' if method == 'P1024' else method
                old_quality = float(prior_matrix[(float(snr), old_method)][:, METRICS.index(metric)].mean())
                ax.plot([512, 1024], [old_quality, indexed[(float(snr), method)][metric]],
                    color=colors[method], label='P_N' if method == 'P1024' else method,
                    marker='x' if '16QAM' in method else '^' if 'D_C' in method else 'o',
                    linestyle='--' if '16QAM' in method else '-', alpha=.8)
            old = [r for r in references if int(float(r['snr_db'])) == snr]
            if old:
                base_points = [(512, float(prior_matrix[(float(snr), 'P512')][:, METRICS.index(metric)].mean())),
                               (1024, indexed[(float(snr), 'P1024')][metric])]
                for point in old:
                    ax.scatter([int(point['N'])], [float(point[metric])], marker='s', color='#777777', label=point['method'])
                    base_points.append((int(point['N']), float(point[metric])))
                ordered = sorted(base_points)
                ax.plot([n for n, _ in ordered], [q for _, q in ordered], ':', color='#999999', linewidth=.8)
            xmax = max([1024] + [int(r['N']) for r in old])
            ax.set(xlabel='Actual complex symbols N', ylabel=metric, title=f'{snr} dB', xlim=(480, xmax + max(32, xmax * .04))); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=5)
    fig.suptitle('Quality versus resources at fixed symbol power; actual N512/N1024 policies and older P models, with distinct training histories')
    save(fig, 'quality_resources_completed_points')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, metric in zip(axes, ('psnr_db', 'lpips_alex', 'dino_cosine')):
        for offset, method in zip(np.linspace(-.15, .15, len(DIGITAL)), DIGITAL):
            cells = sorted((r for r in tradeoff_rows if r['method'] == method and r['metric'] == metric), key=lambda r:r['snr_db'])
            y = np.asarray([r['delta'] for r in cells])
            ax.errorbar(np.asarray([r['snr_db'] for r in cells]) + offset, y,
                yerr=[y - np.asarray([r['ci_low'] for r in cells]), np.asarray([r['ci_high'] for r in cells]) - y],
                fmt='o-', capsize=2, label=method)
        ax.axhline(0, color='#777777', linewidth=.8)
        ax.set(xlabel='SNR (dB)', ylabel='Change in digital - matched P: ' + metric); ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    fig.suptitle('(D1024-P1024)-(D512-P512): shared source bootstrap, separate transmitted observations')
    save(fig, 'digital_tradeoff_change_between_budgets')
    return [str(p.relative_to(results)) for p in sorted(figures.glob('*'))]


def examples(results, raw):
    from PIL import Image, ImageDraw
    rows = {(int(r['source_index']), int(float(r['snr_db'])), r['method']): r for r in raw if int(r['noise_seed']) == 2001}
    files = []
    for source in (0, 25, 50, 75):
        for snr in (4, 13):
            methods = ('P1024', 'D_U_QPSK', 'D_C_QPSK', 'D_U_16QAM', 'D_C_16QAM', 'V_policy')
            path = results / 'examples' / f'source_{source:03d}_snr_{snr}_combined.png'
            images = [('Source', results / 'examples/plugin' / f'source_{source:03d}_snr_{snr}_Source_seed2001.png')]
            for method in methods:
                origin = 'plugin' if method in ('P1024', 'V_policy') else 'digital'
                images.append((method, results / 'examples' / origin / f'source_{source:03d}_snr_{snr}_{method}_seed2001.png'))
            if any(not p.exists() for _, p in images):
                raise ValueError('Missing fixed example image: ' + str([str(p) for _, p in images if not p.exists()]))
            canvas = Image.new('RGB', (256 * len(images), 316), 'white'); draw = ImageDraw.Draw(canvas)
            for col, (method, image_path) in enumerate(images):
                canvas.paste(Image.open(image_path).convert('RGB'), (256 * col, 60))
                draw.text((256 * col + 3, 3), method, fill='black')
                if method != 'Source':
                    r = rows[(source, snr, method)]
                    text = f"{r['decoder_id']} {r['phy_family']} m={r.get('action_m', '')} E={float(r['E']):.1f}\n"
                    text += f"header={r.get('header_ok', '')} CRC={r.get('body_crc_ok', '')} {r.get('policy_action', '')}"
                    draw.multiline_text((256 * col + 3, 19), text, fill='black', spacing=1)
            canvas.save(path); files.append(str(path.relative_to(results)))
    return files


def make_report(results, summary, lookup, records, policy, timing, energy, config, figures, example_files, budget_paired, tradeoff_paired):
    indexed = {(r['snr_db'], r['method']): r for r in summary}
    clear_hd = [r for r in records if r['hypothesis'] == 'H_D' and r['label'] == 'CLEAR_GAIN']
    strict_hd = [r for r in clear_hd if r['strict_same_information_frame_energy']]
    hr = [r for r in records if r['hypothesis'] == 'H_R']
    clear_hr = [r for r in hr if r['label'] == 'CLEAR_GAIN']
    selected = config['plugin']['identity']['selected']
    choice = lambda records: '、'.join(f"{r['snr_db']:g} dB {r['method']} ({'/'.join(r.get('same_branch_clear', r.get('branches', {}).get('clear_gain_branches', [])))})" for r in records) or '无'
    current_budget = config['training_budgets']['N1024']
    previous_budget = config['training_budgets']['N512']
    budget_note = lambda b: '预算截断且末段仍改善，不能称理论收敛' if b['budget_truncated'] else '按既定停止规则结束，不作理论收敛结论'
    lines = ['# N1024 第二阶段及 N512 资源取舍比较', '',
        f"P1024 单个训练 seed=2026093001；完成更新 {current_budget['completed_updates']}；校准选中更新 {selected['step']}；checkpoint SHA-256 `{selected['checkpoint_sha256']}`。{budget_note(current_budget)}。", '',
        f"N512 已完成更新 {previous_budget['completed_updates']}、选中 {previous_budget['selected_updates']}；{budget_note(previous_budget)}。本轮保留其原始模型、策略和结果，完成用户已授权的 N1024 第二阶段，并在相同100源图/三噪声重复上比较两预算。", '',
        '1. **同 Dc、无额外类别的数字分工是否胜过 P1024？** 严格逐帧同能量无条件 QPSK 的明显增益点：' + choice(strict_hd) + '。下面同时给出 PSNR、F 平方和及配对区间。',
        '2. **属于什么资源口径？** 连续与 QPSK 的每帧 E=2048、N=1024；16QAM 保留原星座，其实测帧能量分布另表。全部数字明显信号：' + choice(clear_hd) + '。类别辅助与 16QAM 点分别限定解释。',
        '3. **插件是否同时超过 P、A1、A2？** 同一个质量分支同时超过三控制的明显点：' + choice(clear_hr) + '。真正 BYPASS 不计为修正成功；共同 A1 alpha 的内容差值单列。',
        '4. **D0、类别、失败与计算改变多少结论？** 同 latent 的 D0−Dc、同动作接收前缀的 C−U、生成−prefix、失败率及 CPU 波形→图像时间均列在下表。类别来自原应用/数据标签，通过付费头接收；该来源本身不等于部署可免费取得正确类别。',
        '5. **从 N512 到 N1024 的取舍如何变化？** 后文给出全部同方法跨预算差值，以及 (D1024−P1024)−(D512−P512) 的源图配对区间。这区分数字系统自身改善和匹配连续基线变强，不把两个预算当作独立测试。完成本轮后停止；未自动启动新预算、生成模型训练或旧队列。', '',
        '## 同 Dc 主表', '',
        '所有主输出使用同视觉 Encoder、32×16×16 的 F 及冻结 Dc。先对每张源图三个噪声求均值，再对 100 张源图平均。原坐标 F 平方和除以 8192 才是均方误差。', '',
        '| SNR | 方法 | PSNR dB | LPIPS alex | DINO | F平方和 | DINO<0.6 | LPIPS>0.35 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for snr in SNRS:
        for method in (*POLICIES, *DIGITAL):
            r = indexed[(float(snr), method)]
            latent = value_text(r['latent_sq_err_final']) + f" (有效{r['latent_valid_frames']}/300)"
            lines.append(f"| {snr} | {method} | {r['psnr_db']:.4f} | {r['lpips_alex']:.5f} | {r['dino_cosine']:.5f} | {latent} | {r['dino_lt_0_6']:.4f} | {r['lpips_gt_0_35']:.4f} |")
    lines += ['', '两个阈值比例仅为所设指标阈值的失效比例，不能直接称业务可靠性。数字头失败沿原规则输出RGB=0.5，没有实际latent；像素/LPIPS/DINO统计和区间仍计入全部失败帧，F平方和单独在有效latent条件下统计。主表标明有效帧数，paired.csv的F差值只使用双方有效帧的交集并标明帧数和源图数。这些条件F误差不能解释为全体失败帧的系统F误差。raw m4/m5/m6/m7 为 360/660/1092/1860 bit；数字 header 为 68 个复符号、正文 956 个，CRC16 与 6 tail bit 均计入。本轮不使用算术码均值。', '',
              '## H_D：数字分工与匹配训练 P1024', '',
              '| SNR | 方法 | 标签/条件 | ΔPSNR [95% CI] | ΔLPIPS [95% CI] | ΔDINO [95% CI] | Δspecific [95% CI] | ΔF平方和 [95% CI] |',
              '|---:|---|---|---|---|---|---|---|']
    for r in records:
        if r['hypothesis'] == 'H_D':
            values = [delta_text(lookup[(r['snr_db'], r['method'], 'P1024', metric)], 3 if metric.startswith('latent') else 5)
                      for metric in ('psnr_db', 'lpips_alex', 'dino_cosine', 'delta_specific', 'latent_sq_err_final')]
            branch = '/'.join(r['branches']['clear_gain_branches'] or r['branches']['supported_branches']) or '无支持分支'
            lines.append(f"| {r['snr_db']:g} | {r['method']} | {r['label']} ({branch}); {r['class_condition']}; {r['energy_constraint']} | " + ' | '.join(values) + ' |')
    lines += ['', 'H_D 是感知/语义取舍，未要求 PSNR 获胜。16QAM 的 signal 仅属于当前实际 N/E 工作点；仅 C 获胜则属于类别辅助系统，不能归因于同 F 信息的逐尺度机制胜利。', '',
              '## H_R：同观测插件面对三个控制', '',
              '| SNR | 标签 | 动作 | 同分支明显/支持 | PSNR约束(校准/实测) |', '|---:|---|---|---|---|']
    for r in hr:
        lines.append(f"| {r['snr_db']:g} | {r['label']} | {r['policy_action']} | {r['same_branch_clear']} / {r['same_branch_supported']} | {r['calibration_psnr_feasible']} / {r['observed_psnr_constraint_pass']} (cap {r['psnr_drop_cap_db']}) |")
    lines += ['', '| SNR | V策略对控制 | ΔPSNR | ΔLPIPS | ΔDINO | Δspecific |', '|---:|---|---|---|---|---|']
    for snr in SNRS:
        for control in ('P1024', 'A1_policy', 'A2_policy'):
            values = [delta_text(lookup[(float(snr), 'V_policy', control, m)]) for m in ('psnr_db', 'lpips_alex', 'dino_cosine', 'delta_specific')]
            lines.append(f'| {snr} | {control} | ' + ' | '.join(values) + ' |')
    lines += ['', '明显增益须同一 LPIPS 分支或同一 DINO+specific 分支超过 P1024/A1/A2 全部控制，不能将不同控制的不同指标拼成成功。calibration 选参 PSNR cap 在 13 dB 为 0.2、其余为 0.3 dB；development 实测 ΔPSNR 与约束符合情况只报告代价，不作为第二次筛选门槛。λ=0 仍是量化修正，BYPASS 原样返回 P1024。', '',
              '## 共同权重、类别与 Decoder 归因', '',
              '| SNR | 比较/作用 | ΔPSNR | ΔLPIPS | ΔDINO | ΔF平方和 |', '|---:|---|---|---|---|---|']
    roles = {'common_content', 'same_action_class_effect', 'same_latent_decoder_effect', 'same_observation_generation_effect'}
    pairs = list(dict.fromkeys((r['method'], r['control'], r['comparison_role']) for r in config['paired_metadata'] if r['comparison_role'] in roles))
    for snr in SNRS:
        for method, control, role in pairs:
            values = [delta_text(lookup[(float(snr), method, control, m)], 3 if m.startswith('latent') else 5)
                      for m in ('psnr_db', 'lpips_alex', 'dino_cosine', 'latent_sq_err_final')]
            lines.append(f'| {snr} | {method}−{control} ({role}) | ' + ' | '.join(values) + ' |')
    lines += ['', '共同权重固定使用 A1 calibration alpha，不重选 λ；未旁路候选仅是诊断。C/U 独立动作主表包含动作变化，类别归因采用同动作、同波形/接收前缀的比较。D0 只重渲染已选 Dc 动作的同一 latent，不重选动作或合并两 Decoder 最好值。数字 CRC 失败图均保留，并沿既有 raw hard-candidate 前缀规则处理；生成−prefix 只说明局部作用。', '',
              '## 13 dB 的实际取舍', '',
              '| 输出相对 P1024 | ΔPSNR | ΔLPIPS | ΔDINO | ΔF平方和 |', '|---|---|---|---|---|']
    for method in (*DIGITAL, 'V_policy', 'V_raw_fused', 'V_common', 'V_tok', 'V_lambda1'):
        values = [delta_text(lookup[(13., method, 'P1024', m)], 3 if m.startswith('latent') else 5)
                  for m in ('psnr_db', 'lpips_alex', 'dino_cosine', 'latent_sq_err_final')]
        lines.append(f'| {method} | ' + ' | '.join(values) + ' |')
    lines += ['', '本轮 13 dB 位于训练集合内。未旁路候选不因 development 的某一指标更好而升级为部署策略。', '',
              '## 16QAM 与 QPSK 的完整系统取舍', '',
              '方向为同类别条件的16QAM−QPSK。两PHY家族各自校准m，实际星座帧能量约束不同；下表比较完整系统工作点，不能解释为隔离调制效果或严格同帧能量实验。', '',
              '| SNR | 类别条件 | ΔPSNR [95% CI] | ΔLPIPS [95% CI] | ΔDINO [95% CI] | Δspecific [95% CI] | ΔF平方和 [95% CI] | Δ实际E [95% CI] |',
              '|---:|---|---|---|---|---|---|---|']
    for snr in SNRS:
        for condition in ('U', 'C'):
            values = [delta_text(lookup[(float(snr), 'D_' + condition + '_16QAM', 'D_' + condition + '_QPSK', m)], 3 if m in ('latent_sq_err_final', 'energy') else 5)
                      for m in ('psnr_db', 'lpips_alex', 'dino_cosine', 'delta_specific', 'latent_sq_err_final', 'energy')]
            lines.append(f'| {snr} | {condition} | ' + ' | '.join(values) + ' |')
    lines += ['',
              '## 能量和失败', '', '| SNR | 方法 | 约束 | E均值 | 最小/P05/P95/最大 | 头失败 | CRC失败 |',
              '|---:|---|---|---:|---|---:|---:|']
    for r in energy:
        if r['method'] in (*POLICIES, *DIGITAL):
            s = indexed[(r['snr_db'], r['method'])]
            lines.append(f"| {r['snr_db']:g} | {r['method']} | {r['energy_constraint']} | {r['mean']:.3f} | {r['min']:.3f}/{r['p05']:.3f}/{r['p95']:.3f}/{r['max']:.3f} | {s['header_failure']:.4f} | {s['body_crc_failure']:.4f} |")
    lines += ['', '连续的头/CRC栏不适用，表内计数为0；逐帧文件不适用字段为空。16QAM 等概率星座平均能量为2不保证实际数据帧 E=2048。本实验没有静默逐帧归一化或免费增益。', '',
              '## 接收与发送计算代价', '',
              '| SNR | 方法 | 实际VAR | RX均值/中位/p95 ms | TX均值 ms | 次数 |', '|---:|---|---|---|---|---:|']
    for r in timing:
        lines.append(f"| {r['snr_db']:g} | {r['method']} | {r['ran_VAR']} | {r.get('rx_ms_mean', float('nan')):.3f}/{r.get('rx_ms_median', float('nan')):.3f}/{r.get('rx_ms_p95', float('nan')):.3f} | {r.get('tx_ms_mean', '')} | {r['calls']} |")
    lines += ['', 'RX 主边界为 CPU 已接收波形→CPU 浮点 RGB，包括 P 接收网络与实际修正/生成/解码。噪声抽样在计时外；10张固定源图×2测量，warmup沿用3次。V_raw 若 λ=0 只量化，不能称运行 VAR 的耗时。数字 source encoding 与 TX 另列。', '',
              '## 协议、证据与后续边界', '',
              '- 100个源图为统计单位；每图先平均三噪声，所有比较共用10000组bootstrap索引 seed=20260930，95%百分位区间，未校正多重比较。单点信号属于探索性证据。',
              '- Development 已多轮使用；区间不包含训练 seed 变异。未访问holdout、未用development挑checkpoint、λ、m或停止规则。',
              '- P1024采用真实N1024训练，完整1000cal、五SNR、三噪声选模；200cal重新校准插件。误差方差包含学习和压缩误差，经验融合不称精确物理后验。',
              '- 同 P1024 的后处理复用同一波形/观测；完整数字与连续系统按source/noise重复配对，不声称跨系统同观测。',
              '- 资源图只画已完成且有兼容证据的点；没有以坐标比例代替信息保留量，也没有补画未测曲线。较大预算旧点缺兼容证据时单列。',
              '- 当前授权的N1024第二阶段在交付后结束；不自动启动任何进一步实验、生成模型训练或旧队列。', '',
              f"文件SHA-256：config.json `{sha(results / 'config.json')}`；selected_policy.json `{sha(results / 'selected_policy.json')}`；per_frame.csv `{sha(results / 'per_frame.csv')}`。", '',
        '详细表：source_references.csv、digital_candidates.csv、digital_action_ledger.csv、selected_policy.json、per_frame.csv、summary.csv、paired.csv、decisions.json、energy_summary.csv、timing.csv、plugin_error_stats.csv、budget_paired.csv、tradeoff_change_paired.csv、tradeoff_change_source_means.csv、budget_pairing.csv、budget_comparison_identity.json与token诊断表。', '',
        f"较大预算曲线参考：{config['historical_budget_reference']['status']}，已核查并绘制 {config['historical_budget_reference']['plotted_points']} 个点。仅历史纯连续模型点与P1024形成资源参考；高预算数字/插件未测点没有补线。旧点缺少错配DINO及F误差，不进入新H_R/H_D判定、source specificity或配对区间。它们是不同模型/训练历史的工作点，不能把N视为唯一模型差别。", '']
    historical = config['historical_budget_reference']
    if historical['plotted_points']:
        lines += ['| 历史纯连续参考 | 初始化seed/训练标签seed | 选中/完成更新 | 历史配方差异 |', '|---|---|---|---|']
        for m in historical['models']:
            lines.append(f"| {m['method']} | {m['actual_initialization_seed']}/{m['training_seed_label']} | {m['selected_step']}/{m['completed_updates']} | {m['microbatch_history']}; parent={m['parent']} |")
        lines += ['', 'P512/P1024为真实预算fresh训练 seed2026093001，保持原P2048 fresh配方micro4；旧P2048在20k后登记micro8，旧P4084是原seed2026092303的续训。资源图反映这些实测系统工作点，不是仅N改变的因果实验。', '']
    lines += ['', '## N512 与 N1024：同方法跨预算的实际变化', '',
        '方向为N1024−N512；每个预算各自使用校准冻结的模型、数字m、λ与旁路。N512/E1024与N1024/E2048是固定符号功率下的资源变化；16QAM另保留实际能量。相同噪声seed只定义重复索引，跨N的波形、噪声命名空间和接收观测不同。', '',
        '| SNR | 方法 | ΔPSNR [95% CI] | ΔLPIPS [95% CI] | ΔDINO [95% CI] | Δspecific [95% CI] | ΔF平方和 [95% CI] |',
        '|---:|---|---|---|---|---|---|']
    cross_lookup = {(r['snr_db'], r['method'], r['metric']): r for r in budget_paired}
    for snr in SNRS:
        for method in (*POLICIES, *DIGITAL):
            values = [delta_text(cross_lookup[(snr, 'N1024/' + method, metric)], 3 if metric.startswith('latent') else 5)
                      for metric in ('psnr_db', 'lpips_alex', 'dino_cosine', 'delta_specific', 'latent_sq_err_final')]
            lines.append(f'| {snr} | {method} | ' + ' | '.join(values) + ' |')
    lines += ['', '## 数字相对匹配 P 的优势/代价是否改变', '',
        '下表不是两个独立区间相减。每源图先对三噪声求同预算D−P，再求两个预算的差，最后使用同一10000组源图bootstrap。正ΔDINO表示数字相对连续的语义优势扩大，负值表示缩小；负ΔLPIPS表示数字相对连续的感知代价降低。原点是否通过H_D仍由本预算冻结策略的直接D−P比较决定，跨预算变化不另造获胜门槛。', '',
        '| SNR | 数字系统 | Δ(D−P) PSNR [95% CI] | Δ(D−P) LPIPS [95% CI] | Δ(D−P) DINO [95% CI] | Δ(D−P) specific [95% CI] | Δ(D−P) F平方和 [95% CI] |',
        '|---:|---|---|---|---|---|---|']
    trade_lookup = {(r['snr_db'], r['method'], r['metric']): r for r in tradeoff_paired}
    for snr in SNRS:
        for method in DIGITAL:
            values = [delta_text(trade_lookup[(snr, method, metric)], 3 if metric.startswith('latent') else 5)
                      for metric in ('psnr_db', 'lpips_alex', 'dino_cosine', 'delta_specific', 'latent_sq_err_final')]
            lines.append(f'| {snr} | {method} | ' + ' | '.join(values) + ' |')
    lines += ['', 'F变化只使用四项(D/P×两预算)都有真实latent的noise帧交集，表内标明有效帧数；头失败仍完整计入PSNR/LPIPS/DINO。全部跨预算方法与阈值失效率、实际能量和失败变化见budget_paired.csv。', '',
        '| SNR | N512 无条件同帧QPSK H_D | N1024 无条件同帧QPSK H_D | N512 H_R | N1024 H_R |', '|---:|---|---|---|---|']
    old_gates = config['prior_budget_reference']['original_decisions']
    for snr in SNRS:
        label_for = lambda records, h, m: next(r['label'] for r in records if r['snr_db'] == snr and r['hypothesis'] == h and r['method'] == m)
        labels = [label_for(old_gates, 'H_D', 'D_U_QPSK'), label_for(records, 'H_D', 'D_U_QPSK'),
                  label_for(old_gates, 'H_R', 'V_policy'), label_for(records, 'H_R', 'V_policy')]
        lines.append(f'| {snr} | ' + ' | '.join(labels) + ' |')
    lines += ['', '两预算使用同一反复使用过的development源图，结果相关；共同源图重采样保留这种相关性。单个训练seed和两次独立真实预算训练不能提供训练seed置信区间，也不能将预算差值解释为纯带宽因果效应。P512/P1024若标为预算截断，属于已登记40k限制下仍在改善的系统工作点，不能把当前训练上限称理论收敛。', '',
        '| 基线 | seed | 选中/完成更新 | 停止原因 | 预算截断 | 最后两区间相对改善 |', '|---|---:|---|---|---|---|']
    for b in (previous_budget, current_budget):
        improvements = '/'.join(f'{100 * v:.4f}%' for v in b['final_relative_improvements'])
        lines.append(f"| P{b['N']} | {b.get('training_seed', 2026093001)} | {b['selected_updates']}/{b['completed_updates']} | {b['stop_reason']} | {b['budget_truncated']} | {improvements} |")
    lines += ['']
    if config['engineering_fixture']:
        lines = ['# ENGINEERING / SYNTHETIC：程序接口检查', '',
            '本目录仅用于完整分析、报告、图像和完成回执的工程测试。N1024帧由N512记录的副本改资源标签构造，没有进行P1024实测；不作为H_D/H_R科学结果。', '', *lines]
    for name in [p for p in figures if p.endswith('.png')] + example_files:
        lines += [f'![{Path(name).stem}]({name})', '']
    (results / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main(out, results):
    out, results = Path(out), Path(results)
    plugin_config = read_json(results / 'plugin_config.json')
    plugin_policy = read_json(results / 'plugin_selected_policy.json')
    digital_policy = read_json(results / 'digital_selected_policy.json')
    digital_config = read_json(results / 'digital_config.json')
    if plugin_config['mismatch_permutation'] != digital_config['mismatch_permutation']:
        raise ValueError('DINO mismatched source permutations differ between systems')
    plugin_rows = csv_rows(results / 'plugin_per_frame.csv')
    digital_rows = csv_rows(results / 'digital_per_frame.csv')
    if len(plugin_rows) != 21000 or len(digital_rows) != 24000:
        raise ValueError('Complete plugin21000 and digital24000 measured rows are required')
    raw = normalize_rows(plugin_rows + digital_rows)
    write_csv(results / 'per_frame.csv', raw)
    write_csv(results / 'token_accuracy.csv', csv_rows(results / 'plugin_token_accuracy.csv'))
    time_rows = csv_rows(results / 'plugin_timing.csv') + csv_rows(results / 'digital_timing.csv')
    write_csv(results / 'timing.csv', time_rows)
    policy = dict(plugin=plugin_policy, digital=digital_policy, development_selection=False)
    write_json(results / 'selected_policy.json', policy)
    matrix, means, ids, snrs, methods = source_means(raw)
    expected_methods = {'P1024', 'A1_policy', 'A2_policy', 'V_policy', 'V_lambda1',
        *[prior + '_' + suffix for prior in ('A1', 'A2', 'V') for suffix in ('raw_fused', 'common', 'tok')],
        *DIGITAL, *[method + '_D0' for method in DIGITAL],
        *['D_prefix_' + condition + '_' + family for family in ('QPSK', '16QAM') for condition in ('U', 'C')],
        *['D_C_' + family + '_at_U_action' for family in ('QPSK', '16QAM')],
        *['D_U_' + family + '_at_C_action' for family in ('QPSK', '16QAM')]}
    if set(methods) != expected_methods or len(raw) != 45000:
        raise ValueError('Registered complete thirty-method N1024 grid is required')
    pairs = comparisons(methods)
    receipts = verify_shared_receptions(raw, pairs)
    indices = np.random.default_rng(SEED).integers(100, size=(RESAMPLES, 100), dtype=np.int64)
    summary, paired, lookup = analyze(matrix, snrs, methods, indices, means, raw, ids)
    gates = decisions(snrs, lookup, plugin_policy)
    energy = energy_summary(raw); timing = timing_summary(time_rows)
    prior_raw, prior_matrix, prior_means, prior_methods, prior_identity = prior_budget_data(results, out, plugin_config, raw, ids)
    budget_paired, tradeoff_paired, tradeoff_sources, pairing = cross_budget_analysis(raw, matrix, prior_raw, prior_matrix, methods, ids, indices)
    write_csv(results / 'budget_paired.csv', budget_paired)
    write_csv(results / 'tradeoff_change_paired.csv', tradeoff_paired)
    write_csv(results / 'tradeoff_change_source_means.csv', tradeoff_sources)
    write_csv(results / 'budget_pairing.csv', pairing)
    training_done = read_json(Path(plugin_config['identity']['base']['training']) / 'completion.json')
    current_budget = training_budget(training_done, plugin_config['identity']['selected'], 1024)
    references, reference_identity = compatible_budget_references(results, out)
    if references:
        if reference_identity['source_ids'] != ids:
            raise ValueError('Historical reference and current source ordering differ')
        preprocessing = {r['source_id']: r['preprocessing_id'] for r in raw}
        if reference_identity['preprocessing_ids'] != [preprocessing[sid] for sid in ids]:
            raise ValueError('Historical reference and current preprocessing differ')
        if reference_identity['decoder_state_sha256'] != plugin_config['identity']['models']['decoder']:
            raise ValueError('Historical reference and current Dc differ')
    write_csv(results / 'source_means.csv', means); write_csv(results / 'summary.csv', summary)
    write_csv(results / 'paired.csv', paired + budget_paired + tradeoff_paired); write_csv(results / 'energy_summary.csv', energy)
    write_csv(results / 'timing_summary.csv', timing); write_json(results / 'decisions.json', gates)
    write_json(results / 'shared_reception_checks.json', receipts)
    config = dict(scope='N1024_ONLY_COMPLETED_EXTREME_BANDWIDTH', plugin=plugin_config, digital=digital_config,
        paired_metadata=[dict(method=a, control=b, comparison_role=r) for a, b, r in pairs],
        development_reused=True, automatic_further_experiments=False, automatic_generation_training=False,
        bootstrap_seed=SEED, bootstrap_repeats=RESAMPLES, training_seeds=1,
        no_development_policy_changes=True, zero_tolerance=ZERO,
        historical_budget_reference=reference_identity, prior_budget_reference=prior_identity,
        training_budgets=dict(N512=prior_identity['budget'], N1024=current_budget),
        source_reuse_correlated=True, training_seed_variance_estimated=False,
        engineering_fixture=bool(plugin_config.get('engineering_fixture', False)))
    write_json(results / 'config.json', config)
    tokens = token_summary(results, ids, indices)
    figures = make_figures(results, summary, paired, plugin_policy, digital_policy, references, prior_matrix, tradeoff_paired)
    example_files = examples(results, raw)
    make_report(results, summary, lookup, gates, policy, timing, energy, config, figures, example_files, budget_paired, tradeoff_paired)
    completion = dict(status='ENGINEERING_SYNTHETIC_N1024_ANALYSIS_COMPLETE' if config['engineering_fixture'] else 'N1024_SOURCE_PAIRED_ANALYSIS_COMPLETE', sources=100, noise_repeats=3,
        methods=methods, snrs=snrs, metric_rows=len(raw), plugin_metric_rows=len(plugin_rows), digital_metric_rows=len(digital_rows),
        bootstrap_seed=SEED, bootstrap_resamples=RESAMPLES,
        bootstrap_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(), source_order=ids,
        selected_updates=plugin_config['identity']['selected']['step'], evaluation_training_updates=0,
        figures=figures, examples=example_files, token_summary_rows=len(tokens),
        automatic_further_experiments=False, automatic_generation_training=False, automatic_old_queue_resume=False,
        synthetic=config['engineering_fixture'], training_budgets=config['training_budgets'],
        cross_budget_metric_rows=len(budget_paired), tradeoff_change_rows=len(tradeoff_paired),
        cross_budget_source_pairing_rows=len(pairing), prior_budget_comparison_sha256=sha(results / 'budget_comparison_identity.json'),
        files={name: sha(results / name) for name in ('config.json', 'selected_policy.json', 'per_frame.csv',
            'summary.csv', 'paired.csv', 'decisions.json', 'budget_paired.csv', 'tradeoff_change_paired.csv',
            'budget_pairing.csv', 'budget_comparison_identity.json', 'report.md')})
    write_json(results / 'analysis_completion.json', completion)
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    result = main(args.out, args.results)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'source_order', 'figures', 'examples')}, ensure_ascii=False))
