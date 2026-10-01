"""Sequential oracle ladder and conditional paid residual link; zero training.

Run only after the verified M1_COMPLETE receipt. Calibration chooses every
lambda/projection before development. Oracle assumes a correct prefix/gain;
even feasible dimensions are NOT actual-link results. Complete oracle rows
remain publishable when the actual-link calibration gate fails.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time

import residual_receiver as rx
import common as c
import numpy as np
import torch
from var_comm.scale_channel import encode_packet, indices_to_bits, bits_to_indices
from var_comm.study import seeded_noise

OUT = c.OUT / 'm2'
CONTROLS = ('UNGUIDED', 'LIKELIHOOD', 'STATIC', 'VAR_GUIDED', 'DIRECT')
LAMBDAS = (.25, .5, 1., 2.)
METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine', 'dino_specificity', 'latent_sq_err_final')
SEED = 20261002
CAL_SELECT = 200
SAMPLES = (0, 25, 50, 75)
TIMING = (0, 11, 22, 33, 44, 55, 66, 77, 88, 99)


def require_m1():
    path = c.OUT / 'm1_complete.json'
    if not path.exists():
        raise RuntimeError('Method2 is gated by verified M1_COMPLETE publication')
    receipt = c.read(path)
    publication = receipt.get('publication', {})
    publication_path = c.OUT / 'm1_publication.json'
    commit = publication.get('commit', '')
    if (receipt.get('status') != 'M1_COMPLETE' or receipt.get('synthetic') is not False
            or receipt.get('training_updates') != 0 or publication.get('status') != 'PUSHED'
            or publication.get('checks') != 'PASS' or not re.fullmatch(r'[0-9a-f]{40}', commit)
            or publication.get('remote_commit') != commit or not publication_path.exists()
            or c.read(publication_path) != publication):
        raise RuntimeError('M1 must have a real checked publication with matching verified remote commit')
    c.verify_bindings(publication['source_bindings'])
    return c.sha(path)


def arraysha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def noise(source, dimension, seed):
    return seeded_noise(f'{c.RUN}/M2/measurement/{dimension}|{source}', int(seed), (dimension // 2 + 1, 2))


def complete_receipt(stage, files, **extra):
    c.assert_frozen(extra.pop('loaded'))
    value = dict(status=f'M2_{stage.upper()}_COMPLETE', synthetic=False,
                 training_updates=0, m1_completion_sha256=require_m1(),
                 protocol_sha256=c.sha(c.HERE / 'protocol.json'),
                 files={str(p): c.sha(p) for p in files}, **extra)
    c.write(c.OUT / f'm2_{stage}_complete.json', value)


def prior_tables(tokens):
    answer = []
    for t in tokens:
        counts = torch.bincount(t.flatten(), minlength=4096).double() + .5
        answer.append((counts / counts.sum()).log().float())
    return answer


def split_batch(tokens):
    return list(torch.split(tokens.long(), [x * x for x in rx.PATCH_NUMS], dim=1))


def tensor_save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def load_state(loaded):
    receipt = c.read(OUT / 'statistics_complete.json')
    c.verify_bindings(receipt['source_bindings'])
    if receipt['model_identity'] != loaded['identity']:
        raise RuntimeError('Residual calibration model identity changed')
    for p, h in receipt['files'].items():
        if c.sha(p) != h: raise RuntimeError('Residual calibration tensor changed: ' + p)
    pca = torch.load(OUT / 'channel_pca.pt', map_location='cpu', weights_only=True)
    projections, stats, operators = {}, {}, {}
    for g, channels in rx.PROJECTIONS:
        p = rx.Projection(g, channels, pca['axes'].to(loaded['device']))
        projections[p.name] = p
        stats[p.name] = torch.load(OUT / 'statistics' / f'{p.name}.pt', map_location='cpu', weights_only=True)
        operators[p.name] = rx.Operators(loaded['vae'].quantize, p, check=c.check)
    static = torch.load(OUT / 'static_prior.pt', map_location='cpu', weights_only=True)
    return projections, stats, operators, static


@torch.no_grad()
def fit_state(loaded, d):
    path = OUT / 'statistics_complete.json'
    if path.exists():
        return load_state(loaded)
    if len(d['records']) != 1000:
        raise RuntimeError('Full original1000 calibration required to fit residual statistics')
    q = loaded['vae'].quantize
    tokens = split_batch(d['T'])
    residual = []
    for start in range(0, 1000, 10):
        c.check()
        pre = [t[start:start + 10].to(loaded['device']) for t in tokens[:4]]
        residual.append((d['F'][start:start + 10].to(loaded['device']) - rx.prefix_latent(q, pre)).cpu())
    pca = rx.fit_channel_pca(torch.cat(residual))
    tensor_save(OUT / 'channel_pca.pt', pca)
    tensor_save(OUT / 'static_prior.pt', prior_tables(tokens))
    paths = [OUT / 'channel_pca.pt', OUT / 'static_prior.pt']
    for g, channels in rx.PROJECTIONS:
        c.status('m2_fit', projection=f'g{g}_c{channels}')
        p = rx.Projection(g, channels, pca['axes'].to(loaded['device']))
        fitted = rx.fit_structural_stats(q, d['F'], tokens, p, check=c.check)
        path = OUT / 'statistics' / f'{p.name}.pt'
        tensor_save(path, fitted); paths.append(path)
    c.write(OUT / 'statistics_complete.json', dict(status='M2_CALIBRATION_STATISTICS_COMPLETE',
        synthetic=False, calibration_sources=1000, calibration_source_ids=[r['image_id'] for r in d['records']],
        calibration_bindings=d['bindings'], source_bindings=c.source_bindings(), model_identity=loaded['identity'],
        pca_scope='shared32-channel covariance of true-m4 residual at original16x16 positions',
        mean_variance_scope='true-prefix future residual; scalar isotropic approximation',
        training_updates=0, files={str(p): c.sha(p) for p in paths}))
    return load_state(loaded)


def source_meta(record, i, projection, stage, snr, seed):
    return dict(source_id=record['image_id'], source_index=i, preprocessing_id=record['preprocessing_id'],
                stage=stage, projection=projection.name, g=projection.g, channels=projection.channels,
                snr_db=snr, noise_seed=seed, prefix_m=4, decoder_id='Dc', class_condition='unconditional1000',
                N='', E='', training_updates=0, oracle_side_information=True,
                exact_posterior_claim=False, coordinate_sweeps=1,
                feasible_N512_QPSK=rx.resource_record(512, projection.g, projection.channels, 'QPSK')['feasible'],
                feasible_N512_16QAM=rx.resource_record(512, projection.g, projection.channels, '16QAM')['feasible'],
                feasible_N1024_QPSK=rx.resource_record(1024, projection.g, projection.channels, 'QPSK')['feasible'],
                feasible_N1024_16QAM=rx.resource_record(1024, projection.g, projection.channels, '16QAM')['feasible'])


def score(loaded, d, i, latent, *, images=False):
    # Reuse the exact old development derangement for P comparisons.
    other = c.assets.mismatch_permutation()[i] if len(d['records']) == 100 else (i + 1) % len(d['records'])
    mismatch = d['reference'][other]
    row, image = c.quality(loaded, d['records'][i], latent, d['F'][i], d['reference'][i], mismatch, images)
    row['dino_specificity'] = row['dino_cosine'] - row['dino_mismatched']
    row['mismatch_source_id'] = d['records'][other]['image_id']
    return row, image


def greedy(loaded, prefix):
    arrays = [t[0].detach().cpu().numpy() for t in prefix]
    return c.legacy.complete_latent(loaded['vae'], loaded['var'], arrays, 1000, loaded['device'])


def measure(clean, prefix_map, projection, record, snr, seed):
    z = projection.forward(clean - prefix_map)
    if snr == 'clean':
        return z, 0., dict(gain='', E_analog='', measurement_wave_sha256='', measurement_y_sha256='',
                           assumed_prefix_correct=True, assumed_gain_correct=True, measurement_noise='none')
    wave, gain = rx.analog_transmit(z[0].detach().cpu().numpy())
    perturbation = noise(record['image_id'], projection.dimension, seed)
    y = wave + perturbation * 10. ** (-float(snr) / 20)
    received = torch.as_tensor(y[:-1].ravel() / gain, dtype=torch.float32, device=clean.device)[None]
    return received, 10. ** (-float(snr) / 10) / (gain * gain), dict(gain=gain,
        E_analog=float(np.square(wave).sum()), measurement_wave_sha256=arraysha(wave),
        measurement_y_sha256=arraysha(y), assumed_prefix_correct=True, assumed_gain_correct=True,
        gain_quantization='downward float32; ideal correctly received paid gain in oracle',
        measurement_noise='actual AWGN on normalized paid-dimension waveform; padding ignored')


def run_control(loaded, prefix, observation, projection, stats, operators, static,
                noise_variance, control, lam, base):
    if control == 'UNGUIDED':
        return base
    if control == 'DIRECT':
        pre = rx.prefix_latent(loaded['vae'].quantize, prefix)
        return rx.direct_correction(base, pre, observation, projection, 1.)
    prior = {'VAR_GUIDED': 'VAR', 'STATIC': 'STATIC', 'LIKELIHOOD': 'LIKELIHOOD'}[control]
    return rx.infer(loaded['vae'], loaded['var'], prefix, observation, projection, stats,
        noise_variance=noise_variance, prior=prior, lam=lam, static=static,
        operators=operators, check=c.check)['fhat']


def paired(rows, first, second, *, group='projection'):
    result = []
    groups = sorted({r[group] for r in rows})
    for key in groups:
        data = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r[group] == key and r['control'] in (first, second):
                data[r['source_id']][r['control']].append(r)
        for metric in METRICS:
            differences = []
            for source, controls in data.items():
                if set(controls) != {first, second}:
                    raise RuntimeError('Incomplete source-paired calibration gate')
                a = [float(r[metric]) for r in controls[first] if r.get(metric, '') != '']
                b = [float(r[metric]) for r in controls[second] if r.get(metric, '') != '']
                if a and b:
                    differences.append(float(np.mean(a) - np.mean(b)))
            if not differences:
                continue
            delta = np.asarray(differences)
            indices = np.random.default_rng(SEED).integers(0, len(delta), (10000, len(delta)))
            lo, hi = np.quantile(delta[indices].mean(1), [.025, .975])
            result.append(dict(projection=key, first=first, second=second, metric=metric,
                mean=float(delta.mean()), lower95=float(lo), upper95=float(hi),
                paired_sources=len(delta), unit='source image; noise averaged within source', seed=SEED))
    return result


def choose_lambda(rows, control):
    baselines = defaultdict(list)
    candidates = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['control'] == 'UNGUIDED': baselines[row['projection']].append(row)
        if row['control'] == control: candidates[row['projection']][float(row['lambda'])].append(row)
    answer = {}
    for projection, choices in candidates.items():
        base = {m: np.mean([float(r[m]) for r in baselines[projection]]) for m in METRICS[:3]}
        scored = []
        for lam, values in choices.items():
            avg = {m: float(np.mean([float(r[m]) for r in values])) for m in METRICS[:3]}
            psnr, dino = avg['psnr_db'] - base['psnr_db'], avg['dino_cosine'] - base['dino_cosine']
            scored.append(dict(lambda_value=lam, **avg, selection_feasible=psnr >= .5 and dino >= -.01,
                               dino_feasible=dino >= -.01, delta_psnr=psnr, delta_dino=dino))
        eligible = [r for r in scored if r['selection_feasible']]
        fallback = [r for r in scored if r['dino_feasible']]
        pool = eligible or fallback or scored
        winner = min(pool, key=lambda r: (r['lpips_alex'], r['lambda_value']))
        answer[projection] = dict(selected_lambda=winner['lambda_value'], selected=winner,
                                  candidates=scored, objective='minimum LPIPS subject PSNR+.5 and DINO loss<=.01; disclosed fallback')
    return answer


def gate_from_pairs(pairs):
    grouped = defaultdict(dict)
    for row in pairs: grouped[row['projection']][row['metric']] = row
    decisions = []
    for name, values in grouped.items():
        ps, lp, di = [values[m] for m in METRICS[:3]]
        passed = ps['mean'] >= .5 and ps['lower95'] > 0 and lp['mean'] <= -.02 and lp['upper95'] < 0 and di['mean'] >= -.01
        decisions.append(dict(projection=name, passed=bool(passed), delta_psnr=ps, delta_lpips=lp,
                              delta_dino=di, scope='full1000cal7dB; no development filtering'))
    return dict(status='PASSED' if any(d['passed'] for d in decisions) else 'SKIPPED_GATE_NOT_MET',
                passed=any(d['passed'] for d in decisions), decisions=decisions, snr_db=7,
                calibration_sources=1000, noise_seeds=c.CAL_SEEDS, bootstrap=10000, seed=SEED,
                criteria=dict(psnr_mean_min=.5, psnr_lower95_min_exclusive=0.,
                              lpips_mean_max=-.02, lpips_upper95_max_exclusive=0., dino_mean_min=-.01),
                scope='A budget-feasible oracle gate permits real-link calibration, not a VAR-mechanism claim')


def cached_source(path, registration, producer):
    if path.exists():
        saved = c.read(path)
        if saved['registration_sha256'] != registration:
            raise RuntimeError('Residual source receipt has different frozen inputs')
        return saved['rows']
    rows = producer()
    c.write(path, dict(registration_sha256=registration, rows=rows))
    return rows


@torch.no_grad()
def calibration(loaded):
    require_m1()
    d = c.data('calibration', loaded)
    registration = c.registration(loaded, d, 'm2_calibration', [c.OUT / 'm1_complete.json', c.OUT / 'm1_publication.json'])
    qualify(loaded, d)
    projections, stats, operators, static = fit_state(loaded, d)
    all_tokens = split_batch(d['T'])
    ledger = [rx.resource_record(N, g, ch, phy) for g, ch in rx.PROJECTIONS
              for N in (512, 1024) for phy in ('QPSK', '16QAM')]
    c.csv_rows(c.RESULT / 'm2_resource_ledger.csv', ledger)
    screen = []
    for i in range(CAL_SELECT):
        c.check(); c.status('m2_screen', sources=i, total=CAL_SELECT)
        def produce(i=i):
            record = d['records'][i]
            prefix = [t[i:i + 1].to(loaded['device']) for t in all_tokens[:4]]
            pre = rx.prefix_latent(loaded['vae'].quantize, prefix)
            clean = d['F'][i:i + 1].to(loaded['device'])
            base = greedy(loaded, prefix)
            baseline, _ = score(loaded, d, i, base)
            rows = []
            for name, projection in projections.items():
                for seed in c.CAL_SEEDS:
                    obs, variance, info = measure(clean, pre, projection, record, 7, seed)
                    methods = [('UNGUIDED', 0.), ('LIKELIHOOD', 0.), ('DIRECT', 0.)]
                    methods += [(control, lam) for control in ('VAR_GUIDED', 'STATIC') for lam in LAMBDAS]
                    for control, lam in methods:
                        c.check()
                        metrics = baseline if control == 'UNGUIDED' else score(loaded, d, i,
                            run_control(loaded, prefix, obs, projection, stats[name], operators[name], static,
                                        variance, control, lam, base))[0]
                        rows.append(dict(**source_meta(record, i, projection, 'calibration_screen_reduced_noisy', 7, seed),
                            control=control, **{'lambda': lam}, **info, **metrics))
            return rows
        screen += cached_source(OUT / 'calibration' / 'screen' / f'source_{i:04d}.json', registration, produce)
    screen_path = OUT / 'calibration' / 'screen_per_frame.csv'
    c.csv_rows(screen_path, screen)
    selected_var, selected_static = choose_lambda(screen, 'VAR_GUIDED'), choose_lambda(screen, 'STATIC')
    policy = dict(status='M2_ORACLE_POLICY_FROZEN', training_updates=0,
        development_read=False,
        selected_snr_db=7, selection_sources=200, selection_noise_seeds=c.CAL_SEEDS,
        var=selected_var, static=selected_static, direct_alpha=1., likelihood_lambda=0.,
        screen_sha256=c.sha(screen_path), statistics_receipt_sha256=c.sha(OUT / 'statistics_complete.json'),
        registration_sha256=registration, source_bindings=c.source_bindings())
    policy_path = OUT / 'calibration' / 'policy.json'; c.seal(policy_path, policy)
    names = {f"g{r['g']}_c{r['channels']}" for r in ledger if r['feasible']}
    full = []
    for i in range(1000):
        c.check(); c.status('m2_gate', sources=i, total=1000)
        def produce(i=i):
            record = d['records'][i]; rows = []
            prefix = [t[i:i + 1].to(loaded['device']) for t in all_tokens[:4]]
            pre = rx.prefix_latent(loaded['vae'].quantize, prefix)
            clean = d['F'][i:i + 1].to(loaded['device']); base = greedy(loaded, prefix)
            baseline, _ = score(loaded, d, i, base)
            for name in sorted(names):
                projection = projections[name]
                lam = selected_var[name]['selected_lambda']
                for seed in c.CAL_SEEDS:
                    # Reuse matching screen results with registered source/seed/choice.
                    reused = [r for r in screen if r['source_index'] == i and r['projection'] == name
                              and r['noise_seed'] == seed and (r['control'] == 'UNGUIDED' or
                              (r['control'] == 'VAR_GUIDED' and float(r['lambda']) == lam))] if i < 200 else []
                    if len(reused) == 2:
                        rows += [dict(r, stage='full_calibration_gate_reduced_noisy') for r in reused]
                        continue
                    obs, variance, info = measure(clean, pre, projection, record, 7, seed)
                    guided = run_control(loaded, prefix, obs, projection, stats[name], operators[name], static,
                                         variance, 'VAR_GUIDED', lam, base)
                    for control, metrics in [('UNGUIDED', baseline), ('VAR_GUIDED', score(loaded, d, i, guided)[0])]:
                        rows.append(dict(**source_meta(record, i, projection, 'full_calibration_gate_reduced_noisy', 7, seed),
                            control=control, **{'lambda': lam if control == 'VAR_GUIDED' else 0.}, **info, **metrics))
            return rows
        full += cached_source(OUT / 'calibration' / 'gate' / f'source_{i:04d}.json', registration, produce)
    path = OUT / 'calibration' / 'gate_per_frame.csv'; c.csv_rows(path, full)
    intervals = paired(full, 'VAR_GUIDED', 'UNGUIDED')
    pair_path = c.RESULT / 'm2_gate_paired.csv'; c.csv_rows(pair_path, intervals)
    gate = gate_from_pairs(intervals)
    gate.update(policy_sha256=c.sha(policy_path), per_frame_sha256=c.sha(path), registration_sha256=registration,
                eligible_projections=sorted(names), development_read=False,
                decided_before_actual_link_calibration=True)
    gate_path = OUT / 'calibration' / 'gate.json'; c.seal(gate_path, gate)
    c.write(c.RESULT / 'm2_oracle_policy.json', policy); c.write(c.RESULT / 'm2_gate.json', gate)
    actual_policy = calibrate_actual(loaded, d, (projections, stats, operators, static), policy, gate, registration)
    actual_policy_path = OUT / 'calibration' / 'actual_policy.json'
    c.write(c.RESULT / 'm2_actual_policy.json', actual_policy)
    evidence = [policy_path, gate_path, path, pair_path, screen_path, actual_policy_path,
                OUT / 'statistics_complete.json', OUT / 'qualification.json']
    if gate['passed']: evidence.append(OUT / 'calibration' / 'actual_per_frame.csv')
    complete_receipt('calibration', evidence + [
        c.RESULT / 'm2_resource_ledger.csv'], loaded=loaded, gate_status=gate['status'],
        screen_rows=len(screen), gate_rows=len(full), fitted_calibration_sources=1000,
        all_actual_and_oracle_policies_frozen_before_development=True, development_read=False)


def frozen_policy():
    receipt = c.read(c.OUT / 'm2_calibration_complete.json')
    for p, h in receipt['files'].items():
        if c.sha(p) != h: raise RuntimeError('M2 calibration evidence changed')
    policy = c.read(OUT / 'calibration' / 'policy.json')
    c.verify_bindings(policy['source_bindings'])
    return policy, c.read(OUT / 'calibration' / 'gate.json')


def gain_bits(gain):
    return np.unpackbits(np.frombuffer(struct.pack('>f', float(gain)), dtype=np.uint8))


def bits_gain(bits):
    if len(bits) != 32: raise ValueError('All32 paid gain bits required')
    return float(struct.unpack('>f', np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes())[0])


def header_code(projection):
    return (rx.PREFIX_M << 3) | list(rx.PROJECTIONS).index((projection.g, projection.channels))


def actual_transmit(scales, coordinates, N, projection, phy):
    ledger = rx.resource_record(N, projection.g, projection.channels, phy)
    if not ledger['feasible']: raise ValueError('Residual profile cannot fit registered total budget')
    raw = np.concatenate([np.asarray(x, dtype=np.int64) for x in scales[:4]])
    if len(raw) != 30:
        raise ValueError('Original m4 prefix required')
    #12 paid bits: fixed m4/projection code, no transmitted source class.
    header = indices_to_bits([header_code(projection)], 12)
    analog, gain = rx.analog_transmit(coordinates)
    header_wave = encode_packet(header, 68)['symbols']
    body = c.legacy.phy.packet(indices_to_bits(raw), ledger['N_digital'], phy)
    gain_wave = encode_packet(gain_bits(gain), 32)['symbols']
    wave = np.concatenate((header_wave, body, gain_wave, analog))
    if wave.shape != (N, 2): raise RuntimeError('Residual frame resource mismatch')
    energy = float(np.square(wave).sum())
    if phy == 'QPSK' and not np.isclose(energy, 2 * N, atol=1e-8, rtol=1e-12):
        raise RuntimeError('Residual QPSK total E2N failed')
    ledger.update(E=energy, E_analog=float(np.square(analog).sum()), E_gain=64., E_header=136.,
        E_digital=float(np.square(body).sum()), transmitted_gain=gain,
        energy_constraint='per_frame_2N' if phy == 'QPSK' else 'QPSK_control_and_normalized_analog_plus_actual_16QAM_body',
        waveform_sha256=arraysha(wave), resource_scope='actual paid prefix/gain/measurement/padding frame',
        header_source_bits=12, header_code=header_code(projection))
    return wave, ledger


def actual_receive(y, snr, N, projection, phy):
    """RX has only y, public nominal SNR and previously frozen profile/PHY."""
    ledger = rx.resource_record(N, projection.g, projection.channels, phy)
    y = np.asarray(y, dtype=np.float64)
    if not ledger['feasible'] or y.shape != (N, 2) or not np.isfinite(y).all():
        raise ValueError('Complete finite registered frame required')
    header, hcrc, _ = c.legacy.phy.decode_packet(y[:68], 12, snr, 'QPSK')
    decoded_code = int(bits_to_indices(header, 12)[0])
    header_ok = bool(hcrc and decoded_code == header_code(projection))
    event = dict(header_ok=header_ok, header_crc_ok=bool(hcrc), prefix_crc_ok=False,
        gain_crc_ok=False, gain_fields_valid=False, gain_numeric_valid=False, analog_used=False, prefix=[], gain='',
        observation=None, noise_variance=None, source_error='header_failure',
        decoded_header_code=decoded_code, decoded_mode=4 if header_ok else '', true_class_sent=False,
        observation_sha256=arraysha(y), padding_ignored=True)
    if not header_ok: return event
    end = 68 + ledger['N_digital']
    bits, prefix_crc, _ = c.legacy.phy.decode_packet(y[68:end], 360, snr, phy)
    values = bits_to_indices(bits)
    prefix = list(np.split(values, [1, 5, 14]))
    received_gain_bits, gain_crc, _ = c.legacy.phy.decode_packet(y[end:end + 32], 32, snr, 'QPSK')
    gain = bits_gain(received_gain_bits)
    valid = math.isfinite(gain) and gain > 0
    use = bool(prefix_crc and gain_crc and valid)
    event.update(prefix=prefix, prefix_crc_ok=bool(prefix_crc), gain_crc_ok=bool(gain_crc),
        gain_fields_valid=bool(valid), gain=gain if valid else '', analog_used=use,
        source_error='' if use else 'analog_discarded_prefix_or_gain_failure')
    if use:
        # Last complex is paid energy padding. Never infer gain from it.
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            observation = y[end + 32:-1].ravel() / gain
            variance = 10. ** (-float(snr) / 10) / (gain * gain)
            observation32 = observation.astype(np.float32)
            variance32 = np.float32(variance)
        numeric_valid = (np.isfinite(observation32).all() and np.isfinite(variance32) and variance32 > 0)
        event['gain_numeric_valid'] = bool(numeric_valid)
        if numeric_valid:
            event['observation'] = observation
            event['noise_variance'] = float(variance)
        else:
            event.update(analog_used=False, source_error='analog_discarded_gain_numerical_invalid')
    return event


def event_meta(event):
    return {k: v for k, v in event.items() if k not in ('prefix', 'observation', 'noise_variance')}


def actual_outputs(loaded, event, projection, fitted, operators, static, policy):
    if not event['header_ok']:
        return {control: None for control in CONTROLS}
    prefix = [torch.as_tensor(x, dtype=torch.long, device=loaded['device'])[None] for x in event['prefix']]
    base = greedy(loaded, prefix)
    if not event['analog_used']:
        return {control: base for control in CONTROLS}
    obs = torch.as_tensor(event['observation'], dtype=torch.float32, device=loaded['device'])[None]
    out = {}
    for control in CONTROLS:
        lam = policy['var'][projection.name]['selected_lambda'] if control == 'VAR_GUIDED' else policy['static'][projection.name]['selected_lambda'] if control == 'STATIC' else 0.
        out[control] = run_control(loaded, prefix, obs, projection, fitted, operators, static,
            event['noise_variance'], control, lam, base)
    return out


@torch.no_grad()
def qualify(loaded, d):
    path = OUT / 'qualification.json'
    if path.exists():
        done = c.read(path)
        if done['status'] != 'M2_REAL_WEIGHT_QUALIFICATION_PASS' or done['synthetic']:
            raise RuntimeError('Invalid M2 real-weight qualification')
        c.verify_bindings(done['source_bindings'])
        return
    require_m1(); q = loaded['vae'].quantize
    tokens = split_batch(d['T'])
    prefix = [t[:1].to(loaded['device']) for t in tokens[:4]]
    clean = d['F'][:1].to(loaded['device'])
    axes = torch.eye(32, device=loaded['device'])
    details = []
    for g, ch in ((4, 8), (8, 32)):
        p = rx.Projection(g, ch, axes)
        observation = p.forward(clean - rx.prefix_latent(q, prefix))
        so = rx.ScaleOperator(q, p, 4, check=c.check)
        trial = tokens[4][:1].to(loaded['device']).clone()
        error = observation[0] - so.measured(trial)[0]
        maxima = []
        for position in (0, 12, 24):
            scores = rx.candidate_scores(so, position, int(trial[0, position]), error, 1.)
            direct = []
            for start in range(0, 4096, 64):
                c.check()
                candidate = trial.expand(min(64, 4096 - start), -1).clone()
                candidate[:, position] = torch.arange(start, min(start + 64, 4096), device=trial.device)
                residual = observation - so.measured(candidate)
                direct.append(-.5 * residual.double().square().sum(1))
            direct = torch.cat(direct)
            a, b = scores.double() - scores[0].double(), direct - direct[0]
            maxima.append(float((a - b).abs().max()))
            if not torch.allclose(a, b, atol=.01, rtol=.002):
                raise RuntimeError('Real projected Phi candidate score disagrees with direct forward')
        dummy = dict(mean=torch.zeros(10, p.dimension), scalar_variance=torch.ones(10))
        unguided = rx.infer(loaded['vae'], loaded['var'], prefix, observation, p, dummy, prior='UNGUIDED')['fhat']
        original = greedy(loaded, prefix)
        if not torch.equal(unguided, original):
            raise RuntimeError('M2 unguided does not equal frozen legacy completion')
        if not torch.isfinite(unguided).all(): raise RuntimeError('Nonfinite qualification output')
        if (g, ch) == (4, 8):
            tiny_guided = rx.infer(loaded['vae'], loaded['var'], prefix, observation, p, dummy,
                prior='VAR', lam=1., operators=rx.Operators(q, p, check=c.check), check=c.check)['fhat']
            if not torch.isfinite(tiny_guided).all(): raise RuntimeError('Nonfinite real guided qualification frame')
        details.append(dict(projection=p.name, candidate_positions=[0, 12, 24], vocabulary=4096,
                            max_score_difference=max(maxima), unguided_legacy_exact=True,
                            finite_guided_frame_checked=(g, ch) == (4, 8)))
    p = rx.Projection(4, 8, axes)
    z = p.forward(clean - rx.prefix_latent(q, prefix))[0].cpu().numpy()
    scales = c.split_tokens(d['T'][0].numpy())
    roundtrips = []
    for N in (512, 1024):
        for phy in ('QPSK', '16QAM'):
            wave, ledger = actual_transmit(scales, z, N, p, phy)
            event = actual_receive(wave, 19, N, p, phy)
            if not event['header_ok'] or not event['prefix_crc_ok'] or not event['gain_crc_ok'] or not event['analog_used']:
                raise RuntimeError('Real-weight zero-noise paid residual packet roundtrip failed')
            if not all(np.array_equal(a, b) for a, b in zip(event['prefix'], scales[:4])):
                raise RuntimeError('Zero-noise prefix roundtrip changed true tokens')
            if not np.allclose(event['observation'], z, atol=1e-8, rtol=1e-7):
                raise RuntimeError('Zero-noise paid gain failed to restore projected residual')
            roundtrips.append(dict(N=N, phy_family=phy, E=ledger['E'], gain=ledger['transmitted_gain']))
    c.assert_frozen(loaded)
    c.write(path, dict(status='M2_REAL_WEIGHT_QUALIFICATION_PASS', synthetic=False,
        study_quality_rows=0, role='engineering real-weight qualification only',
        source_id=d['records'][0]['image_id'], model_identity=loaded['identity'],
        source_bindings=c.source_bindings(), m1_completion_sha256=require_m1(),
        forward_checks=details, paid_packet_roundtrips=roundtrips,
        receiver_inputs='prefix/observation/projector/calibration_stats/prior only; no clean F or true missing tokens'))


@torch.no_grad()
def calibrate_actual(loaded, d, state, policy, gate, registration):
    projections, stats, operators, static = state
    passed = {r['projection'] for r in gate['decisions'] if r['passed']}
    selected_path = OUT / 'calibration' / 'actual_policy.json'
    if selected_path.exists(): return c.read(selected_path)
    if not passed:
        answer = dict(status='SKIPPED_GATE_NOT_MET', choices=[], calibration_sources=0,
                      gate_sha256=c.sha(OUT / 'calibration' / 'gate.json'), development_read=False)
        c.seal(selected_path, answer); return answer
    tokens = split_batch(d['T']); allrows = []
    for i in range(CAL_SELECT):
        c.check(); c.status('m2_actual_calibration', sources=i, total=CAL_SELECT)
        def produce(i=i):
            record = d['records'][i]; rows = []
            prefix = [t[i:i + 1].to(loaded['device']) for t in tokens[:4]]
            pre = rx.prefix_latent(loaded['vae'].quantize, prefix)
            clean = d['F'][i:i + 1].to(loaded['device']); scales = c.split_tokens(d['T'][i].numpy())
            for name in sorted(passed):
                p = projections[name]; z = p.forward(clean - pre)[0].cpu().numpy()
                for N in (512, 1024):
                    for phy in ('QPSK', '16QAM'):
                        if not rx.resource_record(N, p.g, p.channels, phy)['feasible']: continue
                        wave, ledger = actual_transmit(scales, z, N, p, phy)
                        for snr in c.SNRS:
                            for seed in c.CAL_SEEDS:
                                perturbation = seeded_noise(f'{c.RUN}/M2/actual/N{N}|{record["image_id"]}', seed, (N, 2))
                                y = wave + perturbation * 10. ** (-float(snr) / 20)
                                event = actual_receive(y, snr, N, p, phy)
                                if not event['header_ok']:
                                    base, guided = None, None
                                else:
                                    received_prefix = [torch.as_tensor(x, dtype=torch.long, device=loaded['device'])[None] for x in event['prefix']]
                                    base = greedy(loaded, received_prefix); guided = base
                                    if event['analog_used']:
                                        obs = torch.as_tensor(event['observation'], dtype=torch.float32, device=loaded['device'])[None]
                                        guided = run_control(loaded, received_prefix, obs, p, stats[name], operators[name], static,
                                            event['noise_variance'], 'VAR_GUIDED', policy['var'][name]['selected_lambda'], base)
                                for control, latent in [('UNGUIDED', base), ('VAR_GUIDED', guided)]:
                                    metrics, _ = score(loaded, d, i, latent)
                                    meta = source_meta(record, i, p, 'actual_link_calibration', snr, seed)
                                    meta.update(oracle_side_information=False, N=N, E=ledger['E'])
                                    rows.append(dict(**meta, control=control, **{'lambda': policy['var'][name]['selected_lambda'] if control == 'VAR_GUIDED' else 0.},
                                        **{k: v for k, v in ledger.items() if k not in meta}, **event_meta(event), **metrics))
            return rows
        allrows += cached_source(OUT / 'calibration' / 'actual' / f'source_{i:04d}.json', registration, produce)
    path = OUT / 'calibration' / 'actual_per_frame.csv'; c.csv_rows(path, allrows)
    summaries = defaultdict(lambda: defaultdict(list))
    for row in allrows:
        key = (int(row['N']), row['phy_family'], float(row['snr_db']), row['projection'])
        summaries[key][row['control']].append(row)
    choices, candidates = [], []
    for (N, phy, snr, projection), controls in summaries.items():
        means = {control: {m: float(np.mean([float(r[m]) for r in rows])) for m in METRICS[:3]}
                 for control, rows in controls.items()}
        candidate = dict(N=N, phy_family=phy, snr_db=snr, projection=projection,
                         guided=means['VAR_GUIDED'], unguided=means['UNGUIDED'])
        candidate['selection_feasible'] = (means['VAR_GUIDED']['psnr_db'] >= means['UNGUIDED']['psnr_db'] - .25
                                          and means['VAR_GUIDED']['dino_cosine'] >= means['UNGUIDED']['dino_cosine'] - .01)
        candidates.append(candidate)
    for N in (512, 1024):
        for phy in ('QPSK', '16QAM'):
            for snr in c.SNRS:
                options = [r for r in candidates if r['N'] == N and r['phy_family'] == phy and r['snr_db'] == snr]
                if not options:
                    choices.append(dict(N=N, phy_family=phy, snr_db=snr, status='SKIPPED_NO_PASSED_FEASIBLE_PROJECTION'))
                    continue
                admissible = [r for r in options if r['selection_feasible']]
                winner = min(admissible or options, key=lambda r: (r['guided']['lpips_alex'], r['projection']))
                choices.append(dict(**winner, status='SELECTED', selection_fallback=not bool(admissible)))
    answer = dict(status='M2_ACTUAL_POLICY_FROZEN', choices=choices, candidates=candidates,
        development_read=False,
        calibration_sources=200, calibration_noise_seeds=c.CAL_SEEDS,
        selection='minimum actual-link LPIPS subject unguided PSNR-.25 and DINO-.01; disclosed fallback',
        lambda_scope='oracle7dB selected VAR/static lambda fixed across actual SNRs',
        source_bindings=c.source_bindings(), per_frame_sha256=c.sha(path),
        oracle_policy_sha256=c.sha(OUT / 'calibration' / 'policy.json'),
        gate_sha256=c.sha(OUT / 'calibration' / 'gate.json'))
    c.seal(selected_path, answer)
    c.write(c.RESULT / 'm2_actual_policy.json', answer)
    return answer


def empty_table(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        csv.writer(handle).writerow(['stage', 'projection', 'control', 'source_id', 'N', 'phy_family', 'snr_db', 'noise_seed'])


@torch.no_grad()
def actual(loaded):
    require_m1(); policy, gate = frozen_policy()
    choices = c.read(OUT / 'calibration' / 'actual_policy.json')
    path = c.RESULT / 'm2_actual_per_frame.csv'
    if not gate['passed']:
        empty_table(path)
        complete_receipt('actual', [path, OUT / 'calibration' / 'actual_policy.json'], loaded=loaded,
                         branch='SKIPPED_GATE_NOT_MET', metric_rows=0, actual_development_read=False)
        return
    c.verify_bindings(choices['source_bindings'])
    selected = [choice for choice in choices['choices'] if choice['status'] == 'SELECTED']
    d = c.data('development', loaded)
    registration = c.registration(loaded, d, 'm2_actual', [c.OUT / 'm2_calibration_complete.json',
        OUT / 'calibration' / 'actual_policy.json'])
    projections, stats, operators, static = load_state(loaded)
    tokens = split_batch(d['T']); allrows = []
    for i, record in enumerate(d['records']):
        c.check(); c.status('m2_actual', sources=i, total=100)
        def produce(i=i, record=record):
            rows = []; waves = {}
            prefix = [t[i:i + 1].to(loaded['device']) for t in tokens[:4]]
            pre = rx.prefix_latent(loaded['vae'].quantize, prefix)
            clean = d['F'][i:i + 1].to(loaded['device']); scales = c.split_tokens(d['T'][i].numpy())
            for choice in selected:
                N, phy, snr, name = choice['N'], choice['phy_family'], choice['snr_db'], choice['projection']
                p = projections[name]; key = (N, phy, name)
                if key not in waves:
                    z = p.forward(clean - pre)[0].cpu().numpy()
                    waves[key] = actual_transmit(scales, z, N, p, phy)
                wave, ledger = waves[key]
                for seed in c.DEV_SEEDS:
                    perturbation = seeded_noise(f'{c.RUN}/M2/actual/N{N}|{record["image_id"]}', seed, (N, 2))
                    y = wave + perturbation * 10. ** (-float(snr) / 20)
                    event = actual_receive(y, snr, N, p, phy)
                    outputs = actual_outputs(loaded, event, p, stats[name], operators[name], static, policy)
                    correct = all(np.array_equal(a, b) for a, b in zip(event['prefix'], scales[:4])) if event['header_ok'] else False
                    for control, latent in outputs.items():
                        save = i in SAMPLES and snr in (4, 13) and seed == 2001
                        metrics, image = score(loaded, d, i, latent, images=save)
                        meta = source_meta(record, i, p, 'actual_link', snr, seed)
                        meta.update(oracle_side_information=False, N=N, E=ledger['E'])
                        lam = policy['var'][name]['selected_lambda'] if control == 'VAR_GUIDED' else policy['static'][name]['selected_lambda'] if control == 'STATIC' else 0.
                        rows.append(dict(**meta, control=control, **{'lambda': lam},
                            **{k: v for k, v in ledger.items() if k not in meta}, **event_meta(event),
                            prefix_actually_correct_offline=correct, offline_truth_never_controls_rx=True, **metrics))
                        if save and image is not None:
                            c.save_image(c.RESULT / 'm2_examples' / f'{i:03d}_N{N}_{phy}_{snr}_{control}.png', image)
            return rows
        allrows += cached_source(OUT / 'development' / 'actual' / f'source_{i:04d}.json', registration, produce)
    expected = len(selected) * 100 * 3 * len(CONTROLS)
    if len(allrows) != expected: raise RuntimeError('Incomplete selected actual residual link grid')
    if allrows: c.csv_rows(path, allrows)
    else: empty_table(path)
    rawpath = OUT / 'development' / 'actual_per_frame.csv'
    rawpath.parent.mkdir(parents=True, exist_ok=True); rawpath.write_bytes(path.read_bytes())
    complete_receipt('actual', [path, rawpath, OUT / 'calibration' / 'actual_policy.json'], loaded=loaded,
        branch='ACTUAL_LINK_EVALUATED', metric_rows=len(allrows), selected_contexts=len(selected),
        ideal_prefix_assumption=False, ideal_gain_assumption=False,
        all_failures_in_quality=True, latent_error_valid_only=True)


def synchronize(loaded):
    if loaded['device'].type == 'cuda': torch.cuda.synchronize(loaded['device'])


@torch.no_grad()
def transmit_online(loaded, record, projection, N=None, phy='QPSK'):
    """CPU RGB -> CPU waveform, with original encoder and source quantization."""
    image = torch.as_tensor(record['pixels'][None], dtype=torch.float32, device=loaded['device']) / 127.5 - 1
    F = loaded['vae'].quant_conv(loaded['vae'].encoder(image))
    tokens = loaded['vae'].quantize.f_to_idxBl_or_fhat(F, to_fhat=False)
    prefix = tokens[:4]
    pre = rx.prefix_latent(loaded['vae'].quantize, prefix)
    z = projection.forward(F - pre)[0].cpu().numpy()
    scales = [t[0].cpu().numpy() for t in tokens]
    if N is None:
        wave, gain = rx.analog_transmit(z)
        return wave, dict(prefix=scales[:4], gain=gain, ideal_prefix_and_gain=True)
    return actual_transmit(scales, z, N, projection, phy)


@torch.no_grad()
def receive_online(loaded, wave, snr, projection, fitted, operators, static, policy, control,
                   *, N=None, phy='QPSK', ideal=None):
    if N is None:
        prefix = [torch.as_tensor(x, dtype=torch.long, device=loaded['device'])[None] for x in ideal['prefix']]
        gain = ideal['gain']
        obs = torch.as_tensor(wave[:-1].ravel() / gain, dtype=torch.float32, device=loaded['device'])[None]
        variance = 10. ** (-float(snr) / 10) / (gain * gain)
        base = greedy(loaded, prefix) if control in ('UNGUIDED', 'DIRECT') else None
        lam = policy['var'][projection.name]['selected_lambda'] if control == 'VAR_GUIDED' else policy['static'][projection.name]['selected_lambda'] if control == 'STATIC' else 0.
        latent = run_control(loaded, prefix, obs, projection, fitted, operators, static, variance, control, lam, base)
    else:
        event = actual_receive(wave, snr, N, projection, phy)
        if not event['header_ok']: latent = None
        else:
            prefix = [torch.as_tensor(x, dtype=torch.long, device=loaded['device'])[None] for x in event['prefix']]
            base = greedy(loaded, prefix) if not event['analog_used'] or control in ('UNGUIDED', 'DIRECT') else None
            latent = base
            if event['analog_used']:
                obs = torch.as_tensor(event['observation'], dtype=torch.float32, device=loaded['device'])[None]
                lam = policy['var'][projection.name]['selected_lambda'] if control == 'VAR_GUIDED' else policy['static'][projection.name]['selected_lambda'] if control == 'STATIC' else 0.
                latent = run_control(loaded, prefix, obs, projection, fitted, operators, static,
                                     event['noise_variance'], control, lam, base)
    if latent is None: return np.full((3, 256, 256), .5, dtype=np.float32)
    return loaded['decoder'](latent.to(loaded['device']))[0].detach().cpu().numpy()


@torch.no_grad()
def timing(loaded):
    require_m1(); policy, gate = frozen_policy()
    d = c.data('development', loaded)
    parity = []
    for i in TIMING:
        c.check()
        image = torch.as_tensor(d['records'][i]['pixels'][None], dtype=torch.float32, device=loaded['device']) / 127.5 - 1
        online_F = loaded['vae'].quant_conv(loaded['vae'].encoder(image))
        online_T = torch.cat(loaded['vae'].quantize.f_to_idxBl_or_fhat(online_F, to_fhat=False), 1)[0].cpu()
        if not torch.allclose(online_F[0].cpu(), d['F'][i], atol=1e-5, rtol=0):
            raise RuntimeError('M2 timing online F differs from registered development F')
        if not torch.equal(online_T, d['T'][i]):
            raise RuntimeError('M2 timing online source tokens differ from registered development tokens')
        parity.append(dict(source_index=i, source_id=d['records'][i]['image_id'],
                           F_max_difference=float((online_F[0].cpu() - d['F'][i]).abs().max()), exact_T=True))
    projections, stats, operators, static = load_state(loaded)
    rows = []; actual_policy = c.read(OUT / 'calibration' / 'actual_policy.json')
    # All controls at7dB for every oracle projection; explicitly ideal endpoints.
    contexts = [dict(projection=name, N=None, phy_family='ideal', snr_db=7, scope='oracle_only_ideal_prefix_gain') for name in projections]
    contexts += [dict(projection=choice['projection'], N=choice['N'], phy_family=choice['phy_family'],
                     snr_db=choice['snr_db'], scope='actual_paid_frame')
                 for choice in actual_policy['choices'] if choice['status'] == 'SELECTED']
    for context in contexts:
        name, N, phy, snr = context['projection'], context['N'], context['phy_family'], context['snr_db']
        projection = projections[name]
        # Freeze/cache linear operators before timing; they depend only on the
        # frozen model/projection, not the image. No image output is cached.
        for k in range(4, 10): operators[name].get(k)
        for i in TIMING:
            record = d['records'][i]
            for control in CONTROLS:
                for repeat in (-1, 0, 1):
                    c.check(); synchronize(loaded); start = time.perf_counter()
                    wave, info = transmit_online(loaded, record, projection, N, phy if N is not None else 'QPSK')
                    synchronize(loaded); tx_ms = 1000 * (time.perf_counter() - start)
                    if N is None: perturbation = noise(record['image_id'], projection.dimension, 2001)
                    else: perturbation = seeded_noise(f'{c.RUN}/M2/actual/N{N}|{record["image_id"]}', 2001, (N, 2))
                    observed = wave + perturbation * 10. ** (-float(snr) / 20)
                    synchronize(loaded); start = time.perf_counter()
                    image = receive_online(loaded, observed, snr, projection, stats[name], operators[name], static,
                        policy, control, N=N, phy=phy if N is not None else 'QPSK', ideal=info if N is None else None)
                    synchronize(loaded); rx_ms = 1000 * (time.perf_counter() - start)
                    if not np.isfinite(image).all(): raise RuntimeError('Timing output is nonfinite')
                    if repeat >= 0:
                        rows.append(dict(**context, source_id=record['image_id'], source_index=i,
                            control=control, repeat=repeat, tx_ms=tx_ms, rx_ms=rx_ms, total_ms=tx_ms + rx_ms,
                            endpoint='CPU_RGB_to_CPU_wave_and_CPU_wave_to_CPU_RGB',
                            excludes='airtime/queue; LPIPS/DINO scoring', noise_seed=2001,
                            deployment_endpoint=N is not None, no_image_output_cache=True))
    path = c.RESULT / 'm2_timing.csv'; c.csv_rows(path, rows)
    complete_receipt('timing', [path], loaded=loaded, timing_rows=len(rows), sources=10,
        repeats=2, warmups_per_source_control_context=1,
        operator_cache_scope='fixed model/projection matrix only; image outputs uncached',
        online_source_parity_checked_outside_timing=parity,
        actual_link_gate=gate['status'], oracle_timing_is_not_deployment_latency=True)


def main(stage):
    # This check precedes any frozen model loading, even for qualification.
    require_m1()
    loaded = c.setup()
    stages = ('calibration', 'evaluation', 'actual', 'timing') if stage == 'all' else (stage,)
    functions = dict(calibration=calibration, evaluation=evaluation, actual=actual, timing=timing)
    for name in stages:
        completion = c.OUT / f'm2_{name}_complete.json'
        if completion.exists():
            receipt = c.read(completion)
            if receipt['status'] != f'M2_{name.upper()}_COMPLETE': raise RuntimeError('Wrong stage receipt')
            for p, h in receipt['files'].items():
                if c.sha(p) != h: raise RuntimeError('Completed M2 output changed')
            continue
        functions[name](loaded)


@torch.no_grad()
def evaluation(loaded):
    require_m1(); policy, gate = frozen_policy()
    d = c.data('development', loaded)
    registration = c.registration(loaded, d, 'm2_evaluation', [c.OUT / 'm2_calibration_complete.json',
        OUT / 'calibration' / 'policy.json', OUT / 'calibration' / 'gate.json',
        OUT / 'calibration' / 'actual_policy.json'])
    projections, stats, operators, static = load_state(loaded)
    all_tokens = split_batch(d['T']); rows = []
    for i, record in enumerate(d['records']):
        c.check(); c.status('m2_evaluation', sources=i, total=100)
        def produce(i=i, record=record):
            result = []
            prefix = [t[i:i + 1].to(loaded['device']) for t in all_tokens[:4]]
            pre = rx.prefix_latent(loaded['vae'].quantize, prefix)
            clean = d['F'][i:i + 1].to(loaded['device']); base = greedy(loaded, prefix)
            base_metrics, base_image = score(loaded, d, i, base, images=i in SAMPLES)
            for name, projection in projections.items():
                for snr in ['clean'] + c.SNRS:
                    for seed in [0] if snr == 'clean' else c.DEV_SEEDS:
                        obs, variance, info = measure(clean, pre, projection, record, snr, seed)
                        stage = ('full_noiseless' if name == 'g8_c32' else 'reduced_noiseless') if snr == 'clean' else 'reduced_noisy'
                        if name == 'g8_c32' and snr != 'clean': stage = 'full_noisy_infeasible_oracle'
                        for control in CONTROLS:
                            c.check()
                            lam = policy['var'][name]['selected_lambda'] if control == 'VAR_GUIDED' else policy['static'][name]['selected_lambda'] if control == 'STATIC' else 0.
                            pic = base_image
                            if control == 'UNGUIDED':
                                metrics = base_metrics
                            else:
                                latent = run_control(loaded, prefix, obs, projection, stats[name], operators[name], static,
                                                     variance, control, lam, base)
                                metrics, pic = score(loaded, d, i, latent, images=i in SAMPLES and snr in ('clean', 4, 13) and seed in (0, 2001))
                            result.append(dict(**source_meta(record, i, projection, stage, snr, seed),
                                control=control, **{'lambda': lam}, direct_alpha=1. if control == 'DIRECT' else '', **info, **metrics))
                            if pic is not None and i in SAMPLES and snr in ('clean', 4, 13) and seed in (0, 2001):
                                c.save_image(c.RESULT / 'm2_examples' / f'{i:03d}_{name}_{snr}_{control}.png', pic)
            return result
        rows += cached_source(OUT / 'development' / 'oracle' / f'source_{i:04d}.json', registration, produce)
    path = OUT / 'development' / 'oracle_per_frame.csv'; c.csv_rows(path, rows)
    expected = 100 * 6 * 5 * (1 + 5 * 3)
    if len(rows) != expected: raise RuntimeError('Incomplete six-projection oracle ladder')
    c.csv_rows(c.RESULT / 'm2_oracle_per_frame.csv', rows)
    complete_receipt('evaluation', [path, c.RESULT / 'm2_oracle_per_frame.csv'], loaded=loaded,
        metric_rows=len(rows), clean_rows=100 * 6 * 5, noisy_rows=100 * 6 * 5 * 5 * 3,
        actual_link_gate=gate['status'], oracle_does_not_prove_actual_link=True,
        development_reused=True, independent_training_seeds=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['calibration', 'evaluation', 'actual', 'timing', 'all'], required=True)
    main(parser.parse_args().stage)

