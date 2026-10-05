"""Pure CPU selection from completed H initial_true200 receiver measurements.

PREPARED ONLY: no CLI, model, decoder, filesystem writes or stage scheduling.
The future caller must bind this module and the consumed completion receipt.
The renderer seals and verifies RGB archives; this core checks their references,
per-frame proof consistency, raw metric-file SHA and the full source/noise grid.
"""
from __future__ import annotations
import hashlib
import json
import math
from fractions import Fraction
from pathlib import PurePosixPath, PureWindowsPath
import re

ARMS = ('H16-R', 'H16-A', 'H64-R', 'H64-A')
SNRS = (13, 19)
SEEDS = (6101, 6102, 6103)
RATES = ('1/2', '2/3', '3/4', '5/6')
EPSILON = 1e-12
FINAL_STATUSES = ('RAW_SOURCE_DECODED', 'ARITHMETIC_SOURCE_DECODED',
                  'WIRE_REJECT_GRAY', 'ARITHMETIC_SOURCE_INVALID_GRAY')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest_bytes(value):
    require(isinstance(value, bytes), 'Original JSON bytes are required for SHA verification')
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def absolute_path(value):
    return isinstance(value, str) and (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute())


def decode_json(value):
    require(isinstance(value, bytes), 'Expected original JSON file bytes')
    return json.loads(value.decode('utf-8-sig'))


def validate_shortlist(shortlist):
    require(shortlist.get('status') == 'H_EXPECTED_PSNR_SHORTLIST_FROZEN'
            and shortlist.get('ready_for_real_calibration') is True, 'Final ready frozen shortlist required')
    ids = shortlist['source_ids']
    require(len(ids) == len(set(ids)) == 200 and all(isinstance(x, str) and x for x in ids), 'Wrong original hash200')
    candidates = shortlist['whole_candidates']; by_key = {}; slots = set(); families = {}
    for c in candidates:
        family = (c['arm'], c['snr_db'])
        require(family in {(a, s) for a in ARMS for s in SNRS}, 'Candidate outside four-arm/main-SNR scope')
        require(type(c['target_m']) is int and c['target_m'] in (6, 7, 8, 9) and c.get('K', 0) == 0,
                'Only frozen whole targetm6..m9 is eligible')
        require(c['q'] == (4 if c['arm'].startswith('H16') else 6) and c['nominal_rate'] in RATES,
                'Frozen arm/PHY resource inconsistent')
        require(type(c['slot']) is int and 0 <= c['slot'] < 16 and c['slot'] not in slots, 'Invalid/duplicate candidate slot')
        require(isinstance(c['candidate_id'], str) and c['candidate_id'] and sha_string(c['policy_key']),
                'Missing frozen policy identity')
        key = (c['candidate_id'], c['snr_db'])
        require(key not in by_key, 'Duplicate frozen candidate/SNR')
        expected_cost = float(c['target_m'] if c['arm'].endswith('-A') else 0)
        require(c['mean_attempted_tx_probability_scales'] == expected_cost, 'Frozen TX probability-scale tie cost differs')
        by_key[key] = c; slots.add(c['slot']); families[family] = families.get(family, 0)+1
    require(set(families) == {(a, s) for a in ARMS for s in SNRS} and all(1 <= n <= 2 for n in families.values()),
            'Each four-arm/SNR family needs exactly one or two frozen candidates')
    require(len(candidates)*200*3 <= 9600, 'Candidate count exceeds registered whole scope')
    return ids, by_key


def validate_receiver_row(row, candidate, ids, outputs):
    index = row['source_index']; seed = row['noise_seed']
    require(type(index) is int and 0 <= index < 200 and row['source_id'] == ids[index], 'Source index/ID differs')
    require(type(seed) is int and seed in SEEDS, 'Wrong calibration noise seed; development is prohibited')
    for key in ('candidate_id', 'arm', 'target_m', 'q', 'nominal_rate', 'snr_db', 'slot'):
        require(row[key] == candidate[key], 'Frame differs from frozen candidate: '+key)
    if 'K' in row:
        require(row['K'] == 0, 'Unexpected partial-source candidate')
    mse, psnr = row['mse'], row['psnr_db']
    require(type(mse) in (int, float) and type(psnr) in (int, float)
            and 0 < mse <= 1 and math.isfinite(mse) and math.isfinite(psnr), 'Invalid per-frame PSNR/MSE')
    require(abs(psnr+10*math.log10(mse)) <= 1e-9, 'PSNR is inconsistent with that frame MSE')
    status = row['source_status']; summary = row['rx_summary']
    require(status in FINAL_STATUSES and type(row['gray']) is bool, 'Unknown/pending/software-error receiver state')
    require(summary['status'] == 'H_ACTUAL_RX_RECONSTRUCTION_COMPLETE'
            and summary['source_decode_complete'] is True and summary['new_packet_decodes'] == 0,
            'Receiver reconstruction is not complete')
    require(summary['source_status'] == status and summary['gray'] is row['gray'], 'Top-level and receiver status disagree')
    require(row['gray'] is (status in ('WIRE_REJECT_GRAY', 'ARITHMETIC_SOURCE_INVALID_GRAY')),
            'Receiver status cannot be silently relabelled gray')
    for key in ('target_image_used_for_reconstruction', 'truth_correction', 'cached_clean_image_used'):
        require(summary[key] is False, 'Receiver used source truth or a clean-output substitute')
    for key in ('image_sha256', 'receiver_view_sha256'):
        require(sha_string(row[key]) and row[key] == summary[key], 'Receiver/image SHA differs: '+key)
    archive, image_key = row['image_archive'], row['image_key']
    require(absolute_path(archive) and sha_string(outputs.get(archive))
            and isinstance(image_key, str) and image_key, 'Actual RGB archive/reference is not sealed')
    if status == 'ARITHMETIC_SOURCE_DECODED':
        require(summary['arithmetic_canonical_attempted'] is True and summary['arithmetic_canonical'] is True
                and summary['received_mode'] == 'arithmetic' and summary['body_parser_accepted'] is True,
                'Arithmetic result lacks successful independent canonical decoding')
    elif status == 'ARITHMETIC_SOURCE_INVALID_GRAY':
        require(summary['arithmetic_canonical_attempted'] is True and summary['arithmetic_canonical'] is False
                and summary['canonical_decode_invalid'] is True and summary['received_mode'] == 'arithmetic',
                'Invalid arithmetic stream lacks a completed canonical rejection')
    elif status == 'RAW_SOURCE_DECODED':
        require(summary['received_mode'] == 'raw' and summary['body_parser_accepted'] is True
                and summary['arithmetic_canonical_attempted'] is False, 'Raw decoded state inconsistent')
    else:
        require(summary['header_accepted'] is False or summary['body_parser_accepted'] is False,
                'A successfully parsed actual payload cannot be called wire rejection')
    return index, seed


def tie_key(candidate):
    return (candidate['mean_attempted_tx_probability_scales'], candidate['q'],
            Fraction(candidate['nominal_rate']), candidate['target_m'], candidate.get('K', 0), candidate['policy_key'])


def choose_one(rows, by_key):
    best = max(r['mean_per_source_psnr_db'] for r in rows)
    tied = [r for r in rows if best-r['mean_per_source_psnr_db'] <= EPSILON]
    return min(tied, key=lambda r: tie_key(by_key[r['candidate_id'], r['snr_db']]))


def select_initial(shortlist_bytes, frame_metrics_bytes, *, frame_metrics_path,
                   render_completion, protocol, interpretation):
    """Return eight selected original policies and complete per-source evidence.

    No input may be incomplete. The caller separately registers this computation
    and verifies the renderer's immutable completion receipt and output archives.
    This pure function checks original JSON bytes and their bound SHA references;
    it does not re-read RGB archives or turn cached expected quality into data.
    """
    require(protocol['schema'] == 'H_CODEC_PROTOCOL_V1' and protocol['status'] == 'FROZEN_BEFORE_DATA'
            and protocol['main_snrs_db'] == list(SNRS)
            and protocol['population']['initial_and_full_cal_noise_seeds'] == list(SEEDS)
            and protocol['calibration']['no_development_selection'] is True, 'Wrong registered H selection protocol')
    require(interpretation['schema'] == 'H_PRESCREEN_INTERPRETATION_V1'
            and interpretation['status'] == 'FROZEN_BEFORE_COARSE_DATA'
            and interpretation['original_protocol_changed'] is False, 'TX-cost/tie interpretation is not frozen')
    receipt = render_completion; outputs = receipt['outputs']
    require(receipt['status'] == 'H_INITIAL_TRUE200_RX_COMPLETE' and receipt['images_scored'] is True
            and receipt['source_decode_complete'] is True and receipt['source_count'] == 200
            and receipt['new_packet_decodes'] == 0 and receipt['development_used'] is False,
            'Actual completed initial_true200 render receipt required')
    require(sha_string(receipt['registration_sha256']), 'Renderer execution registration missing')
    require(receipt['shortlist_sha256'] == digest_bytes(shortlist_bytes), 'Frozen shortlist raw SHA changed')
    require(absolute_path(frame_metrics_path) and outputs.get(frame_metrics_path) == digest_bytes(frame_metrics_bytes),
            'Raw frame-metrics file differs from sealed renderer output')
    shortlist = decode_json(shortlist_bytes); frame_rows = decode_json(frame_metrics_bytes)
    require(isinstance(frame_rows, list), 'Metric file must contain the registered frame row list')
    ids, by_key = validate_shortlist(shortlist)
    require(len(frame_rows) == receipt['frame_count'] == len(by_key)*200*3, 'Incomplete/extra actual frame count')
    grouped = {k: {} for k in by_key}; physical_images = {}
    for row in frame_rows:
        candidate_key = (row['candidate_id'], row['snr_db'])
        require(candidate_key in by_key, 'Measured row is not in the frozen shortlist')
        source_index, seed = validate_receiver_row(row, by_key[candidate_key], ids, outputs)
        key = source_index, seed
        require(key not in grouped[candidate_key], 'Duplicate candidate/source/noise row')
        grouped[candidate_key][key] = row
        image_ref = row['image_archive'], row['image_key']
        if image_ref in physical_images:
            require(physical_images[image_ref] == row['image_sha256'], 'One RGB reference has inconsistent image hashes')
        physical_images[image_ref] = row['image_sha256']
    expected = {(i, seed) for i in range(200) for seed in SEEDS}
    evidence, summaries = [], []
    for key, candidate in sorted(by_key.items(), key=lambda item: item[1]['slot']):
        frames = grouped[key]
        require(set(frames) == expected, 'Exact200-source by3-noise Cartesian grid is incomplete')
        source_psnr, source_mse = [], []
        statuses = {state: 0 for state in FINAL_STATUSES}
        for i, sid in enumerate(ids):
            samples = [frames[i, seed] for seed in SEEDS]
            psnr = math.fsum(r['psnr_db'] for r in samples)/3
            mse = math.fsum(r['mse'] for r in samples)/3
            source_psnr.append(psnr); source_mse.append(mse)
            for row in samples:
                statuses[row['source_status']] += 1
            evidence.append(dict(candidate_id=key[0], snr_db=key[1], arm=candidate['arm'], slot=candidate['slot'],
                source_index=i, source_id=sid, mean_noise_psnr_db=psnr, mean_noise_mse=mse,
                noise_samples=[{field: r[field] for field in ('noise_seed', 'psnr_db', 'mse', 'source_status', 'gray',
                      'image_sha256', 'receiver_view_sha256', 'image_archive', 'image_key')} for r in samples]))
        mean_psnr, mean_mse = math.fsum(source_psnr)/200, math.fsum(source_mse)/200
        summaries.append(dict(candidate_id=key[0], snr_db=key[1], arm=candidate['arm'], slot=candidate['slot'],
            source_count=200, frames=600, mean_per_source_psnr_db=mean_psnr, mean_overall_mse=mean_mse,
            psnr_of_mean_mse_db=-10*math.log10(mean_mse), mean_mse_is_selection_objective=False,
            final_receiver_status_counts=statuses, mean_attempted_tx_probability_scales=candidate['mean_attempted_tx_probability_scales']))
    selected, decisions = [], []
    for snr in SNRS:
        for arm in ARMS:
            family = [r for r in summaries if r['arm'] == arm and r['snr_db'] == snr]
            chosen = choose_one(family, by_key); c = by_key[chosen['candidate_id'], snr]
            selected.append(dict(c, prescreen_selection_rank=c.get('selection_rank'), selection_rank=1,
                initial_true200_mean_psnr_db=chosen['mean_per_source_psnr_db'],
                initial_true200_mean_mse=chosen['mean_overall_mse'],
                selection_scope='ACTUAL_HASH200_ALL_THREE_CALIBRATION_NOISES_MEAN_PER_SOURCE_PSNR'))
            max_score = max(r['mean_per_source_psnr_db'] for r in family)
            decisions.append(dict(arm=arm, snr_db=snr, selected_candidate_id=c['candidate_id'],
                considered_candidate_ids=[r['candidate_id'] for r in family],
                within_epsilon_ids=[r['candidate_id'] for r in family if max_score-r['mean_per_source_psnr_db'] <= EPSILON],
                epsilon_psnr_db=EPSILON,
                tie_cost='Arithmetic target_m; raw0; then q, exact rational rate, target_m/K, lexical policy_key'))
    return dict(status='H_INITIAL_TRUE200_SELECTION_COMPLETE', schema='H_INITIAL_TRUE200_SELECTION_V1',
        source_ids=ids, selected_candidates=selected, all_candidate_summaries=summaries,
        per_source_evidence=evidence, decisions=decisions, source_count=200,
        candidate_count=len(by_key), measured_frames=len(frame_rows), selected_count=8,
        input_proofs=dict(shortlist_sha256=digest_bytes(shortlist_bytes), frame_metrics_path=frame_metrics_path,
             frame_metrics_sha256=digest_bytes(frame_metrics_bytes),
             render_completion_canonical_sha256=digest(receipt), render_registration_sha256=receipt['registration_sha256'],
             protocol_canonical_sha256=digest(protocol), interpretation_canonical_sha256=digest(interpretation)),
        selection_objective='Equal weight noise-mean PSNR per source; all failure frames included',
        metric_proof_scope='Renderer sealed and checked RGB arrays; this pure core verifies metric-file bytes, archive references and matching per-frame receiver hashes, without reading RGB archives again',
        new_candidate_search=False, new_packet_decodes=0, new_visual_inference=0,
        development_used=False, holdout_used=False, full1000_calibration_complete=False,
        execution_registration_created=False)
