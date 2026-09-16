"""Read-only convergence audit of completed trials; no training, inference, parameter search or file deletion."""

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT / 'outputs'
DESTINATION = OUTPUTS / 'CONVERGENCE-AUDIT-20260915'
END = datetime.fromisoformat('2026-09-15T01:28:36+08:00')
METRICS = ('psnr_db', 'ssim', 'lpips', 'dino', 'LPIPS_excess_from_native', 'severe_distortion')
TRIALS = (
    ('RX_ONLY_5000', 'WETOK-INNOVATION-R1-TRAINING', 5000, 'frozen__', 'fixed_E_train_R',
     'WETOK-INNOVATION-R1-EVALUATION/step_0005000', 'WETOK-INNOVATION-R1-ANALYSIS/development_0005000'),
    ('JOINT_5000', 'WETOK-JOINT-SENDER-R1-TRAINING', 5000, 'joint__', 'train_E_and_R',
     'WETOK-JOINT-SENDER-R1-EVALUATION/step_0005000', 'WETOK-JOINT-SENDER-R1-ANALYSIS/development_0005000'),
    ('GRID_5000', 'WETOK-JOINT-GRID-CONTROLS-R1-TRAINING', 5000, 'grid__', 'train_E_and_R',
     'WETOK-JOINT-GRID-CONTROLS-R1-EVALUATION/quality_0005000', 'WETOK-JOINT-GRID-CONTROLS-R1-ANALYSIS/quality_0005000'),
    ('R2_10000', 'WETOK-JOINT-SUFFICIENCY-R2-TRAINING', 10000, 'r2__', 'continued_actual5000_model_and_Adam',
     'WETOK-JOINT-SUFFICIENCY-R2-EVALUATION/quality_0010000', 'WETOK-JOINT-SUFFICIENCY-R2-ANALYSIS/quality_0010000'),
)


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def event_time(record):
    for key in ('completed_local', 'completed_at', 'finished_at', 'finished_local', 'local_time'):
        value = record.get(key)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
                if parsed.tzinfo is not None:
                    return parsed, key
            except ValueError:
                pass
    return None, ''


def verify_receipt(root, manifest):
    path = root / 'completion.json'
    record = read_json(path)
    manifest[str(path)] = sha(path)
    for relative, expected in record.get('output_hashes', {}).items():
        if sha(root / relative) != expected:
            raise RuntimeError(f'closed artifact changed: {root / relative}')
    return record


def inventory():
    listing = subprocess.check_output(['rg', '--files', '--hidden', '--no-ignore', str(OUTPUTS), '-g', 'completion.json',
        '-g', '*.pipeline.json', '-g', 'step_*.json'], text=True).splitlines()
    events, undated = [], []
    for text in listing:
        if '/snapshots/' in text or '/images/' in text:
            continue
        path = Path(text)
        record = read_json(path)
        status = str(record.get('status', ''))
        stamp, field = event_time(record)
        item = {'path': str(path.relative_to(PROJECT)), 'status': status, 'sha256': sha(path)}
        if stamp is None:
            if 'COMPLETE' in status or 'PASS' in status:
                undated.append({**item, 'reason': 'no_explicit_timezone_aware_completion_time_not_inferred_from_mtime'})
            continue
        if END - timedelta(days=1) <= stamp <= END:
            if 'NEEDS_REVIEW' in status or 'FAILED' in status:
                kind = 'failure_or_interruption_retained'
            elif '/milestones/' in text:
                kind = 'training_milestone_not_separate_system_trial'
            elif 'FIGURE' in status or 'PLOT' in status or 'VIEWS' in status:
                kind = 'reporting_not_new_transmission_experiment'
            elif 'CALIBRATION' in status:
                kind = 'calibration_or_audit'
            elif 'TIMING' in status:
                kind = 'timing_or_pipeline'
            else:
                kind = 'evaluation_analysis_or_engineering'
            events.append({**item, 'completed_time': stamp.isoformat(), 'time_field': field, 'kind': kind})
    return sorted(events, key=lambda row: row['completed_time']), undated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: fixed24h completion inventory, closed training/calibration/quality/cost audit; no GPU')
        return
    torch.set_num_threads(2)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    table_root = DESTINATION / 'tables'
    if table_root.exists():
        raise RuntimeError('audit tables already exist; preserve them and use a separate revision instead of overwriting')
    table_root.mkdir()
    events, undated = inventory()
    write_csv(table_root / 'completion_events_24h.csv', events)
    write_csv(table_root / 'artifacts_without_completion_time.csv', undated)
    manifest, experiments, qualifications, curves, costs = {}, [], [], [], []
    for trial, directory, step, prefix, policy, quality_directory, analysis_directory in TRIALS:
        root = OUTPUTS / directory
        milestone_path = root / 'milestones' / f'step_{step:07d}.json'
        milestone = read_json(milestone_path)
        stamp = datetime.fromisoformat(milestone['completed_local'])
        if not END - timedelta(days=1) <= stamp <= END or sha(milestone['checkpoint']) != milestone['checkpoint_sha256']:
            raise RuntimeError('trial is outside the requested window or its checkpoint changed')
        manifest[str(milestone_path)] = sha(milestone_path)
        manifest[milestone['checkpoint']] = milestone['checkpoint_sha256']
        state = torch.load(milestone['checkpoint'], map_location='cpu', weights_only=True)
        metadata = read_json(root / 'metadata.json')
        quality = verify_receipt(OUTPUTS / quality_directory, manifest)
        analysis = verify_receipt(OUTPUTS / analysis_directory, manifest)
        analysis_stamp, unused = event_time(analysis)
        if analysis_stamp is None or not END - timedelta(days=1) <= analysis_stamp <= END:
            raise RuntimeError('a claimed completed system trial has no in-window final analysis')
        initial = 5000 if trial == 'R2_10000' else 0
        experiments.append({'trial': trial, 'arms': len(milestone['selected']), 'training_completed_local': milestone['completed_local'],
            'analysis_completed_local': analysis_stamp.isoformat(), 'full_branch_updates_each': step,
            'increments_in_this_trial_each': step - initial, 'shared_parent_updates': 7000,
            'train_images': 20000, 'calibration_images': 1000, 'development_images': 100,
            'source_population': 'original_development_not_fresh_test', 'new_random_training_seeds': 0,
            'quality_rows_including_reused_references': quality.get('rows'), 'policy': policy,
            'recorded_active_process_hours': float(state['elapsed_seconds']) / 3600,
            'cost_scope': 'cross_day_with_unrecorded_interruption_work_unknown' if trial == 'RX_ONLY_5000' else 'trial_active_process_time_not_GPU_utilization_integral',
            'shared_parent_cost_included': False, 'milestone': str(milestone_path), 'analysis': str(OUTPUTS / analysis_directory)})
        for variant, choice in milestone['selected'].items():
            checkpoint = Path(choice['checkpoint'])
            if not checkpoint.is_absolute():
                checkpoint = root / checkpoint
            if sha(checkpoint) != choice['checkpoint_sha256'] or choice['scope'] != 'full' or int(choice['source_images']) != 1000:
                raise RuntimeError('a displayed learned baseline is not its qualified complete-calibration choice')
            manifest[str(checkpoint)] = choice['checkpoint_sha256']
            history = state['histories'][variant]
            if len(history) != step:
                raise RuntimeError('training history length does not match full opportunities')
            for point in sorted({int(row['step']) for row in state['summaries'] if row['variant'] == variant and row['scope'] == 'full'}):
                subset = [row for row in state['summaries'] if row['variant'] == variant and row['scope'] == 'full' and int(row['step']) == point]
                curves.append({'trial': trial, 'arm': prefix + variant, 'step': point, 'scope': 'full_calibration_only',
                    'snrs_db': '1.0+4.0+7.0+13.0+19.0', 'source_images_per_snr': 1000,
                    **{metric: float(np.mean([float(row[metric]) for row in subset])) for metric in ('lpips', 'psnr_db', 'bit_error_rate', 'state_error')}})
            current_curve = [row for row in curves if row['trial'] == trial and row['arm'] == prefix + variant]
            optimizer = state['optimizers'][variant]
            actual_steps = sorted({int(value['step']) for value in optimizer['state'].values()})
            windows = {}
            for label, subset in (('previous1000', history[-2000:-1000]), ('last1000', history[-1000:])):
                for metric in ('loss', 'mse', 'lpips', 'bits', 'state'):
                    if subset and metric in subset[0]:
                        windows[label + '_' + metric] = float(np.mean([float(row[metric]) for row in subset]))
            qualifications.append({'trial': trial, 'arm': prefix + variant, 'policy': policy,
                'shared_parent_updates': 7000, 'branch_update_opportunities': step, 'selected_step': choice['step'],
                'selected_scope': choice['scope'], 'selected_full_calibration_images': choice['source_images'],
                'checkpoint_qualified': True, 'invalid_warmup_fallback': False, 'checkpoint': str(checkpoint),
                'checkpoint_sha256': choice['checkpoint_sha256'], 'optimizer_step_values': json.dumps(actual_steps),
                'initial_full_calibration_lpips': current_curve[0]['lpips'],
                'previous_full_calibration_step': current_curve[-2]['step'], 'previous_full_calibration_lpips': current_curve[-2]['lpips'],
                'last_full_calibration_step': current_curve[-1]['step'], 'last_full_calibration_lpips': current_curve[-1]['lpips'],
                'last_full_minus_previous_lpips': current_curve[-1]['lpips'] - current_curve[-2]['lpips'],
                'selected_full_calibration_lpips': choice['lpips'],
                'training_windows_are_different_batches_not_convergence_proof': True, **windows})
            costs.append({'trial': trial, 'arm': prefix + variant,
                'recorded_training_hours': float(state['timings']['training'][variant]) / 3600,
                'recorded_calibration_hours': float(state['timings']['calibration'][variant]) / 3600,
                'cost_scope': 'only_increment5000_to10000' if trial == 'R2_10000' else 'entire_branch_after_shared_parent',
                'shared_parent_cost_included': False})
        del state
    write_csv(table_root / 'completed_system_trials.csv', experiments)
    write_csv(table_root / 'training_qualification_and_sufficiency.csv', qualifications)
    write_csv(table_root / 'full_calibration_curves.csv', curves)
    write_csv(table_root / 'training_cost_scopes.csv', costs)
    quality_root = OUTPUTS / 'WETOK-JOINT-SUFFICIENCY-R2-EVALUATION/quality_0010000'
    analysis_root = OUTPUTS / 'WETOK-JOINT-SUFFICIENCY-R2-ANALYSIS/quality_0010000'
    timing_root = OUTPUTS / 'WETOK-JOINT-SUFFICIENCY-R2-EVALUATION/timing_0010000'
    verify_receipt(timing_root, manifest)
    summary = read_csv(analysis_root / 'summary.csv')
    write_csv(table_root / 'quality_all22_methods_all_scopes.csv', summary)
    write_csv(table_root / 'quality_primary.csv', [row for row in summary if row['snrs_db'] == '1.0+4.0+7.0'])
    write_csv(table_root / 'quality_supported_snrs.csv', [row for row in summary if row['snrs_db'] in ('1.0', '4.0', '7.0', '13.0', '19.0')])
    write_csv(table_root / 'paired_R2_existing.csv', read_csv(analysis_root / 'paired.csv'))
    write_csv(table_root / 'deep_fixed_support_summary.csv', read_csv(analysis_root / 'deep_support_summary.csv'))
    write_csv(table_root / 'online_matched15_models.csv', read_csv(timing_root / 'summary.csv'))
    write_csv(table_root / 'online_matched_paired.csv', read_csv(timing_root / 'paired.csv'))
    raw = read_csv(quality_root / 'per_frame.csv')
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in raw}
    base = yaml.safe_load((PROJECT / 'experiments/wetok-comm-v2-20260912/configs/study.yaml').read_text())
    noise_seeds = base['evaluation']['noise_seeds']
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(100, size=(10000, 100))
    paired = []
    for snrs in ([1., 4., 7.], [1.], [4.], [7.], [13.], [19.]):
        for control in ('digital_m8', 'perceptual_deepjscc', 'r2__full_grid_innovation'):
            for metric in METRICS:
                differences = np.array([np.mean([float(lookup[index, snr, seed, 'digital_adaptive'][metric]) -
                    float(lookup[index, snr, seed, control][metric]) for snr in snrs for seed in noise_seeds]) for index in range(100)])
                low, high = np.percentile(differences[draws].mean(1), (2.5, 97.5))
                paired.append({'method': 'digital_adaptive', 'control': control, 'snrs_db': '+'.join(map(str, snrs)),
                    'metric': metric, 'source_images': 100, 'delta': float(differences.mean()), 'ci_low': float(low), 'ci_high': float(high),
                    'scope': 'new_CPU_summary_of_existing_transmissions_not_new_inference_or_independent_test'})
    write_csv(table_root / 'digital_candidate_paired.csv', paired)
    ledgers = []
    for mode, payload in ((7, 1860), (8, 3060), (9, 5088)):
        ledgers.append({'method': f'VAR_m{mode}', 'raw_payload_bits': payload, 'body_CRC_bits': 16, 'body_tail_bits': 6,
            'body_information_bits': payload + 22, 'body_mother_code_bits': 2 * (payload + 22), 'body_transmitted_coded_bits': 5984,
            'header_class_bits': 10, 'header_mode_bits': 2, 'header_CRC_bits': 16, 'header_tail_bits': 6,
            'header_transmitted_coded_bits': 136, 'header_complex_uses': 68, 'data_complex_uses': 2992,
            'total_complex_uses': 3060, 'complex_symbol_energy': 2, 'total_energy': 6120,
            'body_info_to_coded_rate': (payload + 22) / 5984, 'ARQ': False,
            'side_information': 'class_known_at_TX_no_classifier_model_costed_nominal_SNR_given',
            'source_rate_policy': 'development_derived_SNR_thresholds_2.5_and_5.5_dB'})
    write_csv(table_root / 'digital_physical_ledger.csv', ledgers)
    old_digital = read_csv(OUTPUTS / 'VAR-PROGRESSIVE-CHANNEL-001/per_frame.csv')
    phy_times = []
    for snr in (1., 4., 7., 13., 19.):
        values = [float(row['receiver_seconds']) for row in old_digital if row['arm'] == 'whole_adaptive' and float(row['snr_db']) == snr]
        phy_times.append({'method': 'digital_adaptive', 'snr_db': snr, 'samples': len(values), 'PHY_decode_seconds_mean': float(np.mean(values)),
            'full_RX_seconds': '', 'TX_seconds': '', 'scope': 'historical_PHY_only_excludes_VAR_completion_and_image_decoder_not_comparable_to_full_RX'})
    write_csv(table_root / 'digital_online_cost_gap.csv', phy_times)
    exclusions = [
        {'method': 'VAR_small_parallel_B', 'available_updates': 20000, 'selected_step': 10000,
         'eligibility': 'invalid_warmup_fallback', 'use_in_strong_baseline_ranking': False,
         'reason': 'all_candidates_fail_registered_PSNR_constraint', 'evidence': 'reports/token_backbone_result_2026-09-12.md'},
        {'method': 'VAR_enhanced_parallel_B', 'available_updates': 20000, 'selected_step': 10000,
         'eligibility': 'invalid_warmup_fallback', 'use_in_strong_baseline_ranking': False,
         'reason': 'all_candidates_fail_registered_PSNR_constraint', 'evidence': 'reports/token_backbone_result_2026-09-12.md'},
    ]
    write_csv(table_root / 'excluded_unqualified_fallbacks.csv', exclusions)
    manifest[str(quality_root / 'per_frame.csv')] = sha(quality_root / 'per_frame.csv')
    manifest[str(PROJECT / 'reports/token_backbone_result_2026-09-12.md')] = sha(PROJECT / 'reports/token_backbone_result_2026-09-12.md')
    manifest[str(Path(__file__))] = sha(__file__)
    result = {'status': 'READ_ONLY_COMPLETED_TRIAL_CONVERGENCE_AUDIT_READY', 'window_start': (END - timedelta(days=1)).isoformat(),
        'window_end': END.isoformat(), 'completed_system_comparisons': len(experiments), 'completed_model_training_versions': len(qualifications),
        'completion_events_in_window': len(events), 'undated_artifacts_not_assigned_from_mtime': len(undated),
        'new_GPU_inference': False, 'new_training_or_parameter_search': False, 'new_holdout_accessed': False,
        'all_old_artifacts_read_only': True, 'source_artifacts': manifest,
        'output_hashes': {str(path.relative_to(DESTINATION)): sha(path) for path in sorted(table_root.glob('*.csv'))}}
    (DESTINATION / 'audit_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('source_artifacts', 'output_hashes')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
