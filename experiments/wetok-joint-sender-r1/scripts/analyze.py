"""Joint-update effects, structure interactions and strong-control comparisons at equal opportunity."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

import numpy as np

from joint_sender.archive_audit import audit_saved
from joint_sender.common import output_path
from joint_sender.evaluation import fresh_names, load_evaluation, matched_milestone, statistics, support_statistics, validate_diagnostics, validate_selection
from joint_sender.evaluation_io import write_rows
from joint_sender.references import References, read_rows
from wetok_comm.common import PROJECT, artifact_hashes, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.interface_evaluation import METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=5000)
    parser.add_argument('--evaluation-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation_config, config, reference, base, parent_record = load_evaluation()
    if arguments.step != 5000:
        raise ValueError('Joint final attribution requires equal 5000-update opportunities')
    if not arguments.execute:
        print('PLAN ONLY: 33600 outcomes, 2016 paired intervals, 144 Joint/history interactions and 162 support intervals; no GPU')
        return
    milestone, milestone_path, review_path, control, control_path, control_review = matched_milestone(config, reference, 5000)
    references = References(evaluation_config)
    evaluation = (arguments.evaluation_dir or output_path(config, 'evaluation') / 'step_0005000').resolve()
    receipt = json.loads((evaluation / 'completion.json').read_text())
    if (receipt['status'] != 'JOINT_EVALUATION_COMPLETE' or receipt['milestone_sha256'] != sha256(milestone_path) or
        receipt['control_milestone_sha256'] != sha256(control_path) or receipt['reference_qualification_sha256'] != sha256(references.qualification_path) or
        (receipt['rows'], receipt['noiseless_rows'], receipt['support_rows']) != (33600, 900, 600) or
        receipt['frozen_models_before'] != receipt['frozen_models_after'] or not receipt['nine_models_freshly_timed']):
        raise RuntimeError('Joint evaluation receipt does not prove the registered complete comparison')
    verify_sources(receipt['source_hashes'])
    for relative, expected in receipt['output_hashes'].items():
        if sha256(evaluation / relative) != expected:
            raise RuntimeError('Joint evaluation artifact changed')
    rows = read_rows(evaluation / 'per_frame.csv')
    clean = read_rows(evaluation / 'native_reference.csv')
    noiseless = read_rows(evaluation / 'noiseless_mapping.csv')
    support = read_rows(evaluation / 'deep_support_supplement.csv')
    hardware = read_rows(evaluation / 'hardware_telemetry.csv')
    validate_selection(rows, config, milestone, control)
    validate_diagnostics(rows, clean, noiseless, support, config, reference)
    summaries, paired, interactions = statistics(rows, config, reference, base)
    support_summary, support_paired = support_statistics(rows, support, config, reference, base)
    if (len(paired), len(interactions), len(support_paired)) != (2016, 144, 162):
        raise RuntimeError('registered Joint comparisons changed')
    audit = audit_saved(evaluation, rows, noiseless, support, evaluation_config, config, reference, base)
    if (len(hardware) != 200 or {(int(row['image_index']), row['phase']) for row in hardware} !=
        {(index, phase) for index in range(100) for phase in ('before_source', 'after_source')} or len({row['uuid'] for row in hardware}) != 1):
        raise RuntimeError('nine-model hardware timing context is incomplete')
    output = (arguments.output_dir or output_path(config, 'analysis') / 'development_0005000').resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('Joint analysis output escapes the project')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
        EXPERIMENT / 'src/joint_sender/evaluation.py', EXPERIMENT / 'src/joint_sender/references.py', EXPERIMENT / 'src/joint_sender/archive_audit.py'])
    for filename, values in (('summary.csv', summaries), ('paired.csv', paired), ('joint_history_interactions.csv', interactions),
                             ('deep_support_summary.csv', support_summary), ('deep_support_paired.csv', support_paired)):
        write_rows(output / filename, values)
    write_json(output / 'saved_waveform_image_audit.json', audit)
    label = '+'.join(map(str, base['evaluation']['primary_snrs_db']))
    primary = [row for row in summaries if row['snrs_db'] == label]
    write_rows(output / 'primary.csv', primary)
    costs, diagnostics, noiseless_summary = [], [], []
    for name in fresh_names(config, reference):
        variant = name.split('__', 1)[1]
        origin = milestone if name.startswith('joint__') else control
        frames = [row for row in rows if row['arm'] == name]
        first = frames[0]
        costs.append({'arm': name, 'training_policy': first['training_policy'], 'available_updates': 5000,
            'selected_step': origin['selected'][variant]['step'], 'communication_parameters': int(first['communication_parameters']),
            'optimized_parameters': int(first['optimized_parameters']), 'training_seconds': origin['timings']['training'][variant],
            'calibration_seconds': origin['timings']['calibration'][variant], 'sender_state_sha256': first['encoder_sha256'],
            'selected_encoder_differs_from_parent': first['selected_encoder_differs_from_parent'],
            'receiver_grid': '16' if variant in ('single_pass', 'multiscale_no_history') else ('16/16/16' if variant == 'full_grid_innovation' else '4/8/16'),
            'internal_E_calls': 2 if variant in ('full_grid_innovation', 'multiscale_innovation', 'multiscale_prediction_features') else 0})
        for snr in base['evaluation']['snrs_db']:
            subset = [row for row in frames if float(row['snr_db']) == snr]
            diagnostics.append({'arm': name, 'snr_db': snr, **{key: float(np.mean([float(row[key]) for row in subset]))
                for key in ('bit_error_rate', 'feature_mse', 'feature_abs_mean', 'feature_saturation_fraction')}})
        subset = [row for row in noiseless if row['arm'] == name]
        noiseless_summary.append({'arm': name, 'source_images': 100, 'role': 'not_wireless_ranking',
            **{key: float(np.mean([float(row[key]) for row in subset])) for key in (*METRICS, 'bit_error_rate', 'feature_mse')}})
    write_rows(output / 'costs.csv', costs)
    write_rows(output / 'feature_diagnostics.csv', diagnostics)
    write_rows(output / 'noiseless_summary.csv', noiseless_summary)
    recoveries = []
    for path in sorted((PROJECT / 'outputs').glob('WETOK-INNOVATION-R1-RECOVERY-*/audit.json')):
        record = json.loads(path.read_text())
        recoveries.append({'receipt': str(path), 'sha256': sha256(path), 'saved_receiver_step': record['checkpoint_step'],
                           'unobserved_work_after_last_save': record['unobserved_work_after_last_save_not_reconstructed']})
    write_json(output / 'observed_cost_context.json', {'selected_common_parent_lineage_seconds': {
            phase: parent_record['timings'][phase]['single_pass'] for phase in ('representation', 'image', 'calibration')},
        'parent_geometry_study_timings_all_arms_not_each_method_lineage': parent_record['timings'],
        'joint_observed_wall_GPU_hours': milestone['observed_wall_GPU_hours'],
        'frozen_control_observed_wall_GPU_hours': control['observed_wall_GPU_hours'],
        'frozen_shared_TX_seconds_counted_once': control['timings']['shared_TX'],
        'evaluation_observed_wall_hours': receipt['evaluation_observed_wall_hours_all_sessions'],
        'evaluation_time_lower_bound': receipt['time_lower_bound_due_to_unfinished_sessions'], 'receiver_control_recovery_records': recoveries,
        'unobserved_work_is_not_zero_and_downtime_is_not_training_time': True, 'additional_training_repeats_completed': 0})
    text = ['# Joint E/R与R-only：同预算development比较', '',
        '两侧各5000次新增更新机会，完整校准LPIPS选模；100图×七SNR×三噪声，主区间1/4/7。',
        '九模型在本机新测：Joint三基础结构和全部六个冻结E接收器；Joint与R-only并非同y，只保持同源、同噪声、同N/E。', '',
        '| 方法 | PSNR ↑ | LPIPS ↓ | DINO ↑ | 严重失真率 ↓ |', '|---|---:|---:|---:|---:|']
    text.extend(f'| {row["arm"]} | {row["psnr_db"]:.5f} | {row["lpips"]:.6f} | {row["dino"]:.6f} | {row["severe_distortion"]:.4f} |' for row in primary)
    text.extend(['', '## 同结构Joint减R-only', ''])
    for variant in config['variants']:
        row = next(row for row in paired if row['snrs_db'] == label and row['method'] == 'joint__' + variant and
                   row['control'] == 'frozen__' + variant and row['metric'] == 'lpips')
        text.append(f'- {variant}：ΔLPIPS {row["delta"]:+.6f}，逐比较95% CI [{row["ci_low"]:+.6f}, {row["ci_high"]:+.6f}]。')
    text.extend(['', '## 界限', '',
        '- 全网格迭代、预测特征和多尺度创新量冻结参照未删掉；普通全网格不能改名next-scale。',
        '- Joint条件若有增量，仍须对应Joint普通全网格对照，不能只据本三结构宣布排除额外迭代。',
        '- no-history部署仅需一次fine读取，训练辅助阶段已剔除，输出逐位核查；时延包含内部E和最终视觉Decoder。',
        '- 正确native/noiseless是诊断；DINO不参与loss或选模，不能单独代表实例身份。',
        '- 区间未作多重比较校正，不包含训练种子随机性；仍是development，没有新holdout或训练重复。'])
    (output / 'summary.md').write_text('\n'.join(text) + '\n')
    write_json(output / 'completion.json', {'status': 'JOINT_DEVELOPMENT_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'milestone_sha256': sha256(milestone_path), 'control_milestone_sha256': sha256(control_path),
        'evaluation_receipt_sha256': sha256(evaluation / 'completion.json'), 'source_hashes': sources,
        'main_rows': len(rows), 'paired_intervals': len(paired), 'interaction_intervals': len(interactions),
        'support_intervals': len(support_paired), 'new_holdout_accessed': False, 'research_goal_complete': False,
        'output_hashes': artifact_hashes(output)})
    print(json.dumps(primary, indent=2))


if __name__ == '__main__':
    main()
