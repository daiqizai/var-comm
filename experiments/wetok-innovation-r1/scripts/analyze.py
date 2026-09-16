"""Source-image paired attribution for the shared-waveform receiver study."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src'), str(REFERENCE / 'scripts')]

import numpy as np

from train_milestone import write_csv
from innovation_comm.archive_audit import audit_saved_evaluation
from innovation_comm.common import output_path
from innovation_comm.evaluation import audited_milestone, comparison_pairs, load_evaluation, receiver_name, statistics, supplement_statistics, validate_native_reference, validate_selection
from wetok_comm.common import PROJECT, artifact_hashes, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.interface_evaluation import METRICS


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--evaluation-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation_config, config, base, parent_milestone = load_evaluation()
    if arguments.step not in config['calibration']['full_steps'] or arguments.step <= 0:
        raise ValueError('use a registered complete-calibration milestone')
    if not arguments.execute:
        print('PLAN ONLY: 27300 matched rows, 1968 paired intervals and 108 fixed-support comparisons; no GPU or training')
        return
    milestone, milestone_path, review_path = audited_milestone(config, arguments.step)
    evaluation = (arguments.evaluation_dir or output_path(config, 'evaluation') / f'step_{arguments.step:07d}').resolve()
    receipt = json.loads((evaluation / 'completion.json').read_text())
    if (receipt['status'] != 'INNOVATION_EVALUATION_COMPLETE' or receipt['milestone_sha256'] != sha256(milestone_path) or
        receipt['reference_receipt_sha256'] != evaluation_config['reference_receipt_sha256'] or
        receipt['rows'] != 27300 or receipt['noiseless_rows'] != 600 or receipt['separate_fixed_support_rows'] != 600 or
        receipt['frozen_before'] != receipt['frozen_after'] or not receipt['shared_s_y_every_frame']):
        raise RuntimeError('receiver evaluation receipt does not match the complete pinned experiment')
    verify_sources(receipt['source_hashes'])
    for relative, expected in receipt['output_hashes'].items():
        if sha256(evaluation / relative) != expected:
            raise RuntimeError('receiver evaluation artifact changed')
    if (receipt['maximum_power_error'] > 1e-5 or
        receipt['maximum_parent_signal_error'] > evaluation_config['parent_signal_replay_tolerance']):
        raise RuntimeError('shared transmitter or physical budget differs from its reference')
    rows = read_rows(evaluation / 'per_frame.csv')
    clean = read_rows(evaluation / 'native_reference.csv')
    noiseless = read_rows(evaluation / 'noiseless_mapping.csv')
    support = read_rows(evaluation / 'deep_support_supplement.csv')
    hardware = read_rows(evaluation / 'hardware_telemetry.csv')
    if (len(hardware) != 200 or {(int(row['image_index']), row['phase']) for row in hardware} !=
        {(index, phase) for index in range(100) for phase in ('before_source', 'after_source')} or
        len({row['uuid'] for row in hardware}) != 1):
        raise RuntimeError('source-level hardware timing context is incomplete or changed devices')
    validate_native_reference(rows + support, clean, noiseless, config, base)
    validate_selection(rows, milestone)
    for row in rows + noiseless:
        if row['arm'] == receiver_name('multiscale_no_history') and float(row['no_history_prune_feature_max_error']) != 0:
            raise RuntimeError('the optimized no-history reference changed its image input')
    archive_audit = audit_saved_evaluation(evaluation, rows, noiseless, support, evaluation_config, config, base)
    summary, paired = statistics(rows, config, base)
    support_summary, support_paired = supplement_statistics(rows, support, config, base)
    if len(paired) != 1968 or len(support_paired) != 108:
        raise RuntimeError('registered contrasts were added or omitted')
    output = (arguments.output_dir or output_path(config, 'analysis') / f'development_{arguments.step:07d}').resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('receiver analysis output escapes the project')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'src/innovation_comm/evaluation.py',
        EXPERIMENT / 'src/innovation_comm/archive_audit.py',
        EXPERIMENT / 'src/innovation_comm/hardware.py',
        EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'configs/study.yaml', EXPERIMENT / 'docs/evaluation_protocol.md'])
    write_json(output / 'saved_channel_image_audit.json', archive_audit)
    temperatures = [float(row['temperature.gpu']) for row in hardware if row['temperature.gpu'] not in ('N/A', '[N/A]')]
    thermal_samples = sum(any(row[key] == 'Active' for key in
        ('clocks_event_reasons.sw_thermal_slowdown', 'clocks_event_reasons.hw_thermal_slowdown')) for row in hardware)
    write_json(output / 'hardware_timing_context.json', {'telemetry_rows': len(hardware), 'thermal_slowdown_samples': thermal_samples,
        'temperature_min_C': min(temperatures) if temperatures else None, 'temperature_max_C': max(temperatures) if temperatures else None,
        'receiver_order': 'rotating_equal_position_counts_on_the_fixed_2100_frame_grid',
        'clock_power_or_fan_settings_modified': False, 'historical_latency_is_not_current_host_ranking': True})
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'paired.csv', paired)
    write_csv(output / 'deep_support_summary.csv', support_summary)
    write_csv(output / 'deep_support_paired.csv', support_paired)
    label = '+'.join(map(str, base['evaluation']['primary_snrs_db']))
    primary = [row for row in summary if row['snrs_db'] == label]
    write_csv(output / 'primary.csv', primary)
    costs, diagnostic_rows, noiseless_summary = [], [], []
    for variant in config['variants']:
        name = receiver_name(variant)
        method_rows = [row for row in rows if row['arm'] == name]
        sample = method_rows[0]
        training_seconds = milestone['timings']['training'][variant]
        calibration_seconds = milestone['timings']['calibration'][variant]
        costs.append({'arm': name, 'parent_training_updates': 7000, 'available_receiver_updates': arguments.step,
            'selected_receiver_step': milestone['selected'][variant]['step'],
            'communication_parameters': int(sample['communication_parameters']),
            'receiver_trainable_parameters': int(sample['receiver_trainable_parameters']),
            'receiver_training_seconds': training_seconds, 'receiver_calibration_seconds': calibration_seconds,
            'new_receiver_train_and_calibration_GPU_hours': (training_seconds + calibration_seconds) / 3600,
            'shared_TX_all_arms_seconds': milestone['timings']['shared_TX'],
            'shared_TX_all_arms_cost_role': 'one_shared_cost_not_summed_six_times',
            'internal_E_calls_per_receive': 2 if variant in ('multiscale_prediction_features', 'multiscale_innovation', 'full_grid_innovation') else 0,
            'receive_grid_sizes': '16' if variant in ('single_pass', 'multiscale_no_history') else ('16/16/16' if variant == 'full_grid_innovation' else '4/8/16'),
            'no_history_auxiliary_reads_pruned_at_image_inference': variant == 'multiscale_no_history'})
        for snr in base['evaluation']['snrs_db']:
            subset = [row for row in method_rows if float(row['snr_db']) == snr]
            diagnostic = {'arm': name, 'snr_db': snr, 'transmissions': len(subset)}
            for key in ('bit_error_rate', 'feature_mse', 'feature_abs_mean', 'feature_saturation_fraction',
                        'feature_gate_1', 'feature_gate_2', 'residual_noise_ratio_1', 'residual_noise_ratio_2'):
                values = [float(row[key]) for row in subset if row[key] != '']
                if not all(np.isfinite(value) for value in values):
                    raise RuntimeError('nonfinite receiver diagnostic cannot be silently removed')
                diagnostic[key] = float(np.mean(values)) if values else ''
            diagnostic_rows.append(diagnostic)
        subset = [row for row in noiseless if row['arm'] == name]
        noiseless_summary.append({'arm': name, 'source_images': 100, 'role': 'noiseless_nominal19_not_wireless_ranking',
            **{metric: float(np.mean([float(row[metric]) for row in subset])) for metric in (*METRICS, 'bit_error_rate', 'feature_mse')}})
    write_csv(output / 'receiver_costs.csv', costs)
    write_csv(output / 'receiver_diagnostics_by_snr.csv', diagnostic_rows)
    write_csv(output / 'noiseless_summary.csv', noiseless_summary)
    native = {metric: float(np.mean([float(row[metric]) for row in clean])) for metric in ('psnr_db', 'ssim', 'lpips', 'dino')}
    write_json(output / 'native_reference_summary.json', {'source_images': 100, 'role': 'not_a_wireless_method', **native})
    parent_times = parent_milestone['timings']
    accounting = {'shared_parent_single_representation_seconds': parent_times['representation']['single_pass'],
        'shared_parent_single_image_training_seconds': parent_times['image']['single_pass'],
        'shared_parent_single_calibration_seconds': parent_times['calibration']['single_pass'],
        'parent_cost_scope': 'selected_common_lineage_not_all_historical_research',
        'receiver_trial_observed_wall_GPU_hours': milestone['observed_wall_GPU_hours'],
        'receiver_training_plus_calibration_measured_seconds': sum(milestone['timings']['training'].values()) + sum(milestone['timings']['calibration'].values()),
        'shared_TX_seconds_counted_once': milestone['timings']['shared_TX'],
        'evaluation_observed_wall_GPU_hours': receipt['evaluation_observed_wall_hours_all_sessions'],
        'evaluation_time_lower_bound': receipt['evaluation_total_time_is_lower_bound'],
        'calibration_is_not_independent_evidence': True, 'new_holdout_accessed': False,
        'training_seed_count': 1, 'additional_training_repeats_completed': 0}
    write_json(output / 'cost_accounting.json', accounting)
    text = ['# 固定发送器接收研究：development结果', '',
        f'六臂各有{arguments.step}次新增接收更新机会，完整校准选模；100源图×七SNR×三噪声。主区间1/4/7 dB。',
        '六臂使用完全相同s/y；三项旧geometry和四项强系统是系统参考，不冒充同波形因果对照。', '',
        '| 方法 | PSNR ↑ | LPIPS ↓ | DINO ↑ | 严重失真率 ↓ |', '|---|---:|---:|---:|---:|']
    text.extend(f'| {row["arm"]} | {row["psnr_db"]:.5f} | {row["lpips"]:.6f} | {row["dino"]:.6f} | {row["severe_distortion"]:.4f} |' for row in primary)
    text.extend(['', '## 主区间创新量配对差', '', '差值为方法减对照；每图先平均多个噪声与指定SNR，再以源图bootstrap。'])
    text.extend(f'- 对{row["control"]}：LPIPS Δ={row["delta"]:+.6f}，95% CI [{row["ci_low"]:+.6f}, {row["ci_high"]:+.6f}]。'
        for row in paired if row['snrs_db'] == label and row['method'] == receiver_name('multiscale_innovation') and row['metric'] == 'lpips')
    text.extend(['', '## 边界', '',
        '- 接收时延包含内部发送器重编码与最终视觉Decoder；full-grid不等FLOPs。no-history裁去不影响图像的辅助尺度读取，逐帧核对输入完全不变。旧参考时延不是本轮测量。',
        '- 正确native/noiseless均仅为诊断，不进无线排名；严重失真阈值为相对本图native的LPIPS增加至少0.15。',
        '- 未执行新的ISEC复现、训练种子重复、鲁棒性或新holdout；本表不能直接宣布方法独立验证或原创性。',
        '- 全部失败与非理想输出保留。完整PSNR/SSIM/LPIPS/DINO及两项派生指标在summary/paired CSV。'])
    (output / 'summary.md').write_text('\n'.join(text) + '\n')
    write_json(output / 'completion.json', {'status': 'INNOVATION_DEVELOPMENT_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'evaluation_receipt_sha256': sha256(evaluation / 'completion.json'),
        'milestone_sha256': sha256(milestone_path), 'calibration_review_sha256': sha256(review_path),
        'main_rows': len(rows), 'main_pairwise_relationships': len(comparison_pairs(config)), 'main_paired_intervals': len(paired),
        'fixed_support_rows': len(support), 'fixed_support_paired_intervals': len(support_paired),
        'selected_steps': {name: choice['step'] for name, choice in milestone['selected'].items()},
        'source_hashes': sources, 'new_holdout_accessed': False, 'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print(json.dumps(primary, indent=2))


if __name__ == '__main__':
    main()
