"""Source-image paired analysis and training/selection audit of WeTok A0/A1."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np

from wetok_comm.common import PROJECT, artifact_hashes, now, output_path, settings, sha256, snapshot, verify_sources, write_json
from wetok_comm.training import paired_batches

METRICS = ('psnr_db', 'ssim', 'lpips', 'dino', 'LPIPS_excess_from_native', 'severe_distortion')


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def statistics(rows, config):
    arms = config['arms'] + ['wetok_8PSK_FEC', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc']
    snrs, seeds = config['evaluation']['snrs_db'], config['evaluation']['noise_seeds']
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    expected = {(index, snr, seed, arm) for index in range(100) for snr in snrs for seed in seeds for arm in arms}
    if set(lookup) != expected or len(lookup) != len(rows):
        raise RuntimeError('wireless evaluation grid incomplete or duplicated')
    for index in range(100):
        for snr in snrs:
            for seed in seeds:
                paired = [lookup[index, snr, seed, arm] for arm in arms]
                if len({row['image_id'] for row in paired}) != 1 or len({row['noise_sha256'] for row in paired}) != 1:
                    raise RuntimeError('source/noise pairing mismatch')
                if any(int(row['total_complex_uses']) != 3060 or abs(float(row['total_energy']) - 6120) > 1e-5 for row in paired):
                    raise RuntimeError('wireless resource ledger changed')
    summaries, intervals = [], []
    resampled = np.random.default_rng(config['evaluation']['bootstrap_seed']).integers(100, size=(config['evaluation']['bootstrap_resamples'], 100))
    comparisons = [('multiscale_conditioned', 'multiscale_no_history'), ('multiscale_conditioned', 'single_pass'),
                   ('multiscale_no_history', 'single_pass')]
    comparisons.extend((method, control) for method in ('single_pass', 'multiscale_conditioned')
                       for control in ('wetok_8PSK_FEC', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc'))
    for group in [config['evaluation']['primary_snrs_db'], *[[snr] for snr in snrs]]:
        label = '+'.join(map(str, group))
        per_source = {arm: {metric: np.array([np.mean([float(lookup[index, snr, seed, arm][metric]) for snr in group for seed in seeds])
                                                    for index in range(100)]) for metric in METRICS} for arm in arms}
        for arm in arms:
            selected = [lookup[index, snr, seed, arm] for index in range(100) for snr in group for seed in seeds]
            row = {'snr_db': label, 'arm': arm, 'source_images': 100, 'transmissions': len(selected),
                   **{metric: float(values.mean()) for metric, values in per_source[arm].items()},
                   'source_averaged_LPIPS_p90': float(np.percentile(per_source[arm]['lpips'], 90))}
            for key in ('online_TX_seconds', 'receiver_seconds'):
                values = [float(item[key]) for item in selected if item[key] != '']
                row[key] = float(np.mean(values)) if values else ''
            summaries.append(row)
        for method, control in comparisons:
            for metric in METRICS:
                difference = per_source[method][metric] - per_source[control][metric]
                low, high = np.percentile(difference[resampled].mean(1), [2.5, 97.5])
                intervals.append({'snr_db': label, 'method': method, 'control': control, 'metric': metric,
                                  'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summaries, intervals


def training_audit(training, milestone, config):
    until = milestone['updates_per_arm']
    histories = {arm: read_csv(training / arm / 'training.csv')[:until] for arm in config['arms']}
    if any(len(history) != until for history in histories.values()):
        raise RuntimeError('unequal training opportunities at the evaluated milestone')
    for batch in paired_batches(config, 20000, 0, until):
        expected_phase = 'representation' if batch['step'] < config['training']['representation_updates'] else 'joint'
        rates = set()
        for arm in config['arms']:
            row = histories[arm][batch['step']]
            if int(row['step']) != batch['step'] + 1 or int(row['optimizer_step']) != batch['step'] + 1 or row['batch_sha256'] != batch['fingerprint']:
                raise RuntimeError('source/noise/optimizer training pairing changed')
            if row['phase'] != expected_phase or float(row['power_max_error']) > 1e-5:
                raise RuntimeError('training phase or physical power changed')
            if expected_phase == 'representation' and (row['mse'] or row['lpips']):
                raise RuntimeError('representation stage used image losses')
            rates.add(float(row['learning_rate']))
        if len(rates) != 1:
            raise RuntimeError('comparison arms received different LR schedules')
    calibration_ids = json.loads((output_path(config, 'cache') / 'calibration_ids.json').read_text())
    expected_grid = {(identifier, float(snr)) for identifier in calibration_ids for snr in config['channel']['snrs_db']}
    full_steps = [config['training']['representation_updates']] + list(range(config['training']['milestone_stride'], until + 1, config['training']['milestone_stride']))
    checked_calibration_rows = 0
    for arm in config['arms']:
        selected = milestone['selected'][arm]
        if selected is None or selected['scope'] != 'full' or selected['source_images'] != 1000:
            raise RuntimeError('warmup/monitor fallback used as a comparison model')
        if sha256(training / selected['checkpoint']) != selected['checkpoint_sha256']:
            raise RuntimeError('selected checkpoint changed')
        candidates = []
        for step in full_steps:
            values = read_csv(training / arm / f'full_{step:07d}.csv')
            actual = {(row['image_id'], float(row['snr_db'])) for row in values}
            if len(values) != len(expected_grid) or actual != expected_grid:
                raise RuntimeError('full-calibration population or SNR grid changed')
            if any(row['noiseless'] not in ('False', 'false', '0') for row in values):
                raise RuntimeError('noiseless diagnostic was used for normal calibration selection')
            score = float(np.mean([float(row['lpips']) for row in values]))
            if not np.isfinite(score):
                raise RuntimeError('nonfinite full-calibration objective')
            candidates.append((score, step))
            checked_calibration_rows += len(values)
        score, step = min(candidates)
        if selected['step'] != step or abs(selected['lpips'] - score) > 1e-12:
            raise RuntimeError('selected model does not minimize full-calibration LPIPS with earliest tie-breaking')
    return {'matched_updates_per_arm': until, 'all_training_batches_replayed': True,
            'learning_rates_matched': True, 'no_fallback_comparators': True,
            'full_calibration_rows_checked': checked_calibration_rows, 'selection_recomputed': True}


def plot_curves(output, summary, calibration, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(18, 4))
    for axis, metric in zip(axes, ('psnr_db', 'ssim', 'lpips', 'dino')):
        for arm in config['arms'] + ['wetok_8PSK_FEC', 'digital_adaptive', 'perceptual_deepjscc']:
            rows = [row for row in summary if row['arm'] == arm and '+' not in str(row['snr_db'])]
            axis.plot([float(row['snr_db']) for row in rows], [row[metric] for row in rows], marker='o', label=arm)
        axis.set(xlabel='SNR (dB)', ylabel=metric)
        axis.grid(alpha=.25)
    axes[-1].legend(fontsize=6)
    figure.suptitle('Development; equal 3060 complex uses and total energy 6120')
    figure.tight_layout()
    figure.savefig(output / 'quality_vs_snr.png', dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric in zip(axes, ('lpips', 'psnr_db', 'bit_error_rate')):
        for arm in config['arms']:
            rows = [row for row in calibration if row['arm'] == arm and row['scope'] == 'full']
            steps = sorted({int(row['step']) for row in rows})
            values = [np.mean([float(row[metric]) for row in rows if int(row['step']) == step]) for step in steps]
            axis.plot(steps, values, marker='o', label=arm)
        axis.set(xlabel='Retained optimizer updates', ylabel=metric)
        axis.grid(alpha=.25)
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / 'calibration.png', dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT / 'configs/study.yaml')
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--evaluation-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config = settings(arguments.config)
    if not arguments.execute:
        print('PLAN ONLY: audit and source-image paired statistics')
        return
    training = output_path(config, 'training')
    milestone_path = training / 'milestones' / f'step_{arguments.milestone:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    verify_sources(milestone['source_hashes'])
    evaluation = (arguments.evaluation_dir or output_path(config, 'evaluation') / f'step_{arguments.milestone:07d}').resolve()
    receipt = json.loads((evaluation / 'completion.json').read_text())
    if receipt['status'] != 'EVALUATION_COMPLETE' or receipt['milestone_sha256'] != sha256(milestone_path):
        raise RuntimeError('evaluated models differ from the registered milestone')
    verify_sources(receipt['source_hashes'])
    for relative, expected in receipt['output_hashes'].items():
        if sha256(evaluation / relative) != expected:
            raise RuntimeError('evaluation artifact checksum mismatch')
    audit = training_audit(training, milestone, config)
    summaries, paired = statistics(read_csv(evaluation / 'per_frame.csv'), config)
    output = (arguments.output_dir or output_path(config, 'analysis') / f'step_{arguments.milestone:07d}').resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('analysis must stay in VAR_COMM outputs')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), arguments.config, *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py'))])
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'paired.csv', paired)
    label = '+'.join(map(str, config['evaluation']['primary_snrs_db']))
    primary = [row for row in summaries if row['snr_db'] == label]
    write_csv(output / 'primary.csv', primary)
    calibration = [row for row in read_csv(training / 'calibration_summary.csv') if int(row['step']) <= arguments.milestone]
    plot_curves(output, summaries, calibration, config)
    lines = [f'# WeTok A0/A1：{arguments.milestone}步研究里程碑', '',
             '原100张development，三噪声、七SNR；主1/4/7 dB。固定3060 complex uses、总能量6120；不是新正式test，也不是研发终点。',
             '新方法为原生±1 WeTok接收接口，全部阶段读取完整y，不称逐包渐进。每臂参数/更新机会一致，no-history与conditioned具有相同阶段数。', '',
             '## 主区间结果', '| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | DINO↑ | 严重失真概率↓ |', '|---|---:|---:|---:|---:|---:|']
    for row in primary:
        lines.append(f'| {row["arm"]} | {row["psnr_db"]:.5f} | {row["ssim"]:.5f} | {row["lpips"]:.6f} | {row["dino"]:.6f} | {row["severe_distortion"]:.4f} |')
    lines.extend(['', '严重失真预定义为LPIPS相对正确native重建增加≥0.15；此真值只用于评测，未用于RX选择。', '',
                  '## 主要机制差', '| 方法 | 对照 | LPIPS差 | 95% CI |', '|---|---|---:|---|'])
    for row in paired:
        if row['snr_db'] == label and row['metric'] == 'lpips':
            lines.append(f'| {row["method"]} | {row["control"]} | {row["delta"]:+.6f} | [{row["ci_low"]:+.6f}, {row["ci_high"]:+.6f}] |')
    lines.extend(['', '## 归因与资格',
        '必须先看conditioned对no-history，不能只用多阶段对single-pass证明历史条件有用。只胜旧VAR不归为通信机制；先看同WeTok对照。',
        '选模为完整1000图校准平均LPIPS最小；各SNR PSNR仍须报告。相对A0下降超过0.2dB时标为取舍，不称无代价提高。未拿不合格warmup fallback作强对照。',
        'native_reference.csv是正确Fq参考，不是无线链，也不是数学质量上界。noiseless_mapping.csv使用零噪声/名义19dB，独立于AWGN排名，用于分离映射损失。',
        'WeTok数字参考是固定8PSK+soft-bit Viterbi，不宣称最优数字系统。CRC失败候选依然用于图像；原数字/Deep保留实际开销，Deep5/6已知条件异常不当作新方法优势。', '',
        '## 选模与下一里程碑'])
    for arm, choice in milestone['selected'].items():
        lines.append(f'- {arm}: 完整校准选择step {choice["step"]}，LPIPS {choice["lpips"]:.6f}。')
    lines.extend([f'- 当前共同image LR：{milestone["joint_rate"]}；无明显校准增益里程碑数：{milestone["no_gain_milestones"]}。',
                  '- 根据calibration而非development选checkpoint。结合曲线决定延长、配方修订或封存；不能因达到本里程碑自动停止研究。', '',
                  '## 成本与审计', f'训练进程观测GPU小时：{milestone["observed_training_gpu_hours"]:.5f}；本次评测：{receipt["GPU_hours_this_session"]:.5f}，不等于租赁账单。',
                  '在线TX含真实WeTok Encoder，RX含所有串行通信读取与Frozen Decoder；历史对照时延不混排。',
                  f'审计：{json.dumps(audit, ensure_ascii=False)}。全部14700无线行保留；按源图平均噪声/SNR后bootstrap，训练随机性仍需额外重复种子验证。'])
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'audit.json', {'status': 'MILESTONE_AUDIT_PASS', **audit, 'wireless_rows': 14700, 'paired_rows': len(paired)})
    verify_sources(sources)
    write_json(output / 'completion.json', {'status': 'MILESTONE_ANALYSIS_COMPLETE', 'completed_local': now(), 'milestone': arguments.milestone,
        'source_hashes': sources, 'evaluation_receipt_sha256': sha256(evaluation / 'completion.json'),
        'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print('MILESTONE_ANALYSIS_COMPLETE_REVIEW_AND_CONTINUE_RESEARCH', flush=True)


if __name__ == '__main__':
    main()
