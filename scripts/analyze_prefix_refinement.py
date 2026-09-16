#!/usr/bin/env python3
"""Audit matched training/selection, summarize paired development results, and plot real calibration points."""

import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import numpy as np
import yaml

from var_comm.study import artifact_hashes, create_output, sha256, snapshot, verify_artifacts, verify_snapshot, write_csv, write_json

METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine')


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def audit_training(training, receipt, config):
    histories = {branch: read_csv(training / branch / 'training.csv') for branch in config['branches']}
    for branch, history in histories.items():
        if [int(row['additional_step']) for row in history] != list(range(1, 10001)):
            raise RuntimeError('training update budget mismatch')
        for row in history:
            step = int(row['additional_step'])
            expected_phase = 'continuation' if branch == 'continuation' else ('prefix' if step <= 2000 else 'joint')
            if row['phase'] != expected_phase or int(row['optimizer_step']) != 10000 + step or int(row['additional_images_seen']) != step * 4:
                raise RuntimeError('stage schedule, optimizer step, or exposure count changed')
            if float(row['teacher_probability']) != 0 or float(row['learning_rate']) != 3e-5:
                raise RuntimeError('teacher context or learning-rate schedule changed')
            if not np.isfinite(float(row['gradient_norm'])) or float(row['power_max_error']) > 1e-5:
                raise RuntimeError('invalid gradient or data power')
            weights = config['training'][expected_phase + '_weights']
            reconstructed = sum(weight * float(row[metric]) for metric, weight in weights.items() if weight)
            np.testing.assert_allclose(float(row['loss']), reconstructed, rtol=1e-6, atol=1e-6)
            if expected_phase == 'prefix' and (row['mse'] or row['lpips'] or int(row['image_loss_active'])):
                raise RuntimeError('prefix-only training reported an active RGB loss')
    if any(first['batch_sha256'] != second['batch_sha256'] for first, second in zip(histories['continuation'], histories['two_stage'])):
        raise RuntimeError('branches did not receive identical minibatches and noise')
    initial = json.loads((training / 'initialization.json').read_text())
    for key in ('model_state_sha256', 'optimizer_state_sha256'):
        if len(set(initial[key].values())) != 1:
            raise RuntimeError('branches started from different states')
    if set(initial['trainable_parameters'].values()) != {1508750}:
        raise RuntimeError('communication parameter count changed')
    candidates = []
    calibration_keys = None
    base_psnr = None
    for branch in config['branches']:
        eligible = []
        for step in range(0, 10001, 1000):
            rows = read_csv(training / branch / f'calibration_step_{step:05d}.csv')
            keys = {(row['image_id'], float(row['snr_db'])) for row in rows}
            if len(rows) != 5000 or len(keys) != 5000 or len({row['image_id'] for row in rows}) != 1000:
                raise RuntimeError('calibration population incomplete')
            if calibration_keys is None:
                calibration_keys = keys
            if keys != calibration_keys:
                raise RuntimeError('calibration population changed over training')
            for row in rows:
                np.testing.assert_allclose(float(row['psnr_db']), -10 * np.log10(max(float(row['mse']), 1e-12)), rtol=0, atol=1e-10)
            psnr = {str(snr): float(np.mean([float(row['psnr_db']) for row in rows if float(row['snr_db']) == snr])) for snr in config['selection']['snrs_db']}
            if base_psnr is None:
                base_psnr = psnr
            if step == 0:
                for snr in base_psnr:
                    np.testing.assert_allclose(psnr[snr], base_psnr[snr], rtol=0, atol=1e-12)
            mean_lpips = float(np.mean([float(row['lpips']) for row in rows]))
            passes = all(psnr[snr] >= base_psnr[snr] - 0.2 for snr in psnr)
            candidate = {'branch': branch, 'additional_step': step, 'mean_lpips': mean_lpips, 'psnr_guard_pass': passes,
                         'maximum_SNR_PSNR_drop_db': max(base_psnr[snr] - psnr[snr] for snr in psnr),
                         **{f'psnr_{snr}dB': value for snr, value in psnr.items()}}
            candidates.append(candidate)
            if passes:
                eligible.append(candidate)
        chosen = min(eligible, key=lambda item: (item['mean_lpips'], item['additional_step']))
        selected = receipt['selected'][branch]
        if chosen['additional_step'] != selected['additional_step']:
            raise RuntimeError('selected checkpoint violates the registered LPIPS/PSNR rule')
        np.testing.assert_allclose(chosen['mean_lpips'], selected['mean_lpips'], rtol=0, atol=1e-12)
        if sha256(training / selected['checkpoint']) != selected['checkpoint_sha256']:
            raise RuntimeError('selected checkpoint bytes changed')
    monitor = read_csv(training / 'gradient_monitor.csv')
    expected = {(branch, step, group, metric) for branch in config['branches'] for step in (0, 2000)
                for group in ('encoder', 'reader') for metric in ('mse', 'lpips', 'token_ce', 'state')}
    observed = {(row['branch'], int(row['completed_additional_step']), row['parameter_group'], row['component']) for row in monitor}
    if len(monitor) != len(expected) or observed != expected:
        raise RuntimeError('gradient monitoring schedule incomplete')
    for row in monitor:
        np.testing.assert_allclose(float(row['weighted_norm']), float(row['weight']) * float(row['unweighted_norm']), rtol=1e-12, atol=1e-12)
    return candidates, {'matched_training_updates_per_branch': 10000, 'calibration_rows_checked': 110000,
                        'gradient_monitor_rows_checked': len(monitor), 'source_image_selection_rule': 'PASS'}


def audit_evaluation(evaluation, config):
    rows = read_csv(evaluation / 'per_frame.csv')
    spec = importlib.util.spec_from_file_location('refinement_evaluation', ROOT / 'scripts/evaluate_prefix_refinement.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary, paired = module.statistics(rows, config)
    saved_paired = read_csv(evaluation / 'paired_quality.csv')
    if len(saved_paired) != len(paired):
        raise RuntimeError('paired comparison count changed')
    maximum_error = 0.0
    for actual, expected in zip(paired, saved_paired):
        for key in ('snr_db', 'method', 'control', 'metric'):
            if actual[key] != expected[key]:
                raise RuntimeError('paired comparison indexing changed')
        for key in ('delta', 'ci_low', 'ci_high'):
            maximum_error = max(maximum_error, abs(actual[key] - float(expected[key])))
    if maximum_error > 1e-12:
        raise RuntimeError('paired statistics do not reproduce')
    old_rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row
                for row in read_csv(ROOT / config['evaluation']['old_evaluation'] / 'per_frame.csv')}
    controls = {'starting_next_scale': 'prefix_next_scale', 'digital_m8': 'digital_m8', 'digital_adaptive': 'digital_adaptive'}
    for row in rows:
        if int(row['total_complex_uses']) != 3060 or int(row['header_uses']) != 68 or int(row['data_uses']) != 2992:
            raise RuntimeError('communication budget changed')
        key = (int(row['image_index']), float(row['snr_db']), int(row['seed']))
        reference = old_rows[key + (controls.get(row['arm'], 'prefix_next_scale'),)]
        if reference['noise_sha256'] != row['noise_sha256']:
            raise RuntimeError('source-image channel noise pairing changed')
        if row['arm'] in controls:
            for metric in METRICS:
                if float(row[metric]) != float(reference[metric]):
                    raise RuntimeError('frozen strong-control quality changed')
    return rows, summary, paired, {'development_rows_checked': len(rows), 'paired_rows_recomputed': len(paired), 'maximum_paired_error': maximum_error}


def make_plots(output, calibration, candidates, summary, config, selections):
    colors = {'continuation': '#377eb8', 'two_stage': '#e41a1c'}
    figure, axes = pyplot.subplots(2, 3, figsize=(13, 7))
    names = [('lpips', 'Hard calibration LPIPS'), ('psnr_db', 'Hard calibration PSNR'), ('token_error_rate', 'Prefix token error rate'),
             ('token_ce', 'Prefix CE'), ('state_error', 'Normalized cumulative state error'), ('maximum_SNR_PSNR_drop_db', 'Worst per-SNR PSNR drop from step 0')]
    for axis, (metric, title) in zip(axes.flatten(), names):
        for branch, color in colors.items():
            steps = list(range(0, 10001, 1000))
            source = candidates if metric == 'maximum_SNR_PSNR_drop_db' else calibration
            values = [np.mean([float(row[metric]) for row in source if row['branch'] == branch and int(row['additional_step']) == step]) for step in steps]
            axis.plot(steps, values, 'o-', color=color, label=branch)
            if metric == 'lpips':
                selected_step = selections[branch]['additional_step']
                axis.plot(selected_step, values[steps.index(selected_step)], 'o', markersize=12, markerfacecolor='none', color=color)
        axis.axvline(2000, linestyle=':', color='#777777')
        if metric == 'maximum_SNR_PSNR_drop_db':
            axis.axhline(0.2, linestyle='--', color='black', label='PSNR guard')
        axis.set(title=title, xlabel='Additional optimizer updates')
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.suptitle('1000 fixed calibration images / five fixed SNRs / same noise / own history\n11 real checkpoints; stage boundary at 2000; DINO not used for selection')
    figure.tight_layout()
    figure.savefig(output / 'calibration.png', dpi=170)
    pyplot.close(figure)
    figure, axes = pyplot.subplots(1, 3, figsize=(13, 4))
    all_colors = {**colors, 'starting_next_scale': '#ff7f00', 'digital_m8': '#333333', 'digital_adaptive': '#4daf4a'}
    for axis, metric in zip(axes, METRICS):
        for arm, color in all_colors.items():
            selected = sorted([row for row in summary if row['arm'] == arm], key=lambda row: float(row['snr_db']))
            axis.plot([float(row['snr_db']) for row in selected], [float(row[metric]) for row in selected], 'o-', color=color, label=arm)
        axis.set(title=metric, xlabel='SNR (dB)', xticks=config['evaluation']['snrs_db'])
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    figure.suptitle('Development 100 images x 3 noises; 3060 complex uses; frozen selections and strong digital controls')
    figure.tight_layout()
    figure.savefig(output / 'development.png', dpi=170)
    pyplot.close(figure)
    figure, axes = pyplot.subplots(3, 8, figsize=(22, 8))
    for scale in range(1, 9):
        for row_index, metric in enumerate(('ce', 'ter', 'state')):
            axis = axes[row_index, scale - 1]
            for branch, color in colors.items():
                steps = list(range(0, 10001, 1000))
                values = [np.mean([float(row[f'{metric}_r{scale}']) for row in calibration if row['branch'] == branch and int(row['additional_step']) == step]) for step in steps]
                axis.plot(steps, values, color=color)
            axis.axvline(2000, linestyle=':', color='#777777')
            axis.set_title(f'r{scale} {metric}')
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.2)
    figure.suptitle('Per-scale diagnostics on the same full calibration population; no scale-CE reweighting')
    figure.tight_layout()
    figure.savefig(output / 'scale_diagnostics.png', dpi=150)
    pyplot.close(figure)


def main():
    config = yaml.safe_load((ROOT / 'configs/prefix_refinement.yaml').read_text())
    training, evaluation = ROOT / config['outputs']['training'], ROOT / config['outputs']['evaluation']
    receipt = verify_artifacts(training, 'completion.json')
    evaluation_receipt = verify_artifacts(evaluation, 'completion.json')
    verify_snapshot(receipt['source_hashes'])
    verify_snapshot(evaluation_receipt['source_hashes'])
    if evaluation_receipt['training_receipt_sha256'] != sha256(training / 'completion.json'):
        raise RuntimeError('evaluation is not bound to this completed training')
    candidates, training_checks = audit_training(training, receipt, config)
    rows, summary, paired, evaluation_checks = audit_evaluation(evaluation, config)
    output = create_output(ROOT / config['outputs']['analysis'])
    sources = snapshot(output, [Path(__file__), ROOT / 'configs/prefix_refinement.yaml'])
    write_csv(output / 'calibration_selection.csv', candidates)
    primary = []
    for arm in config['branches'] + config['evaluation']['controls']:
        selected = [row for row in rows if row['arm'] == arm and float(row['snr_db']) in config['evaluation']['primary_snr_db']]
        primary.append({'arm': arm, 'source_images': 100, 'transmissions': len(selected),
                        **{metric: float(np.mean([float(row[metric]) for row in selected])) for metric in METRICS}})
    write_csv(output / 'primary_quality.csv', primary)
    primary_paired = [row for row in paired if row['snr_db'] == '1.0+4.0+7.0']
    write_csv(output / 'primary_paired.csv', primary_paired)
    make_plots(output, read_csv(training / 'calibration_summary.csv'), candidates, summary, config, receipt['selected'])
    lines = ['# 两阶段prefix恢复与等预算续训：结果', '', f"训练实际结束时间：{receipt['local_completed']}。本报告对应2026-09-08注册协议，不以注册日期替代执行日期。", '',
             '两分支从相同next-scale模型及Adam状态出发，各新增10000更新/40000曝光；两阶段为2000 prefix + 8000 joint，始终自身历史。',
             '固定m8、3060 complex uses；DINO不参与训练或选择。新增训练配方是一个整体对照，不能单独归因给state损失或阶段化。', '',
             '## 校准选择', '', '| 分支 | 选中新增步数 | 阶段 | 校准LPIPS |', '|---|---:|---|---:|']
    for branch, selected in receipt['selected'].items():
        lines.append(f"| {branch} | {selected['additional_step']} | {selected['stage']} | {selected['mean_lpips']:.8f} |")
    lines.extend(['', '每1000步、完整1000图×五SNR校准；按平均LPIPS选择，且每SNR PSNR相对共同起点最多下降0.2 dB。包含step0候选。',
                  '完整阶段边界/终点结果见calibration_selection.csv与calibration_summary.csv；若选中prefix阶段或step0，不把它包装成末端联合微调的成绩。', '',
                  '## 主1/4/7 dB开发结果', '', '| 方法 | PSNR ↑ | LPIPS ↓ | DINO ↑ |', '|---|---:|---:|---:|'])
    for row in primary:
        lines.append(f"| {row['arm']} | {row['psnr_db']:.6f} | {row['lpips_alex']:.6f} | {row['dino_cosine']:.6f} |")
    lines.extend(['', '## 两阶段相对对照的配对差', '', '差为two_stage减control；先每图平均三噪声及主SNR，再按100张source-image bootstrap。', '',
                  '| 对照 | 指标 | 差值 | 95% CI |', '|---|---|---:|---|'])
    for row in primary_paired:
        if row['method'] == 'two_stage':
            lines.append(f"| {row['control']} | {row['metric']} | {row['delta']:+.8f} | [{row['ci_low']:+.8f}, {row['ci_high']:+.8f}] |")
    lines.extend(['', '## 判读', ''])
    for control in ['continuation', 'starting_next_scale', 'digital_m8', 'digital_adaptive']:
        result = next(row for row in primary_paired if row['method'] == 'two_stage' and row['control'] == control and row['metric'] == 'lpips_alex')
        interpretation = 'LPIPS改善，逐项95%配对区间小于零' if result['ci_high'] < 0 else ('LPIPS更差，逐项95%配对区间大于零' if result['ci_low'] > 0 else 'LPIPS配对区间跨零，未取得明确优势')
        lines.append(f'- two_stage对{control}：{interpretation}。')
    lines.extend(['', '只有胜过续训对照，不等于胜过强数字系统；三指标存在取舍时，不宣称整体胜出。本轮仍为原development，不是新正式测试。', '',
                  '## 计算与复核', '', f"实际计时：{json.dumps(receipt['timing'], ensure_ascii=False)}。阶段A跳过后缀/RGB，因此不声称等FLOPs。", '',
                  f"训练/选择检查：{json.dumps(training_checks, ensure_ascii=False)}。", f"开发配对复算：{json.dumps(evaluation_checks, ensure_ascii=False)}。", '',
                  '原数字与未续训对照逐帧质量未改写；新模型评测使用原hard接收实现，另有新state追踪前向重放一致性检查。',
                  '本次自动审计核对产物SHA、预算、采样配对、选模与统计；不冒充另一次全量模型推理或全量独立图像指标重算。', '',
                  '图：calibration.png、scale_diagnostics.png、development.png。逐SNR完整表与区间见本轮EVAL目录summary.csv、paired_quality.csv。'])
    report = '\n'.join(lines) + '\n'
    (output / 'report.md').write_text(report)
    report_path = ROOT / 'reports/prefix_refinement_result.md'
    if report_path.exists():
        raise FileExistsError('refusing to overwrite an existing refinement result report')
    report_path.write_text(report)
    write_json(output / 'audit.json', {'status': 'TRAINING_SELECTION_PAIRING_AUDIT_PASS', **training_checks, **evaluation_checks})
    verify_snapshot(sources)
    write_json(output / 'completion.json', {'status': 'PREFIX_REFINEMENT_ANALYSIS_COMPLETE', 'local_completed': datetime.now().astimezone().isoformat(),
               'source_hashes': sources, 'training_receipt_sha256': sha256(training / 'completion.json'),
               'evaluation_receipt_sha256': sha256(evaluation / 'completion.json'), 'report_sha256': sha256(report_path),
               'output_hashes': artifact_hashes(output)})
    progress = ROOT / 'PROGRESS.md'
    existing = progress.read_text()
    heading = '## 两阶段prefix恢复：训练、开发评测与统计复核完成'
    if heading not in existing:
        note = heading + '\n\n各分支新增10000更新已完成；以实际运行receipt日期为准。结果及强数字对照见`reports/prefix_refinement_result.md`。\n' + \
               '当前完成的是这次固定预算实验，不自动追加续训或改变VAR通信主线。\n\n'
        progress.write_text(existing.replace('# VAR 通信当前进度\n\n', '# VAR 通信当前进度\n\n' + note, 1))
    print(report, flush=True)


if __name__ == '__main__':
    main()
