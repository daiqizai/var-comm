"""Causal loss-weight comparison with parent-state and calibration audits."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import numpy as np
import torch

from wetok_comm.bit_support import arm_definitions, load_repair, optimizer_digest, repair_output
from wetok_comm.common import PROJECT, artifact_hashes, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.training import module_sha256, new_network, paired_batches

METRICS = ('psnr_db', 'ssim', 'lpips', 'dino', 'LPIPS_excess_from_native', 'severe_distortion')


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparisons(repair):
    result = []
    for variant in repair['variants']:
        result.extend([(f'bit_support__{variant}', f'joint_original__{variant}'),
                       (f'bit_support__{variant}', f'old_selected__{variant}'),
                       (f'joint_original__{variant}', f'old_selected__{variant}')])
    for recipe in repair['recipes']:
        result.extend([(f'{recipe}__multiscale_conditioned', f'{recipe}__multiscale_no_history'),
                       (f'{recipe}__multiscale_conditioned', f'{recipe}__single_pass'),
                       (f'{recipe}__multiscale_no_history', f'{recipe}__single_pass')])
    result.extend((f'bit_support__{variant}', control) for variant in ('single_pass', 'multiscale_conditioned')
                  for control in ('wetok_8PSK_FEC', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc'))
    return result


def statistics(rows, repair, base):
    names = list(arm_definitions(repair)) + ['wetok_8PSK_FEC', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc']
    names += ['old_selected__' + variant for variant in repair['variants']]
    snrs, seeds = base['evaluation']['snrs_db'], base['evaluation']['noise_seeds']
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    expected = {(index, snr, seed, arm) for index in range(100) for snr in snrs for seed in seeds for arm in names}
    if set(lookup) != expected or len(lookup) != len(rows):
        raise RuntimeError('recipe evaluation grid incomplete or duplicated')
    for index in range(100):
        for snr in snrs:
            for seed in seeds:
                selected = [lookup[index, snr, seed, name] for name in names]
                if len({row['image_id'] for row in selected}) != 1 or len({row['noise_sha256'] for row in selected}) != 1:
                    raise RuntimeError('recipe source/noise mismatch')
                if any(int(row['total_complex_uses']) != 3060 or abs(float(row['total_energy']) - 6120) > 1e-5 or
                       int(row['header_uses']) + int(row['data_uses']) != 3060 for row in selected):
                    raise RuntimeError('recipe comparison resource drift')
    resampled = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(100, size=(base['evaluation']['bootstrap_resamples'], 100))
    summary, paired = [], []
    for group in [base['evaluation']['primary_snrs_db'], *[[snr] for snr in snrs]]:
        label = '+'.join(map(str, group))
        values = {name: {metric: np.array([np.mean([float(lookup[index, snr, seed, name][metric]) for snr in group for seed in seeds])
                                         for index in range(100)]) for metric in METRICS} for name in names}
        for name in names:
            selected = [lookup[index, snr, seed, name] for index in range(100) for snr in group for seed in seeds]
            row = {'snr_db': label, 'arm': name, 'source_images': 100, 'transmissions': len(selected),
                   **{metric: float(metric_values.mean()) for metric, metric_values in values[name].items()}}
            for field in ('receiver_seconds', 'online_TX_seconds'):
                valid = [float(item[field]) for item in selected if item[field] != '']
                row[field] = float(np.mean(valid)) if valid else ''
            summary.append(row)
        for method, control in comparisons(repair):
            for metric in METRICS:
                difference = values[method][metric] - values[control][metric]
                low, high = np.percentile(difference[resampled].mean(1), [2.5, 97.5])
                paired.append({'snr_db': label, 'method': method, 'control': control, 'metric': metric,
                               'delta': float(difference.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summary, paired


def audit_training(repair, base, training, milestone, parent):
    definitions = arm_definitions(repair)
    additional = milestone['additional_updates_per_arm']
    initial = json.loads((training / 'initialization.json').read_text())
    original = torch.load(parent / repair['parent_optimizer_checkpoint'], map_location='cpu', weights_only=True)
    for variant in repair['variants']:
        model = new_network(base, variant, 'cpu')
        model.load_state_dict(original['models'][variant], strict=True)
        model_hash = module_sha256(model)
        opt_hash = optimizer_digest(original['optimizers'][variant])
        for recipe in repair['recipes']:
            name = f'{recipe}__{variant}'
            if initial['models'][name] != model_hash or initial['optimizers'][name] != opt_hash:
                raise RuntimeError('recipe pair did not restore the exact parent model/Adam values')
        del model
    del original
    histories = {name: read_csv(training / name / 'training.csv')[:additional] for name in definitions}
    if any(len(rows) != additional for rows in histories.values()):
        raise RuntimeError('recipe training update budget differs across arms')
    for local, batch in enumerate(paired_batches(base, 20000, repair['parent_step'], repair['parent_step'] + additional)):
        for name, definition in definitions.items():
            row = histories[name][local]
            if int(row['global_step']) != batch['step'] + 1 or int(row['optimizer_step']) != batch['step'] + 1 or row['batch_sha256'] != batch['fingerprint']:
                raise RuntimeError('recipe data/noise/Adam trajectory not matched')
            if float(row['learning_rate']) != repair['training']['learning_rate'] or float(row['bits_weight']) != definition['weights']['bits']:
                raise RuntimeError('recipe LR or registered loss weight changed')
            if not np.isfinite(float(row['loss'])) or float(row['power_max_error']) > 1e-5:
                raise RuntimeError('nonfinite loss or physical violation')
    ids = json.loads((parent.parent / base['outputs']['cache'] / 'calibration_ids.json').read_text())
    expected = {(identifier, float(snr)) for identifier in ids for snr in base['channel']['snrs_db']}
    checked = 0
    for name in definitions:
        candidates = []
        for step in range(0, additional + 1, 2500):
            rows = read_csv(training / name / f'full_{step:07d}.csv')
            if len(rows) != len(expected) or {(row['image_id'], float(row['snr_db'])) for row in rows} != expected:
                raise RuntimeError('recipe calibration grid changed')
            candidates.append((float(np.mean([float(row['lpips']) for row in rows])), step))
            checked += len(rows)
        score, step = min(candidates)
        selected = milestone['selected'][name]
        if selected['step'] != step or abs(selected['lpips'] - score) > 1e-12 or selected['scope'] != 'full':
            raise RuntimeError('recipe selection not based on complete calibration')
        if sha256(training / selected['checkpoint']) != selected['checkpoint_sha256']:
            raise RuntimeError('recipe selected checkpoint changed')
    return {'matched_additional_updates_per_arm': additional, 'parent_model_and_Adam_exact': True,
            'paired_batches_replayed': True, 'full_calibration_rows_checked': checked, 'selection_recomputed': True}


def plots(output, summary, calibration, repair, base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, variant in zip(axes, repair['variants']):
        for recipe in repair['recipes']:
            name = f'{recipe}__{variant}'
            rows = [row for row in calibration if row['arm'] == name and row['scope'] == 'full']
            steps = sorted({int(row['additional_step']) for row in rows})
            axis.plot(steps, [np.mean([float(row['lpips']) for row in rows if int(row['additional_step']) == step]) for step in steps], marker='o', label=recipe)
        axis.set(title=variant, xlabel='Additional matched updates', ylabel='Full-calibration LPIPS')
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / 'calibration_recipe_pairs.png', dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, metric in zip(axes, ('psnr_db', 'lpips', 'dino')):
        names = ['bit_support__multiscale_conditioned', 'joint_original__multiscale_conditioned',
                 'old_selected__multiscale_conditioned', 'digital_adaptive', 'perceptual_deepjscc']
        for name in names:
            rows = [row for row in summary if row['arm'] == name and '+' not in row['snr_db']]
            axis.plot([float(row['snr_db']) for row in rows], [row[metric] for row in rows], marker='o', label=name)
        axis.set(xlabel='SNR (dB)', ylabel=metric)
        axis.grid(alpha=.25)
    axes[-1].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output / 'quality_vs_snr.png', dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXPERIMENT / 'configs/bit_support.yaml')
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--evaluation-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    repair, base, parent, parent_receipt = load_repair(arguments.config)
    if not arguments.execute:
        print('PLAN ONLY: audit paired bit-weight continuation and compare to historical best')
        return
    training = repair_output(repair, 'training')
    milestone_path = training / 'milestones' / f'additional_{arguments.milestone:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    verify_sources(milestone['source_hashes'])
    evaluation = arguments.evaluation_dir or repair_output(repair, 'evaluation') / f'additional_{arguments.milestone:07d}'
    receipt = json.loads((evaluation / 'completion.json').read_text())
    if receipt['status'] != 'EVALUATION_COMPLETE' or receipt['rows'] != 27300 or receipt['milestone_sha256'] != sha256(milestone_path):
        raise RuntimeError('recipe evaluation belongs to a different milestone')
    verify_sources(receipt['source_hashes'])
    for relative, expected in receipt['output_hashes'].items():
        if sha256(evaluation / relative) != expected:
            raise RuntimeError('recipe evaluation artifact changed')
    audit = audit_training(repair, base, training, milestone, parent)
    summary, paired = statistics(read_csv(evaluation / 'per_frame.csv'), repair, base)
    output = arguments.output_dir or repair_output(repair, 'analysis') / f'additional_{arguments.milestone:07d}'
    if not output.resolve().is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('analysis output must remain in VAR_COMM')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), arguments.config, EXPERIMENT / repair['base_config'],
        EXPERIMENT / 'docs/bit_support_protocol.md', *sorted((EXPERIMENT / 'src/wetok_comm').glob('*.py'))])
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'paired.csv', paired)
    label = '+'.join(map(str, base['evaluation']['primary_snrs_db']))
    primary = [row for row in summary if row['snr_db'] == label]
    write_csv(output / 'primary.csv', primary)
    calibration = [row for row in read_csv(training / 'calibration_summary.csv') if int(row['additional_step']) <= arguments.milestone]
    plots(output, summary, calibration, repair, base)
    lines = ['# Native bit-preservation：配对训练目标修订', '',
        f'同5000模型和Adam状态，每臂新增{arguments.milestone}步；唯一变化是joint bit BCE权重0.01→1。其余损失、native接口、网络和物理N/E不变。',
        '本次为development研究里程碑，不代表全新holdout验证或整个研究完成。', '',
        '## 主1/4/7 dB', '| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | DINO↑ | 严重失真↓ |', '|---|---:|---:|---:|---:|---:|']
    for row in primary:
        lines.append(f'| {row["arm"]} | {row["psnr_db"]:.5f} | {row["ssim"]:.5f} | {row["lpips"]:.6f} | {row["dino"]:.6f} | {row["severe_distortion"]:.4f} |')
    lines.extend(['', '## 原配方/修订/历史最佳的主要LPIPS差', '| 方法 | 对照 | 差值 | 95% CI |', '|---|---|---:|---|'])
    for row in paired:
        if row['snr_db'] == label and row['metric'] == 'lpips':
            lines.append(f'| {row["method"]} | {row["control"]} | {row["delta"]:+.6f} | [{row["ci_low"]:+.6f}, {row["ci_high"]:+.6f}] |')
    lines.extend(['', '## 选模与边界'])
    for name, choice in milestone['selected'].items():
        lines.append(f'- {name}: additional {choice["step"]} / global {choice["global_step"]}, full calibration LPIPS {choice["lpips"]:.6f}.')
    lines.extend(['', 'old_selected为原实验按校准选中的2000表示恢复端点，不是5000退化模型。若只赢joint_original、不赢old_selected，只能称修复退化，不能称整体通信进步。',
        '先看结构内配方差，再看各配方内conditioned对no_history。无合格或异常模型不使用弱fallback制造收益。',
        '全部无线行保留，native参考与noiseless映射单列；参数/训练随机性与源图噪声统计分开，主结论仍需后续独立验证。',
        'N=3060/E=6120，WeTok无类别header；老数字及Deep实际开销分列，旧时延不混排。Deep5/6既有条件异常不当新方法优势。', '',
        '## 成本与复核', f'训练进程GPU小时：{milestone["observed_GPU_hours"]:.5f}；评测：{receipt["GPU_hours_this_session"]:.5f}。',
        f'复核：{json.dumps(audit, ensure_ascii=False)}；源图级bootstrap，不把重复噪声当独立图像。'])
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'audit.json', {'status': 'BIT_SUPPORT_AUDIT_PASS', **audit, 'wireless_rows': 27300, 'paired_rows': len(paired)})
    verify_sources(sources)
    write_json(output / 'completion.json', {'status': 'BIT_SUPPORT_ANALYSIS_COMPLETE', 'completed_local': now(),
        'source_hashes': sources, 'evaluation_receipt_sha256': sha256(evaluation / 'completion.json'),
        'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print('BIT_SUPPORT_ANALYSIS_COMPLETE_REVIEW_AND_CONTINUE', flush=True)


if __name__ == '__main__':
    main()
