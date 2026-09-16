"""Plot only completed, audited R2 quality results without reselection or new inference."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
GRID = EXPERIMENT.parent / 'wetok-joint-grid-controls-r1'
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(GRID / 'src'), str(JOINT / 'src'), str(JOINT / 'scripts'),
    str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from finish_registered_trial import validate_receipt
from joint_sender.evaluation_io import write_rows
from sufficiency.evaluation import all_names, audited_endpoint, comparison_pairs, evaluation_output, load_evaluation, new_names, previous_name
from sufficiency.references import read_rows
from wetok_comm.common import artifact_hashes, now, sha256, snapshot, write_json


SYSTEM_CONTROLS = ('digital_m8', 'digital_adaptive', 'wetok_8PSK_FEC', 'perceptual_deepjscc')
LABELS = {'r2__single_pass': 'R2 single', 'r2__multiscale_state_history': 'R2 multiscale state',
    'r2__full_grid_state_history': 'R2 full-grid state', 'r2__full_grid_innovation': 'R2 full-grid innovation',
    'digital_m8': 'VAR digital m8', 'digital_adaptive': 'VAR digital adaptive',
    'wetok_8PSK_FEC': 'WeTok digital 8PSK+FEC', 'perceptual_deepjscc': 'Perceptual DeepJSCC'}
COLORS = dict(zip(LABELS, ('#4C78A8', '#F58518', '#54A24B', '#B279A2', '#72B7B2', '#111111', '#E45756', '#9D755D')))


def display_tables(summary, paired, config, original, grid, reference, base):
    names = all_names(config, original, grid, reference)
    scopes = ['+'.join(map(str, base['evaluation']['primary_snrs_db'])), *[str(float(snr)) for snr in base['evaluation']['snrs_db']]]
    lookup = {(row['arm'], row['snrs_db']): row for row in summary}
    expected = {(name, scope) for name in names for scope in scopes}
    if len(summary) != len(expected) or set(lookup) != expected or any(int(row['source_images']) != 100 for row in summary):
        raise RuntimeError('the complete quality summary lost a source or method/SNR condition')
    methods = new_names(config) + list(SYSTEM_CONTROLS)
    supported = [lookup[name, str(float(snr))] for snr in base['channel']['snrs_db'] for name in methods]
    primary = [lookup[name, scopes[0]] for name in methods]
    contrast = {(row['method'], row['control'], row['snrs_db'], row['metric']): row for row in paired}
    if len(contrast) != len(paired):
        raise RuntimeError('quality contrast table contains duplicate intervals')

    def require_pairs(pairs, scope):
        result = []
        for method, control in pairs:
            key = method, control, scopes[0], 'lpips'
            if key not in contrast:
                raise RuntimeError('a mandatory primary quality comparison was omitted')
            row = contrast[key]
            if (row['comparison_scope'] != scope or not all(np.isfinite(float(row[key])) for key in ('delta', 'ci_low', 'ci_high')) or
                float(row['ci_low']) > float(row['ci_high'])):
                raise RuntimeError('a primary contrast has invalid provenance or uncertainty')
            result.append(row)
        return result

    matched = require_pairs([pair for pair in comparison_pairs(config, original, grid, reference) if pair[1].startswith('r2__')],
        'matched10000_structure_comparison')
    training = require_pairs([(name, previous_name(name.split('__', 1)[1])) for name in new_names(config)],
        'extra_training_effect_not_new_mechanism')
    systems = require_pairs([(name, control) for control in SYSTEM_CONTROLS for name in new_names(config)],
        'historical_or_system_reference_not_equal10000_training')
    return primary, supported, matched, training, systems


def selected_table(rows, config):
    result = []
    for name in new_names(config):
        subset = [row for row in rows if row['arm'] == name]
        choices = {(int(row['selected_step']), row['checkpoint_sha256'], int(row['available_updates'])) for row in subset}
        if len(choices) != 1 or next(iter(choices))[2] != 10000:
            raise RuntimeError('quality figure changed checkpoint selection or training opportunity')
        step, checkpoint, available = next(iter(choices))
        result.append({'arm': name, 'available_total_updates': available, 'selected_step': step,
            'selected_checkpoint_from_previous_trial': step <= 5000, 'checkpoint_sha256': checkpoint})
    return result


def save_figure(figure, output, name):
    figure.savefig(output / (name + '.png'), dpi=170, bbox_inches='tight')
    figure.savefig(output / (name + '.pdf'), bbox_inches='tight')
    plt.close(figure)


def forest(axis, rows, labels):
    for position, row in enumerate(rows):
        axis.hlines(position, float(row['ci_low']), float(row['ci_high']), color=COLORS[row['method']], linewidth=2)
        axis.scatter(float(row['delta']), position, color=COLORS[row['method']], s=35, zorder=3)
    axis.axvline(0, color='black', linestyle='--', linewidth=1)
    axis.set_yticks(range(len(rows)), labels)
    axis.invert_yaxis()
    axis.set_xlabel('Delta LPIPS (method - control); left is better')


def draw_figures(output, tables, config, base):
    primary, supported, matched, training, systems = tables
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    methods = new_names(config) + list(SYSTEM_CONTROLS)
    markers = ('o', 's', '^', 'D', 'v', 'x', '*', 'P')
    for axis, metric, title in zip(axes.flat, ('lpips', 'psnr_db', 'dino', 'severe_distortion'),
        ('LPIPS (lower)', 'PSNR dB (higher)', 'DINO similarity (higher)', 'Severe distortion fraction (lower)')):
        for name, marker in zip(methods, markers):
            rows = [row for row in supported if row['arm'] == name]
            axis.plot([float(row['snrs_db']) for row in rows], [float(row[metric]) for row in rows],
                color=COLORS[name], marker=marker, linewidth=1.7, label=LABELS[name])
        axis.set_xticks(base['channel']['snrs_db'])
        axis.set_xlabel('Actual SNR dB')
        axis.set_title(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
    figure.suptitle('Development: 100 sources x 3 noises; N=3060 complex uses, E=6120\nFive supported SNR points; all seven SNRs and failures remain in the archived data')
    figure.tight_layout(rect=(0, .09, 1, .92))
    save_figure(figure, output, 'supported_snr_quality')
    figure, axis = plt.subplots(figsize=(11, 5))
    forest(axis, matched, [LABELS[row['method']] + ' - ' + LABELS[row['control']] for row in matched])
    axis.set_title('Matched 10000-opportunity structures: primary 1/4/7 dB\nSource-image paired 95% intervals; not training-seed uncertainty')
    figure.tight_layout()
    save_figure(figure, output, 'matched_structure_primary_lpips')
    figure, axis = plt.subplots(figsize=(10, 4))
    forest(axis, training, [LABELS[row['method']] + ' - own5000' for row in training])
    axis.set_title('Extra training effect only: primary 1/4/7 dB\nNot a new communication mechanism')
    figure.tight_layout()
    save_figure(figure, output, 'extra_training_primary_lpips')
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    for axis, control in zip(axes.flat, SYSTEM_CONTROLS):
        rows = [row for row in systems if row['control'] == control]
        forest(axis, rows, [LABELS[row['method']] for row in rows])
        axis.set_title('Control: ' + LABELS[control])
    figure.suptitle('System references: primary 1/4/7 dB; same physical resources\nDifferent backbones/training histories: not matched-architecture causal comparisons')
    figure.tight_layout(rect=(0, 0, 1, .90))
    save_figure(figure, output, 'system_controls_primary_lpips')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, config, original, grid, reference, base, parent = load_evaluation()
    if not arguments.execute:
        print('PLAN ONLY: completed10000 quality+CPU audit -> labeled figures/tables; no new inference, tests or selection')
        return
    milestone, milestone_path, review_path, sources = audited_endpoint(config, 10000)
    quality = evaluation_output(evaluation, 'quality')
    analysis = evaluation_output(evaluation, 'quality_analysis')
    quality_receipt, analysis_receipt = quality / 'completion.json', analysis / 'completion.json'
    validate_receipt(quality_receipt, 'R2_QUALITY_COMPLETE', {'r2_milestone_sha256': sha256(milestone_path), 'r2_review_sha256': sha256(review_path)})
    validate_receipt(analysis_receipt, 'R2_QUALITY_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE',
        {'r2_milestone_sha256': sha256(milestone_path), 'quality_receipt_sha256': sha256(quality_receipt)})
    summary, paired = read_rows(analysis / 'summary.csv'), read_rows(analysis / 'paired.csv')
    tables = display_tables(summary, paired, config, original, grid, reference, base)
    selected = selected_table(read_rows(quality / 'per_frame.csv'), config)
    output = analysis.parent / 'quality_figures_0010000'
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output, [Path(__file__)])
    for name, rows in zip(('display_primary.csv', 'supported_snr.csv', 'matched_primary_lpips.csv', 'extra_training_primary_lpips.csv', 'system_primary_lpips.csv'), tables):
        write_rows(output / name, rows)
    write_rows(output / 'all_method_summary.csv', summary)
    write_rows(output / 'selected_checkpoints.csv', selected)
    draw_figures(output, tables, config, base)
    lines = ['# R2完整development质量展示', '',
        '100源图×三个固定噪声；主区间1/4/7 dB。统计、图像/波形与选择均来自已完成审计，不新增推理或选模。', '',
        '所有22方法/七SNR保留在all_method_summary.csv。主要曲线画1/4/7/13/19五个支持点；5/6 dB强Deep支持点补充仍在原分析目录，不利用旧离支持点异常宣称优势。', '',
        '| 方法 | PSNR ↑ | LPIPS ↓ | DINO ↑ | 严重失真率 ↓ |', '|---|---:|---:|---:|---:|']
    lines.extend(f'| {LABELS[row["arm"]]} | {float(row["psnr_db"]):.5f} | {float(row["lpips"]):.6f} | {float(row["dino"]):.6f} | {float(row["severe_distortion"]):.4f} |' for row in tables[0])
    lines.extend(['', 'matched_structure是同10000机会结构对照；extra_training只衡量自身相对5000；system_controls含不同骨干和训练历史，三类不可混称。',
        '选择可能仍落在较早完整校准点，实际步数和checkpoint SHA见selected_checkpoints.csv；拥有10000机会不等于部署的权重一定来自10000。',
        '区间未做多重比较校正，也不包含训练种子方差；视觉骨干预训练与指标独立性须区分，DINO不能单独证明实例保真。没有新正式holdout，不能据本报告自动宣布研究完成。'])
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'completion.json', {'status': 'AUDITED_R2_QUALITY_FIGURES_READY_NOT_RESEARCH_COMPLETE', 'completed_local': now(),
        'quality_receipt_sha256': sha256(quality_receipt), 'analysis_receipt_sha256': sha256(analysis_receipt),
        'source_hashes': source_hashes, 'all_methods_retained': 22, 'GPU_used': False, 'new_inference': False,
        'selection_changed': False, 'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print(output)


if __name__ == '__main__':
    main()
