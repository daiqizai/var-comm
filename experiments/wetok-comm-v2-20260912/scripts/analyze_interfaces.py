"""Full same-budget interface comparisons, accounting and development-only reporting."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import numpy as np
import torch

from train_milestone import write_csv
from wetok_comm.common import PROJECT, artifact_hashes, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.deep_support import SUPPORT_NAME, supplement_statistics
from wetok_comm.interface_evaluation import FrozenImageReferences, load_evaluation_config, statistics
from wetok_comm.interface_study import interface_definitions, interface_output
from wetok_comm.training import paired_batches


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def training_costs(study, base, parent, training, milestone):
    checkpoint = Path(milestone['checkpoint'])
    if sha256(checkpoint) != milestone['optimizer_checkpoint_sha256']:
        raise RuntimeError('paired optimizer milestone changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    count = milestone['additional_updates_per_arm']
    definitions = interface_definitions(study)
    if saved['completed_additional_updates'] != count:
        raise RuntimeError('milestone counters differ')
    initial = json.loads((training / 'initialization.json').read_text())
    if not initial['all_optimizers_fresh_and_equal'] or initial['inherited_Adam_moments']:
        raise RuntimeError('the registered common fresh-optimizer history differs')
    for local, batch in enumerate(paired_batches(base, 20000, study['parent_step'], study['parent_step'] + count)):
        for name in definitions:
            row = saved['histories'][name][local]
            if row['batch_sha256'] != batch['fingerprint'] or row['Adam_step'] != local + 1:
                raise RuntimeError('data/noise/Adam pairing differs in the retained history')
    parent_seconds = {}
    for variant in study['variants']:
        history = read_rows(parent / variant / 'training.csv')[:study['parent_step']]
        if len(history) != study['parent_step'] or [int(row['step']) for row in history] != list(range(1, study['parent_step'] + 1)):
            raise RuntimeError('original common-parent training exposure is incomplete')
        parent_seconds[variant] = sum(float(row['step_seconds']) for row in history)
    costs = []
    for name, definition in definitions.items():
        if len(saved['histories'][name]) != count or any(int(value['step']) != count for value in saved['optimizers'][name]['state'].values()):
            raise RuntimeError('retained budgets or optimizer counters differ across arms')
        costs.append({'arm': name, **definition, 'common_parent_updates': study['parent_step'],
            'new_retained_updates': count, 'selected_additional_step': milestone['selected'][name]['step'],
            'common_plus_new_exposures': (study['parent_step'] + count) * base['training']['effective_batch_size'],
            'communication_parameters': sum(value.numel() for value in saved['models'][name].values()),
            'common_parent_training_seconds': parent_seconds[definition['variant']],
            'additional_training_seconds': saved['timings']['training'][name],
            'additional_calibration_seconds': saved['timings']['calibration'][name]})
    return costs, sum(parent_seconds.values()) / 3600


def plot_quality(output, summary, study, support_summary=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for metric in ('psnr_db', 'lpips', 'dino'):
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for axis, variant in zip(axes, study['variants']):
            names = [f'{interface}__{variant}' for interface in study['interfaces']]
            names += ['old_selected__' + variant, 'wetok_8PSK_FEC', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc']
            for name in names:
                rows = sorted((row for row in summary if row['arm'] == name and '+' not in row['snrs_db']), key=lambda row: float(row['snrs_db']))
                style = '-' if name in names[:3] else '--'
                label = name.split('__', 1)[0]
                axis.plot([float(row['snrs_db']) for row in rows], [row[metric] for row in rows],
                          linestyle=style, marker='o', markersize=3, label=label)
            if support_summary:
                supported = sorted((row for row in support_summary if row['arm'] == SUPPORT_NAME and '+' not in row['snrs_db']),
                                   key=lambda row: float(row['snrs_db']))
                axis.scatter([float(row['snrs_db']) for row in supported], [row[metric] for row in supported],
                             marker='D', facecolors='none', edgecolors='black', s=42, label='Deep fixed-support supplement', zorder=4)
            axis.set(title=variant, xlabel='SNR (dB)', ylabel=metric)
            axis.grid(alpha=.2)
        axes[-1].legend(fontsize=7, loc='best')
        figure.suptitle('Development; 3060 complex uses / total energy 6120\nDeep raw 5/6 dB has known condition-support issues; diamonds show the frozen correction')
        figure.savefig(output / f'quality_{metric}.png', dpi=170)
        figure.savefig(output / f'quality_{metric}.pdf')
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--evaluation-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation_config, study, base, parent = load_evaluation_config()
    if not arguments.execute:
        print('PLAN ONLY: 33600 wireless rows, 63 comparisons/3024 source-image paired intervals; no new inference or selection')
        return
    torch.set_num_threads(2)
    training = interface_output(study, 'training')
    milestone_path = training / 'milestones' / f'additional_{arguments.milestone:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    verify_sources(milestone['source_hashes'])
    evaluation = (arguments.evaluation_dir or interface_output(study, 'evaluation') / f'additional_{arguments.milestone:07d}').resolve()
    receipt = json.loads((evaluation / 'completion.json').read_text())
    if receipt['status'] != 'INTERFACE_EVALUATION_COMPLETE' or receipt['milestone_sha256'] != sha256(milestone_path):
        raise RuntimeError('completed evaluation differs from the selected training milestone')
    verify_sources(receipt['source_hashes'])
    for relative, expected in receipt['output_hashes'].items():
        if sha256(evaluation / relative) != expected:
            raise RuntimeError('completed evaluation artifact changed')
    rows = read_rows(evaluation / 'per_frame.csv')
    references = FrozenImageReferences(evaluation_config, study)
    for path in {Path(row['image_archive']) for row in rows if row['image_store'] == 'frozen_reference'}:
        relative = str(path.relative_to(references.root))
        if sha256(path) != references.receipt['output_hashes'][relative]:
            raise RuntimeError('referenced historical image archive changed after evaluation')
    clean = read_rows(evaluation / 'native_reference.csv')
    clean_lookup = {int(row['image_index']): row for row in clean}
    if len(clean) != 100 or set(clean_lookup) != set(range(100)):
        raise RuntimeError('native source-image diagnostic reference is incomplete')
    for row in rows:
        if row['image_id'] != clean_lookup[int(row['image_index'])]['image_id']:
            raise RuntimeError('image reference differs from its true source group')
        expected = float(row['lpips']) - float(clean_lookup[int(row['image_index'])]['lpips'])
        if abs(expected - float(row['LPIPS_excess_from_native'])) > 1e-10:
            raise RuntimeError('additional native-relative distortion was not computed per source')
        if row['arm'] in milestone['selected'] and int(row['selected_additional_step']) != milestone['selected'][row['arm']]['step']:
            raise RuntimeError('a development row used a different checkpoint from calibration selection')
    summary, paired = statistics(rows, study, base)
    support_summary, support_paired = supplement_statistics(rows, read_rows(evaluation / 'deep_support_supplement.csv'), study, base)
    costs, common_parent_hours = training_costs(study, base, parent, training, milestone)
    output = (arguments.output_dir or interface_output(study, 'analysis') / f'development_{arguments.milestone:07d}').resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('analysis output escapes the project')
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/interface_evaluation.yaml',
        EXPERIMENT / 'configs/interface_study.yaml', EXPERIMENT / 'src/wetok_comm/interface_evaluation.py',
        EXPERIMENT / 'configs/deep_support.yaml', EXPERIMENT / 'src/wetok_comm/deep_support.py'])
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'paired.csv', paired)
    write_csv(output / 'training_costs.csv', costs)
    write_csv(output / 'deep_support_summary.csv', support_summary)
    write_csv(output / 'deep_support_paired.csv', support_paired)
    primary_label = '+'.join(map(str, base['evaluation']['primary_snrs_db']))
    primary = [row for row in summary if row['snrs_db'] == primary_label]
    write_csv(output / 'primary.csv', primary)
    noiseless = read_rows(evaluation / 'noiseless_mapping.csv')
    if len(noiseless) != 900 or len({(int(row['image_index']), row['arm']) for row in noiseless}) != 900:
        raise RuntimeError('noiseless mapping diagnostic grid incomplete or duplicated')
    diagnostic = []
    for name in interface_definitions(study):
        values = [row for row in noiseless if row['arm'] == name]
        if {int(row['image_index']) for row in values} != set(range(100)):
            raise RuntimeError('noiseless source coverage differs across interfaces')
        diagnostic.append({'arm': name, 'condition': 'noiseless_nominal19_not_wireless_ranking',
            **{key: float(np.mean([float(row[key]) for row in values])) for key in ('psnr_db', 'ssim', 'lpips', 'dino', 'bit_error_rate', 'feature_mse', 'feature_abs_mean')}})
    write_csv(output / 'noiseless_summary.csv', diagnostic)
    plot_quality(output, summary, study, support_summary)
    lines = [f'# WeTok接收接口：追加{arguments.milestone}步development比较', '',
        '100源图、七SNR、三个配对噪声；主1/4/7dB。33600行不是33600个独立样本；95%区间对源图平均值bootstrap。',
        '同3060 complex uses / 总能量6120。连续接口从训练到评测直接decode有界特征，不称恢复了同样的数字token。', '',
        '## 主区间', '| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | DINO↑ | 严重失真率↓ |', '|---|---:|---:|---:|---:|---:|']
    for row in primary:
        lines.append(f'| {row["arm"]} | {row["psnr_db"]:.5f} | {row["ssim"]:.5f} | {row["lpips"]:.6f} | {row["dino"]:.6f} | {row["severe_distortion"]:.4f} |')
    lines += ['', '## 同接口条件历史与接口差', '| 方法 | 对照 | ΔLPIPS | 95% CI |', '|---|---|---:|---|']
    definitions = interface_definitions(study)
    for row in paired:
        if row['snrs_db'] == primary_label and row['metric'] == 'lpips' and row['method'] in definitions and row['control'] in definitions:
            lines.append(f'| {row["method"]} | {row["control"]} | {row["delta"]:+.6f} | [{row["ci_low"]:+.6f}, {row["ci_high"]:+.6f}] |')
    lines += ['', '## 选择与成本', '选模只用完整calibration LPIPS；实际研究已投入的训练不按选中step缩短记账。所有分支统一fresh Adam，不伪称继承了未保存的2000步moment。']
    lines += ['600行固定支持Deep已用相同source/实际SNR/raw_noise重新生成，逐图与旧冻结记录核对；单列deep_support_summary/paired，不替换原始七点或主1/4/7结果。']
    for name, selected in milestone['selected'].items():
        lines.append(f'- {name}：选中追加{selected["step"]}；全校准LPIPS {selected["lpips"]:.6f}。')
    lines += [f'- 本轮九臂观测占用：{milestone["observed_wall_GPU_hours"]:.5f}小时；共同2000父历史的实测训练主体（每结构计一次）：{common_parent_hours:.5f}小时。',
        f'- 本次评测进程占用：{receipt["evaluation_wall_hours_this_session"]:.5f}小时；不是租赁账单。逐臂计算及校准开销见training_costs.csv。',
        f'- 评测全部{receipt["evaluation_sessions"]}个session的已记录占用：{receipt["evaluation_observed_wall_hours_all_sessions"]:.5f}小时；若存在未正常关闭session则仅为下界：{receipt["evaluation_total_time_is_lower_bound"]}。',
        '- 新模型在线TX包含视觉Encoder/量化/转换，RX包含全部串行读取与冻结Decoder；历史时延不与新计时混排。时延优势须后续统一重测强对照。', '',
        '## 解释边界',
        '- 接口差不等于条件结构增量；首先看同接口conditioned−no_history。所有阶段读取完整y，不是逐包渐进。',
        '- 若选中追加0，必须承认该方法没有从新增训练中取得校准收益；不能把初始化/接口差包装成续训成功或以失败fallback制造结构优势。',
        '- 相对同接口single_pass，逐SNR PSNR低于0.2dB的情况列入psnr_tradeoffs.csv；不隐藏图像失真换感知的代价。',
        '- 旧Deep在5/6dB的已知条件问题不作为主增益证据；主区间固定1/4/7。WeTok数字固定8PSK/FEC不宣称是最优源信道编码。',
        '- 正确native与noiseless输出仅用于诊断；thresholded-logit BER不代表连续分支可靠交付了源bits。',
        '- 这仍是development、一个训练种子；不是独立holdout确认，也没有因此完成整个通信研究。']
    tradeoffs = []
    for row in paired:
        if (row['metric'] == 'psnr_db' and row['method'] in definitions and row['control'] in definitions
            and row['control'].endswith('__single_pass')):
            if definitions[row['method']]['interface'] == definitions[row['control']]['interface']:
                tradeoffs.append({**row, 'drop_over_0p2_dB': row['delta'] < -.2})
    write_csv(output / 'psnr_tradeoffs.csv', tradeoffs)
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'completion.json', {'status': 'INTERFACE_DEVELOPMENT_REVIEW_READY_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'wireless_rows': len(rows), 'paired_intervals': len(paired),
        'separate_fixed_support_rows': 600, 'separate_fixed_support_intervals': len(support_paired),
        'source_hashes': source_hashes, 'evaluation_receipt_sha256': sha256(evaluation / 'completion.json'),
        'output_hashes': artifact_hashes(output), 'research_goal_complete': False})
    print(output / 'report.md')


if __name__ == '__main__':
    main()
