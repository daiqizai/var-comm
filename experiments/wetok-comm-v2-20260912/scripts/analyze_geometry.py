"""Geometry, conditional-history and interaction analysis under matched training budgets."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import numpy as np

from train_milestone import write_csv
from wetok_comm.common import PROJECT, artifact_hashes, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.geometry_evaluation import geometry_statistics, geometry_supplement_statistics, load_geometry_evaluation, method_definitions
from wetok_comm.geometry_study import geometry_output


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total', type=int, required=True)
    parser.add_argument('--evaluation-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation_config, config, base, qualification = load_geometry_evaluation()
    if arguments.total != 7000:
        raise ValueError('final development attribution is registered at matched total7000/image5000')
    if not arguments.execute:
        print('PLAN ONLY: 21000 rows, 1008 paired intervals, 48 geometry/history interactions, plus fixed-support controls')
        return
    training = geometry_output(config, 'training')
    milestone_path = training / 'milestones' / f'total_{arguments.total:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    verify_sources(milestone['source_hashes'])
    review_path = geometry_output(config, 'analysis') / f'calibration_total_{arguments.total:07d}/completion.json'
    review = json.loads(review_path.read_text())
    if review['data_noise_phase_Adam_power_selection_audit'] != 'PASS' or review['milestone_sha256'] != sha256(milestone_path):
        raise RuntimeError('matched training/calibration audit has not passed')
    evaluation = (arguments.evaluation_dir or geometry_output(config, 'evaluation') / f'total_{arguments.total:07d}').resolve()
    receipt = json.loads((evaluation / 'completion.json').read_text())
    if receipt['status'] != 'GEOMETRY_EVALUATION_COMPLETE' or receipt['milestone_sha256'] != sha256(milestone_path):
        raise RuntimeError('geometry evaluation does not match the selected milestone')
    verify_sources(receipt['source_hashes'])
    for relative, expected in receipt['output_hashes'].items():
        if sha256(evaluation / relative) != expected:
            raise RuntimeError('geometry evaluation artifact changed')
    rows = read_rows(evaluation / 'per_frame.csv')
    clean = read_rows(evaluation / 'native_reference.csv')
    clean_lookup = {int(row['image_index']): row for row in clean}
    if len(clean) != 100 or set(clean_lookup) != set(range(100)):
        raise RuntimeError('correct native diagnostic does not cover the full source population')
    controls = json.loads((PROJECT / 'outputs' / config['control_training'] / config['control_milestone']).read_text())
    definitions = method_definitions(config)
    for row in rows:
        reference = clean_lookup[int(row['image_index'])]
        if row['image_id'] != reference['image_id'] or abs(float(row['lpips']) - float(reference['lpips']) - float(row['LPIPS_excess_from_native'])) > 1e-10:
            raise RuntimeError('native-relative quality anchor differs across source groups')
        if row['arm'] in definitions:
            definition = definitions[row['arm']]
            choice = (controls['selected']['continuous_mean__' + definition['variant']] if definition['geometry'] == '153x40'
                      else milestone['selected'][definition['variant']])
            if int(row['selected_image_step']) != choice['step']:
                raise RuntimeError('a development row used a different checkpoint from full calibration selection')
    summary, paired, interactions = geometry_statistics(rows, config, base)
    support_summary, support_paired = geometry_supplement_statistics(rows, read_rows(evaluation / 'deep_support_supplement.csv'), config, base)
    old_rows = [row for row in rows if row.get('geometry') == '153x40']
    if len(old_rows) != 6300 or max(float(row['control_replay_pixel_error']) for row in old_rows) > evaluation_config['control_pixel_replay_max_error']:
        raise RuntimeError('old geometry was not independently re-rendered within the registered tolerance')
    output = (arguments.output_dir or geometry_output(config, 'analysis') / f'development_total_{arguments.total:07d}').resolve()
    if not output.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('geometry analysis output escaped the project')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'src/wetok_comm/geometry_evaluation.py',
        EXPERIMENT / 'configs/geometry_evaluation.yaml', EXPERIMENT / 'configs/geometry_study.yaml'])
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'paired.csv', paired)
    write_csv(output / 'geometry_history_interactions.csv', interactions)
    write_csv(output / 'deep_support_summary.csv', support_summary)
    write_csv(output / 'deep_support_paired.csv', support_paired)
    label = '+'.join(map(str, base['evaluation']['primary_snrs_db']))
    primary = [row for row in summary if row['snrs_db'] == label]
    write_csv(output / 'primary.csv', primary)
    prefix_root = PROJECT / 'outputs' / base['outputs']['training']
    costs = []
    for name, definition in definitions.items():
        variant = definition['variant']
        if definition['geometry'] == '153x40':
            prefix = read_rows(prefix_root / variant / 'training.csv')[:2000]
            representation_seconds = sum(float(row['step_seconds']) for row in prefix)
            image_seconds = controls['timings']['training']['continuous_mean__' + variant]
            calibration_seconds = controls['timings']['calibration']['continuous_mean__' + variant]
            calibration_scope = 'image_phase_only_prefix_calibration_not_separately_measured'
        else:
            representation_seconds = milestone['timings']['representation'][variant]
            image_seconds = milestone['timings']['image'][variant]
            calibration_seconds = milestone['timings']['calibration'][variant]
            calibration_scope = 'all_candidate_calibration_including_representation_monitor'
        sample = next(row for row in rows if row['arm'] == name)
        costs.append({'arm': name, 'geometry': definition['geometry'], 'parameters': int(sample['communication_parameters']),
            'representation_updates': 2000, 'image_updates': 5000, 'selected_image_step': int(sample['selected_image_step']),
            'representation_training_seconds': representation_seconds, 'image_training_seconds': image_seconds,
            'calibration_seconds': calibration_seconds, 'calibration_scope': calibration_scope,
            'training_history_reused': definition['geometry'] == '153x40'})
    write_csv(output / 'training_costs.csv', costs)
    noiseless = read_rows(evaluation / 'noiseless_mapping.csv')
    if len(noiseless) != 600 or len({(int(row['image_index']), row['arm']) for row in noiseless}) != 600:
        raise RuntimeError('geometry noiseless diagnostic grid incomplete')
    diagnostics = [{'arm': name, **{metric: float(np.mean([float(row[metric]) for row in noiseless if row['arm'] == name]))
                    for metric in ('psnr_db', 'ssim', 'lpips', 'dino', 'bit_error_rate', 'feature_mse')}} for name in definitions]
    write_csv(output / 'noiseless_summary.csv', diagnostics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('psnr_db', 'lpips', 'dino'):
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for axis, variant in zip(axes, config['variants']):
            names = [f'geometry153x40__{variant}', f'geometry204x30__{variant}', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc']
            for name in names:
                selected = sorted((row for row in summary if row['arm'] == name and '+' not in row['snrs_db']), key=lambda row: float(row['snrs_db']))
                axis.plot([float(row['snrs_db']) for row in selected], [row[metric] for row in selected], marker='o', markersize=3,
                          linestyle='-' if name.startswith('geometry') else '--', label=name.split('__', 1)[0])
            supported = sorted((row for row in support_summary if row['arm'] == 'perceptual_deepjscc_fixed_support' and '+' not in row['snrs_db']), key=lambda row: float(row['snrs_db']))
            axis.scatter([float(row['snrs_db']) for row in supported], [row[metric] for row in supported], marker='D',
                         facecolors='none', edgecolors='black', label='Deep fixed support', s=40)
            axis.set(title=variant, xlabel='SNR (dB)', ylabel=metric)
            axis.grid(alpha=.2)
        axes[-1].legend(fontsize=7)
        figure.suptitle('Matched N3060 / E6120; geometry changes parameters and compute\nDeep original 5/6 dB is unsupported; diamonds retain the frozen support correction')
        figure.savefig(output / f'quality_{metric}.png', dpi=170)
        plt.close(figure)
    lines = ['# 匹配geometry开发评测', '',
        '总7000=2000表示+5000图像训练机会，源/噪声/选择规则匹配；旧训练历史复用，旧模型在线推理已重新计时与复渲染。',
        '主1/4/7dB，原100 development，非新holdout。每图3060 complex uses / 总归一化能量6120。', '',
        '| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | DINO↑ | 严重失真率↓ |', '|---|---:|---:|---:|---:|---:|']
    for row in primary:
        lines.append(f'| {row["arm"]} | {row["psnr_db"]:.5f} | {row["ssim"]:.5f} | {row["lpips"]:.6f} | {row["dino"]:.6f} | {row["severe_distortion"]:.4f} |')
    lines += ['', '## 归因边界',
        '- 先看同结构geometry差，再看各geometry内conditioned−no_history；交互项单列，不能把共同packing收益全归给next-scale。',
        '- 204×30参数与memory计算更多，必须结合新旧同场计时；初始线性秩不是整个非线性码的容量或质量上界。',
        '- Six geometry models were evaluated from paid observations only; continuous features are not reported as reliable native bits.',
        '- 新旧控制预算都为2000+5000，不按选中checkpoint步数少算实际训练。旧prefix校准耗时没有单独分解，不把缺失值写成0。',
        '- 所有失败图保留、原Deep异常点与合理支持点同时保留；未根据development换checkpoint。',
        '- 这里仍是一个训练seed的development证据，不等于研究完成。强对照、重复训练与独立holdout仍须按总目标验证。', '',
        f'候选实际训练进程占用：{milestone["observed_wall_GPU_hours"]:.5f}小时；本次评测全部session已记录占用：{receipt["evaluation_observed_wall_hours_all_sessions"]:.5f}小时。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'completion.json', {'status': 'GEOMETRY_DEVELOPMENT_REVIEW_READY_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'wireless_rows': len(rows), 'paired_intervals': len(paired), 'interaction_intervals': len(interactions),
        'support_intervals': len(support_paired), 'source_hashes': sources,
        'evaluation_receipt_sha256': sha256(evaluation / 'completion.json'), 'output_hashes': artifact_hashes(output),
        'research_goal_complete': False})
    print(output / 'report.md')


if __name__ == '__main__':
    main()
