"""Source-image paired Grid statistics and CPU artifact checks, with no shared-runtime ranking."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

import numpy as np

from grid_controls.archive_audit import audit_saved
from grid_controls.common import output_path
from grid_controls.evaluation import audited_grid, load_evaluation, learned_names, statistics, support_statistics, validate_diagnostics, validate_reference_diagnostics, validate_selection
from grid_controls.references import References, read_rows
from joint_sender.evaluation_io import write_rows
from wetok_comm.common import artifact_hashes, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.interface_evaluation import METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=5000)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, config, original, reference, base, parent = load_evaluation()
    if not arguments.execute:
        print('PLAN ONLY: completed Grid quality -> paired statistics/CPU audit; no new inference or timing claims')
        return
    milestone, milestone_path, review_path, original_binding = audited_grid(config, original, reference, arguments.step)
    references = References(evaluation, config, original, reference)
    root = output_path(config, 'evaluation') / 'quality_0005000'
    receipt_path = root / 'completion.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt['status'] != 'GRID_QUALITY_COMPLETE' or receipt['grid_milestone_sha256'] != sha256(milestone_path) or
        receipt['grid_review_sha256'] != sha256(review_path) or receipt['original_joint_evaluation_sha256'] != references.receipt_sha or
        not receipt['quality_only_no_current_timing'] or receipt['rows'] != 37800):
        raise RuntimeError('Grid quality is incomplete or bound to a different trial')
    verify_sources(receipt['source_hashes'])
    for relative, expected in receipt['output_hashes'].items():
        if sha256(root / relative) != expected:
            raise RuntimeError('completed Grid quality artifact changed')
    rows, clean, noiseless, support = [read_rows(root / filename) for filename in
        ('per_frame.csv', 'native_reference.csv', 'noiseless_mapping.csv', 'deep_support_supplement.csv')]
    references.validate_reuse(rows)
    validate_selection(rows, milestone)
    validate_diagnostics(rows, clean, noiseless, support, config, original, reference, base)
    validate_reference_diagnostics(noiseless, support, references)
    audit = audit_saved(root, rows, clean, noiseless, support, config, base)
    summaries, paired = statistics(rows, config, original, reference, base)
    support_summary, support_paired = support_statistics(rows, support, config, original, reference, base)
    if len(paired) != evaluation['paired_intervals'] or len(support_paired) != evaluation['support_paired_intervals']:
        raise RuntimeError('registered Grid contrasts were omitted')
    output = output_path(config, 'analysis') / 'quality_0005000'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
        *sorted((EXPERIMENT / 'src/grid_controls').glob('*.py'))])
    label = '+'.join(map(str, base['evaluation']['primary_snrs_db']))
    primary = [row for row in summaries if row['snrs_db'] == label]
    for filename, values in (('summary.csv', summaries), ('primary.csv', primary), ('paired.csv', paired),
        ('deep_support_summary.csv', support_summary), ('deep_support_paired.csv', support_paired)):
        write_rows(output / filename, values)
    no_noise_summary = []
    for name in learned_names(config, original, reference):
        subset = [row for row in noiseless if row['arm'] == name]
        no_noise_summary.append({'arm': name, 'source_images': 100, 'role': 'not_wireless_ranking',
            **{key: float(np.mean([float(row[key]) for row in subset])) for key in (*METRICS, 'bit_error_rate', 'feature_mse')}})
    write_rows(output / 'noiseless_summary.csv', no_noise_summary)
    write_json(output / 'saved_quality_audit.json', audit)
    text = ['# Grid5000固定预算质量比较', '', '100张development、七SNR、三噪声；所有18方法保留。当前表没有时延，速度由独立阶段测量。', '',
        '| 方法 | PSNR ↑ | LPIPS ↓ | DINO ↑ | 严重失真率 ↓ |', '|---|---:|---:|---:|---:|']
    text.extend(f'| {row["arm"]} | {row["psnr_db"]:.5f} | {row["lpips"]:.6f} | {row["dino"]:.6f} | {row["severe_distortion"]:.4f} |' for row in primary)
    main = next(row for row in paired if row['snrs_db'] == label and row['method'] == 'joint__multiscale_state_history'
        and row['control'] == 'grid__full_grid_state_history' and row['metric'] == 'lpips')
    text.extend(['', f'主要对照：multiscale state−普通full-grid state LPIPS {main["delta"]:+.6f}，95% CI [{main["ci_low"]:+.6f}, {main["ci_high"]:+.6f}]。', '',
        '普通迭代不是next-scale；同参数/阶段数不等于同FLOPs，size描述与空间网格尚未单独分离。',
        '区间没有覆盖训练种子随机性或作多重比较校正；没有新holdout。不能把本表当作独占时延排名或研究完成证明。'])
    (output / 'summary.md').write_text('\n'.join(text) + '\n')
    write_json(output / 'completion.json', {'status': 'GRID_QUALITY_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'grid_milestone_sha256': sha256(milestone_path), 'quality_receipt_sha256': sha256(receipt_path),
        'reference_receipt_sha256': references.receipt_sha, 'main_rows': len(rows), 'paired_intervals': len(paired),
        'support_intervals': len(support_paired), 'source_hashes': sources, 'new_holdout_accessed': False,
        'contains_latency_ranking': False, 'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print(json.dumps(primary, indent=2))


if __name__ == '__main__':
    main()
