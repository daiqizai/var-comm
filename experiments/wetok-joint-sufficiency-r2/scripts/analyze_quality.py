"""Separate matched10000 structure contrasts from additional-training effects and historical comparisons."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
GRID = EXPERIMENT.parent / 'wetok-joint-grid-controls-r1'
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(GRID / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

import numpy as np

from joint_sender.evaluation_io import write_rows
from sufficiency.archive_audit import audit_saved
from sufficiency.evaluation import audited_endpoint, evaluation_output, learned_names, load_evaluation, statistics, support_statistics, validate_diagnostics, validate_reference_diagnostics, validate_selection
from sufficiency.references import References, read_rows
from wetok_comm.common import artifact_hashes, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.interface_evaluation import METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, config, original, grid, reference, base, parent = load_evaluation()
    if not arguments.execute:
        print('PLAN ONLY: completed R2 quality -> CPU audit and correctly scoped paired contrasts; no inference or selection')
        return
    milestone, milestone_path, review_path, source_binding = audited_endpoint(config, arguments.step)
    references = References(evaluation, config, original, grid, reference)
    root = evaluation_output(evaluation, 'quality')
    receipt_path = root / 'completion.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt['status'] != 'R2_QUALITY_COMPLETE' or receipt['r2_milestone_sha256'] != sha256(milestone_path) or
        receipt['r2_review_sha256'] != sha256(review_path) or receipt['reference_quality_sha256'] != references.receipt_sha or
        receipt['rows'] != evaluation['main_rows'] or not receipt['quality_only_no_current_timing']):
        raise RuntimeError('R2 quality is incomplete or belongs to different frozen models/references')
    verify_sources(receipt['source_hashes'])
    for relative, expected in receipt['output_hashes'].items():
        if sha256(root / relative) != expected:
            raise RuntimeError('completed R2 quality artifact changed')
    rows, clean, noiseless, support = [read_rows(root / filename) for filename in
        ('per_frame.csv', 'native_reference.csv', 'noiseless_mapping.csv', 'deep_support_supplement.csv')]
    references.validate_reuse(rows)
    validate_reference_diagnostics(noiseless, support, references)
    validate_selection(rows, milestone)
    validate_diagnostics(rows, clean, noiseless, support, config, original, grid, reference, base)
    audit = audit_saved(root, rows, clean, noiseless, support, config, base, references)
    summaries, paired = statistics(rows, config, original, grid, reference, base)
    support_summary, support_paired = support_statistics(rows, support, config, original, grid, reference, base)
    if len(paired) != evaluation['paired_intervals'] or len(support_paired) != evaluation['support_paired_intervals']:
        raise RuntimeError('R2 required comparisons were omitted')
    output = evaluation_output(evaluation, 'quality_analysis')
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
        *sorted((EXPERIMENT / 'src/sufficiency').glob('*.py'))])
    label = '+'.join(map(str, base['evaluation']['primary_snrs_db']))
    primary = [row for row in summaries if row['snrs_db'] == label]
    for filename, values in (('summary.csv', summaries), ('primary.csv', primary), ('paired.csv', paired),
        ('extra_training_effects.csv', [row for row in paired if row['comparison_scope'] == 'extra_training_effect_not_new_mechanism']),
        ('matched10000_structures.csv', [row for row in paired if row['comparison_scope'] == 'matched10000_structure_comparison']),
        ('deep_support_summary.csv', support_summary), ('deep_support_paired.csv', support_paired)):
        write_rows(output / filename, values)
    diagnostics = []
    for name in learned_names(config, original, grid, reference):
        subset = [row for row in noiseless if row['arm'] == name]
        diagnostics.append({'arm': name, 'source_images': 100, 'role': 'not_wireless_ranking',
            **{key: float(np.mean([float(row[key]) for row in subset])) for key in (*METRICS, 'bit_error_rate', 'feature_mse')}})
    write_rows(output / 'noiseless_summary.csv', diagnostics)
    write_json(output / 'saved_quality_audit.json', audit)
    text = ['# R2同机会充分性质量结果', '',
        '四R2臂各10000更新机会；旧18方法保持各自5000及既有训练历史，不冒充同10000预算。时间由独立阶段测量。', '',
        '| 方法 | PSNR ↑ | LPIPS ↓ | DINO ↑ | 严重失真率 ↓ |', '|---|---:|---:|---:|---:|']
    text.extend(f'| {row["arm"]} | {row["psnr_db"]:.5f} | {row["lpips"]:.6f} | {row["dino"]:.6f} | {row["severe_distortion"]:.4f} |' for row in primary)
    main = next(row for row in paired if row['snrs_db'] == label and row['method'] == 'r2__multiscale_state_history'
        and row['control'] == 'r2__full_grid_state_history' and row['metric'] == 'lpips')
    text.extend(['', f'主要匹配结构对照：multiscale−普通full-grid LPIPS {main["delta"]:+.6f}，CI[{main["ci_low"]:+.6f},{main["ci_high"]:+.6f}]。', '',
        'extra_training_effects.csv只量化各臂相对自身5000的变化，不是新机制；普通迭代不改名next-scale。',
        '图像级bootstrap不包含训练种子随机性，未作多重比较校正，没有新正式holdout；不能据本轮自动宣布研究完成。'])
    (output / 'summary.md').write_text('\n'.join(text) + '\n')
    write_json(output / 'completion.json', {'status': 'R2_QUALITY_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE', 'completed_local': now(),
        'r2_milestone_sha256': sha256(milestone_path), 'quality_receipt_sha256': sha256(receipt_path),
        'reference_quality_sha256': references.receipt_sha, 'main_rows': len(rows), 'paired_intervals': len(paired),
        'support_intervals': len(support_paired), 'source_hashes': source_hashes, 'new_holdout_accessed': False,
        'contains_latency_ranking': False, 'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print(json.dumps(primary, indent=2))


if __name__ == '__main__':
    main()
