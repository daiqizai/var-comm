"""CPU audit and paired statistics for complete R3 quality, with the vector comparison explicitly isolated."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
for directory in ('wetok-comm-v2-20260912', 'wetok-innovation-r1', 'wetok-joint-sender-r1', 'wetok-joint-grid-controls-r1', 'wetok-joint-sufficiency-r2'):
    sys.path.insert(0, str(EXPERIMENT.parent / directory / 'src'))
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np

from joint_sender.evaluation_io import write_rows
from vector_control.archive_audit import audit_saved
from vector_control.evaluation import audited_endpoint, evaluation_output, learned_names, load_evaluation, statistics, support_statistics, validate_diagnostics, validate_reference_diagnostics, validate_selection
from vector_control.references import References, read_rows
from wetok_comm.common import artifact_hashes, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.interface_evaluation import METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    evaluation, config, r2, original, grid, reference, base, parent = load_evaluation()
    if not arguments.execute:
        print('PLAN ONLY: complete R3 quality -> CPU image/channel audit and scoped paired statistics; no inference or selection')
        return
    milestone, milestone_path, review_path, qualified = audited_endpoint(config, r2, grid, arguments.step)
    references = References(evaluation, config, r2, original, grid, reference)
    root = evaluation_output(evaluation, 'quality')
    receipt_path = root / 'completion.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt['status'] != 'R3_QUALITY_COMPLETE' or receipt['r3_milestone_sha256'] != sha256(milestone_path) or
        receipt['r3_review_sha256'] != sha256(review_path) or receipt['reference_quality_sha256'] != references.receipt_sha or
        receipt['rows'] != evaluation['main_rows'] or not receipt['quality_only_no_current_timing']):
        raise RuntimeError('R3 quality is incomplete or belongs to a different model/source selection')
    verify_sources(receipt['source_hashes'])
    for relative, expected in receipt['output_hashes'].items():
        if sha256(root / relative) != expected:
            raise RuntimeError('a completed R3 quality artifact changed')
    rows, clean, noiseless, support = [read_rows(root / filename) for filename in
        ('per_frame.csv', 'native_reference.csv', 'noiseless_mapping.csv', 'deep_support_supplement.csv')]
    references.validate_reuse(rows)
    validate_reference_diagnostics(noiseless, support, references)
    validate_selection(rows, milestone)
    validate_diagnostics(rows, clean, noiseless, support, config, r2, original, grid, reference, base)
    audit = audit_saved(root, rows, clean, noiseless, support, config, base, references)
    summaries, paired = statistics(rows, config, r2, original, grid, reference, base)
    support_summary, support_paired = support_statistics(rows, support, config, r2, original, grid, reference, base)
    if (len(paired) != evaluation['paired_intervals'] or len(support_paired) != evaluation['support_paired_intervals'] or
        sum(row['measurement_scope'] == 'new_R3_comparison' for row in support_paired) != evaluation['new_support_paired_intervals']):
        raise RuntimeError('R3 required comparison scope/count changed')
    output = evaluation_output(evaluation, 'quality_analysis')
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/evaluation.yaml', EXPERIMENT / 'docs/evaluation_protocol.md',
        *sorted((EXPERIMENT / 'src/vector_control').glob('*.py'))])
    label = '+'.join(map(str, base['evaluation']['primary_snrs_db']))
    primary = [row for row in summaries if row['snrs_db'] == label]
    for filename, values in (('summary.csv', summaries), ('primary.csv', primary), ('paired.csv', paired),
        ('matched_vector_construction.csv', [row for row in paired if row['comparison_scope'] == 'matched10000_vector_construction_parameter_and_E_call_comparison']),
        ('same10000_structure_references.csv', [row for row in paired if row['comparison_scope'] == 'same10000_structure_reference_not_equal_parameters_or_E_calls']),
        ('deep_support_summary.csv', support_summary), ('deep_support_paired.csv', support_paired)):
        write_rows(output / filename, values)
    diagnostics = []
    for name in learned_names(config, r2, original, grid, reference):
        subset = [row for row in noiseless if row['arm'] == name]
        diagnostics.append({'arm': name, 'source_images': 100, 'role': 'not_wireless_ranking',
            **{key: float(np.mean([float(row[key]) for row in subset])) for key in (*METRICS, 'bit_error_rate', 'feature_mse')}})
    write_rows(output / 'noiseless_summary.csv', diagnostics)
    write_json(output / 'saved_quality_audit.json', audit)
    text = ['# R3完整向量构造对照', '',
        '相同10000机会、参数和E回算的主要对照：新prediction向量−R2 residual。两者仍有同样定义的标量残差门控，不是完全移除残差，也不是固定同一y的纯接收器实验。', '',
        '| 方法 | PSNR ↑ | LPIPS ↓ | DINO ↑ | 严重失真率 ↓ |', '|---|---:|---:|---:|---:|']
    text.extend(f'| {row["arm"]} | {row["psnr_db"]:.5f} | {row["lpips"]:.6f} | {row["dino"]:.6f} | {row["severe_distortion"]:.4f} |' for row in primary)
    main = next(row for row in paired if row['snrs_db'] == label and row['control'] == evaluation['matched_vector_control'] and row['metric'] == 'lpips')
    text.extend(['', f'主LPIPS差 {main["delta"]:+.6f}，逐比较95% CI[{main["ci_low"]:+.6f},{main["ci_high"]:+.6f}]。', '',
        '其它R2模型虽有10000机会，但未必同参数/E回算；旧18方法是历史或系统参考，不冒充匹配训练预算。',
        '区间不包含训练种子不确定性，未作多重比较校正；没有新holdout。整体向量变化包含幅度/方向，不单独归因纯方向或精确似然。普通全网格不改名next-scale。'])
    (output / 'summary.md').write_text('\n'.join(text) + '\n')
    write_json(output / 'completion.json', {'status': 'R3_QUALITY_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE', 'completed_local': now(),
        'r3_milestone_sha256': sha256(milestone_path), 'quality_receipt_sha256': sha256(receipt_path),
        'reference_quality_sha256': references.receipt_sha, 'main_rows': len(rows), 'paired_intervals': len(paired),
        'support_intervals': len(support_paired), 'new_support_intervals': evaluation['new_support_paired_intervals'],
        'source_hashes': sources, 'contains_latency_ranking': False, 'new_holdout_accessed': False,
        'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print(json.dumps(primary, indent=2))


if __name__ == '__main__':
    main()
