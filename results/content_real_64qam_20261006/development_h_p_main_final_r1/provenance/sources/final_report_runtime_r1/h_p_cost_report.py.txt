"""Pure projection of sealed H/P statistics and separate H component costs.

This module has no file loader, CLI, model, PHY, bootstrap or publication code.
The future caller must verify every output SHA and normal owner/worker closure
before passing data. Logical checks here do not establish that process proof.
No successful experiment or final system completion receipt is produced.
"""
from __future__ import annotations
import ast
import copy
import math

ROLES = ('H_WHOLE_SYSTEM', 'H_RAW_PARTIAL_SYSTEM', 'H_FIXED_M7_ATTRIBUTION')
MISSING = ('mse', 'resnet50_top1_probability', 'confidently_wrong')
METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine', 'dinov2_vitl14_cosine',
    'clip_image_cosine', 'dists', 'dreamsim', 'ms_ssim', 'resnet50_top1_label',
    'resnet50_top1_source_prediction', 'semantic_error', 'convnext_top1_label',
    'convnext_top1_source_prediction', 'convnext_source_prediction_agreement',
    'convnext_source_correct_to_wrong', 'convnext_source_wrong_to_correct',
    'dino_mismatched', 'dino_specificity')
NOISE = {'H': [6201, 6202, 6203], 'P': [2001, 2002, 2003]}
FIXED16 = [0, 25, 50, 75, 4, 21, 24, 29, 33, 41, 52, 60, 64, 87, 92, 95]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sequence(value):
    return list(ast.literal_eval(value) if isinstance(value, str) else value)


def interval(row):
    require(int(row['source_count']) == 100 and int(row['noise_count']) == 3
        and int(row['frame_count']) == 300 and int(row['bootstrap_replicates']) == 10000
        and int(row['bootstrap_seed']) == 2026100605, 'Original source-level interval required')
    values = [float(row[k]) for k in ('mean', 'ci_low', 'ci_high')]
    require(all(math.isfinite(x) for x in values) and values[1] <= values[2], 'Invalid finite interval')


def point_catalog(rows):
    require(len(rows) == 18, 'Every frozen H point, including raw partial controls, is required')
    out = {}
    for row in rows:
        slot = int(row['development_slot']); p = row['point_id']
        require(0 <= slot < 18 and p == f'H18_SLOT_{slot:02d}' and p not in out, 'Exact unique slots required')
        role = ROLES[0] if slot < 8 else ROLES[1] if slot < 10 else ROLES[2]
        require(row['role'] == role and int(row['snr_db']) in (13, 19) and row['candidate_id'], 'Frozen role/SNR absent')
        out[p] = row
    for role in ROLES:
        group = [r for r in out.values() if r['role'] == role]
        if role == ROLES[1]:
            require({int(r['snr_db']) for r in group} == {13, 19}, 'Both raw-partial system controls required')
        else:
            require({(int(r['snr_db']), r['arm']) for r in group} ==
                {(s, a) for s in (13, 19) for a in ('H16-R', 'H16-A', 'H64-R', 'H64-A')},
                'Four fixed arms at both SNRs required')
    return out


def count_table(rows, catalog, denominator):
    totals = {p: 0 for p in catalog}
    for row in rows:
        p = row['point_id']; require(p in totals, 'Unknown point in count table')
        require(int(row['denominator']) == denominator and int(row['count']) >= 0,
                'TX source and RX frame denominators must remain separate')
        for key in ('role', 'candidate_id'):
            require(row[key] == catalog[p][key], 'Count identity changed')
        require(int(row['snr_db']) == int(catalog[p]['snr_db']), 'Count SNR changed')
        totals[p] += int(row['count'])
    require(all(n == denominator for n in totals.values()), 'Missing sources/frames/failures in counts')


def costs(timing):
    if timing is None:
        return {'status': 'PENDING_NORMAL_TIMING_CLOSURE', 'components': [], 'historical_PHY_rows': [],
                'P_online_cost': 'NOT_MEASURED', 'MAIN_online_cost': 'NOT_AVAILABLE', 'end_to_end_latency': None}
    done, summary, history = timing['completion'], timing['component_summary'], timing['historical_phy_event_windows']
    require(done['status'] == 'H_FIXED16_ONLINE_COMPONENTS_COMPLETE'
        and done['source_count'] == 16 and done['component_case_count'] == 288
        and done['H_policy_snr_points'] == 18 and done['source_indices'] == FIXED16
        and done['noise_seed'] == 6201 and done['warmup_repetitions'] == 1
        and done['measured_repetitions'] == 3 and done['exact_TX_parity'] is True
        and done['exact_RX_parity'] is True, 'Original fixed16 timing receipt required')
    for key in ('new_packet_decodes', 'new_noise_draws', 'new_quality_samples', 'new_metric_calls', 'budget_writes'):
        require(done[key] == 0, 'Timing may not add quality/channel samples')
    for key in ('exclusive_PHY_latency_measured', 'PHY_encode_measured', 'end_to_end_latency_measured'):
        require(done[key] is False, 'Unregistered complete PHY/end-to-end cost claim')
    require(summary['status'] == 'H_FIXED16_COMPONENT_SUMMARY' and summary['source_count'] == 16
        and summary['component_case_count'] == 288 and summary['uncached'] is True
        and summary['warmups_excluded'] is True and summary['historical_PHY_can_be_added_to_GPU_totals'] is False,
        'Uncached component scope changed')
    require(all(summary[k] is None for k in ('PHY_encode', 'exclusive_PHY_decode', 'end_to_end_latency')),
            'Missing timing cannot be filled with zero or evaluation runtime')
    require(history['status'] == 'HISTORICAL_PHY_EVENT_WINDOWS_READ_ONLY'
        and history['new_packet_decodes'] == 0 and history['standalone_PHY_measured'] is False
        and history['compatible_for_end_to_end_sum'] is False, 'Historical PHY scope changed')
    seen = set(); directions = set()
    for row in summary['components']:
        slot, side, name = int(row['development_slot']), row['side'], row['component']
        key = (slot, side, name)
        require(0 <= slot < 18 and side in ('TX', 'RX') and key not in seen, 'Duplicate/unknown component')
        require(row['source_count'] == 16 and row['repetitions_per_source'] == 3
            and row['warmups_excluded'] is True and row['inclusive'] ==
            (name in ('TX_source_total', 'RX_source_and_image_total')), 'Component averaging/nesting changed')
        require(all(math.isfinite(float(row[k])) and float(row[k]) >= 0
            for k in ('mean_seconds', 'median_source_seconds')), 'Invalid cost')
        seen.add(key); directions.add((slot, side))
    require(directions == {(s, d) for s in range(18) for d in ('TX', 'RX')}, 'Incomplete eighteen-point cost grid')
    for row in history['rows']:
        require(row['timing_scope'] == 'CONCURRENT_CPU_EVENT_WINDOW_WITH_BOOKKEEPING'
            and row['exclusive_PHY_latency'] is False and row['callback_only'] is False,
            'Historical concurrent event cannot become exclusive PHY latency')
    return dict(status='H_COMPONENTS_ONLY', components=[copy.deepcopy({k: v for k, v in row.items()
            if k != 'source_means'}) for row in summary['components']],
        historical_PHY_rows=copy.deepcopy(history['rows']), P_online_cost='NOT_MEASURED',
        MAIN_online_cost='NOT_AVAILABLE', end_to_end_latency=None,
        nested_totals_must_not_be_added=True, historical_PHY_must_not_be_added=True,
        scope='Instrumented, synchronized, uncached source/visual components; neither optimized production nor end-to-end latency')


def project(hp, h18, timing=None):
    """Logical projection only; callers must separately admit normal closures and file SHA maps."""
    done = hp['completion']; hd = h18['completion']
    require(done['status'] == 'H_P_FIXED_SOURCE_COMPARISON_COMPLETE_MAIN_PENDING'
        and done['source_count'] == 100 and done['H_frames'] == 5400 and done['P_frames'] == 600
        and done['comparison_count'] == 18 and done['exact_original_pixel_pairs'] == 100
        and done['noise_seeds'] == NOISE and done['original_rows_modified'] is False
        and done['policy_selection'] is False and done['MAIN_complete'] is False
        and done['H_success_evaluated'] is False, 'Completed fixed H/P bridge required; MAIN remains missing')
    require(hd['status'] == 'H18_DEVELOPMENT_DESCRIPTIVE_AGGREGATION_COMPLETE'
        and hd['source_count'] == 100 and hd['frame_count'] == 5400 and hd['points'] == 18
        and hd['comparison_count'] == 26 and hd['policy_selection'] is False, 'Original H-only summary required')
    catalog = point_catalog(h18['point_catalog'])
    points = hp['point_identity']; expected_points = set(catalog) | {'P1024_SNR_13', 'P1024_SNR_19'}
    require(set(points) == expected_points, 'All eighteen H and two P points required')
    metrics = set(METRICS); admission = hp['common_metric_admission']
    require(admission['status'] == 'FROZEN_H_P_COMMON_METRIC_ADMISSION' and admission['used_for_selection'] is False,
            'Original precomparison admission required')
    require({m for m, r in admission['metrics'].items() if r['status'] == 'ADMITTED'} == metrics
        and {m for m, r in admission['metrics'].items() if r['status'] == 'MISSING_IN_P'} == set(MISSING),
        'Do not fill the three original P missing metrics')
    expected_pairs = {(p, f"P1024_SNR_{int(r['snr_db'])}") for p, r in catalog.items()}
    def signature(row):
        return (row['point_id'], int(row['development_slot']), int(row['snr_db']),
                row['role'], row['arm'], row['candidate_id'])
    require(hp['comparison_plan']['status'] == 'FROZEN_H18_TO_SAME_SNR_P18_PAIRS_BEFORE_SCORE_ROWS'
        and {signature(row) for row in hp['comparison_plan']['H_catalog']} ==
            {signature(row) for row in catalog.values()}, 'Original H/P and H-only policy catalogs differ')
    require({(r['method'], r['reference']) for r in hp['comparison_plan']['comparisons']} == expected_pairs
        and len(hp['comparison_plan']['comparisons']) == 18, 'Only predeclared eighteen H-P pairs allowed')
    for p, spec in points.items():
        branch = 'H' if p in catalog else 'P'
        snr = int(catalog[p]['snr_db']) if p in catalog else int(p.rsplit('_', 1)[1])
        require(spec['branch'] == branch and int(spec['snr_db']) == snr
            and sequence(spec['noise_seeds']) == NOISE[branch], 'Point/noise identity changed')
    seen = set()
    for row in hp['summary']:
        key = (row['point_id'], row['metric']); require(key not in seen, 'Duplicate mean')
        require(key[0] in points and key[1] in metrics, 'Undeclared mean')
        spec = points[key[0]]; require(row['branch'] == spec['branch'] and int(row['snr_db']) == int(spec['snr_db'])
            and row['metric_identity'] == spec['metric_identity'][key[1]]
            and sequence(row['noise_seeds']) == NOISE[spec['branch']], 'Mean identity differs')
        interval(row); seen.add(key)
    require(seen == {(p, m) for p in points for m in metrics}, 'Complete 360 summary cells required')
    seen = set()
    for row in hp['paired']:
        a, b, m = row['method'], row['reference'], row['metric']; key = (a, b, m)
        require((a, b) in expected_pairs and m in metrics and key not in seen, 'New/duplicate comparison forbidden')
        require(int(row['snr_db']) == int(points[a]['snr_db']) == int(points[b]['snr_db'])
            and row['metric_identity'] == points[a]['metric_identity'][m] == points[b]['metric_identity'][m]
            and row['delta_definition'] == 'method minus reference'
            and sequence(row['method_noise_seeds']) == NOISE['H']
            and sequence(row['reference_noise_seeds']) == NOISE['P']
            and row['frame_level_noise_pairing_claimed'] in (False, 'False'), 'Original source-level H-P pairing changed')
        interval(row); seen.add(key)
    require(seen == {(a, b, m) for a, b in expected_pairs for m in metrics}, 'Complete 324 paired cells required')
    for name, den in (('tx_source_scale', 100), ('rx_received_scale', 300), ('receiver_status', 300)):
        count_table(h18[name], catalog, den)
    return dict(status='PURE_REPORT_PROJECTION_ONLY_REQUIRES_EXTERNAL_NORMAL_CLOSURE_AND_SHA_ADMISSION',
        normal_closure_verified_by_this_module=False, new_bootstrap_computed=False, new_pairs_created=False,
        MAIN='MISSING_NOT_ADMITTED', final_system_conclusion='UNRESOLVED_MAIN_NOT_COMPLETE',
        H_success_evaluated=False, point_catalog=copy.deepcopy(h18['point_catalog']),
        summary=copy.deepcopy(hp['summary']), paired=copy.deepcopy(hp['paired']),
        common_metric_admission=copy.deepcopy(admission), missing_P_metrics=list(MISSING),
        tx_source_scale=copy.deepcopy(h18['tx_source_scale']), rx_received_scale=copy.deepcopy(h18['rx_received_scale']),
        receiver_status=copy.deepcopy(h18['receiver_status']), costs=costs(timing),
        interpretation=dict(raw64_whole='Whole-scale calibration winner only; not strongest raw64',
            raw64_partial='Retain both all-integer-K calibration controls; do not choose whole/partial using development',
            fixed_m7='Attribution controls, m7 K0 rate1/2; not a newly selected system',
            intervals='Predeclared descriptive pointwise95%; no multiplicity-adjusted global winner claim',
            noise='H6201/6202/6203 and P2001/2002/2003 retained; pair source means, not noise frames',
            precision='H float64 PSNR; retained P float32 PSNR. Preserve per-metric batch/precision provenance.'))


def markdown(view):
    """Readable table draft; full precision and every interval remain in projected tables."""
    require(view['normal_closure_verified_by_this_module'] is False and view['MAIN'] == 'MISSING_NOT_ADMITTED',
            'Pure projector must not manufacture a final completion claim')
    means = {(r['point_id'], r['metric']): r for r in view['summary']}
    deltas = {(r['method'], r['metric']): r for r in view['paired']}
    labels = dict(zip(ROLES, ('整尺度校准策略', '部分尺度 raw64 校准对照', '固定 m7 归因对照')))
    text = ['# H/P 指标与运行成本汇总草稿', '',
        '**MAIN 尚未完成；本表不能给出 H 相对完整主系统的最终结论。**', '',
        '同一 100 源。保留全部 18 个冻结 H 点，不按开发集挑选赢家。'
        '整尺度 H64-R 仅为整尺度校准赢家；两条部分尺度 raw64 对照并列保留。', '',
        'H 使用 6201–6203，P 使用原 2001–2003；各自先按源平均三次噪声，再配对。'
        '区间直接引用原 10000 次源级 bootstrap（种子 2026100605），为未作多重比较校正的描述性点级 95% 区间。', '',
        'H 的 PSNR 保留 float64 计算，P 保留原 float32 标量。指标模型、定义、预处理与批量差异见原准入表。'
        'P 缺少 MSE、ResNet 预测概率和 confidently_wrong，保持缺测。', '']
    for role in ROLES:
        text += ['## ' + labels[role], '',
            '| SNR | 冻结点 | PSNR | LPIPS | CLIP | ConvNeXt 原图预测一致率 | PSNR 差值 H−P [95% CI] |',
            '|---:|---|---:|---:|---:|---:|---|']
        for item in sorted(view['point_catalog'], key=lambda r: int(r['development_slot'])):
            if item['role'] != role:
                continue
            p = item['point_id']; d = deltas[p, 'psnr_db']
            vals = [f"{float(means[p, m]['mean']):.5f}" for m in
                ('psnr_db', 'lpips_alex', 'clip_image_cosine', 'convnext_source_prediction_agreement')]
            text.append(f"| {item['snr_db']} | {p}: {item['arm']} | " + ' | '.join(vals)
                + f" | {float(d['mean']):+.5f} [{float(d['ci_low']):+.5f}, {float(d['ci_high']):+.5f}] |")
        text += ['']
    text += ['## P 参考点', '', '| SNR | PSNR | LPIPS | CLIP | ConvNeXt 原图预测一致率 |',
        '|---:|---:|---:|---:|---:|']
    for snr in (13, 19):
        vals = [f"{float(means[f'P1024_SNR_{snr}', m]['mean']):.5f}" for m in
            ('psnr_db', 'lpips_alex', 'clip_image_cosine', 'convnext_source_prediction_agreement')]
        text.append(f'| {snr} | ' + ' | '.join(vals) + ' |')
    text += ['', '全部 18 项共同指标及其均值区间、H−P 配对区间保留在逐指标表。'
        '原 H 内部 26 组对照继续作为独立附表，不增加 H64-A 对部分尺度 raw64 的新配对检验。', '',
        '## 尺度回退与接收结果', '',
        'TX 码长/尺度回退按每点 100 源统计；RX 实际接受尺度及失败按每点 300 帧统计。'
        '接收拒绝不算作尺度 0，也不从分母删除。固定 m7 对照用于归因。', '', '## 运行成本', '']
    if view['costs']['status'] == 'PENDING_NORMAL_TIMING_CLOSURE':
        text += ['在线组件计时尚待正常闭合，暂不填数。']
    else:
        text += ['H 组件表保留固定 16 源×18 点、6201 噪声、1 次预热及 3 次实测的原统计。'
            'TX/RX 总计是包含子项的计时，不与子项相加；VAR 概率与 CDF 子项单列。',
            '历史 PHY 时间窗含并发与记账开销，只作独立诊断，不与 GPU 组件拼成端到端时间。']
    text += ['P 的在线成本缺测，MAIN 的在线成本尚缺；不计算 H/P 加速比。'
        '当前没有独占完整 PHY 或端到端延迟，不用离线评测耗时代替。', '',
        '本文件由纯表格投影生成；正式输出前仍须外部核验各阶段正常退出及全部输入输出 SHA。', '']
    return '\n'.join(text)
