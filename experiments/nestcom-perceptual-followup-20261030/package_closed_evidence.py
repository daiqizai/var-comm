"""Package closed E1/E3/E4 evidence; no model, PHY, or statistics reruns."""
from followup_common import *
from collections import Counter, defaultdict
import shutil


def seal_closed(role):
    result = closed(role)
    control = L / (role + '_control')
    owner = read(control / 'owner.json')['owner']
    assert ident(owner['pid']) is None, ('Owner still exists', role)
    assert not (L / role / 'failure.json').exists(), role
    return dict(completion=pin(L / role / 'completion.json'),
                actual_wait=pin(control / 'actual_wait.json'),
                owner_gone=True, child_gone=True, result=result)


def copy_exact(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        f.write(source.read_bytes())
    assert sha(source) == sha(target)
    return dict(source=pin(source), output=pin(target), transformation='byte copy')


def main():
    roles = ['inventory_core', 'inventory_external', 'E1_core_dists',
             'E1_external_dists', 'E1_external_clip', 'E1_statistics',
             'E3_phy', 'E3_rx', 'E3_three', 'E3_clip', 'E3_dists', 'E3_statistics',
             'E4_interventions', 'E4_three', 'E4_clip', 'E4_dists', 'E4_statistics']
    receipts = {role: seal_closed(role) for role in roles}
    root = D / 'closed_evidence_v1'
    root.mkdir(exist_ok=False)
    proto = read(D / 'protocol.json')
    manifest = dict(status='E1_E3_E4_CLOSED_ONLY', protocol=pin(D / 'protocol.json'),
                    baseline_HEAD=proto['baseline_HEAD'], closed_roles=receipts,
                    new_model_calls=0, new_PHY=0, bootstrap_reruns=0,
                    E2_E5_E6_status='NOT_CLAIMED_COMPLETE_BY_THIS_EXPORT', files=[])
    for name in ('E1_dists', 'E3_granularity', 'E4_receiver_interventions'):
        stage = {'E1_dists':'e1', 'E3_granularity':'e3',
                 'E4_receiver_interventions':'e4'}[name]
        (root / stage).mkdir()
        save(root / stage / (name + '_protocol.json'),
             dict(parent=pin(D / 'protocol.json'), design=proto['design'][name],
                  source_sets=pin(D / 'source_sets.json')))
    frames = rows(D / 'e1/frame_metrics.csv')
    assert len(frames) == 66000
    keys = ('method', 'N', 'snr_db', 'source_id', 'channel_repeat')
    assert len({tuple(v[k] for k in keys) for v in frames}) == 66000
    fields = ['source_id', 'source_index', 'method', 'N', 'snr_db',
              'channel_repeat', 'noise_seed', 'generation_seed',
              'reference_sha256', 'pixel_sha256', 'dists_pair_key',
              'receiver_status', 'metric_origin', 'clip_origin', *METRICS]
    csvout(root / 'e1/dists_per_frame.csv',
           [{k:v.get(k,'') for k in fields} for v in frames])
    native = {}
    for v in frames:
        k = v['dists_pair_key']
        info = {s:v.get(s,'') for s in ['reference_sha256', 'pixel_sha256',
                'image', 'archive', 'image_key', 'image_slot', 'hash_rule', 'native_status']}
        if k in native:
            assert all(native[k][s] == info[s] for s in ['reference_sha256','pixel_sha256'])
        else:
            native[k] = dict(dists_pair_key=k, **info)
    csvout(root / 'e1/native_output_inventory.csv', list(native.values()))
    manifest['files'].append(dict(source=pin(D/'e1/frame_metrics.csv'),
        output=pin(root/'e1/dists_per_frame.csv'),
        transformation='Column projection; original scalar strings retained; native payload paths moved to keyed inventory',
        rows=66000, unique_native_metric_pairs=len(native)))
    mappings = {
        'e1/source_means.csv':'e1/dists_source_means.csv',
        'e1/summary.csv':'e1/full_five_metric_summary.csv',
        'e1/statistics_identity.json':'e1/statistics_identity.json',
        'dists_model_and_preprocess.json':'e1/dists_model_and_preprocess.json',
        'e1/failure_counts_native_v2.csv':'e1/failure_counts_native_v2.csv',
        'e1/failure_count_annotation.json':'e1/failure_count_annotation.json',
        'e1/capacity_summary.csv':'e1/capacity_summary.csv',
        'e1/frozen_policy_metadata.json':'e1/frozen_policy_metadata.json',
        'e3/frame_metrics.csv':'e3/granularity_per_frame.csv',
        'e3/source_means.csv':'e3/granularity_per_source.csv',
        'e3/summary.csv':'e3/granularity_summary.csv',
        'e3/paired_deltas.csv':'e3/granularity_paired_deltas.csv',
        'e3/statistics_identity.json':'e3/statistics_identity.json',
        'e3/actual_compute_and_reuse.json':'e3/actual_compute_and_reuse.json',
        'e3/REPORT_zh.md':'e3/REPORT_zh.md',
        'e4/frame_metrics.csv':'e4/receiver_intervention_per_frame.csv',
        'e4/source_means.csv':'e4/receiver_intervention_per_source.csv',
        'e4/summary.csv':'e4/receiver_intervention_summary.csv',
        'e4/paired_deltas.csv':'e4/receiver_intervention_paired_deltas.csv',
        'e4/statistics_identity.json':'e4/statistics_identity.json',
        'e4/actual_compute_and_reuse.json':'e4/actual_compute_and_reuse.json',
        'e4/REPORT_zh.md':'e4/REPORT_zh.md'}
    for src,dst in mappings.items():
        manifest['files'].append(copy_exact(D/src, root/dst))
    summary, paired = rows(D/'e1/summary.csv'), rows(D/'e1/paired_deltas.csv')
    assert len(summary) == len(paired) == 220
    csvout(root/'e1/dists_summary.csv', [v for v in summary if v['metric']=='dists'])
    csvout(root/'e1/dists_paired_deltas.csv', [v for v in paired if v['metric']=='dists'])
    copy_exact(D/'e1/paired_deltas.csv', root/'e1/full_five_metric_paired_deltas.csv')
    efficiency = rows(D/'e3/efficiency.csv')
    e3frames = rows(D/'e3/frame_metrics.csv')
    cost = []
    for v in efficiency:
        g = [x for x in e3frames if x['method']==v['method']]
        assert len(g)==300
        cost.append(dict(**v, new_independent_terminations=sum(int(x['new_independent_terminations']) for x in g),
                         new_prior_forwards_for_fitting=sum(int(x['new_prior_forwards_for_fitting']) for x in g),
                         counter_scope='quality replay reuses terminated streams; fresh cost measured only by E5'))
    csvout(root/'e3/granularity_cost_counters.csv', cost)
    save(root/'e4/final_scale_identity_check.json', read(D/'e4/statistics_identity.json')['final_scale_audit'])
    counts = Counter(v['clip_origin'] for v in frames)
    save(root/'e1/actual_compute_and_reuse.json', dict(logical=66000,
         newly_scored_DISTS_pairs=12589+15664,
         globally_unique_DISTS_pair_keys=len(native),
         DISTS_pair_calls_repeated_across_stage_boundaries=12589+15664-len(native),
         DISTS_logical_positions_above_actual_pair_calls=66000-12589-15664,
         new_reconstructions=0, new_PHY=0, original_existing_four_metric_scalars_preserved=True,
         exception='Previously absent external 7-dB CLIP scored in matching Torch2.10 route',
         CLIP_origin_counts=dict(counts), new_CLIP_pairs=4011,
         CLIP_call_ledger=receipts['E1_external_clip']['result']['ledger'],
         actual_receipts={r:receipts[r] for r in roles if r.startswith('E1_')}))
    lines=['E1 完整五指标核对与解释', '',
           '状态：已实际闭合。66,000 个逻辑结果，44 个方法/预算/SNR 单元，每单元 500 源×3 原信道重复。',
           '每源先平均，再按原 500 源 bank 给逐点均值和直接配对区间。这里不重新运行统计。',
           f'两个阶段合计实际 DISTS 评分 28,253 对，各阶段内去重；全局唯一评分键 {len(native)}，阶段间重复评分键数量 {12589+15664-len(native)}。保留实际账目，不追认全局去重。外部 7 dB 原缺失 CLIP 补 4,011 对，其他既有四指标原值保留。',
           '全部图像来自原生浮点结果，无新重建或无线试验。原灰图计数只看存储指针，漏了 11 个实际保存为数组的 BPG 灰图；独立像素身份修订保留原始文件。', '',
           '解释：NeST 对 WHOLE 在七个点的五指标配对区间均有利。对旧 Quarter，DISTS/DreamSim 六点区间支持，N1024/7 dB 跨零。',
           '对 Ada 的距离优势集中在低预算：N1024/10 和 N512/19 两距离均有利于 Ada；N1024/19 DISTS 有利于 Ada、DreamSim 跨零；N1024/7 DISTS 有利于 NeST、DreamSim 跨零。',
           '完整系统间 tokenizer、prior、decoder、条件和历史噪声身份不同，不解释为表示顺序的独立因果效应。来源区间是逐点、未做多重比较校正。', '',
           '以下为新 NeST 对 WHOLE、Quarter、Ada 的完整五指标配对结果；PSNR/CLIP 正差有利，距离负差有利。']
    for v in paired:
        if v['candidate']=='final_NeST' and v['reference'] in ('WHOLE_final_receiver','original_NeST','Ada_frozen'):
            lines.append(f"N={v['N']} SNR={v['snr_db']} NeST−{v['reference']} {v['metric']}: {v['mean_delta']} [{v['ci95_low']}, {v['ci95_high']}]")
    (root/'e1/REPORT_zh.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    manifest['large_unmodified_provenance'] = [pin(D/x) for x in
        ['e1/frame_metrics.csv','input_inventory_core.csv','input_inventory_external.csv']]
    manifest['storage_note'] = 'Large original CSV provenance remains at its pinned PFS path; compact per-frame and keyed native inventory are publication copies, not recomputed science.'
    manifest['exports'] = [pin(p) for p in sorted(root.rglob('*')) if p.is_file()]
    save(root/'delivery_manifest.json', manifest)
    print(json.dumps(dict(output=str(root), e1_logical=66000, native_pairs=len(native),
                          e3_logical=1200,e4_logical=1800,new_scientific_calls=0)))


if __name__=='__main__':
    main()
