"""Source-level summaries and registered paired contrasts for historical metrics."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import numpy as np
from history_common import read, write, sha, identity, verify, write_csv, normalize_metadata, original_fields

METRICS = ('psnr_db', 'lpips_alex', 'dino_cosine',
    'new_dinov2_vitl14_cosine', 'new_clip_image_cosine', 'new_dists',
    'new_dreamsim', 'new_ms_ssim', 'new_resnet50_top1_label',
    'new_resnet50_top1_source_prediction')
LOWER = {'lpips_alex', 'new_dists', 'new_dreamsim'}
EXTRA_METRICS = ('mse', 'dino_mismatched', 'dino_specificity', 'latent_sq_err_base',
    'latent_sq_err_post', 'latent_sq_err_final', 'latent_sq_error', 'latent_squared_error',
    'fused_latent_squared_error', 'zero_erasure_proxy_sq_error', 'F_error', 'F_mse',
    'f_error', 'f_mse', 'recovery_mse', 'feature_mse')
FIELDS = ('study', 'method', 'model_id', 'model_identity_sha256', 'training_seed', 'N', 'snr_db', 'snr_definition', 'phy',
    'decoder', 'condition', 'projection', 'profile', 'oracle', 'reference_only', 'label_conditioned')


def snr_label(value):
    """Normalize numeric group labels only; raw source fields stay untouched."""
    text = str(value)
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    return format(number.normalize(), 'f') if number.is_finite() else text


def context(row):
    m = normalize_metadata(row, json.loads(row['history_metadata_json']))
    values = dict(study=row['history_study'], method=m['method_id'],
        model_id=m.get('checkpoint_sha256') or m.get('model_id') or row.get('checkpoint_sha256') or row.get('model_id') or '',
        model_identity_sha256=m.get('model_identity_sha256', ''),
        training_seed=m.get('training_seed', row.get('training_seed', '')),
        N=m.get('N', row.get('N', '')),
        snr_db='' if m.get('channel_uses_not_applicable') is True else
            snr_label(row.get(m.get('snr_field', 'snr_db'), m.get('snr_db', ''))),
        snr_definition=m.get('snr_definition') or m.get('snr_field', 'snr_db'),
        phy=row.get('phy_family') or row.get('modulation') or row.get('mcs') or m.get('phy', ''),
        decoder=m.get('decoder') or m.get('decoder_id') or row.get('decoder_id', ''),
        condition=m.get('condition') or row.get('condition') or m.get('scope', ''),
        projection=row.get('projection', m.get('projection', '')),
        profile=m.get('profile') or row.get('profile') or row.get('noise_profile') or '',
        oracle=bool(m.get('oracle', False)), reference_only=bool(m['reference_only']),
        label_conditioned=m['label_conditioned'],
        classification_main_eligible=bool(m.get('classification_main_eligible',
            not m['label_conditioned'] and not m['reference_only'] and not m['oracle'] and m.get('is_main_conclusion', True))),
        output_role=m.get('output_role', 'reference' if m['reference_only'] else 'main'))
    if not values['method']:
        raise RuntimeError('Historical method name missing')
    return values


class Bootstrap:
    def __init__(self, replicates=10000, seed=20261002):
        self.replicates, self.seed, self.cache = replicates, seed, {}

    def interval(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise RuntimeError('Bootstrap requires finite, complete source means')
        n = len(values)
        if n not in self.cache:
            self.cache[n] = np.random.default_rng(self.seed).integers(0, n, size=(self.replicates, n))
        estimates = values[self.cache[n]].mean(axis=1)
        lo, hi = np.quantile(estimates, [.025, .975])
        return dict(mean=float(values.mean()), ci_low=float(lo), ci_high=float(hi), sources=n)


def collect(root, jobs, queue=None):
    from history_inheritance import study_paths, verify_inherited
    queue = queue or {}
    groups, contexts, input_bindings = defaultdict(list), {}, {}
    for job in jobs:
        study = job['study']
        study_out, study_result = study_paths(root, queue, study)
        if study in queue.get('inherited_studies', {}):
            verify_inherited(root, queue, study)
            item = queue['inherited_studies'][study]
            input_bindings[item['receipt_path']] = item['receipt_sha256']
        p = study_out/'completion.json'
        done = read(p)
        if (done.get('status') != 'HISTORICAL_STUDY_METRICS_COMPLETE' or done.get('parity_passed') is not True
                or done.get('synthetic') is not False or done.get('training_updates') != 0
                or done.get('policy_selection_updates') != 0):
            raise RuntimeError('Incomplete historical study: ' + study)
        verify(done['inputs']); verify(done['outputs'])
        table = study_result/'metrics_per_frame.csv'
        count = 0; seen = set()
        with table.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['history_replay_parity_passed'] != 'True':
                    raise RuntimeError('Unverified historical metric row')
                if row['history_row_id'] in seen:
                    raise RuntimeError('Duplicate historical row identity in exported table')
                seen.add(row['history_row_id'])
                if row['history_study'] != study:
                    raise RuntimeError('Historical table study differs')
                c = context(row)
                key = identity(c)
                contexts[key] = c
                groups[key].append(row)
                count += 1
        if count != done['frames'] or len({r['history_source_id'] for rows in groups.values()
                for r in rows if r['history_study'] == study}) != done['sources']:
            raise RuntimeError('Historical table row count differs')
        input_bindings[str(p)] = sha(p)
        input_bindings[str(table)] = sha(table)
    return groups, contexts, input_bindings


def group_means(rows, metric):
    bysource = defaultdict(list)
    missing = 0
    for row in rows:
        metadata = json.loads(row['history_metadata_json'])
        column = metadata.get('original_metric_columns', {}).get(metric, metric) if not metric.startswith('new_') else metric
        value = row.get(column, '')
        if value == '':
            missing += 1; continue
        bysource[row['history_source_id']].append(float(value))
    if missing == len(rows):
        return None
    if missing:
        raise RuntimeError('Partial metric coverage cannot be silently dropped: '+metric)
    if not all(np.isfinite(value).all() for value in bysource.values()):
        raise RuntimeError('Nonfinite metric in historical table')
    return {source: float(np.mean(values)) for source, values in sorted(bysource.items())}


def metrics_for(rows):
    mapped = set()
    for row in rows:
        mapped.update(json.loads(row['history_metadata_json']).get('original_metric_columns', {}))
    # Arbitrary numeric policy/budget fields are never treated as quality metrics.
    return list(METRICS)+[m for m in EXTRA_METRICS if m in mapped or any(r.get(m, '') != '' for r in rows)]


def direction(metric):
    if metric == 'dino_mismatched':
        return 'diagnostic'
    return 'lower' if metric in LOWER or metric in EXTRA_METRICS and metric != 'dino_specificity' else 'higher'


def reference_map(rows):
    result = {}
    for row in rows:
        source, fingerprint = row['history_source_id'], row['history_reference_sha256']
        if source in result and result[source] != fingerprint:
            raise RuntimeError('A historical source has multiple target pixel identities')
        result[source] = fingerprint
    return result


def coverage_rows(groups, contexts):
    rows = []
    for key, frames in groups.items():
        references = reference_map(frames)
        counts = defaultdict(int)
        for frame in frames:
            counts[frame['history_source_id']] += 1
        energy=[]
        for frame in frames:
            metadata=json.loads(frame['history_metadata_json'])
            if metadata.get('channel_uses_not_applicable') is True:
                continue
            val=next((frame[k] for k in ('E','actual_total_energy','total_energy') if frame.get(k) not in ('',None)), metadata.get('E'))
            if val not in (None, ''):
                energy.append(float(val))
        if energy and (len(energy)!=len(frames) or not np.isfinite(energy).all()):
            raise RuntimeError('Partial/nonfinite resource energy ledger')
        rows.append(dict(group_id=key, **contexts[key], sources=len(references), frames=len(frames),
            repeats_min=min(counts.values()), repeats_max=max(counts.values()),
            original_metrics_available=';'.join(m for m in metrics_for(frames) if not m.startswith('new_') and group_means(frames,m) is not None),
            source_population_sha256=identity(sorted(references.items())),
            E_mean=float(np.mean(energy)) if energy else '', E_min=min(energy) if energy else '',
            E_max=max(energy) if energy else '', energy_frames=len(energy)))
    return rows


def unified_index(root):
    result = root/'results/unified_metrics_20261002'
    paths = [result/n for n in ('metrics_summary.csv', 'metrics_paired_intervals.csv', 'METRICS_REPORT.md')]
    if not all(p.is_file() for p in paths):
        raise RuntimeError('Completed original six-study report/index is missing')
    return [dict(scope='N512/N1024/M1/M1_RATE/M2_ORACLE/M2_ACTUAL', kind=p.stem,
                 path=str(p), sha256=sha(p), recomputed=False) for p in paths]


def matches(actual, selector):
    return all(str(actual.get(k, '')) == str(v) for k, v in selector.items())


def analyse(root, queue_path):
    root = Path(root)
    queue = read(queue_path)
    verify(queue['source_bindings'])
    coverage = queue['coverage_manifest']
    if sha(coverage['path']) != coverage['sha256']:
        raise RuntimeError('Historical coverage manifest changed')
    groups, contexts, inputs = collect(root, queue['jobs'], queue)
    coverage_spec = read(coverage['path'])
    if 'selected_studies' in coverage_spec:
        expected_studies = coverage_spec['selected_studies']
        if set(expected_studies) != {j['study'] for j in queue['jobs']}:
            raise RuntimeError('Selected scope and execution roster differ')
        for study, spec in expected_studies.items():
            frames = [r for k,g in groups.items() if contexts[k]['study'] == study for r in g]
            if (len(frames) != spec['frames']
                    or len({r['history_source_id'] for r in frames}) != spec['sources']):
                raise RuntimeError('Selected scope coverage is incomplete: '+study)
    prior_index = unified_index(root)
    inputs.update({x['path']:x['sha256'] for x in prior_index})
    inputs[str(coverage['path'])] = coverage['sha256']
    boot = Bootstrap()
    summary, source_means, means = [], [], {}
    for key, rows in groups.items():
        reference_map(rows)
        for metric in metrics_for(rows):
            values = group_means(rows, metric)
            if values is None:
                continue
            means[key, metric] = values
            summary.append(dict(group_id=key, **contexts[key], metric=metric,
                direction=direction(metric), frames=len(rows), **boot.interval(list(values.values()))))
            source_means.extend(dict(group_id=key, source_id=source, metric=metric, mean=value) for source, value in values.items())
    paired = []
    # Contrasts are fixed before execution. Matching never uses measured scores.
    for contrast in queue.get('contrasts', []):
        left = [k for k, c in contexts.items() if matches(c, contrast['A'])]
        right = [k for k, c in contexts.items() if matches(c, contrast['B'])]
        if not left or not right:
            raise RuntimeError('A registered historical contrast has no rows')
        matched = 0
        for a in left:
            for b in right:
                ca, cb = contexts[a], contexts[b]
                match_fields = set(contrast.get('match_fields', ['N', 'snr_db', 'decoder'])) | {'snr_definition'}
                if any(str(ca[f]) != str(cb[f]) for f in match_fields):
                    continue
                refs_a, refs_b = reference_map(groups[a]), reference_map(groups[b])
                if refs_a != refs_b:
                    raise RuntimeError('Registered paired contrast has different source images or preprocessing')
                matched += 1
                for metric in metrics_for(groups[a]):
                    x, y = means.get((a, metric)), means.get((b, metric))
                    if x is None or y is None:
                        continue
                    if set(x) != set(y):
                        raise RuntimeError('Registered paired contrast lacks matched sources')
                    interval = boot.interval([x[s] - y[s] for s in sorted(x)])
                    paired.append(dict(name=contrast['name'], group_A=a, group_B=b, metric=metric,
                        delta_mean=interval.pop('mean'), **interval,
                        pairing='source image; original noise repeats averaged within source'))
        if not matched:
            raise RuntimeError('Registered contrast has no compatible resource/SNR groups: '+contrast['name'])
    out = root / 'results/historical_metrics_r3_20261003'
    write_csv(out / 'summary.csv', summary)
    curve_files=[]
    if 'selected_studies' in coverage_spec:
        from history_curves import generate
        curve_files=[Path(p) for p in generate(root,out/'summary.csv',
            root/'results/unified_metrics_20261002/metrics_summary.csv',out/'figures')]
    write_csv(out / 'per_source_means.csv', source_means)
    covered = coverage_rows(groups, contexts)
    write_csv(out / 'coverage.csv', covered)
    write(out / 'prior_six_study_index.json', dict(status='INDEX_ONLY', recomputed=False, files=prior_index))
    if paired:
        write_csv(out / 'paired.csv', paired)
    else:
        write(out / 'paired_status.json', dict(status='NO_REGISTERED_CONTRASTS', paired_conclusions=False))
    report = ['# 历史已完成方法：补充独立指标', '',
        f'本登记队列已完成 {len(queue["jobs"])} 个研究、{len(groups)} 个方法/资源组、{sum(map(len, groups.values()))} 行评测。', '',
        ('R3逐源验证并只读继承R2前七项45,000行，新增评测剩余五项11,700行。原科学行及模型身份保持原登记，继承证明见provenance/inherited_R2。'
         if queue.get('inherited_studies') else '本报告沿用各项原始登记与完成证明。'), '',
        '**本报告完成用户2026-10-03筛选后的范围：必补项目先完成，再补第5、6、8项的指定工作点。**',
        '完整范围以冻结 coverage_manifest 为准。第7、10、12–16、18–20、22项及新低带宽外部模型训练已排除；不能把这些项目写成已补评。', '',
        '## 覆盖与结果', '',
        '[全部指标与区间](../results/historical_metrics_r3_20261003/summary.csv) · '
        '[源图均值](../results/historical_metrics_r3_20261003/per_source_means.csv) · '
        '[逐组覆盖](../results/historical_metrics_r3_20261003/coverage.csv)', '',
        '[质量—资源曲线与逐点来源](../results/historical_metrics_r3_20261003/figures/curve_provenance.json)', '',
        '| 研究 | 方法数 | 资源/条件组 | 源图数 | 原始结果行数 |',
        '|---|---:|---:|---:|---:|']
    for study in sorted({c['study'] for c in contexts.values()}):
        keys = [k for k,c in contexts.items() if c['study'] == study]
        sources = {r['history_source_id'] for k in keys for r in groups[k]}
        report.append(f'| {study} | {len({contexts[k]["method"] for k in keys})} | {len(keys)} | {len(sources)} | {sum(len(groups[k]) for k in keys)} |')
    report += ['', '## 比较口径', '',
        '所有旧模型、策略、源图、噪声及原始行保留；新指标列使用 new_ 前缀。物理信道 SNR 和等效 latent SNR 分组，不能把数值相同当作同一信道条件。',
        '区间先在每张源图内平均原始噪声重复，再按源图重采样 10,000 次。配对比较只执行登记的对照，要求源图和参考像素完全一致。区间不包括训练种子不确定性。',
        '分类主结论限无类别、非 oracle、非参考的主结果；带真实类别、D0、oracle 及诊断输出保持标记。',
        '新增指标为 DINOv2 ViT-L/14、OpenAI CLIP ViT-L/14 图像余弦、DISTS、DreamSim、MS-SSIM，以及独立 ResNet-50 V2 的真实标签准确率和原图预测一致率。原有DINO是DINOv2 ViT-S/14，LPIPS是AlexNet v0.1，版本记录在每行元数据中。',
        '已有 F 恢复误差和错配特异性指标在存在时纳入汇总；缺失项保持缺失。', '',
        '无信道参考每种输出只有100张源图，不复制三份噪声。m8无类别补全为本轮新增确定性推断，单列其来源；其余旧指标逐帧保留并核验。',
        '历史N2048/N3060/N4084数字最终策略为付费类别D_C；早期N3060使用D0。它们不能标成当前同Dc、无类别D_U曲线。SwinJSCC的作者假设边信息与共同计费口径分开。', '',
        '## 原六项研究索引', '',
        'N512、N1024、M1、M1_RATE、M2_ORACLE、M2_ACTUAL 的原评测见 '
        '[原补充指标报告](../results/unified_metrics_20261002/METRICS_REPORT.md)。'
        '本报告仅保存其文件与校验值索引，不重算、不假合并结果。', '',
        '本轮训练更新为 0，策略选择更新为 0；未启用新 holdout。KID 留待最终 holdout，FID 未评估。', '']
    report_path = root / 'reports/historical_metrics_r3_20261003.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(report), encoding='utf-8')
    paths = [out / 'summary.csv', out / 'per_source_means.csv', out/'coverage.csv',
             out/'prior_six_study_index.json', report_path]
    paths.extend(curve_files)
    paths.append(out / ('paired.csv' if paired else 'paired_status.json'))
    write(out / 'analysis_completion.json', dict(status='COMPLETE', synthetic=False, groups=len(groups),
        registered_queue_complete=True, selected_scope_complete='selected_studies' in coverage_spec, full_history_coverage_complete=False,
        frames=sum(map(len, groups.values())), paired_rows=len(paired), bootstrap_replicates=10000,
        inputs={**inputs, str(queue_path): sha(queue_path)}, outputs={str(p): sha(p) for p in paths}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--queue-registration', required=True)
    a = p.parse_args()
    analyse(a.root, a.queue_registration)
