"""Format the completed Kodak24 CSVs; no inference or statistical recomputation."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/generalization_kodak_20261009'
DATA = OUT / 'analysis_v1'
METHODS = ('VAR_UNCONDITIONAL', 'P1024', 'BPG_ADAPTIVE', 'SWIN80K')
METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine', 'convnext_top1_source_prediction')
NAMES = dict(VAR_UNCONDITIONAL='Proposed: partial-scale digital + unconditional VAR',
             P1024='Latent continuous JSCC', BPG_ADAPTIVE='Adaptive-resolution BPG + LDPC',
             SWIN80K='SwinJSCC-80k (adapted)')
TITLES = ('PSNR (dB)', 'LPIPS', 'DINOv2-L cosine', 'ConvNeXt agreement (%)')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def run():
    completion = json.loads((DATA / 'completion.json').read_text(encoding='utf8'))
    assert completion['status'] == 'KODAK24_SOURCE_STATISTICS_AND_PLOTS_COMPLETE_V1'
    assert completion['draws_calls'] == 1 and completion['bootstrap_seed'] == 2026100901
    assert completion['bootstrap_replicates'] == 10000
    metric_path = OUT / 'metrics_v1/completion.json'
    metric = json.loads(metric_path.read_text(encoding='utf8'))
    metric_binding = [v for k, v in completion['inputs'].items() if Path(k).name == 'completion.json']
    assert metric_binding == [sha(metric_path)]
    assert metric['status'] == 'KODAK24_FOUR_METRICS_COMPLETE_V1'
    assert metric['frame_count'] == 864 and metric['metric_rows'] == 3456
    assert metric['unique_reconstruction_scores'] == 580 and metric['reference_preparations'] == 24
    receipts = {m: json.loads((OUT / 'actual_receipts' / (m + '.json')).read_text(encoding='utf8')) for m in METHODS}
    for m in ('VAR_UNCONDITIONAL', 'P1024', 'SWIN80K'):
        r = receipts[m]
        assert r['status'] == 'COMPLETE' and r['frames'] == 216 and r['source_count'] == 24
        assert not r['true_labels_used'] and not r['policy_selection'] and r['training_updates'] == 0
        assert r['original_ledger_mutated'] is False
    bpg = receipts['BPG_ADAPTIVE']
    assert bpg['status'] == 'KODAK24_ADAPTIVE_BPG_ACTUAL_CHILDREN_WAIT_ZERO'
    assert bpg['actual_children_waited'] and bpg['worker_exit_codes'] == [0, 0]
    assert bpg['frame_count'] == bpg['decoded'] == 216 and bpg['source_count'] == 24
    assert bpg['root_budget_before'] == bpg['root_budget_after'] and bpg['independent_ledger']['unresolved'] == 0
    assert bpg['training_updates'] == bpg['calibration_calls'] == 0 and not bpg['policy_selection']
    calls = {m: receipts[m]['packet_attempts'] for m in ('VAR_UNCONDITIONAL', 'P1024', 'SWIN80K')}
    calls['BPG_ADAPTIVE'] = bpg['independent_ledger']['total']
    assert calls == dict(VAR_UNCONDITIONAL=432, P1024=0, SWIN80K=216, BPG_ADAPTIVE=432)
    assert receipts['VAR_UNCONDITIONAL']['VAR_null_embedding_calls'] == 216
    pins = {}
    for name in ['summary.csv', 'paired.csv', 'status_counts.csv']:
        path = DATA / name
        bound = [value for key, value in completion['outputs'].items() if Path(key).name == name]
        assert len(bound) == 1 and sha(path) == bound[0]
        pins[name] = bound[0]
    summary, paired = rows(DATA / 'summary.csv'), rows(DATA / 'paired.csv')
    assert len(summary) == 48 and len(paired) == 36
    s = {(r['method'], int(r['snr_db']), r['metric']): r for r in summary}
    p = {(r['reference'], int(r['snr_db']), r['metric']): r for r in paired}
    assert len(s) == 48 and len(p) == 36
    for table in [summary, paired]:
        for r in table:
            assert int(r['source_count']) == 24 and int(r['noise_count']) == 3 and int(r['frame_count']) == 72
            assert float(r['ci_low']) <= float(r['mean']) <= float(r['ci_high'])
    for (ref, snr, metric), r in p.items():
        assert r['method'] == 'VAR_UNCONDITIONAL' and r['delta_definition'] == 'method minus reference'
        assert r['frame_level_noise_pairing_claimed'] == 'False'
        assert math.isclose(float(r['mean']), float(s['VAR_UNCONDITIONAL', snr, metric]['mean']) -
                            float(s[ref, snr, metric]['mean']), abs_tol=1e-12)

    def formatted(r, ci=True, sign=False):
        metric = r['metric']; scale = 100 if metric == METRICS[3] else 1
        precision = 2 if metric in (METRICS[0], METRICS[3]) else 4
        def number(k):
            return format(float(r[k]) * scale, ('+' if sign else '') + f'.{precision}f')
        return number('mean') + (f" [{number('ci_low')}, {number('ci_high')}]" if ci else '')

    def delta(ref, snr, metric):
        return formatted(p[ref, snr, metric], sign=True)

    for metric in METRICS[2:]:
        assert all(float(r['ci_low']) > 0 for r in paired if r['metric'] == metric)
    assert float(p['P1024', 4, 'lpips_alex']['ci_low']) > 0
    assert float(p['SWIN80K', 19, 'psnr_db']['ci_low']) < 0 < float(p['SWIN80K', 19, 'psnr_db']['ci_high'])

    english = [
        'We evaluated all 24 Kodak images using one predetermined 256 by 256 center crop per image, '
        'N=1024, SNRs of 4, 10, and 19 dB, and three fixed noise realizations per source and method. '
        'All weights and source/channel policies remained frozen, and VAR completion used the unconditional null embedding. '
        'Across all three SNRs and all three reference methods, the source-paired 95% intervals were strictly positive '
        'for DINOv2-L similarity and ConvNeXt source-prediction agreement. Relative to latent continuous JSCC, the '
        'DINOv2-L gains were ' + ', '.join(formatted(p['P1024', snr, METRICS[2]], ci=False, sign=True) for snr in (4, 10, 19)) +
        ', and agreement gains were ' + ', '.join(formatted(p['P1024', snr, METRICS[3]], ci=False, sign=True) for snr in (4, 10, 19)) +
        ' percentage points, respectively. These metrics measure feature and prediction consistency with the source, not ground-truth semantic accuracy.',
        'The gains involve a pixel-fidelity tradeoff. At 4 dB, the proposed method had worse LPIPS than latent continuous JSCC '
        f"(difference {delta('P1024', 4, 'lpips_alex')}) and lower PSNR than all three references. "
        'Its LPIPS was lower than that of adaptive-resolution BPG + LDPC and adapted SwinJSCC-80k at every tested SNR, '
        'and lower than latent continuous JSCC at 10 and 19 dB. PSNR remained below adaptive-resolution BPG + LDPC '
        'at all three SNRs and below adapted SwinJSCC-80k at 4 and 10 dB. At 19 dB, PSNR exceeded latent continuous JSCC '
        f"by {delta('P1024', 19, 'psnr_db')} dB, while the difference from adapted SwinJSCC-80k was "
        f"{delta('SWIN80K', 19, 'psnr_db')} dB; the latter interval crosses zero and establishes neither a directional difference nor equivalence.",
        'Intervals are pointwise percentile source-bootstrap intervals from one shared set of 10,000 resamples '
        '(seed 2026100901), after averaging three noises within each source. No old result was re-bootstrapped. '
        'Swin at 19 dB is outside its training and calibration range. The 24 fixed center crops provide limited '
        'cross-collection evidence, not full-resolution Kodak evaluation or proof of non-overlap with pretrained-model '
        'training data. The intervals are not multiplicity-adjusted and do not establish universal superiority.'
    ]

    text = ['# Kodak24 frozen-method results', '', '## 中文结论', '',
        '- 这批结果支持特征一致性与源图预测一致性的改善，但不支持“全面胜出”。DINOv2-L 和 ConvNeXt 源图预测一致率相对三个基线、三个 SNR 的全部 9 项源级配对区间均严格高于零。它们是代理指标，不是真实语义标签准确率。',
        f"- 4 dB 必须保留的不利结果：本文相对潜空间连续 JSCC 的 LPIPS 差值为 {delta('P1024', 4, 'lpips_alex')}，正值表示更差；PSNR 相对连续 JSCC、BPG 和 Swin 分别为 {delta('P1024', 4, 'psnr_db')}、{delta('BPG_ADAPTIVE', 4, 'psnr_db')}、{delta('SWIN80K', 4, 'psnr_db')} dB。",
        '- 相对 BPG，本文三个 SNR 的 PSNR 均更低，同时 LPIPS、DINOv2-L 和源图预测一致率均更有利。相对潜空间连续 JSCC，10 dB 仍有 PSNR 损失，19 dB 四项指标的点态配对区间均有利。',
        f"- 19 dB 相对 Swin 的 PSNR 差值为 {delta('SWIN80K', 19, 'psnr_db')} dB，区间跨零，不能据此宣称胜出、持平或等价。Swin 的 19 dB 属于训练及校准范围外工作点。",
        '- 只有 24 张预先固定的中心裁剪图；每图三个噪声先平均，随后按来源图配对。不同方法没有共享同一次实际接收观测。不能扩展为原尺寸 Kodak、任意域泛化或预训练集完全无重叠的结论。',
        '', '## Manuscript-ready English', '']
    for paragraph in english:
        text += [paragraph, '']
    text += ['', '## Single-method means and existing 95% intervals', '',
             'All numbers below are formatted directly from `analysis_v1/summary.csv`. Agreement and its interval are multiplied by 100. No scientific value or interval is recomputed.', '',
             '| SNR (dB) | Method | PSNR (dB) | LPIPS | DINOv2-L cosine | Agreement (%) |',
             '|---:|---|---:|---:|---:|---:|']
    for snr in (4, 10, 19):
        for method in METHODS:
            text.append(f'| {snr} | {NAMES[method]} | ' + ' | '.join(formatted(s[method, snr, metric]) for metric in METRICS) + ' |')
    text += ['', '## Source-paired differences and existing 95% intervals', '',
             'Direction: proposed minus reference. Negative LPIPS favors proposed; positive PSNR, DINOv2-L, and agreement favor proposed. Agreement differences are percentage points (pp), not relative percentages. Conclusions use these paired intervals, not overlap between single-method intervals.', '',
             '| SNR (dB) | Reference | ΔPSNR (dB) | ΔLPIPS | ΔDINOv2-L | ΔAgreement (pp) |',
             '|---:|---|---:|---:|---:|---:|']
    for snr in (4, 10, 19):
        for ref in METHODS[1:]:
            text.append(f'| {snr} | {NAMES[ref]} | ' + ' | '.join(formatted(p[ref, snr, metric], sign=True) for metric in METRICS) + ' |')
    text += ['', '## Execution and failure accounting', '',
             'The four local actual completion receipts confirm 864 completed reconstruction frames and 1080 new PHY decoder calls: 432 proposed, 432 BPG, 216 Swin, and zero latent continuous JSCC. The proposed receipt records 216 audited VAR null-embedding calls, no true labels, and no training. BPG records two child exits of zero, 216 decoded frames, zero unresolved ledger attempts, no calibration, and an unchanged original root ledger. Actual scoring completed 580 unique reconstruction evaluations plus 24 source preparations; exact same-source reconstruction pixels were reused without dropping any frame. The metric completion, itself hash-bound by the analysis completion, records 3456 metric rows. The analysis records 48 summaries, 36 paired contrasts, 1152 source means, and exactly one new-data bootstrap draw array.',
             '', 'Execution receipts: [proposed](actual_receipts/VAR_UNCONDITIONAL.json), [latent continuous JSCC](actual_receipts/P1024.json), [BPG](actual_receipts/BPG_ADAPTIVE.json), [Swin](actual_receipts/SWIN80K.json), and [metrics](metrics_v1/completion.json). The independent review reads these top-level actual counts; it does not claim every remote waveform or reconstruction archive has been copied locally.',
             '', 'The independently read `status_counts.csv` retains four body-CRC-rejected frames at 10 dB for the proposed method (`BODY_CRC_REJECT_KEEP`), together with all 68 ordinary received frames at that point. All 72 frames at the other proposed points and all reference points are present. All 216 BPG frames are decoded. These status statements do not replace detailed per-frame receiver evidence.',
             '', '## Read-only audit and provenance', '',
             '- Verified all 48 summary and 36 paired rows against the actual analysis completion hashes; exact method/SNR/metric coverage, 24-source/three-noise/72-frame fields, finite ordered confidence intervals, and VAR-minus-reference means agree with the CSVs.',
             '- Every DINOv2-L and agreement paired interval excludes zero in the favorable direction. The low-SNR adverse LPIPS/PSNR results and the Swin 19 dB PSNR interval crossing zero are explicitly retained.',
             '- This writer only reads completed CSVs and formats prose/tables. It performs no model call, channel simulation, bootstrap, policy selection, or modification of original result files.',
             '- Independent PNG inspection found complete labels, legends, intervals, negative differences, and correct percent/percentage-point units. The main plot preserves the hollow Swin 19 dB marker and dashed 10-to-19 dB segment.', '']
    text += [f'- `{name}` SHA-256: `{digest}`' for name, digest in pins.items()]
    text += [f'- `metrics_v1/completion.json` SHA-256: `{sha(metric_path)}`']
    text += [f'- `actual_receipts/{m}.json` SHA-256: `{sha(OUT / "actual_receipts" / (m + ".json"))}`' for m in METHODS]
    text += ['', 'Recreate only these prose files: `python experiments/generalization_kodak_20261009/write_results.py`.', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(text), encoding='utf8')
    tex = ['% Formatted from the completed summary.csv and paired.csv; no new statistics.',
           '% Means/intervals use the full-precision CSVs, rounded only for presentation.']
    for name, digest in pins.items():
        tex.append(f'% {name} SHA256 {digest}')
    for i, paragraph in enumerate(english):
        paragraph = paragraph.replace('256 by 256', r'$256\times256$').replace('N=1024', r'$N=1024$').replace('95%', r'95\%')
        tex += ['', (r'\paragraph{Frozen transfer to Kodak24.} ' if i == 0 else '') + paragraph]
    (OUT / 'manuscript_results.tex').write_text('\n'.join(tex) + '\n', encoding='utf8')
    print(json.dumps(dict(status='PROSE_FORMATTED_FROM_ACTUAL_CSV',
                         summary_rows=48, paired_rows=36, new_bootstrap_calls=0,
                         files=[str(OUT / 'RESULTS.md'), str(OUT / 'manuscript_results.tex')])))


if __name__ == '__main__':
    run()
