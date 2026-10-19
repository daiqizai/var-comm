"""Deterministic manuscript assembly from completed, saved results only.

No model, PHY, score, resampling, or timing entry is imported or executed.
Run from any directory in the checked-out repository.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def rows(rel):
    with (ROOT / rel).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def desc(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def write(rel, s):
    p = HERE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding='utf-8', newline='\n')


def main():
    sources = {
        'paper/figures_redesign_v2/fig2_main_1x4.pdf': 'figures/main_common500.pdf',
        'paper/figures_redesign_v2/fig3_paired_two_panels.pdf': 'figures/historical_paired_common500.pdf',
        'paper/figures_redesign_v2/plot_inputs/fig2_plot_data.csv': 'evidence/common500_main_points.csv',
        'paper/figures_redesign_v2/plot_inputs/fig3_plot_data.csv': 'evidence/historical_common500_paired_points.csv',
        'paper/figures_redesign_v2/input_manifest.json': 'evidence/original_figure_input_manifest.json',
        'paper/figures_redesign_v2/review/dreamsim_identity.json': 'evidence/dreamsim_identity.json',
        'results/entropy_main_common500_20261013/analysis_v1/paired.csv': 'evidence/historical_common500_main_paired.csv',
        'results/arpc_external_baseline_20261019_v1/a3_metrics_v1/matched100_summary.csv': 'evidence/arpc_n1024_summary.csv',
        'results/arpc_external_baseline_20261019_v1/a3_metrics_v1/nest_minus_arpc_paired.csv': 'evidence/arpc_n1024_paired.csv',
        'results/arpc_n2048_validation_20261019_v1/metrics_v1/summary.csv': 'evidence/n2048_fixed_summary.csv',
        'results/arpc_n2048_validation_20261019_v1/metrics_v1/paired_differences.csv': 'evidence/n2048_fixed_paired.csv',
        'results/t1c_m10_test_20261016_v1/delivery/summary_19dB.csv': 'evidence/t1c_19db_summary.csv',
        'results/t1c_m10_test_20261016_v1/delivery/paired_19dB.csv': 'evidence/t1c_19db_paired.csv',
        'results/t1c_m10_test_20261016_v1/delivery/actual_fallback_counts.csv': 'evidence/t1c_actual_fallback_counts.csv',
        'results/baseline_strength_20261011/timing/six_method_v6_delivery_v1/paper_timing_table.csv': 'evidence/h800_old_version_timing.csv',
        'results/quality_metrics_supplement_20261014_v1/hifi_fixed16/summary.csv': 'evidence/hifi_final_fixed16_summary.csv',
        'results/arpc_n2048_adaptive_20261019_v1/calibration_diagnostics/endpoint_means.csv': 'evidence/cal100_clean_endpoint_means.csv',
        'results/arpc_n2048_adaptive_20261019_v1/calibration_diagnostics/policy_capacity_utilization.csv': 'evidence/cal100_capacity_utilization.csv',
        'results/n1024_consistency_20261019_v1/family_snr_summary.csv': 'evidence/n1024_source_utilization_summary.csv',
        'results/n1024_consistency_20261019_v1/completion.json': 'evidence/n1024_consistency_audit_completion.json',
        'results/n1024_consistency_20261019_v1/repair_budget.json': 'evidence/n1024_consistency_repair_budget.json',
        'results/arpc_n2048_adaptive_20261019_v1/figures_calibration_v2/cal100_diagnostics.pdf': 'figures/calibration_payload.pdf',
        'paper/figures/hifi_final_fixed16_20261012/selected_three_final_link.pdf': 'figures/fixed16_examples.pdf',
        'paper/figures/hifi_final_fixed16_20261012/provenance.json': 'evidence/fixed16_image_provenance.json',
    }
    delivery = 'results/n1024_consistency_20261019_v1/repair_v1/delivery/'
    for name in ('summary_common500.csv', 'paired_common500.csv',
                 'source_means_common500.csv', 'frame_scores_9000.csv',
                 'actual_fallback_counts.csv', 'RX_status_counts.csv',
                 'closure_audit.json', 'manifest.json', 'completion.json'):
        sources[delivery + name] = 'evidence/n1024_repair/' + name
    for name in ('figure_style_v2.json', 'resolved_style.json'):
        sources['paper/figures_redesign_v2/' + name] = 'style/' + name
    manifest = []
    for src, dst in sources.items():
        a, b = ROOT / src, HERE / dst
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a, b)
        assert a.read_bytes() == b.read_bytes()
        manifest.append({'source': desc(a), 'copy': desc(b)})

    # Retain raw rows verbatim. Replace only entropy rows from the closed
    # same-version delivery; sign inversion preserves direct paired intervals.
    repair_manifest = json.loads((ROOT / (delivery + 'manifest.json')).read_text())
    assert desc(ROOT / (delivery + 'manifest.json'))['sha256'] == '8d6a29a210c300082f2ad14bc99b7334d2138d0b1f17d7f92195d21b31f4e235'
    for d in repair_manifest['files']:
        p = ROOT / delivery / Path(d['path']).name
        a = desc(p)
        assert a['sha256'] == d['sha256'] and a['bytes'] == d['bytes']
    point_rows = rows('paper/figures_redesign_v2/plot_inputs/fig3_plot_data.csv')
    repaired_pairs = rows(delivery + 'paired_common500.csv')
    source_sha = desc(ROOT / (delivery + 'paired_common500.csv'))['sha256']
    for r in point_rows:
        if r['method'] != 'Entropy-coded':
            continue
        matches = [(i + 2, v) for i, v in enumerate(repaired_pairs)
                   if v['snr_db'] == r['snr_db'] and v['metric'] == r['metric']]
        assert len(matches) == 1
        line, v = matches[0]
        assert v['population'] == 'common500' and v['sources'] == '500' and v['noise_per_source'] == '3'
        r.update(input_path=delivery + 'paired_common500.csv', input_csv_line=str(line),
                 input_sha256=source_sha, version='n1024_same_version_repair_20261019',
                 row_identifier=json.dumps({k: v[k] for k in ('population', 'snr_db', 'comparison', 'metric')}, sort_keys=True),
                 original_mean=v['mean'], original_ci_low=v['ci_low'], original_ci_high=v['ci_high'])
        if r['metric'] == 'dreamsim':
            r.update(mean=str(-float(v['mean'])), ci_low=str(-float(v['ci_high'])),
                     ci_high=str(-float(v['ci_low'])), transform='negate and reverse interval endpoints')
        else:
            r.update(mean=v['mean'], ci_low=v['ci_low'], ci_high=v['ci_high'], transform='identity')
        r['bootstrap_recomputed'] = 'False'
    point_path = HERE / 'evidence/common500_paired_points.csv'
    with point_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)

    metrics = [
        ('psnr_db', r'PSNR (dB) $\uparrow$', 3, 1),
        ('lpips_alex', r'LPIPS $\downarrow$', 4, 1),
        ('dists', r'DISTS $\downarrow$', 4, 1),
        ('dreamsim', r'DreamSim $\downarrow$', 4, 1),
        ('dinov2_vitl14_cosine', r'DINO-L $\uparrow$', 4, 1),
        ('convnext_top1_source_prediction', r'Agreement (\%) $\uparrow$', 1, 100),
        ('ms_ssim', r'MS-SSIM $\uparrow$', 4, 1),
    ]
    ext = rows('results/arpc_external_baseline_20261019_v1/a3_metrics_v1/matched100_summary.csv')
    emethods = sorted({r['method'] for r in ext})
    amethod = next(m for m in emethods if 'ARPC' in m.upper())
    def get(data, method, snr, metric):
        found = [r for r in data if r['method'] == method and int(r['snr_db']) == snr and r['metric'] == metric]
        assert len(found) == 1, (method, snr, metric)
        return found[0]
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Separate matched100 comparison at $N=1024$. Each cell is NeST-Com / ARPC adapted. All seven metrics and all 300 logical frames per method/SNR are included. Full-precision means, marginal intervals and direct paired intervals are supplied in the accompanying CSVs.}',
             r'\label{tab:arpc1024}', r'\small\setlength{\tabcolsep}{8pt}',
             r'\begin{tabular}{lrrr}', r'\toprule',
             r'Metric & 4 dB: NeST / ARPC & 10 dB: NeST / ARPC & 19 dB: NeST / ARPC \\', r'\midrule']
    for metric, label, nd, scale in metrics:
        vals = []
        for snr in (4, 10, 19):
            a = float(get(ext, 'EC_VAR_PARTIAL', snr, metric)['mean']) * scale
            b = float(get(ext, amethod, snr, metric)['mean']) * scale
            vals.append(f'{a:.{nd}f} / {b:.{nd}f}')
        lines.append(label + ' & ' + ' & '.join(vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    write('tables/arpc_n1024.tex', '\n'.join(lines) + '\n')

    fixed = rows('results/arpc_n2048_validation_20261019_v1/metrics_v1/summary.csv')
    pair = rows('results/arpc_n2048_validation_20261019_v1/metrics_v1/paired_differences.csv')
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Fixed complete-representation reference at $N=2048$, 19 dB, on the same 100 sources. Both methods correctly deliver 300/300 objects. Differences are NeST minus ARPC; intervals are direct source-paired 95\% intervals. Negative differences favor NeST for downward metrics.}',
             r'\label{tab:fixed2048}', r'\small\setlength{\tabcolsep}{9pt}',
             r'\begin{tabular}{lrrr}', r'\toprule',
             r'Metric & NeST full m10 & ARPC full e7 & Difference [95\% CI] \\', r'\midrule']
    for metric, label, nd, scale in metrics:
        a = float(get(fixed, 'NEST_FULL_M10_N2048', 19, metric)['mean']) * scale
        b = float(get(fixed, 'ARPC_FULL_E7_N2048', 19, metric)['mean']) * scale
        d = [r for r in pair if int(r['snr_db']) == 19 and r['metric'] == metric]
        assert len(d) == 1
        d = d[0]
        val = [float(d[k]) * scale for k in ('mean', 'ci_low', 'ci_high')]
        lines.append(label + f' & {a:.{nd}f} & {b:.{nd}f} & {val[0]:.{nd}f} [{val[1]:.{nd}f}, {val[2]:.{nd}f}]' + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    write('tables/n2048_fixed.tex', '\n'.join(lines) + '\n')

    timing = rows('results/baseline_strength_20261011/timing/six_method_v6_delivery_v1/paper_timing_table.csv')
    names = [('RAW_WHOLE', 'Whole-scale (raw)'), ('RAW_PARTIAL', 'NeST-Com (raw)'),
             ('EC_VAR_WHOLE', 'Whole-scale (entropy)'), ('EC_VAR_PARTIAL', 'NeST-Com (entropy)'),
             ('P100000', 'Latent-space JSCC'), ('Swin200000', 'SwinJSCC (adapted)')]
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Existing shared-H800 mean latency (ms), $N=1024$, 19 dB, fixed16, batch one. Entropy rows are the old, pre-m10 version. Three fresh measured repetitions per source give 48 measurements per row; 4/10-dB rows and all components remain in the accompanying CSV.}',
             r'\label{tab:timing}', r'\footnotesize\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{lrrr}', r'\toprule', r'Method & TX & RX & E2E \\', r'\midrule']
    for method, label in names:
        r = next(r for r in timing if r['method'] == method and int(r['snr_db']) == 19)
        values = [float(r[k]) for k in ('TX_mean_ms', 'RX_mean_ms', 'software_e2e_mean_ms')]
        lines.append(label + ' & ' + ' & '.join(f'{x:.1f}' for x in values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('tables/timing.tex', '\n'.join(lines) + '\n')

    utilization = rows('results/n1024_consistency_20261019_v1/family_snr_summary.csv')
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Same-version common500 source utilization, including arithmetic termination. Each cell is whole / partial. Capacity excludes the paid 13-bit length and CRC16; source statistics count each image once. These are source-layout statistics, separate from repaired wireless quality.}',
             r'\label{tab:utilization}', r'\footnotesize\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{lrrr}', r'\toprule',
             r'SNR (dB) & Capacity (bit) & Utilization (\%) & Spare (bit) \\', r'\midrule']
    for snr in (4, 10, 19):
        vals = {}
        for family in ('WHOLE', 'PARTIAL'):
            found = [r for r in utilization if r['family'] == family and int(r['snr_db']) == snr]
            assert len(found) == 1 and int(found[0]['sources']) == 500
            vals[family] = found[0]
        a, b = vals['WHOLE'], vals['PARTIAL']
        assert a['source_capacity_bits'] == b['source_capacity_bits']
        u = [100 * float(v['mean_utilization']) for v in (a, b)]
        spare = [float(v['mean_remaining_bits']) for v in (a, b)]
        lines.append(f"{snr} & {a['source_capacity_bits']} & {u[0]:.2f} / {u[1]:.2f} & {spare[0]:.1f} / {spare[1]:.1f}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('tables/n1024_utilization.tex', '\n'.join(lines) + '\n')
    assembled = {'kind': 'SAVED_RESULT_ASSEMBLY_ONLY',
        'scientific_calls': 0, 'bootstrap_calls': 0, 'files': manifest,
        'derived_paired_points': {'file': desc(point_path),
            'operation': 'Exact saved entropy paired rows plus unchanged raw rows; DreamSim sign inversion only',
            'new_bootstrap_draws': 0}}
    if (HERE / 'evidence_relocation.json').exists():
        assembled['evidence_relocation'] = desc(HERE / 'evidence_relocation.json')
    write('input_manifest.json', json.dumps(assembled, indent=2) + '\n')
    print(json.dumps({'copied_inputs': len(manifest), 'tables': 4, 'scientific_calls': 0}))


if __name__ == '__main__':
    main()
