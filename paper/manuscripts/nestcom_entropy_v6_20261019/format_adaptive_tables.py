"""Format saved B3 means/direct intervals; no scoring or resampling."""
from pathlib import Path
import csv

P = Path(__file__).resolve().parent
E = P / 'evidence/n2048_adaptive'


def main():
    with (E / 'B3_summary.csv').open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (E / 'B3_paired_differences.csv').open(encoding='utf8', newline='') as stream:
        paired = list(csv.DictReader(stream))
    assert len(rows) == 28 and len(paired) == 14
    means = {(r['method'], int(r['snr_db']), r['metric']): r for r in rows}
    differences = {(int(r['snr_db']), r['metric']): r for r in paired}
    metrics = [
        ('psnr_db', r'PSNR (dB) $\uparrow$', 3, 1),
        ('lpips_alex', r'LPIPS $\downarrow$', 4, 1),
        ('dists', r'DISTS $\downarrow$', 4, 1),
        ('dreamsim', r'DreamSim $\downarrow$', 4, 1),
        ('dinov2_vitl14_cosine', r'DINO-L $\uparrow$', 4, 1),
        ('convnext_top1_source_prediction', r'Agreement (\%) $\uparrow$', 1, 100),
        ('ms_ssim', r'MS-SSIM $\uparrow$', 4, 1),
    ]
    lines = [r'\begin{table*}[t]', r'\centering',
        r'\caption{Frozen-policy adaptive test at $N=2048$ on external\_common100. Each mean first averages three noises per source. $\Delta$ is NeST minus ARPC; brackets are the saved direct source-paired 95\% intervals. Negative differences favor NeST for downward metrics. Agreement differences are percentage points. The fixed-full 19-dB reference is separate.}',
        r'\label{tab:adaptive2048}', r'\footnotesize\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{lrrrrrr}', r'\toprule',
        r' & \multicolumn{3}{c}{4 dB} & \multicolumn{3}{c}{10 dB} \\',
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
        r'Metric & NeST & ARPC & $\Delta$ [95\% CI] & NeST & ARPC & $\Delta$ [95\% CI] \\',
        r'\midrule']
    for metric, label, digits, scale in metrics:
        values = [label]
        for snr in (4, 10):
            for method in ('NEST_ADAPTIVE_N2048', 'ARPC_ADAPTIVE_N2048'):
                row = means[(method, snr, metric)]
                assert int(row['source_count']) == 100
                values.append(f"{float(row['mean']) * scale:.{digits}f}")
            row = differences[(snr, metric)]
            assert int(row['source_count']) == 100
            mean, low, high = (float(row[k]) * scale for k in ('mean', 'ci_low', 'ci_high'))
            values.append(f'{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]')
        lines.append(' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (P / 'tables/n2048_adaptive_quality.tex').write_text('\n'.join(lines) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
