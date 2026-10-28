"""Export Table III from frozen paired values; no inference or resampling."""
import argparse
import csv
import io
from pathlib import Path

P = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true', help='Check existing exports without writing')
args = parser.parse_args()
with (P / 'inputs/paired_deltas.csv').open(encoding='utf-8', newline='') as f:
    rows = {(int(r['snr_db']), r['metric']): r for r in csv.DictReader(f)
            if r['comparison'] == 'NEST_minus_ADA'}
columns = ['snr_db', 'comparison', 'sources', 'noise_per_source'] + [
    f'{m}_{k}' for m in ('psnr_db', 'dreamsim') for k in ('mean', 'ci_low', 'ci_high')]
out = io.StringIO(newline='')
writer = csv.DictWriter(out, fieldnames=columns, lineterminator='\n')
writer.writeheader()
tex = r'''\begin{table}[t]
\caption{Reduced-budget comparison on common500 at $N=512$. $\Delta=\mathrm{NeST}-\mathrm{Ada}$; brackets give source-paired 95\% CIs. Positive PSNR and negative DreamSim favor NeST.}
\label{tab:reduced}
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ccc}
\toprule
SNR & $\Delta$PSNR (dB) & $\Delta$DreamSim \\
\midrule
'''
for snr in (4, 10, 19):
    data = dict(snr_db=snr, comparison='NEST_minus_ADA', sources=500, noise_per_source=3)
    values = []
    for metric, precision in (('psnr_db', 3), ('dreamsim', 4)):
        r = rows[(snr, metric)]
        assert r['sources'] == '500' and r['noise_per_source'] == '3'
        for k in ('mean', 'ci_low', 'ci_high'):
            data[f'{metric}_{k}'] = r[k]  # Preserve CSV text precision.
        values.append([format(float(r[k]), f'+.{precision}f') for k in ('mean', 'ci_low', 'ci_high')])
    writer.writerow(data)
    a, b = values
    tex += f'{snr} & ${a[0]}$ & ${b[0]}$ ' + '\\\\\n'
    tex += f' & $[{a[1]},{a[2]}]$ & $[{b[1]},{b[2]}]$ ' + '\\\\\n'
tex += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
for suffix, content in (('csv', out.getvalue()), ('tex', tex)):
    path = P / f'tables/table_iii.{suffix}'
    if args.check:
        assert path.read_text(encoding='utf-8') == content, str(path)
    else:
        path.write_text(content, encoding='utf-8', newline='\n')
print('PASS: three rows, six paired means and twelve CI bounds match frozen CSV.')
