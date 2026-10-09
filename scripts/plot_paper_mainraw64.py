#!/usr/bin/env python3
"""Plot published CSV estimates only; no model, channel or bootstrap computation."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

COMMIT = '252176e041758ecb2d3e81b6fde5b587e7e17bb7'
DATA_REL = 'results/main_raw64_20261007/final_common500_r6'
SOURCE_SHA = {
    'summary.csv': '5899fe10ccf2e7fe93ef36c59b9e14c6622f542d58ee1c0bbf0c8332a5577859',
    'paired.csv': 'dd8221b1fe8aa9d86f9cbf78ea281fd2da860e2ffdcd6fb83daa487eebc0c235',
    'points.json': '1efbbf1cc4d1dc4359f005bc234a324e830072dc4e1a4b57b5ad4eac92fbc9ce',
    'pairs.json': '524f95dfec6278bdfe9b8fc03765bd2f678854816e1ad2e73a38bb854a485468',
    'fairness_notes.json': 'fbdde67b6ce1d7862f24b4e7510da8562e74b43244d3770febd3c2c4fe56c968',
    'REPORT.md': 'f3e69c7baff10b185d68dbc3061fdd4e7cf64111d897bb5669eb924afc5f21c1',
    'evidence/unified_holdout_final_freeze.json': '097ca08886761bf076ebab4d35765b7327d2b32d53355e0306ce85763a20a169',
}
SNRS = (1, 4, 7, 10, 13, 19)
PROPOSED = 'RAW64_PARTIAL_VAR_COMPLETION_SNR_{snr}'
METHODS = (
    (PROPOSED, 'Proposed (raw + partial + VAR)', '#0072B2', '-', 'o'),
    ('P1024_SNR_{snr}', 'P1024', '#D55E00', '--', 's'),
    ('SWIN80K_N1024_SNR_{snr}', 'SwinJSCC-80k (adapted)', '#009E73', '-.', '^'),
)
METRICS = (
    dict(key='psnr_db', name='PSNR', slug='psnr', scale=1,
         ylabel='PSNR (dB) ↑', delta='ΔPSNR (dB) ↑', unit='dB'),
    dict(key='lpips_alex', name='LPIPS', slug='lpips', scale=1,
         ylabel='LPIPS ↓', delta='ΔLPIPS ↓', unit='LPIPS'),
    dict(key='dinov2_vitl14_cosine', name='DINOv2-L', slug='dinov2_l', scale=1,
         ylabel='Cosine similarity ↑', delta='ΔDINOv2-L ↑', unit='cosine similarity'),
    dict(key='convnext_top1_source_prediction', name='ConvNeXt', slug='convnext_agreement', scale=100,
         ylabel='Prediction agreement (%) ↑', delta='ΔConvNeXt agreement\n(percentage points) ↑', unit='%'),
)
GROUPS = (
    ('fig02_main_N1024', None, 'N = 1024'),
    ('fig03_partial_vs_whole', 'RAW64_WHOLE_VAR_COMPLETION_SNR_{snr}', 'PARTIAL − WHOLE'),
    ('fig04_var_vs_direct', 'RAW64_PARTIAL_DIRECT_DC_SNR_{snr}', 'VAR completion − Direct decoding'),
)

CAPTIONS = r'''% Existing published estimates only; no statistics were recomputed.
% Use \caption{\MainRawMainCaption} etc. within your paper's figure environment.
\newcommand{\MainRawMainCaption}{Main results on the frozen 500-source holdout at
$N=1024$ transmitted complex channel symbols, including the charged header.
Proposed (raw + partial + VAR), P1024, and SwinJSCC-80k (adapted) are compared in
PSNR, LPIPS-Alex, DINOv2-ViT-L/14 cosine similarity, and ConvNeXt prediction
agreement with the original source prediction (not classification accuracy).
Higher is better except for LPIPS. Means and shaded pointwise 95\% confidence
intervals are read directly from the published summary CSV. Each source first
averages its three frozen noise realizations; the existing 10,000-replicate
bootstrap weights the 500 sources equally, without multiplicity correction.
19 dB is outside Swin's training and calibration range. Its 19 dB marker is hollow
and its 13--19 dB segment is dashed. Swin uses six body channels (768 symbols) and
a 256-symbol protected header; this frozen adaptation and fixed 80k checkpoint
are not asserted to be the official optimum or fully converged. Equal source
count and total $N$ do not imply equal training, search, or header-protection
budgets. Only the three methods available in this holdout are plotted.}

\newcommand{\MainRawPartialCaption}{Partial-scale ablation on the same frozen
500-source holdout. Points, straight connecting lines, and error bars show the
published source-paired mean differences and pointwise 95\% confidence intervals
for PARTIAL $-$ WHOLE, with VAR completion used in both arms. The dashed horizontal
line marks zero. Positive differences favor PARTIAL for PSNR, DINOv2-L cosine
similarity, and ConvNeXt prediction agreement; negative differences favor PARTIAL
for LPIPS. ConvNeXt differences are percentage points, not relative percentages.
The intervals come directly from the paired CSV, not from subtraction of
single-method intervals. Both schemes have the same modulation and coding
permissions. PARTIAL can fall back to a whole-scale configuration: the final
WHOLE calibration winner remains eligible, and $K>0$ is not forced at every SNR.
The exact zero increments at 7 dB and the small or zero increments at 13 dB are
retained. Per-source three-noise means and the existing 500-source bootstrap are
used without recomputation or holdout reselection. This contrast is separate
from the VAR-completion ablation; the two sets of increments are not added.}

\newcommand{\MainRawCompletionCaption}{VAR-completion ablation on the frozen
500-source holdout, with the PARTIAL transmission configuration held fixed.
Points, straight connecting lines, and error bars show the published source-paired
mean differences and pointwise 95\% confidence intervals for VAR completion $-$
Direct decoding; the dashed horizontal line marks zero. Both arms use the same
actually received tokens from each transmission and the same frozen decoder
$D_c$. Direct decoding does not generate untransmitted residuals: their residual
embedding contribution is set to zero. This is not filling unknown tokens with
codebook index zero, and Direct receives no additional source-image ground truth.
Positive differences favor VAR completion for PSNR, DINOv2-L cosine similarity,
and ConvNeXt prediction agreement; negative differences favor VAR completion
for LPIPS. ConvNeXt differences and their intervals are expressed in percentage
points. All estimates and intervals are read from the existing paired CSV after
the original three-noise source averaging and 500-source bootstrap; no new
bootstrap is performed. This contrast is not added to PARTIAL $-$ WHOLE.}
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return [dict(row, source_row=str(i)) for i, row in enumerate(csv.DictReader(f), 2)]


def index_rows(rows, fields):
    index = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        index.setdefault(key, []).append(row)
    return index


def select_group(group, summary, paired, points, pairs):
    name, reference, _ = group
    selected, issues = [], []
    for metric in METRICS:
        templates = [PROPOSED] if reference else [m[0] for m in METHODS]
        for template in templates:
            for snr in SNRS:
                method = template.format(snr=snr)
                ref = reference.format(snr=snr) if reference else ''
                key = (method, ref, metric['key'], str(snr)) if reference else (method, metric['key'])
                matches = (paired if reference else summary).get(key, [])
                identity = f'{method} | {ref or "single method"} | {metric["key"]} | SNR={snr}'
                if len(matches) != 1:
                    issues.append(f'{identity}: expected one row, found {len(matches)}'); continue
                row = matches[0]
                try:
                    assert int(row['snr_db']) == snr
                    assert (int(row['source_count']), int(row['noise_count']), int(row['frame_count'])) == (500, 3, 1500)
                    assert int(row['bootstrap_replicates']) == 10000 and int(row['bootstrap_seed']) == 2026100701
                    assert row['bootstrap_unit'] == 'source after original3-noise mean'
                    assert method in points and points[method]['snr_db'] == snr
                    assert row['metric_identity'] == points[method]['metric_identity'][metric['key']]
                    seeds = json.loads(row['method_noise_seeds' if reference else 'noise_seeds'])
                    assert seeds == points[method]['noise_seeds'] and len(seeds) == 3
                    if reference:
                        assert row['delta_definition'] == 'method minus reference'
                        assert (method, ref) in pairs and metric['key'] in pairs[(method, ref)]['metrics']
                        assert ref in points and points[ref]['snr_db'] == snr
                        assert points[ref]['metric_identity'][metric['key']] == row['metric_identity']
                        assert json.loads(row['reference_noise_seeds']) == points[ref]['noise_seeds']
                    mean, low, high = [Decimal(row[k]) for k in ('mean', 'ci_low', 'ci_high')]
                    assert all(v.is_finite() for v in (mean, low, high)) and low <= mean <= high
                except (AssertionError, ValueError, KeyError) as exc:
                    issues.append(f'{identity}: metadata/CI validation failed ({type(exc).__name__})'); continue
                record = dict(row, figure=name, source_csv='paired.csv' if reference else 'summary.csv',
                              method=method, reference=ref, display_scale=str(metric['scale']),
                              display_unit='percentage points' if reference and metric['scale'] == 100 else metric['unit'])
                for field in ('mean', 'ci_low', 'ci_high'):
                    record['plot_' + field] = str(Decimal(row[field]) * metric['scale'])
                selected.append(record)
    return selected, issues


def render(group, rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    import numpy as np

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5,
        'axes.labelsize': 8.5, 'axes.titlesize': 9, 'axes.linewidth': 0.65,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': COMMIT, 'axes.unicode_minus': True,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'path.simplify': False,
    })
    name, reference, contrast = group
    saved = []
    handles = [Line2D([], [], color=c, linestyle=ls, marker=mk, linewidth=1.35,
                      markersize=4, label=label) for _, label, c, ls, mk in METHODS]

    def panel(ax, metric, letter=None):
        relevant = [r for r in rows if r['metric'] == metric['key']]
        lows, highs = [], []
        for template, label, color, ls, marker in (METHODS[:1] if reference else METHODS):
            rr = sorted([r for r in relevant if r['method'] in [template.format(snr=s) for s in SNRS]], key=lambda r: int(r['snr_db']))
            assert len(rr) == 6 and tuple(int(r['snr_db']) for r in rr) == SNRS
            x = np.asarray(SNRS, dtype=float)
            y, lo, hi = [np.asarray([float(r['plot_' + f]) for r in rr]) for f in ('mean', 'ci_low', 'ci_high')]
            lows.extend(lo); highs.extend(hi)
            if reference:
                ax.errorbar(x, y, yerr=np.vstack((y - lo, hi - y)), color=color, linestyle=ls,
                            marker=marker, linewidth=1.35, markersize=4.4, markeredgewidth=0.85,
                            capsize=3, capthick=0.9, elinewidth=1.05, zorder=4)
            else:
                ax.fill_between(x, lo, hi, facecolor=color, alpha=0.13, linewidth=0, zorder=1)
                if template.startswith('SWIN'):
                    ax.plot(x[:5], y[:5], color=color, linestyle=ls, marker=marker,
                            linewidth=1.35, markersize=4.4, markeredgewidth=0.85, zorder=3)
                    ax.plot(x[4:], y[4:], color=color, linestyle='--', linewidth=1.35, zorder=3)
                    ax.plot(x[-1:], y[-1:], color=color, linestyle='None', marker=marker,
                            markersize=5.1, markerfacecolor='white', markeredgewidth=1.2, zorder=5)
                else:
                    ax.plot(x, y, color=color, linestyle=ls, marker=marker,
                            linewidth=1.35, markersize=4.4, markeredgewidth=0.85, zorder=4)
        if reference:
            ax.axhline(0, color='#656565', linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
            lows.append(0); highs.append(0)
        low, high = min(lows), max(highs)
        span = high-low if high > low else max(abs(high)*0.2, 0.01)
        ax.set_ylim(low - span*0.09, high + span*0.09)
        ax.set_xlim(0.3, 19.7); ax.set_xticks(SNRS); ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(metric['delta'] if reference else metric['ylabel'], labelpad=5)
        ax.set_title((f'({letter}) ' if letter else '') + metric['name'], loc='left', fontweight='semibold', pad=7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        fmt = ScalarFormatter(useOffset=False); fmt.set_powerlimits((-3, 4)); ax.yaxis.set_major_formatter(fmt)
        ax.grid(color='#DFE3E6', linewidth=0.55); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        for edge in ('left', 'bottom'): ax.spines[edge].set_color('#535353')

    def export(fig, stem):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer(); bounds = fig.bbox
        texts = []
        for ax in fig.axes:
            texts += [ax.xaxis.label, ax.yaxis.label, ax._left_title, ax.yaxis.get_offset_text(),
                      *ax.get_xticklabels(), *ax.get_yticklabels()]
        if fig._suptitle: texts.append(fig._suptitle)
        boxes = [text.get_window_extent(renderer) for text in texts if text.get_visible() and text.get_text()]
        boxes += [legend.get_window_extent(renderer) for legend in fig.legends]
        assert all(b.x0 >= -0.5 and b.y0 >= -0.5 and b.x1 <= bounds.x1+0.5 and b.y1 <= bounds.y1+0.5 for b in boxes), 'Text clipped: '+stem
        for ext in ('pdf', 'svg', 'png'):
            p = out / (stem + '.' + ext)
            kw = {'dpi': 600} if ext == 'png' else {}
            if ext == 'pdf': kw['metadata'] = dict(Title=stem, Author='VAR_COMM', Subject='Published holdout500 CSV estimates; '+COMMIT)
            fig.savefig(p, format=ext, **kw)
            saved.append(p.name)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7, 4.9), dpi=150)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.105, top=0.87, wspace=0.34, hspace=0.57)
    if reference:
        fig.suptitle(contrast, fontsize=9, fontweight='semibold', y=0.975)
    else:
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=3,
                   frameon=False, handlelength=2.6, columnspacing=1.35)
    for ax, metric, letter in zip(axes.flat, METRICS, 'abcd'): panel(ax, metric, letter)
    export(fig, name)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(3.45, 2.65 if reference else 3.0), dpi=150)
        fig.subplots_adjust(left=0.215, right=0.975, bottom=0.185 if reference else 0.165,
                            top=0.795 if reference else 0.705)
        if reference:
            fig.suptitle(contrast, fontsize=8.5, fontweight='semibold', y=0.985)
        else:
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.51, 0.995),
                       ncol=1, frameon=False, handlelength=2.6, fontsize=7.9, labelspacing=0.35)
        panel(ax, metric); export(fig, name + '_' + metric['slug'])
    return saved


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root/DATA_REL)
    parser.add_argument('--output-dir', type=Path, default=root/'paper/figures/mainraw64_holdout500')
    args = parser.parse_args(); data = args.data_dir.resolve(); out = args.output_dir.resolve()
    assert out != data and not out.is_relative_to(data), 'Output must be independent of scientific results'
    out.mkdir(parents=True, exist_ok=True)
    bad = [n for n,h in SOURCE_SHA.items() if not (data/n).is_file() or sha(data/n) != h]
    if bad:
        write_json(out/'validation.json', dict(status='STOPPED_SOURCE_IDENTITY',commit=COMMIT,missing_or_changed_files=bad))
        raise SystemExit('Source files do not match published commit: '+', '.join(bad))
    sr, pr = read_csv(data/'summary.csv'), read_csv(data/'paired.csv')
    summary = index_rows(sr, ('point_id','metric')); paired = index_rows(pr, ('method','reference','metric','snr_db'))
    points = json.loads((data/'points.json').read_text()); pair_list = json.loads((data/'pairs.json').read_text())
    pairs = {(p['method'],p['reference']):p for p in pair_list}
    assert len(pairs) == len(pair_list), 'Duplicated comparison metadata'
    all_rows, issues, files, counts = [], {}, [], {}
    for group in GROUPS:
        rows, errors = select_group(group, summary, paired, points, pairs)
        if errors: issues[group[0]] = errors; continue
        all_rows.extend(rows); counts[group[0]] = len(rows)
        files.extend(render(group, rows, out))
    fields = ['figure','source_csv','source_row','point_id','method','reference','snr_db','metric','mean','ci_low','ci_high',
              'display_scale','display_unit','plot_mean','plot_ci_low','plot_ci_high','source_count','noise_count','delta_definition']
    fields += sorted(set().union(*(r.keys() for r in all_rows)) - set(fields)) if all_rows else []
    with (out/'plot_data.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(all_rows)
    (out/'captions.tex').write_text(CAPTIONS,encoding='utf-8')
    links = '\n'.join(f'| {g[0]} | [{ext.upper()}]({g[0]}.{ext}) |' for g in GROUPS for ext in ('pdf','svg','png'))
    (out/'README.md').write_text(f'''# Frozen holdout500 paper figures

Data commit: `{COMMIT}`. Source directory: `{DATA_REL}/`.
All seven input files are SHA256-pinned in the plotting script and unchanged after export.
No development, MAIN16, H, C-REAL, BPG, HiFi or other bandwidth results are used.

Run from the actual VAR_COMM repository root (existing environment):
```sh
outputs/UNIFIED-METRICS-20261002/environment/bin/python -B scripts/plot_paper_mainraw64.py
```
An ordinary Python environment with matplotlib and NumPy can run the same script.
Use `--data-dir PATH --output-dir PATH` for a byte-identical local data mirror.

## Identity and units

| Method | Exact point template | Color / style / marker |
| --- | --- | --- |
| Proposed (raw + partial + VAR) | RAW64_PARTIAL_VAR_COMPLETION_SNR_{{snr}} | blue / solid / circle |
| P1024 | P1024_SNR_{{snr}} | vermillion / dashed / square |
| SwinJSCC-80k (adapted) | SWIN80K_N1024_SNR_{{snr}} | green / dash-dot / triangle |

SNR is numeric: 1,4,7,10,13,19. Swin's 19 dB marker is hollow and 13--19 dB is dashed.
DINOv2-L uses `dinov2_vitl14_cosine`, never `dino_cosine`.
ConvNeXt uses `convnext_top1_source_prediction`, agreement with the source prediction, not label accuracy.
ConvNeXt means AND CI endpoints are scaled by exactly100: percent for Fig.2, percentage points for Figs.3--4.
LPIPS differences retain their original sign; negative is better.

Fig.3 reads PARTIAL_VAR_COMPLETION minus WHOLE_VAR_COMPLETION from paired.csv.
Fig.4 reads PARTIAL_VAR_COMPLETION minus PARTIAL_DIRECT_DC from paired.csv.
Both comparisons use the matching SNR suffix. Their increments are never added.
The exact zero at7dB and small/zero values at13dB are included.

## Files and precision

15 figures (3 composites +12 individually laid-out single panels), each in PDF, SVG and600dpi PNG.
Composites are7 by4.9in; single panels are3.45in wide with their own legend/margins.
PDF uses vector paths and embedded TrueType fonts; SVG retains editable text.
Y limits cover all plotted CI endpoints with padding; delta panels include zero.
No smoothing, fitted curves, broken axes or per-point number labels are used.

`plot_data.csv` preserves original CSV `mean`, `ci_low`, `ci_high` text precision.
The `plot_*` columns record the exact Decimal unit conversion used for display;
`source_csv` and `source_row` identify the original row. Lines only connect measured SNR points.
Existing500-source/three-noise means and pointwise95% intervals are read, never recomputed.
No training, inference, channel simulation, bootstrap or strategy selection is invoked.

`captions.tex` defines three English caption commands. `validation.json` records checks and output SHAs.
Missing or duplicate rows stop their affected figure group and are listed precisely; no measurements are requested.
The publication filters are unchanged; files are generated in the working tree without committing.

| Composite | Export |
| --- | --- |
{links}
''',encoding='utf-8')
    assert all(sha(data/n)==h for n,h in SOURCE_SHA.items()), 'Scientific input changed during plotting'
    import matplotlib, numpy
    validation = dict(status='COMPLETE' if not issues else 'PARTIAL_MISSING_DATA',commit=COMMIT,
                      source_sha256=SOURCE_SHA,source_counts=dict(summary=len(sr),paired=len(pr)),
                      figure_row_counts=counts,missing_or_invalid=issues,snr_order=SNRS,
                      source_count=500,noise_count=3,scientific_recomputation=False,
                      pdf_fonttype=42,svg_text_editable=True,png_dpi=600,
                      matplotlib=matplotlib.__version__,numpy=numpy.__version__,
                      outputs={n:sha(out/n) for n in files+['captions.tex','plot_data.csv','README.md']})
    write_json(out/'validation.json',validation)
    print(json.dumps(dict(status=validation['status'],figure_count=len(files)//3,exports=len(files),plotted_rows=len(all_rows),issues=issues)))
    return 1 if issues else 0


if __name__=='__main__':
    sys.exit(main())
