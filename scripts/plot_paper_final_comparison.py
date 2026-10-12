"""Plot published common500 estimates only; no models or new statistics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SNR = [1, 4, 7, 10, 13, 19]
METRICS = [
    ('psnr_db', 'PSNR (dB) ↑', 'psnr', 1),
    ('lpips_alex', 'LPIPS ↓', 'lpips', 1),
    ('dinov2_vitl14_cosine', 'DINOv2-L cosine similarity ↑', 'dinov2_l', 1),
    ('convnext_top1_source_prediction', 'ConvNeXt prediction agreement (%) ↑', 'convnext_agreement', 100),
]
STYLES = {
    'proposed': ('Partial-scale digital + VAR (ours)', '#0072B2', '-', 'o'),
    'latent': ('Latent continuous JSCC (100k)', '#D55E00', '--', 's'),
    'swin': ('SwinJSCC (adapted, 200k)', '#009E73', '-.', '^'),
    'bpg': ('Adaptive-resolution BPG + LDPC', '#CC79A7', (0, (5, 2, 1, 2)), 'v'),
}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    with p.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / 'paper/figures/final_paper_comparisons_20261012'
    out.mkdir(parents=True, exist_ok=True)
    old = root / 'results/main_raw64_20261007/final_common500_r6/summary.csv'
    new = root / 'results/baseline_strength_20261011/paired_supplement_v1/common500/summary.csv'
    bpg = root / 'results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1/comparison_summary.csv'
    complete = json.loads((new.parent / 'completion.json').read_text())
    assert sha(new) == complete['outputs']['summary.csv']
    tables = {p: read(p) for p in (old, new, bpg)}
    definitions = [
        ('proposed', old, 'point_id', 'RAW64_PARTIAL_VAR_COMPLETION_SNR_{}'),
        ('latent', new, 'method', 'P100000'),
        ('swin', new, 'method', 'Swin200000'),
        ('bpg', bpg, 'point_id', 'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024_SNR_{}'),
    ]
    rows = []
    for method, path, key, template in definitions:
        for metric, label, slug, scale in METRICS:
            for snr in SNR:
                identity = template.format(snr)
                match = [r for r in tables[path] if r[key] == identity and r['metric'] == metric and float(r['snr_db']) == snr]
                assert len(match) == 1, (method, metric, snr, len(match))
                r = match[0]
                assert int(r['source_count']) == 500 and int(r['noise_count']) == 3
                vals = [float(r[k]) for k in ('ci_low', 'mean', 'ci_high')]
                assert np.isfinite(vals).all() and vals[0] <= vals[1] <= vals[2]
                if method == 'proposed':
                    same = [v for v in tables[bpg] if v['point_id'] == identity and v['metric'] == metric]
                    assert len(same) == 1
                    assert all(float(r[k]) == float(same[0][k]) for k in ('mean', 'ci_low', 'ci_high'))
                rows.append(dict(population='common500', N=1024, method=method,
                    public_label=STYLES[method][0], point_id=identity, metric=metric,
                    snr_db=snr, mean=r['mean'], ci_low=r['ci_low'], ci_high=r['ci_high'],
                    source_count=500, noise_count=3, display_scale=scale,
                    source_csv=path.relative_to(root).as_posix(), source_sha256=sha(path)))
    assert len(rows) == 96
    with (out / 'plot_data.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
        'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65})

    def panel(ax, metric, label, slug, scale):
        bounds = []
        for method, (legend, color, line, marker) in STYLES.items():
            rr = [r for r in rows if r['method'] == method and r['metric'] == metric]
            assert [r['snr_db'] for r in rr] == SNR
            mean, low, high = [np.array([float(r[k]) for r in rr]) * scale for k in ('mean', 'ci_low', 'ci_high')]
            ax.fill_between(SNR, low, high, color=color, alpha=.13, linewidth=0)
            ax.plot(SNR, mean, color=color, linestyle=line, marker=marker,
                    markersize=4, linewidth=1.4, label=legend)
            bounds.extend(low); bounds.extend(high)
        span = max(bounds) - min(bounds)
        ax.set_ylim(min(bounds)-.07*span, max(bounds)+.07*span)
        ax.set_xlim(.4, 19.6); ax.set_xticks(SNR)
        ax.set_xlabel('SNR (dB)'); ax.set_ylabel(label)
        ax.grid(color='#E5E5E5', linewidth=.5); ax.set_axisbelow(True)

    def save(fig, name):
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(out / f'{name}.{ext}', dpi=600, facecolor='white', bbox_inches='tight', pad_inches=.06)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.2))
    fig.subplots_adjust(left=.10, right=.99, bottom=.10, top=.83, hspace=.37, wspace=.33)
    for ax, spec in zip(axes.flat, METRICS):
        panel(ax, *spec)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .99), ncol=2, frameon=False)
    save(fig, 'fig_main_N1024_common500')
    for spec in METRICS:
        fig, ax = plt.subplots(figsize=(3.5, 3.3))
        fig.subplots_adjust(left=.19, right=.98, bottom=.15, top=.72)
        panel(ax, *spec)
        ax.legend(loc='lower left', bbox_to_anchor=(-.17, 1.05), frameon=False, fontsize=7.5)
        save(fig, 'fig_main_N1024_common500_' + spec[2])
    caption = ('Performance at N=1024 on the original common500 cohort (500 sources, three noise realizations per source). '
        'Lines show published means; shaded bands are existing pointwise 95\\% source-level bootstrap intervals. '
        'Failures remain included. Latent continuous JSCC uses the calibration-selected 100k checkpoint; '
        'adapted SwinJSCC uses the selected 200k checkpoint. Swin was initialized from 80k weights with fresh optimizer, '
        'sampling order and RNG, then trained for 120k additional updates over 1--19 dB. It stopped at budget while still improving; '
        '19 dB is within this updated range. The old 80k adaptation had an out-of-range 19 dB point and is not plotted here. '
        'Adaptive-resolution BPG uses calibration-only selection. These later baseline comparisons reuse the original cohort, '
        'not a newly blind test. ConvNeXt agreement is agreement with the source-image prediction, not classification accuracy. '
        'DINOv2-L is ViT-L/14 cosine similarity. HiFi-DiffCom is omitted because no matching common500 result exists.')
    (out / 'captions.tex').write_text('\\newcommand{\\FinalMainComparisonCaption}{' + caption + '}\n', encoding='utf-8')
    (out / 'README.md').write_text('''# Main-paper comparison figures

The new main figure compares the proposed raw partial-scale method with selected latent continuous JSCC (100k), adapted SwinJSCC (200k), and adaptive-resolution BPG + LDPC. It uses only the original common500 cohort at N1024, six numeric SNRs, four metrics, and existing source-level 95% intervals. No training, inference, channel simulation, policy selection, or bootstrap was performed. Single-column versions use separate layouts.

Original main results: commit 252176e041758ecb2d3e81b6fde5b587e7e17bb7; final selected-baseline supplement: 3ccfe6e9fb2c914f1ac999733e61ccd3d0493015. Exact CSV paths, hashes, identity mappings and unrounded numbers are in plot_data.csv. Proposed values were also checked against the unchanged anchors in the adaptive-BPG comparison. New100 results are not mixed into this figure. The adaptation and range changes are specified in captions.tex.

Reproduce from the repository root:

```sh
python scripts/plot_paper_final_comparison.py
```

PNG previews are 600 dpi; PDF curves are vector; SVG text remains editable.

## Qualitative panels for the paper

The existing five-column and four-column panels are at ../hifi_development_N1024_13dB_group02/. These are fixed development sources 4,21,24,29 at N1024/13dB, not common500. They use historical latent 40k and Swin80k models, including the HiFi reconstruction conditioned on that same Swin reception. They must not be presented as images from the final 100k/200k baselines. Use as explicitly historical adapted-baseline illustrations or supplementary material; they are not a matched qualitative validation of the final main curves. No newer matched HiFi panel is asserted here.

All methods share source pixels and working point. The fixed first noise is used: 2001 for latent/Swin/HiFi and 6201 for digital methods; cross-family draws are not identical. No per-method best-noise selection. Mechanism panels retain the same received partial tokens and decoder, zero absent residual contribution (not codebook index 0), and no extra source truth. Original provenance, captions and image hashes remain alongside those files. Existing mainraw64_holdout500 fig03/fig04 are the unchanged paired mechanism ablations.
''', encoding='utf-8')
    (out / 'validation.json').write_text(json.dumps({'status': 'PASS', 'rows': len(rows),
        'each_method_metric_snr': SNR, 'source_count': 500, 'noise_count': 3,
        'exact_proposed_anchor_match': True, 'new_scientific_calls': 0,
        'inputs': {p.relative_to(root).as_posix(): sha(p) for p in tables},
        'outputs': {p.name: sha(p) for p in out.iterdir() if p.suffix in ('.png', '.pdf', '.svg', '.csv')}}, indent=2))
    print(json.dumps({'output': str(out), 'rows': len(rows), 'figures': 5}))

if __name__ == '__main__':
    main()
