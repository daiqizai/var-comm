"""Plot sealed scale-up statistics only; never infer, train, or bootstrap."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

REG_SHA = '86119c506d6c82d43d75028ac5810cf12fea372488ff1d33b23bfe8c368ce4ac'
ENDS = [(7, 0), (7, 25), (8, 0), (8, 84), (8, 126), (9, 0), (9, 128), (10, 0)]
SEEDS = ['2026102201', '2026102202']
STYLES = [
    ('current_dc', 'Frozen current Dc', '#767676', '--', 's'),
    ('old_recipe_control', 'Old-recipe control', '#B64342', ':', 'D'),
    ('multiprefix_unconditional', 'Multi-prefix', '#3775BA', '-', 'o'),
    ('multiprefix_full_anchor', 'Multi-prefix + full anchor', '#5E966B', '-.', '^'),
]
METRICS = ['mse', 'psnr_db', 'lpips_alex', 'ms_ssim', 'utility']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    p = Path(path)
    assert p.is_file() and not p.is_symlink(), p
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def checked(desc):
    p = Path(desc['path'])
    assert pin(p) == desc, p
    return p


def read(path):
    return json.loads(Path(path).read_text())


def csvread(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def save_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def process_gone(identity):
    p = Path('/proc') / str(identity['pid']) / 'stat'
    if not p.exists():
        return True
    ticks = int(p.read_text().rsplit(')', 1)[1].split()[19])
    return ticks != identity['start_ticks']


def load_closed(root, run):
    """Read only the sealed CPU delivery; no active training-grid audit."""
    assert sha(root / 'registration.json') == REG_SHA
    wait = read(run / 'summary_dispatch_v1/actual_wait.json')
    assert type(wait['returncode']) is int and wait['returncode'] == 0
    assert wait['actual_wait'] is True
    assert not any(wait.get(k, False) for k in ['timed_out', 'terminated', 'killed'])
    assert process_gone(wait['process'])
    assert not (run / 'summary_dispatch_v1/failure.json').exists()
    delivery = root / 'delivery_v1'
    manifest = read(delivery / 'manifest.json')
    expected = {'source_metrics.csv', 'summary.csv', 'paired_differences.csv',
                'endpoint_rates.csv', 'closure_audit.json'}
    assert {Path(d['path']).name for d in manifest['files']} == expected
    for desc in manifest['files']:
        assert Path(desc['path']).parent.resolve() == delivery.resolve()
        checked(desc)
    closure = read(delivery / 'closure_audit.json')
    assert closure['status'] == 'PASS' and closure['registration']['sha256'] == REG_SHA
    assert closure['source_CI_does_not_estimate_training_seed_distribution'] is True
    assert len(closure['jobs']) == 6 and closure['main_training_samples'] == 240000
    assert closure['bootstrap_seed'] == 2026102209
    summary = csvread(delivery / 'summary.csv')
    paired = csvread(delivery / 'paired_differences.csv')
    rates = csvread(delivery / 'endpoint_rates.csv')
    assert len(summary) == 400 and len(paired) == 720 and len(rates) == 8
    for rows in [summary, paired]:
        for row in rows:
            assert int(row['sources']) == 200 and row['metric'] in METRICS
            assert row['subset'] == 'historical_calibration_new_validation200'
            assert all(math.isfinite(float(row[k])) for k in ['mean', 'ci_low', 'ci_high'])
            assert float(row['ci_low']) <= float(row['ci_high'])
    smap = {(r['arm'], r['seed'], int(r['m']), int(r['K']), r['metric']): r for r in summary}
    pmap = {(r['comparison'], r['seed'], int(r['m']), int(r['K']), r['metric']): r for r in paired}
    assert len(smap) == len(summary) and len(pmap) == len(paired)
    for arm, *_ in STYLES:
        for seed in (['frozen'] if arm == 'current_dc' else [*SEEDS, 'mean_of_2']):
            for m, k in ENDS:
                for metric in METRICS:
                    assert (arm, seed, m, k, metric) in smap
    rmap = {(int(r['m']), int(r['K'])): r for r in rates}
    assert set(rmap) == set(ENDS)
    x = [float(rmap[e]['mean_bits_per_pixel']) for e in ENDS]
    assert all(math.isfinite(v) and v > 0 for v in x)
    assert all(a < b for a, b in zip(x, x[1:]))
    return smap, pmap, x, manifest


def export(fig, out, name, plt, mpl):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden_ticks = set()
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            low, high = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    hidden_ticks.update([tick.label1, tick.label2])
    for text in fig.findobj(match=mpl.text.Text):
        if text not in hidden_ticks and text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, (name, text.get_text(), box)
            assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, (name, text.get_text(), box)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(out / f'{name}.{ext}', dpi=600, facecolor='white')
    fig.savefig(out / f'{name}_print.png', dpi=160, facecolor='white')
    # Rendering is deterministic QA preparation, not an actual human visual review.
    from PIL import Image
    with Image.open(out / f'{name}_print.png') as im:
        im.convert('L').save(out / f'{name}_grayscale.png')
    plt.close(fig)


def main(root, run):
    root, run = Path(root), Path(run)
    smap, pmap, rates, manifest = load_closed(root, run)
    import numpy as np
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    out = root / 'figures_v3'
    out.mkdir(exist_ok=False)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5,
        'axes.labelsize': 8.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })

    def series(ax, rows, color, line, marker, label):
        mean, lo, hi = (np.array([float(r[k]) for r in rows]) for k in ['mean', 'ci_low', 'ci_high'])
        # Draw actual saved CI endpoints independently of the point estimate.
        # Percentile intervals are not assumed to contain their mean.
        ax.errorbar(rates, (lo + hi) / 2, yerr=(hi - lo) / 2, fmt='none',
                    color=color, capsize=2, elinewidth=.8, capthick=.8, zorder=2)
        ax.plot(rates, mean, color=color, ls=line, marker=marker, ms=4,
                lw=1.5, markerfacecolor='white' if marker == 's' else color,
                markeredgewidth=.8, zorder=3, label=label)

    def decorate(ax):
        ax.tick_params(length=3, width=.6)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis='y', color='#dddddd', lw=.4, zorder=0)
        ax.margins(x=.045)

    def summary_rows(arm, seed, metric):
        return [smap[arm, 'frozen' if arm == 'current_dc' else seed, m, k, metric] for m, k in ENDS]

    metric_titles = [('psnr_db', 'PSNR (dB) ↑'), ('lpips_alex', 'LPIPS-Alex ↓'), ('ms_ssim', 'MS-SSIM ↑')]
    limits = {}
    for metric, _ in metric_titles:
        values = [float(r[k]) for r in smap.values() if r['metric'] == metric for k in ['mean', 'ci_low', 'ci_high']]
        lo, hi = min(values), max(values)
        margin = max(.07 * (hi - lo), 1e-4)
        limits[metric] = (lo - margin, hi + margin)
    for seed, name in [('mean_of_2', 'rd_mean_two_seeds'), (SEEDS[0], 'rd_seed_2026102201'), (SEEDS[1], 'rd_seed_2026102202')]:
        fig, axes = plt.subplots(1, 3, figsize=(178 / 25.4, 70 / 25.4))
        fig.subplots_adjust(left=.075, right=.985, bottom=.23, top=.74, wspace=.36)
        for i, (ax, (metric, title)) in enumerate(zip(axes, metric_titles)):
            for arm, label, color, line, marker in STYLES:
                series(ax, summary_rows(arm, seed, metric), color, line, marker, label)
            ax.set_title(f'({chr(97+i)}) {title}', loc='left')
            ax.set_ylim(limits[metric]); decorate(ax)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.51, 1.0),
                   frameon=False, columnspacing=1.4, handlelength=2.8)
        fig.supxlabel('Mean terminated arithmetic rate (bits/pixel)', y=.045, fontsize=8.5)
        export(fig, out, name, plt, mpl)

    panels = [
        ('paired_vs_frozen', [(s[0] + '_minus_current_dc', {'old_recipe_control': 'Control − frozen', 'multiprefix_unconditional': 'Multi-prefix − frozen', 'multiprefix_full_anchor': 'Anchored − frozen'}[s[0]], s[2], s[4]) for s in STYLES[1:]]),
        ('paired_internal', [
            ('multiprefix_unconditional_minus_old_recipe_control', 'Multi − control', '#3775BA', 'o'),
            ('multiprefix_full_anchor_minus_old_recipe_control', 'Anchor − control', '#5E966B', '^'),
            ('multiprefix_full_anchor_minus_multiprefix_unconditional', 'Anchor − multi', '#4D4D4D', 'D'),
        ]),
    ]
    for name, comparisons in panels:
        fig, axes = plt.subplots(2, 3, figsize=(178 / 25.4, 108 / 25.4))
        fig.subplots_adjust(left=.12, right=.985, bottom=.14, top=.9, hspace=.5, wspace=.52)
        for col, (comparison, label, color, marker) in enumerate(comparisons):
            for row, metric in enumerate(['psnr_db', 'lpips_alex']):
                ax = axes[row, col]
                values = [pmap[comparison, 'mean_of_2', m, k, metric] for m, k in ENDS]
                series(ax, values, color, '-', marker, label)
                ax.axhline(0, color='#555555', ls='--', lw=.75, zorder=0)
                ax.set_title(f'({chr(97+row*3+col)}) {label}', loc='left', fontsize=8)
                ax.set_ylabel('ΔPSNR (dB) ↑' if row == 0 else 'ΔLPIPS-Alex ↓')
                decorate(ax)
        fig.supxlabel('Mean terminated arithmetic rate (bits/pixel)', y=.035, fontsize=8.5)
        export(fig, out, name, plt, mpl)

    caption = '''# Saved-statistics figure captions

Rate–distortion figures: eight registered complete/within-scale endpoints on 200 validation sources from the historical calibration population, excluding the previous pilot100. The horizontal coordinate is the population mean actual terminated arithmetic rate, without radio framing or a wireless budget claim. Each trained arm completed 10,000 updates, batch four. The overview averages the two registered training seeds per source before using the saved source-bootstrap interval. The seed-specific figures use identical axis limits. All error bars are saved pointwise 95% source-bootstrap intervals; no statistics are recomputed by this renderer.

Paired figures: the first named arm minus its comparator, averaged across the two registered seeds per source. The first paired figure compares all trained arms with frozen current Dc. The second compares the trained arms internally; Multi denotes multi-prefix, Anchor denotes multi-prefix plus full anchor, and control denotes old-recipe control. Intervals are saved direct source-paired differences. Positive ΔPSNR and negative ΔLPIPS favor the first arm. Zero, unfavorable differences and crossing intervals are retained. These source intervals do not estimate training-seed population uncertainty.

The full-anchor arm has additional frozen-teacher and full-student forward computation. Equal updates and main-source exposures do not imply equal FLOPs. All five scalar metrics, all endpoints and both individual seeds remain in the complete delivery CSVs. The historical calibration results do not replace the frozen WCL model and do not establish blind-test or wireless performance.
'''
    (out / 'captions.md').write_text(caption, encoding='utf-8')
    save_json(out / 'provenance.json', dict(
        registration_sha256=REG_SHA, source=pin(Path(__file__)),
        input_manifest=pin(root / 'delivery_v1/manifest.json'), input_files=manifest['files'],
        summary_actual_wait=pin(run / 'summary_dispatch_v1/actual_wait.json'),
        styles=STYLES, endpoints=ENDS, rates_bits_per_pixel=rates,
        sizes_mm={'rd': [178, 70], 'paired': [178, 108]}, font_points=[8, 8.5, 9], dpi=600,
        skill='C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md',
        references=['references/design-theory.md', 'references/common-patterns.md', 'references/demos.md'],
        visual_review='PENDING_ACTUAL_OPEN', new_inference=0, new_bootstrap=0,
    ))
    save_json(out / 'manifest.json', dict(files=[pin(p) for p in sorted(out.iterdir()) if p.is_file()]))
    print(json.dumps(dict(status='RENDERED_VISUAL_REVIEW_PENDING', output=str(out), manifest=pin(out / 'manifest.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    main(args.results, args.run)
