"""Draw saved same-version entropy contrasts and unchanged historical raw rows.

The approved Figure 3 visual design is retained. No metric or bootstrap call.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'evidence/common500_paired_points.csv'
S = json.loads((ROOT / 'style/figure_style_v2.json').read_text())
C = json.loads((ROOT / 'style/resolved_style.json').read_text())
OUT = ROOT / 'figures'
outputs, checks = [], []


def descriptor(path):
    b = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(b),
            'sha256': hashlib.sha256(b).hexdigest()}


def configure():
    availability, chosen = {}, None
    for font in S['font_fallback']:
        try:
            availability[font] = font_manager.findfont(font, fallback_to_default=False)
        except ValueError:
            availability[font] = None
        if chosen is None and availability[font]:
            chosen = font
    assert chosen
    plt.rcParams.update({'font.family': chosen, 'font.size': 7.3,
        'axes.labelsize': 8.3, 'xtick.labelsize': 7.3, 'ytick.labelsize': 7.3,
        'legend.fontsize': 7.3, 'axes.linewidth': .7,
        'axes.edgecolor': S['palette']['axis'], 'text.color': S['palette']['text'],
        'axes.labelcolor': S['palette']['text'], 'xtick.color': S['palette']['axis'],
        'ytick.color': S['palette']['axis'], 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'svg.hashsalt': 'nestcom-same-version-paired-v1',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'axes.unicode_minus': True})
    return {'actual_font': chosen, 'availability': availability,
            'matplotlib_version': matplotlib.__version__, 'font_files_distributed': False}


def draw(ax, metric, gray=False):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=S['palette']['grid'], linewidth=.45)
    ax.tick_params(direction='out', length=2.3, width=.65, pad=2)
    ax.set_xlabel('SNR (dB)', labelpad=3)
    for name, style in S['paired_series'].items():
        selected = sorted((r for r in rows if r['method'] == name and r['metric'] == metric),
                          key=lambda r: int(r['snr_db']))
        assert len(selected) == 3
        x = [i + style['offset'] for i in range(3)]
        y = [float(r['mean']) for r in selected]
        lo = [float(r['ci_low']) for r in selected]
        hi = [float(r['ci_high']) for r in selected]
        assert all(C['fig3_limits'][metric][0] < a <= b < C['fig3_limits'][metric][1]
                   for a, b in zip(lo, hi))
        color = ('#333333' if name == 'Entropy-coded' else '#777777') if gray else style['color']
        ax.errorbar(x, y, yerr=[[b-a for a, b in zip(lo, y)], [b-a for a, b in zip(y, hi)]],
                    fmt='none', ecolor=color, elinewidth=.9, capsize=2.6, capthick=.9, zorder=3)
        ax.plot(x, y, ls='none', marker=style['marker'], ms=4.5,
                color=color, mfc=color, mec='white', mew=.55, zorder=4)
    ax.set(xlim=[-.42, 2.42], xticks=[0, 1, 2], xticklabels=['4', '10', '19'],
           ylim=C['fig3_limits'][metric], yticks=C['fig3_ticks'][metric])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f' if metric == 'dreamsim' else '%.1f'))


def title(fig, ax, letter, label, y_mm):
    w, h = fig.get_size_inches() * 25.4
    left = ax.get_position().x0
    fig.text(left, y_mm/h, f'({letter})', fontsize=8.8, ha='left', va='top')
    fig.text(left + 5.3/w, y_mm/h, label, fontsize=8.3, ha='left', va='top')


def export(fig, name, width, height, extensions):
    fig.canvas.draw()
    renderer, canvas = fig.canvas.get_renderer(), fig.bbox
    overflow = []
    for obj in fig.findobj(Text):
        if obj.get_visible() and obj.get_text():
            b = obj.get_window_extent(renderer)
            if b.x0 < -.5 or b.y0 < -.5 or b.x1 > canvas.x1+.5 or b.y1 > canvas.y1+.5:
                overflow.append(obj.get_text())
    assert not overflow, overflow
    for legend in fig.legends:
        b = legend.get_window_extent(renderer)
        assert not any(b.overlaps(ax.get_window_extent(renderer)) for ax in fig.axes)
    checks.append({'figure': name, 'width_mm': width, 'height_mm': height,
                   'text_overflow': overflow, 'legend_overlaps_axes': False})
    for ext in extensions:
        p = OUT / (name + '.' + ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {'Creator': 'NeST-Com deterministic saved paired statistics'}
        if ext == 'pdf':
            meta.update(CreationDate=None, ModDate=None)
        elif ext == 'svg':
            meta['Date'] = None
        fig.savefig(p, dpi=600, bbox_inches=None, metadata=meta)
        if ext == 'svg':
            p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines()) + '\n', encoding='utf8')
        outputs.append(descriptor(p))
    plt.close(fig)


with SOURCE.open(newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 12
assert all(r['bootstrap_recomputed'] == 'False' for r in rows)
assert all(r['version'] == 'n1024_same_version_repair_20261019'
           for r in rows if r['method'] == 'Entropy-coded')
fonts = configure()
for gray in (False, True):
    w, h = C['fig3_width_mm'], C['fig3_height_mm']
    left, right, gap, bottom, top = 9.2, 3.1, 12.0, 10.8, 37.2
    panel_width = (w-left-right-gap)/2
    fig = plt.figure(figsize=(w/25.4, h/25.4), dpi=160)
    for j, metric in enumerate(('psnr_db', 'dreamsim')):
        ax = fig.add_axes([(left+j*(panel_width+gap))/w, bottom/h, panel_width/w, (top-bottom)/h])
        draw(ax, metric, gray)
        title(fig, ax, 'ab'[j], ('ΔPSNR (dB)', 'ΔDreamSim')[j], 42.0)
    handles = []
    for name, style in S['paired_series'].items():
        color = ('#333333' if name == 'Entropy-coded' else '#777777') if gray else style['color']
        handles.append(Line2D([], [], label=name, ls='none', marker=style['marker'],
                              ms=4.5, mfc=color, mec='white', mew=.55))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1-.7/h), ncol=2,
               frameon=False, handletextpad=.35, columnspacing=1.4, borderaxespad=0)
    export(fig, 'paired_common500_gray' if gray else 'paired_common500', w, h,
           ('png',) if gray else ('pdf', 'svg', 'png'))
for j, metric in enumerate(('psnr_db', 'dreamsim')):
    w, h = 45., 50.
    fig = plt.figure(figsize=(w/25.4, h/25.4), dpi=160)
    ax = fig.add_axes([9/w, 11/h, 33/w, 26/h])
    draw(ax, metric)
    title(fig, ax, 'ab'[j], ('ΔPSNR (dB)', 'ΔDreamSim')[j], 45)
    export(fig, 'panels/paired_' + metric, w, h, ('pdf', 'svg', 'png'))
(OUT/'paired_common500_manifest.json').write_text(json.dumps({
    'schema': 'MANUSCRIPT_SAME_VERSION_PAIRED_PLOT_V1', 'input': descriptor(SOURCE),
    'style_inputs': [descriptor(ROOT/'style'/n) for n in ('figure_style_v2.json', 'resolved_style.json')],
    'fonts': fonts, 'layout_checks': checks, 'outputs': outputs,
    'scientific_calls': 0, 'bootstrap_calls': 0, 'metric_calls': 0,
    'raw_rows': 'Unchanged historical saved data and intervals',
    'entropy_rows': 'Closed same-version repair; 19 dB retains T1-C m10',
    'axis_limits_changed': False}, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print(json.dumps({'exports': len(outputs), 'scientific_calls': 0}))
