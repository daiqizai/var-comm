"""Render closed, pinned formal summaries; no inference or statistical estimation.

Usage: python -B draw.py --manifest INPUT_MANIFEST --output NEW_DIRECTORY
The manifest and CSVs are created only after the real statistics run closes.
No source images, models, bootstrap banks or scientific arrays are loaded here.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

HERE = Path(__file__).resolve().parent
SNRS = (4, 10, 19)
METRICS = (
    ('psnr_db', '(a) PSNR (dB) ↑', (13, 25)),
    ('lpips_alex', '(b) LPIPS ↓', (0, .8)),
    ('dreamsim', '(c) DreamSim ↓', (0, .65)),
    ('clip_image_cosine', '(d) CLIP-I ↑', (.5, 1.0)),
)
ORDER = ('NEST', 'P100000', 'Swin200000', 'BPG', 'ARPC_ADAPTED', 'ADA_PREDCLASS_FIXEDFRAME')
ALIASES = {'EC_VAR_PARTIAL': 'NEST', 'NEST_PARTIAL': 'NEST',
           'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024': 'BPG', 'BPG_ADAPTIVE': 'BPG'}


def pin(path):
    b = path.read_bytes()
    return dict(path=str(path), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def checked(root, descriptor):
    p = (root / descriptor['local_path']).resolve()
    assert p.is_relative_to(root.resolve()), 'Input path escapes the package'
    actual = pin(p)
    assert all(actual[k] == descriptor[k] for k in ('bytes', 'sha256'))
    return p


def options(s):
    line = s['linestyle']
    if isinstance(line, list):
        line = (line[0], tuple(line[1]))
    return dict(color=s['color'], marker=s['marker'], linestyle=line,
                linewidth=s['linewidth'], markersize=s['markersize'],
                markerfacecolor=s['color'] if s['filled'] else 'white',
                markeredgecolor=s['color'], markeredgewidth=1)


def save_csv(path, rows):
    with path.open('x', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def draw(rows, population, out, style):
    count = {'common500': 500, 'kodak24': 24}[population]
    rows = [r for r in rows if r['method'] != 'NEST_WHOLE']
    by_key = {}
    for r in rows:
        method = ALIASES.get(r['method'], r['method'])
        key = (method, int(r['snr_db']), r['metric'])
        assert key not in by_key and method in ORDER
        assert r['population'] == population and int(r['source_count']) == count
        assert int(r['logical_frames']) == count * 3
        expected_origin = 'NEW_ADA_FROM_EXISTING_BANK' if method == 'ADA_PREDCLASS_FIXEDFRAME' else 'EXISTING_FROZEN_SUMMARY'
        assert r['statistics_origin'] == expected_origin
        by_key[key] = r
    assert set(by_key) == {(m, s, metric) for m in ORDER for s in SNRS for metric, _, _ in METRICS}
    mm = style['figure_mm']
    fig, axes = plt.subplots(1, 4, figsize=(mm[0] / 25.4, mm[1] / 25.4))
    fig.subplots_adjust(left=.05, right=.992, bottom=.19, top=.70, wspace=.31)
    used, panels = [], []
    for ax, (metric, title, original_limits) in zip(axes, METRICS):
        ax.set_title(title, loc='left', fontsize=8.5, pad=6)
        ax.set_xticks(SNRS)
        ax.set_xlim(2.8, 20.2)
        ax.set_xlabel('SNR (dB)', labelpad=3)
        ax.tick_params(pad=2.5)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#E5E5E5', linewidth=.4)
        all_ci, artists = [], []
        for method in ORDER:
            selected = [by_key[method, snr, metric] for snr in SNRS]
            triples = [[float(r[k]) for k in ('mean', 'ci_low', 'ci_high')] for r in selected]
            assert all(math.isfinite(v) for t in triples for v in t)
            assert all(t[1] <= t[0] <= t[2] for t in triples)
            s = style['methods'][method]
            # Exact interval endpoints, under opaque markers, with no x offset.
            err = ax.errorbar(SNRS, [t[0] for t in triples],
                yerr=[[t[0]-t[1] for t in triples], [t[2]-t[0] for t in triples]],
                fmt='none', ecolor=s['color'], elinewidth=.9, capsize=2.6, capthick=.9, zorder=3)
            line, = ax.plot(SNRS, [t[0] for t in triples], zorder=4, **options(s))
            assert list(line.get_xdata()) == list(SNRS)
            assert list(line.get_ydata()) == [t[0] for t in triples]
            for snr, segment, triple in zip(SNRS, err[2][0].get_segments(), triples):
                expected = ((snr, triple[1]), (snr, triple[2]))
                assert all(abs(float(segment[i][j]) - expected[i][j]) < 1e-14 for i in range(2) for j in range(2))
            all_ci.extend(v for t in triples for v in t[1:])
            artists.append(dict(method=method, exact_mean_and_ci=triples))
            used.extend(selected)
        # Preserve old limits unless actual new CIs require more space; never clip.
        low, high = original_limits
        pad = max(.01, (high-low)*.04)
        low = min(low, min(all_ci)-pad)
        high = max(high, max(all_ci)+pad)
        if metric in ('lpips_alex', 'dreamsim') and min(all_ci) >= 0:
            low = 0
        ax.set_ylim(low, high)
        panels.append(dict(metric=metric, ylim=[low, high], artists=artists))
    handles = [Line2D([], [], label=style['methods'][m]['label'], **options(style['methods'][m])) for m in ORDER]
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.52,.995),
               handlelength=2.2, handletextpad=.5, columnspacing=1.7, fontsize=8.2,
               labelspacing=.6, borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(lambda a: isinstance(a, matplotlib.text.Text)):
        if text.get_visible() and text.get_text():
            b = text.get_window_extent(renderer)
            assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.x1+1 and b.y1 <= fig.bbox.y1+1, text.get_text()
    stem = 'four_metrics_' + population
    assert len(used) == 72
    save_csv(out / (stem+'_data.csv'), used)
    for ext in ('pdf', 'svg', 'png'):
        kwargs = dict(dpi=600)
        if ext == 'pdf':
            kwargs['metadata'] = dict(CreationDate=None, ModDate=None)
        elif ext == 'svg':
            kwargs['metadata'] = dict(Date=None)
        fig.savefig(out / (stem+'.'+ext), **kwargs)
    svg = out / (stem+'.svg')
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n', encoding='utf8')
    root = ET.parse(svg).getroot()
    assert not root.findall('.//{http://www.w3.org/2000/svg}image')
    assert len(root.findall('.//{http://www.w3.org/2000/svg}text')) > 10
    im = Image.open(out / (stem+'.png')).convert('RGB')
    size = tuple(round(v/25.4*96) for v in mm)
    im.resize(size, Image.Resampling.LANCZOS).save(out / (stem+'_print_size.png'), dpi=(96,96))
    im.convert('L').resize(size, Image.Resampling.LANCZOS).save(out / (stem+'_grayscale.png'), dpi=(96,96))
    im.thumbnail((1800,900), Image.Resampling.LANCZOS)
    im.save(out / (stem+'_preview.png'))
    plt.close(fig)
    return dict(population=population, source_count=count, exact_points=72, panels=panels,
                size_mm=mm, svg_raster_images=0, editable_text=True,
                visual_review='PENDING_ACTUAL_OPEN', new_statistics=0, new_model_calls=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    m = json.loads(args.manifest.read_text())
    assert m['schema'] == 'C_FORMAL_FIGURE_INPUTS_V1'
    assert m['closed_statistics_verified'] is True
    assert len(m['repository_commit']) == 40
    root = args.manifest.parent
    wait = json.loads(checked(root,m['actual_wait']).read_text())
    done = json.loads(checked(root,m['completion']).read_text())
    assert wait['actual_wait'] is True and type(wait['returncode']) is int and wait['returncode'] == 0
    assert all(wait[k] is False for k in ('timed_out','terminated','killed')) and wait['metadata_error'] is None
    assert done['status'] == 'C_FORMAL_SCORES_AND_SOURCE_PAIRED_STATISTICS_COMPLETE'
    assert done['registration'] == wait['registration'] and done['ledger']['unresolved'] == 0
    inputs = {}
    for pop in ('common500','kodak24'):
        d = m['populations'][pop]
        matches = [x for x in done['populations'][pop]['outputs'] if x['path'].endswith('/combined_summary.csv')]
        assert len(matches) == 1 and all(matches[0][k] == d[k] for k in ('bytes','sha256'))
        with checked(root,d).open(newline='', encoding='utf8') as f:
            inputs[pop] = list(csv.DictReader(f))
    style = json.loads((HERE/'style.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.frameon':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white',
        'savefig.facecolor':'white','svg.hashsalt':'ada-fixed-frame-formal-v1'})
    args.output.mkdir(parents=True, exist_ok=False)
    checks = [draw(inputs[pop],pop,args.output,style) for pop in ('common500','kodak24')]
    record = dict(input_manifest=pin(args.manifest),script=pin(Path(__file__)),style=pin(HERE/'style.json'),
                  figures=checks,visual_review='PENDING_ACTUAL_OPEN',outputs=[pin(q) for q in sorted(args.output.iterdir())])
    (args.output/'render_validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    main()
