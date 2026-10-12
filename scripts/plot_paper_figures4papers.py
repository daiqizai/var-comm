"""Layout-only figures4papers skill trial on already published VAR_COMM figures."""
import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 8.5
    axes_linewidth: float = 1.0
    use_tex: bool = False
    font_family: tuple = ('DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif')

PALETTE = {'blue_main': '#0F4D92', 'red_strong': '#B64342',
           'teal': '#42949E', 'violet': '#9A4D8E'}
METHODS = {
    'proposed': ('Partial-scale digital + VAR (ours)', 'blue_main', '-', 'o'),
    'latent': ('Latent continuous JSCC (100k)', 'red_strong', '--', 's'),
    'swin': ('SwinJSCC (adapted, 200k)', 'teal', '-.', '^'),
    'bpg': ('Adaptive-resolution BPG + LDPC', 'violet', ':', 'D'),
}
METRICS = [('psnr_db', 'PSNR ↑', 'dB', 1),
           ('lpips_alex', 'LPIPS ↓', 'Distance', 1),
           ('dinov2_vitl14_cosine', 'DINOv2-L ↑', 'Cosine similarity', 1),
           ('convnext_top1_source_prediction', 'ConvNeXt agreement ↑', 'Prediction agreement (%)', 100)]
SNR = [1, 4, 7, 10, 13, 19]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def apply_publication_style(style=FigureStyle()):
    available = {font.name for font in font_manager.fontManager.ttflist}
    families = [name for name in style.font_family if name in available or name == 'sans-serif']
    plt.rcParams.update({'font.family': families, 'font.size': style.font_size,
        'axes.linewidth': style.axes_linewidth, 'axes.spines.top': False,
        'axes.spines.right': False, 'legend.frameon': False, 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'text.usetex': style.use_tex, 'axes.labelsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'savefig.facecolor': 'white'})

def finalize_figure(fig, out_path, formats=('pdf', 'svg', 'png'), dpi=600):
    for ext in formats:
        fig.savefig(out_path.with_suffix('.' + ext), dpi=dpi, bbox_inches='tight', pad_inches=.045)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--individual-dir', type=Path)
    args = ap.parse_args(); root = args.root.resolve()
    prior = root/'paper/figures/final_paper_comparisons_20261012'
    original = root/'paper/figures/hifi_development_N1024_13dB_group02'
    images = args.individual_dir or original/'individual'
    out = root/'paper/figures/figures4papers_trial_20261012'
    out.mkdir(parents=True, exist_ok=True)
    source = prior/'plot_data.csv'
    assert sha(source) == json.loads((prior/'validation.json').read_text())['outputs']['plot_data.csv']
    with source.open(newline='', encoding='utf-8-sig') as f: rows = list(csv.DictReader(f))
    assert len(rows) == 96
    for row in rows:
        assert row['population'] == 'common500' and int(row['N']) == 1024
        assert int(row['source_count']) == 500 and int(row['noise_count']) == 3
        assert float(row['ci_low']) <= float(row['mean']) <= float(row['ci_high'])
    shutil.copy2(source, out/'plot_data.csv')
    apply_publication_style()

    def panel(ax, spec, letter):
        metric, label, unit, scale = spec; limits = []
        for method, (name, color, linestyle, marker) in METHODS.items():
            group = sorted([r for r in rows if r['method'] == method and r['metric'] == metric], key=lambda r: float(r['snr_db']))
            assert [int(r['snr_db']) for r in group] == SNR
            y, lo, hi = [np.asarray([float(r[k]) for r in group]) * scale for k in ('mean', 'ci_low', 'ci_high')]
            assert all(v.ndim == 1 and len(v) == len(SNR) for v in (y, lo, hi))
            ax.fill_between(SNR, lo, hi, color=PALETTE[color], alpha=.12, linewidth=0)
            ax.plot(SNR, y, color=PALETTE[color], linestyle=linestyle, marker=marker,
                    linewidth=1.7, markersize=4.0, markeredgewidth=.7, label=name)
            limits.extend(lo); limits.extend(hi)
        margin=.07*(max(limits)-min(limits)); ax.set_ylim(min(limits)-margin, max(limits)+margin)
        ax.set_xlim(.5, 19.5); ax.set_xticks(SNR); ax.set_xlabel('SNR (dB)'); ax.set_ylabel(unit)
        ax.set_title(f'({letter})  {label}', loc='left', fontsize=9, fontweight='bold', pad=7)
        ax.grid(axis='y', color='#E5E5E5', linewidth=.5); ax.set_axisbelow(True)
        ax.tick_params(width=.8, length=3)

    fig = plt.figure(figsize=(7, 4.6), layout='constrained')
    gs = fig.add_gridspec(3, 2, height_ratios=[.31, 1, 1], hspace=.06, wspace=.10)
    legend_ax=fig.add_subplot(gs[0,:]); legend_ax.set_axis_off()
    axes=[fig.add_subplot(gs[i//2+1,i%2]) for i in range(4)]
    for ax,spec,letter in zip(axes,METRICS,'abcd'): panel(ax,spec,letter)
    legend_ax.legend(*axes[0].get_legend_handles_labels(), loc='center', ncol=2,
        fontsize=8, handlelength=2.7, columnspacing=1.7)
    finalize_figure(fig,out/'fig_main_N1024_common500')
    for spec,letter in zip(METRICS,'abcd'):
        fig=plt.figure(figsize=(3.5,3.05),layout='constrained')
        gs=fig.add_gridspec(2,1,height_ratios=[.42,1]); legend_ax=fig.add_subplot(gs[0]);legend_ax.set_axis_off()
        ax=fig.add_subplot(gs[1]);panel(ax,spec,letter)
        legend_ax.legend(*ax.get_legend_handles_labels(),loc='center',fontsize=7,handlelength=2.5)
        finalize_figure(fig,out/('fig_main_'+spec[0]))

    image_bindings={}
    def photo_grid(columns, name, subtitle):
        n=len(columns)
        fig=plt.figure(figsize=(7,7*4/n+.69))
        gs=fig.add_gridspec(4,n,left=.035,right=.997,bottom=.007,top=.90,wspace=.022,hspace=.022)
        for row,letter in enumerate('abcd'):
            for col,(key,label) in enumerate(columns):
                path=images/f'example_{letter}_{key}.png'; image_bindings[path.name]=sha(path)
                im=np.asarray(Image.open(path).convert('RGB'));assert im.shape==(256,256,3)
                ax=fig.add_subplot(gs[row,col]);ax.imshow(im,interpolation='none');ax.set_axis_off()
                if row==0:ax.set_title(label,fontsize=8,fontweight='bold',pad=5)
                if col==0:ax.text(-.035,.5,f'({letter})',transform=ax.transAxes,ha='right',va='center',fontsize=8)
        fig.text(.52,.985,'N = 1024  |  SNR = 13 dB',ha='center',va='top',fontsize=9,fontweight='bold')
        fig.text(.52,.957,subtitle,ha='center',va='top',fontsize=7,color='#4D4D4D')
        finalize_figure(fig,out/name)
    photo_grid([('original','Original'),('swin','SwinJSCC\n(adapted, 80k)'),
        ('continuous_latent_jscc','Latent continuous\nJSCC (40k)'),('hifi_diffcom','HiFi-DiffCom\n(adapted)'),
        ('partial_scale_with_var','Partial digital + VAR\n(ours)')],
        'fig_qualitative_historical_development','Historical development examples · fixed sources and first noise realization')
    photo_grid([('original','Original'),('complete_scales_with_var','Complete scales\n+ VAR'),
        ('partial_scale_without_completion','Partial scales\nwithout completion'),
        ('partial_scale_with_var','Partial scales\n+ VAR (ours)')],
        'fig_mechanism_development','Development · last two columns share received tokens and frozen decoder')
    for name in ('selection_and_integrity.json','provenance.csv'):
        shutil.copy2(original/name,out/name)
    (out/'captions.tex').write_text((prior/'captions.tex').read_text()+'\n'+(original/'captions.tex').read_text(),encoding='utf-8')
    (out/'README.md').write_text('''# figures4papers style trial

Uses the scientific-figure-making skill from https://github.com/ChenLiu-1996/figures4papers/tree/main/scientific-figure-making (SKILL.md, design-theory.md, api.md and common-patterns.md). Adapted conventions: FigureStyle, apply_publication_style, finalize_figure, deep-blue key method, warm-red/teal/violet comparators, clean spines, dedicated legend area, consistent line weights and vector text. The user's 7-inch double-column and 8–9pt final-size typography requirements take precedence over the skill's large-canvas defaults. Different markers/line patterns retain grayscale readability. Shaded bands use published intervals, not estimated deviations.

This is a separate layout trial, preserving all original figure files. plot_data.csv is an exact byte copy of the preceding96-row common500 main-figure CSV. No new scientific calls, statistics, point selection, cropping, smoothing or image enhancement. Four independent quantitative panels accompany the2x2 composite. The qualitative panels retain all four fixed sources and the same256x256 display pixels. Only layout and public labels change.

IMPORTANT: performance curves use final latent100k/Swin200k, while the HiFi-matched qualitative panel is historical development with latent40k/Swin80k. It is explicitly labeled and cannot be described as final-baseline holdout reconstruction. Digital mechanism examples are also development. Fixed first noise2001 for latent/Swin/HiFi and6201 for digital; cross-family noise draws differ. Same-SNR HiFi only; no substitutions. Original captions and source/noise/receiver proofs accompany this trial.

Reproduce from the repository root with matplotlib, NumPy and Pillow:

```sh
python scripts/plot_paper_figures4papers.py --individual-dir PATH_TO_EXISTING_GROUP02_INDIVIDUAL_PNGS
```

Default individual PNG path: paper/figures/hifi_development_N1024_13dB_group02/individual. These display assets remain in the local figure backups. No inference is required. PDF/SVG quantitative curves are vector; photographs are embedded rasters, with vector labels. PNG previews are600dpi.
''',encoding='utf-8')
    (out/'validation.json').write_text(json.dumps({'status':'PASS','data_csv_identical':sha(source)==sha(out/'plot_data.csv'),
        'data_sha256':sha(source),'new_scientific_calls':0,'qualitative_image_sha256':image_bindings,
        'input_validation_sha256':sha(prior/'validation.json'),
        'outputs':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.pdf','.svg','.png')}},indent=2))
    print(json.dumps({'output':str(out),'figures':7,'new_scientific_calls':0}))

if __name__=='__main__': main()
