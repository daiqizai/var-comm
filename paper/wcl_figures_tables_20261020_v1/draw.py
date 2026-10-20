"""Render frozen CSV values only. No inference, decoder, or bootstrap is imported.

Run: python -B draw.py [--output PATH]
Requires numpy, matplotlib and Pillow. The five byte-pinned CSVs are shipped in
inputs/. All statistics are read, never re-estimated. Output may be regenerated.
"""
from pathlib import Path
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SNR = [4, 10, 19]
S = json.loads((HERE/'style.json').read_text())
MANIFEST = json.loads((HERE/'input_manifest.json').read_text())
AUDIT = {'baseline_commit': S['baseline_commit'], 'new_scientific_calls': 0,
         'new_bootstrap_draws': 0, 'panels': [], 'table_rows': [], 'exports': []}


def text(path, value):
    path.write_bytes(value.encode('utf8'))


def read(name):
    item = next(r for r in MANIFEST['inputs'] if r['local_path'] == 'inputs/'+name)
    p = HERE/item['local_path']
    b = p.read_bytes()
    assert len(b) == item['bytes'] and hashlib.sha256(b).hexdigest() == item['sha256'], name
    with p.open(newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))


def one(rows, **filters):
    found = [r for r in rows if all(r[k] == str(v) for k, v in filters.items())]
    assert len(found) == 1, (filters, len(found))
    return found[0]


def triple(row, prefix=''):
    a = np.array([float(row[prefix+k]) for k in ['mean','ci_low','ci_high']])
    assert np.isfinite(a).all() and a[1] <= a[0] <= a[2]
    return a


def rc():
    plt.rcParams.update({
        'font.family': S['font_family'], 'font.size': S['axis_pt'],
        'axes.labelsize': S['axis_pt'], 'axes.titlesize': S['title_pt'],
        'xtick.labelsize': S['tick_pt'], 'ytick.labelsize': S['tick_pt'],
        'legend.fontsize': S['legend_pt'], 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'text.color': '#222222', 'axes.labelcolor': '#222222',
        'xtick.color': '#333333', 'ytick.color': '#333333',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'wcl-frozen-a5fc2de0-v1', 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'axes.unicode_minus': True,
    })


def line_options(spec):
    ls = spec['linestyle']
    if isinstance(ls, list): ls = (ls[0], tuple(ls[1]))
    return dict(color=spec['color'], linestyle=ls, linewidth=spec['linewidth'],
                marker=spec['marker'], markersize=spec['markersize'],
                markerfacecolor=spec['color'] if spec['filled'] else 'white',
                markeredgecolor=spec['color'], markeredgewidth=1.0)


def base_axis(ax, xlabel=True):
    ax.set_xlim(2.8,20.2)
    ax.set_xticks(SNR)
    if xlabel: ax.set_xlabel('SNR (dB)', labelpad=3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#E5E5E5', linewidth=.4)
    ax.tick_params(pad=3)


def error_curve(ax, x, data, spec, connect=True):
    # Draw the exact CI under opaque markers; no x shift and no inflated CIs.
    means,lo,hi = data.T
    ax.errorbar(x,means,yerr=np.vstack((means-lo,hi-means)),fmt='none',
                ecolor=spec['color'],elinewidth=.95,capsize=3.0,capthick=.95,zorder=3)
    opts=line_options(spec)
    if not connect:opts['linestyle']='none'
    ax.plot(x,means,zorder=4,**opts)


def export(fig, out, stem, mm):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=fig.bbox
    # Check text and legends in final physical dimensions; skip empty tick text.
    for artist in fig.findobj(lambda a: isinstance(a, matplotlib.text.Text)):
        if not artist.get_visible() or not artist.get_text():continue
        box=artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= bounds.x1+1 and box.y1 <= bounds.y1+1, (stem,artist.get_text(),box,bounds)
    for suffix in ['pdf','svg','png']:
        kw = {'dpi':S['dpi']}
        if suffix == 'pdf':kw['metadata']={'Creator':'Frozen CSV rendering with Matplotlib','CreationDate':None,'ModDate':None}
        if suffix == 'svg':kw['metadata']={'Date':None}
        fig.savefig(out/(stem+'.'+suffix),**kw)  # no tight crop: exact requested page size
    p=out/(stem+'.svg')
    # Whitespace-only XML normalization, before hashing/review, preserves vectors.
    text(p,'\n'.join(line.rstrip(' \t') for line in p.read_text().splitlines())+'\n')
    root=ET.parse(p).getroot()
    assert not root.findall('.//{http://www.w3.org/2000/svg}image')
    assert len(root.findall('.//{http://www.w3.org/2000/svg}text')) > 10
    im=Image.open(out/(stem+'.png')).convert('RGB')
    im.convert('L').save(out/(stem+'_grayscale.png'),dpi=(600,600))
    # A 96 dpi preview represents the same physical page, not added detail.
    size=tuple(round(v/25.4*96) for v in mm)
    im.resize(size,Image.Resampling.LANCZOS).save(out/(stem+'_print_size.png'),dpi=(96,96))
    im.convert('L').resize(size,Image.Resampling.LANCZOS).save(out/(stem+'_print_size_grayscale.png'),dpi=(96,96))
    AUDIT['exports'].append({'stem':stem,'size_mm':mm,'png_pixels':list(im.size),'editable_svg_text':True,'svg_raster_images':0,'text_bounds_checked':True})
    plt.close(fig)


def figure2(out):
    rows=read('figure2_data.csv')
    assert len(rows)==60
    specs=[('psnr_db','(a) PSNR (dB) ↑',(13,25),[13,16,19,22,25]),
           ('lpips_alex','(b) LPIPS-Alex ↓',(0,.8),[0,.2,.4,.6,.8]),
           ('dreamsim','(c) DreamSim ↓',(0,.65),[0,.2,.4,.6]),
           ('convnext_top1_source_prediction','(d) Prediction agreement (%) ↑',(0,100),[0,25,50,75,100])]
    mm=S['figure2_mm']
    fig,axes=plt.subplots(2,2,figsize=np.array(mm)/25.4)
    fig.subplots_adjust(left=.085,right=.98,bottom=.115,top=.79,wspace=.25,hspace=.43)
    for i,(metric,title,limits,ticks) in enumerate(specs):
        ax=axes.flat[i];base_axis(ax,xlabel=i>=2)
        ax.set_title(title,loc='left',pad=7)
        ax.set_ylim(*limits);ax.set_yticks(ticks)
        if metric in ['lpips_alex','dreamsim']:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v:.1f}'))
        values={}
        for method in S['method_order']:
            selected=[one(rows,method=method,metric=metric,snr_db=x) for x in SNR]
            assert all(r['sources']=='500' and r['noise_per_source']=='3' for r in selected)
            data=np.array([triple(r,'display_') for r in selected])
            assert np.all(data[:,1]>=limits[0]) and np.all(data[:,2]<=limits[1]),(metric,method,'CI clipping')
            if metric=='convnext_top1_source_prediction':
                assert all(r['display_scale']=='100' for r in selected)
            error_curve(ax,SNR,data,S['methods'][method])
            values[method]=data.tolist()
        AUDIT['panels'].append({'panel':'figure2_'+metric,'source':'inputs/figure2_data.csv',
            'columns':['display_mean','display_ci_low','display_ci_high'],'x':SNR,'y_limits':limits,'values':values})
    # Row-major reading order with matplotlib's column-major legend filling.
    order=[S['method_order'][i] for i in [0,3,1,4,2]]
    handles=[Line2D([],[],label=S['methods'][m]['label'],**line_options(S['methods'][m])) for m in order]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.53,.99),
               handlelength=2.5,handletextpad=.55,columnspacing=1.7,labelspacing=.65,borderaxespad=0)
    export(fig,out,'figure2',mm)


def figure3(out):
    paired=read('paired_common500.csv');util=read('family_snr_summary.csv')
    data=np.array([triple(one(paired,population='common500',comparison='PARTIAL_minus_WHOLE',metric='psnr_db',snr_db=x)) for x in SNR])
    mm=S['figure3_mm'];fig,axes=plt.subplots(1,2,figsize=np.array(mm)/25.4)
    fig.subplots_adjust(left=.085,right=.98,bottom=.205,top=.77,wspace=.27)
    ax=axes[0];base_axis(ax)
    ax.set_title('(a) PSNR gain (dB)',loc='left',pad=8)
    ax.set_ylim(-.05,.85);ax.set_yticks([0,.2,.4,.6,.8])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.1f}'))
    ax.axhline(0,color='#666666',linewidth=.7,linestyle=(0,(3,2)),zorder=2)
    error_curve(ax,SNR,data,S['methods']['EC_VAR_PARTIAL'],connect=False)
    assert data[:,1].min()>-.05 and data[:,2].max()<.85
    AUDIT['panels'].append({'panel':'figure3_psnr_gain','source':'inputs/paired_common500.csv','comparison':'PARTIAL_minus_WHOLE',
        'metric':'psnr_db','x':SNR,'columns':['mean','ci_low','ci_high'],'values':data.tolist(),'interval':'direct source-paired 95% CI'})
    ax=axes[1]
    ax.set_title('(b) Source-capacity utilization (%)',loc='left',pad=8)
    ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100]);ax.set_xticks(range(3),SNR)
    ax.set_xlabel('SNR (dB)',labelpad=3);ax.set_axisbelow(True);ax.yaxis.grid(True,color='#E5E5E5',linewidth=.4)
    widths=.31;values={}
    for family,label,color,hatch,offset in [('WHOLE','Whole-scale only','#767676','///',-.18),('PARTIAL','NeST-Com','#3775BA',None,.18)]:
        selected=[one(util,family=family,snr_db=x) for x in SNR]
        assert all(r['sources']=='500' for r in selected)
        ys=[float(Decimal(r['mean_utilization'])*Decimal(100)) for r in selected]
        bars=ax.bar(np.arange(3)+offset,ys,widths,label=label,color=color,edgecolor='white' if hatch else '#275B93',hatch=hatch,linewidth=.6,zorder=3)
        for b,y in zip(bars,ys):ax.text(b.get_x()+b.get_width()/2,y+1.4,f'{y:.1f}',ha='center',va='bottom',fontsize=8)
        values[family]=ys
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.23),ncol=2,handlelength=1.4,columnspacing=1.2,fontsize=8.2,borderaxespad=0)
    AUDIT['panels'].append({'panel':'figure3_utilization','source':'inputs/family_snr_summary.csv','column':'mean_utilization',
        'display_scale':100,'values':values,'error_bars':False,'p05_p95_used':False})
    export(fig,out,'figure3',mm)


def table(out):
    adaptive=read('n2048_adaptive_paired.csv');full=read('n2048_full_paired.csv')
    lines=[r'% Requires booktabs. Data frozen at a5fc2de0c0039689f2999ad0ed6a1c12c1487d06.',
      r'\begin{table}[t]',r'\centering',r'\caption{Source-paired NeST--ARPC differences at $N=2048$.}',
      r'\label{tab:n2048-paired}',r'\begingroup',r'\fontsize{8.5}{10}\selectfont',r'\setlength{\tabcolsep}{3pt}',
      r'\renewcommand{\arraystretch}{1.15}',r'\begin{tabular}{@{}lcc@{}}',r'\toprule',
      r'Condition & \shortstack{$\Delta$PSNR (dB)\\[1pt][95\% CI]} & \shortstack{$\Delta$DISTS\\[1pt][95\% CI]} \\',r'\midrule']
    for snr in SNR:
        rows=full if snr==19 else adaptive
        comparison='NeST_full_m10_minus_ARPC_full_e7' if snr==19 else 'NeST_adaptive_minus_ARPC_adaptive'
        condition=r'\shortstack[l]{19 dB\\full representations}' if snr==19 else str(snr)+' dB adaptive'
        cells=[]
        for metric,nd in [('psnr_db',3),('dists',4)]:
            row=one(rows,comparison=comparison,snr_db=snr,metric=metric)
            assert row['source_count']=='100'
            triple(row)
            vals=[format(Decimal(row[k]),f'.{nd}f') for k in ['mean','ci_low','ci_high']]
            cells.append(r'\shortstack{$'+vals[0]+r'$\\[1pt]$['+vals[1]+', '+vals[2]+r']$}')
            AUDIT['table_rows'].append({'snr_db':snr,'condition':'full representations' if snr==19 else 'adaptive',
                'source':'inputs/n2048_full_paired.csv' if snr==19 else 'inputs/n2048_adaptive_paired.csv',
                'comparison':comparison,'metric':metric,'mean':row['mean'],'ci_low':row['ci_low'],'ci_high':row['ci_high'],'display':vals})
            if snr==19 and metric=='dists':assert Decimal(row['ci_low'])>0
        lines.append(condition+' & '+' & '.join(cells)+r' \\[3pt]')
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\vspace{3pt}',r'\begin{minipage}{\columnwidth}',
      r'$\Delta = \mathrm{NeST}-\mathrm{ARPC}$. Matched100; three noise realizations per source; 300/300 correctly delivered frames per method and condition. Intervals are direct source-paired 95\% CIs. Positive $\Delta$PSNR favors NeST; positive $\Delta$DISTS favors ARPC. Adaptive 4/10 dB and full-representation 19 dB use different frozen configurations.',
      r'\end{minipage}',r'\endgroup',r'\end{table}','']
    text(out/'table_i.tex','\n'.join(lines))
    text(out/'table_i_preview.tex',r'''\documentclass[10pt]{article}
\usepackage[paperwidth=100mm,paperheight=90mm,left=5.75mm,right=5.75mm,top=5mm,bottom=5mm]{geometry}
\usepackage{booktabs}
\pagestyle{empty}
\renewcommand{\thetable}{\Roman{table}}
\begin{document}
\input{table_i.tex}
\end{document}
''')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE)
    out=parser.parse_args().output;out.mkdir(parents=True,exist_ok=True)
    assert MANIFEST['baseline_commit']==S['baseline_commit']
    rc();figure2(out);figure3(out);table(out)
    AUDIT['runtime']={'python':__import__('sys').version,'matplotlib':matplotlib.__version__,'numpy':np.__version__}
    text(out/'render_audit.json',json.dumps(AUDIT,indent=2)+'\n')
    print(json.dumps({'status':'RENDERED_PENDING_VISUAL_REVIEW','output':str(out),'panels':len(AUDIT['panels']),'table_metric_cells':len(AUDIT['table_rows']),'new_scientific_calls':0}))


if __name__=='__main__':main()
