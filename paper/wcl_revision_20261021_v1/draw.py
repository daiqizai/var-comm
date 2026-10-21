"""Render frozen CSV values only. No inference, decoder, or bootstrap is imported.

Run: python -B draw.py [--output PATH]
Requires numpy, matplotlib and Pillow. The byte-pinned inputs are shipped in
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
        'svg.hashsalt': 'wcl-revision-20261021', 'savefig.facecolor': 'white',
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
    raster_count=len(root.findall('.//{http://www.w3.org/2000/svg}image'))
    assert raster_count == (12 if stem == 'figure3' else 0), raster_count
    assert len(root.findall('.//{http://www.w3.org/2000/svg}text')) > 10
    im=Image.open(out/(stem+'.png')).convert('RGB')

    # A 96 dpi preview represents the same physical page, not added detail.
    size=tuple(round(v/25.4*96) for v in mm)
    im.resize(size,Image.Resampling.LANCZOS).save(out/(stem+'_print_size.png'),dpi=(96,96))
    im.convert('L').resize(size,Image.Resampling.LANCZOS).save(out/(stem+'_print_size_grayscale.png'),dpi=(96,96))
    AUDIT['exports'].append({'stem':stem,'size_mm':mm,'png_pixels':list(im.size),'editable_svg_text':True,'svg_raster_images':raster_count,'text_bounds_checked':True})
    im.thumbnail((1500,1100),Image.Resampling.LANCZOS)
    im.save(out/(stem+'_preview.png'))
    plt.close(fig)



def verify_inputs():
    for item in MANIFEST['inputs']:
        b=(HERE/item['local_path']).read_bytes()
        assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'],item
    AUDIT['input_files_verified']=len(MANIFEST['inputs'])

def write_csv(p, rows, fields=None):
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        w.writeheader();w.writerows(rows)

def figure2(out):
    rows=read('eight_metric_summary.csv')
    specs=[('psnr_db','(a) PSNR (dB) ↑',(13,25),[13,16,19,22,25]),
           ('lpips_alex','(b) LPIPS ↓',(0,.8),[0,.2,.4,.6,.8]),
           ('dreamsim','(c) DreamSim ↓',(0,.65),[0,.2,.4,.6]),
           ('clip_image_cosine','(d) CLIP-I ↑',(.5,1.0),[.5,.6,.7,.8,.9,1.0])]
    mm=S['figure2_mm'];fig,axes=plt.subplots(1,4,figsize=np.array(mm)/25.4)
    fig.subplots_adjust(left=.047,right=.99,bottom=.21,top=.72,wspace=.30)
    used=[]
    for ax,(metric,title,limits,ticks) in zip(axes,specs):
        base_axis(ax);ax.set_title(title,loc='left',pad=6,fontsize=8.5)
        ax.set_ylim(*limits);ax.set_yticks(ticks)
        if metric!='psnr_db':ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.1f}'))
        values={}
        for method in S['method_order']:
            selected=[one(rows,method=method,metric=metric,snr_db=x) for x in SNR]
            assert all(r['source_count']=='500' for r in selected)
            data=np.array([triple(r) for r in selected]);error_curve(ax,SNR,data,S['methods'][method])
            assert np.all(data[:,1]>=limits[0]) and np.all(data[:,2]<=limits[1])
            assert np.array_equal(ax.lines[-1].get_ydata(),data[:,0])
            for x,seg,d in zip(SNR,ax.collections[-1].get_segments(),data):
                assert np.allclose(seg,[[x,d[1]],[x,d[2]]],rtol=0,atol=1e-14)
            values[method]=data.tolist();used.extend(selected)
        AUDIT['panels'].append(dict(metric=metric,values=values,actual_artists_verified=True))
    handles=[Line2D([],[],label=S['methods'][m]['label'],**line_options(S['methods'][m])) for m in S['method_order']]
    fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.51,.99),
        handlelength=2.1,handletextpad=.45,columnspacing=1.15,fontsize=8.3,borderaxespad=0)
    assert len(used)==60
    write_csv(out/'figure2_data.csv',used);AUDIT['figure2_exact_means_and_CIs']=60
    export(fig,out,'figure2',mm)

def figure3(out):
    rows=read('display_manifest.csv')
    metric_rows=read('figure3_metrics.csv')
    assert len(metric_rows)==10
    original_manifest=json.loads((HERE/'inputs/examples_manifest.json').read_bytes())
    columns=[('original','Original'),('P100000','Latent JSCC'),('Swin200000','SwinJSCC'),
      ('BPG','BPG + LDPC'),('ARPC','ARPC + LDPC'),('NEST','NeST-Com')]
    subjects=[(3,95,'n07860988/ILSVRC2012_val_00010848_n07860988','Pastries'),
              (1,21,'n02105641/ILSVRC2012_val_00018762_n02105641','White long-haired dog')]
    mm=S['figure3_mm'];fig=plt.figure(figsize=np.array(mm)/25.4)
    left=1.5;gap=1.3;side=(mm[0]-2*left-5*gap)/6
    placed=[];native_hashes=[]
    for c,(_,label) in enumerate(columns):
        fig.text((left+c*(side+gap)+side/2)/mm[0],.98,label,ha='center',va='top',fontsize=8.2)
    for row,(ordinal,idx,sid,subject) in enumerate(subjects):
        fullbottom=[40.0,5.0][row]
        for c,(method,label) in enumerate(columns):
            p=HERE/'inputs/images'/f'{ordinal:02d}_{method}.png';im=Image.open(p).convert('RGB')
            assert im.size==(256,256)
            sh=hashlib.sha256(p.read_bytes()).hexdigest()
            d=next(v for v in original_manifest['files'] if v['path'].endswith('/'+p.name))
            assert sh==d['sha256']
            x=(left+c*(side+gap))/mm[0]
            ax=fig.add_axes([x,fullbottom/mm[1],side/mm[0],side/mm[1]])
            ax.imshow(np.asarray(im),interpolation='none');ax.axis('off')
            native_hashes.append(hashlib.sha256(np.asarray(im).tobytes()).hexdigest())
            entry=dict(row=row+1,column=c+1,subject=subject,source_index=idx,source_id=sid,
                method=method,display_name=label,image_path=str(p.relative_to(HERE)).replace('\\','/'),
                image_png_sha256=sh,full_view=True)
            lab=['(a) Pastries','(b) Dog'][row]
            if method!='original':
                v=one(rows,source_index=idx,method=method)
                assert v['source_id']==sid and v['N']=='1024' and v['snr_db']=='10' and v['image_png_sha256']==sh
                entry.update({k:v[k] for k in ['version','N','snr_db','noise_seed','noise_namespace','receive_status','actual_profile','float_image_path','float_image_file_sha256','psnr_db','lpips_alex']})
                m=one(metric_rows,source_index=idx,method=method,timing_case=f's10_i{idx:04d}_r3')
                for k in ['source_id','snr_db','float_image_path','float_image_file_sha256',
                          'timing_float_array_sha256','image_png_sha256','lpips_alex']:
                    assert m[k]==v[k],(idx,method,k)
                entry.update(dreamsim=m['dreamsim'],timing_case=m['timing_case'],
                    timing_float_array_sha256=m['timing_float_array_sha256'],
                    metric_registration_sha256=m['metric_registration_sha256'])
                lab=f"{float(v['lpips_alex']):.3f} / {float(m['dreamsim']):.3f}"
            fig.text(x+side/2/mm[0],(fullbottom-1.25)/mm[1],lab,ha='center',va='top',fontsize=8.1)
            entry['display_label']=lab;placed.append(entry)
    write_csv(out/'figure3_mapping.csv',placed,list(dict.fromkeys(k for v in placed for k in v)))
    export(fig,out,'figure3',mm)
    import io,base64
    root=ET.parse(out/'figure3.svg').getroot();actual=[]
    for v in root.findall('.//{http://www.w3.org/2000/svg}image'):
        im=Image.open(io.BytesIO(base64.b64decode(v.attrib['{http://www.w3.org/1999/xlink}href'].split(',',1)[1]))).convert('RGB')
        assert im.size == (256,256)
        actual.append(hashlib.sha256(np.asarray(im).tobytes()).hexdigest())
    assert sorted(actual)==sorted(native_hashes)
    AUDIT['figure3']=dict(source_indices=[95,21],row_labels=['(a) Pastries','(b) Dog'],full_views=12,ROI_views=0,native_pixels_verified=True,
      full_field_of_view=True,whole_image_scores=True,new_inference=0,
      displayed_metrics=['lpips_alex','dreamsim'],lower_is_better=[True,True],
      exact_metric_rows_verified=10,lpips_reused_verbatim=True,
      dreamsim_source='results/figure3_dog_20261021_v1/delivery/figure3_metrics.csv')

def cell(v,nd):
    return r'\shortstack{$'+format(Decimal(v['mean']),f'.{nd}f')+r'$\\[1pt]$['+format(Decimal(v['ci_low']),f'.{nd}f')+', '+format(Decimal(v['ci_high']),f'.{nd}f')+r']$}'
def table(out,stem,caption,label,header,body,notes,columns):
    lines=[r'% Requires booktabs. Values and intervals read from frozen CSV.',r'\begin{table}[t]',r'\centering',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begingroup',r'\fontsize{8}{9.5}\selectfont',
        r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{@{}'+columns+r'@{}}',r'\toprule',header+r' \\',r'\midrule']
    lines+=body+[r'\bottomrule',r'\end{tabular}',r'\endgroup',r'\par\smallskip',
      r'\begin{minipage}{\linewidth}\fontsize{7.5}{9}\selectfont',notes,r'\end{minipage}',r'\end{table}']
    text(out/(stem+'.tex'),'\n'.join(lines)+'\n')
def tables(out):
    pairs=read('n1024_paired.csv');util=read('n1024_utilization.csv');rows=[];body=[]
    for snr in SNR:
        w=one(util,family='WHOLE',snr_db=snr);p=one(util,family='PARTIAL',snr_db=snr)
        vals=[one(pairs,population='common500',comparison='PARTIAL_minus_WHOLE',snr_db=snr,metric=m) for m in ['psnr_db','dreamsim']]
        row=dict(snr_db=snr,whole_utilization=w['mean_utilization'],partial_utilization=p['mean_utilization'],source_capacity_bits=w['source_capacity_bits'])
        for m,v in zip(['psnr','dreamsim'],vals):
            assert v['sources']=='500';triple(v)
            for k in ['mean','ci_low','ci_high']:row[m+'_'+k]=v[k]
        rows.append(row)
        body.append(f"{snr} & {100*float(w['mean_utilization']):.2f} & {100*float(p['mean_utilization']):.2f} & "+cell(vals[0],3)+' & '+cell(vals[1],4)+r' \\[3pt]')
    write_csv(out/'table_i.csv',rows)
    table(out,'table_i','Effect of within-scale endpoints at $N=1024$.','tab:within-scale',
      r'SNR & \shortstack{WHOLE\\util. (\%)} & \shortstack{PARTIAL\\util. (\%)} & \shortstack{$\Delta$PSNR (dB)\\{[95\% CI]}} & \shortstack{$\Delta$DreamSim\\{[95\% CI]}}',body,
      r'$\Delta=\mathrm{PARTIAL}-\mathrm{WHOLE}$, common500, three noise realizations per source. Intervals use direct source-paired differences. Utilization is terminated entropy-bitstream length divided by usable source capacity after the 13-bit length and 16-bit CRC fields. Negative $\Delta$DreamSim favors PARTIAL.','r r r c c')
    rows=[];body=[]
    for snr in SNR:
        vals=[one(read('n2048_full_paired.csv' if snr==19 else 'n2048_adaptive_paired.csv'),snr_db=snr,metric=m,
          comparison='NeST_full_m10_minus_ARPC_full_e7' if snr==19 else 'NeST_adaptive_minus_ARPC_adaptive') for m in ['psnr_db','dreamsim']]
        cond=f'{snr} dB '+('full representations' if snr==19 else 'adaptive')
        row=dict(condition=cond,snr_db=snr,comparison='NeST_minus_ARPC',sources=100)
        for m,v in zip(['psnr','dreamsim'],vals):
            assert v['source_count']=='100';triple(v)
            for k in ['mean','ci_low','ci_high']:row[m+'_'+k]=v[k]
        rows.append(row);body.append(cond+' & '+cell(vals[0],3)+' & '+cell(vals[1],4)+r' \\[3pt]')
    write_csv(out/'table_ii.csv',rows)
    table(out,'table_ii','Paired differences at $N=2048$.','tab:larger-budget',
      r'Condition & \shortstack{$\Delta$PSNR (dB)\\{[95\% CI]}} & \shortstack{$\Delta$DreamSim\\{[95\% CI]}}',body,
      r'$\Delta=\mathrm{NeST}-\mathrm{ARPC}$. Matched100, three noise realizations per source, 300/300 delivered frames for each method and condition. The 4/10-dB rows use adaptive endpoints; the 19-dB row uses full representations. Negative $\Delta$DreamSim favors NeST.','l c c')
    frozen=read('timing_summary.csv');rows=[];body=[]
    for m,label in [('NEST','NeST-Com'),('P100000','Latent JSCC'),('Swin200000','SwinJSCC'),('BPG','BPG + LDPC'),('ARPC','ARPC + LDPC')]:
        v=one(frozen,method=m,snr_db=10);assert v['sources']=='16' and v['measurements']=='48'
        a,b,c=[float(v[k]) for k in ['tx_ms_mean','rx_ms_mean','total_ms_mean']];assert abs(a+b-c)<1e-9
        rows.append(v);body.append(label+f' & {a:.2f} & {b:.2f} & {c:.2f}'+r' \\')
    write_csv(out/'table_iii.csv',rows)
    table(out,'table_iii','Software processing latency at 10 dB (ms/image).','tab:latency',
      'Method & TX & RX & Total',body,
      r'Shared NVIDIA RTX PRO 6000 Blackwell, GPU 1, six CPU cores, batch size 1; 16 fixed development sources, three warmups and three measured repetitions per setting (48 measurements). Total is TX+RX software processing time, excluding air-interface time and model loading. Original method-specific precision is retained; TF32 is off.','l r r r')
    AUDIT['tables']=dict(I=3,II=3,III=5,statistics_recomputed=False,latency_remeasured=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    verify_inputs();rc();figure2(a.output);figure3(a.output);tables(a.output)
    text(a.output/'data_validation.json',json.dumps(AUDIT,indent=2)+'\n')
    print('PASS: 60 means/CIs, 12 full-view images, no ROI, 3 frozen tables')
if __name__=='__main__':main()
