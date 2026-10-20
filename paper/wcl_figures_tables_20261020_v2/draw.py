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
        'svg.hashsalt': 'wcl-frozen-a08e251-v2', 'savefig.facecolor': 'white',
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
    assert len(rows)==120
    specs=[('psnr_db','(a) PSNR (dB) ↑',(13,25),[13,16,19,22,25]),
           ('lpips_alex','(b) LPIPS ↓',(0,.8),[0,.2,.4,.6,.8]),
           ('dreamsim','(c) DreamSim ↓',(0,.65),[0,.2,.4,.6]),
           ('clip_image_cosine','(d) CLIP-I ↑',(.5,1.0),[.5,.6,.7,.8,.9,1.0])]
    mm=S['figure2_mm']; fig,axes=plt.subplots(2,2,figsize=np.array(mm)/25.4)
    fig.subplots_adjust(left=.085,right=.98,bottom=.115,top=.79,wspace=.25,hspace=.43)
    used=[]
    for i,(metric,title,limits,ticks) in enumerate(specs):
        ax=axes.flat[i];base_axis(ax,xlabel=i>=2)
        ax.set_title(title,loc='left',pad=7);ax.set_ylim(*limits);ax.set_yticks(ticks)
        if metric!='psnr_db':ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.1f}'))
        values={}
        for method in S['method_order']:
            selected=[one(rows,method=method,metric=metric,snr_db=x) for x in SNR]
            assert all(r['source_count']=='500' for r in selected)
            data=np.array([triple(r) for r in selected])
            assert np.all(data[:,1]>=limits[0]) and np.all(data[:,2]<=limits[1]),(metric,method,'CI clipping')
            error_curve(ax,SNR,data,S['methods'][method])
            # Inspect the actual line and errorbar artists against CSV values.
            line=ax.lines[-1]
            assert np.array_equal(line.get_xdata(),SNR)
            assert np.array_equal(line.get_ydata(),data[:,0])
            segments=ax.collections[-1].get_segments()
            assert len(segments)==3
            for x,seg,d in zip(SNR,segments,data):
                assert np.allclose(seg,[[x,d[1]],[x,d[2]]],rtol=0,atol=1e-14)
            values[method]=data.tolist();used.extend(selected)
        AUDIT['panels'].append(dict(panel='figure2_'+metric,source='inputs/eight_metric_summary.csv',
            columns=['mean','ci_low','ci_high'],x=SNR,y_limits=limits,values=values,
            actual_artist_values_verified=True))
    order=[S['method_order'][i] for i in [0,3,1,4,2]]
    handles=[Line2D([],[],label=S['methods'][m]['label'],**line_options(S['methods'][m])) for m in order]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.53,.99),
               handlelength=2.5,handletextpad=.55,columnspacing=1.7,labelspacing=.65,borderaxespad=0)
    assert len(used)==60
    write_csv(out/'figure2_data.csv',used)
    AUDIT['figure2_exact_mean_ci_cells']=60
    AUDIT['figure2_data_strings_preserved']=True
    export(fig,out,'figure2',mm)

def figure3(out):
    rows=read('display_manifest.csv')
    original_manifest=json.loads((HERE/'inputs/examples_manifest.json').read_bytes())
    columns=[('original','Original'),('P100000','Latent JSCC'),('Swin200000','SwinJSCC'),
             ('BPG','BPG + LDPC'),('ARPC','ARPC + LDPC'),('NEST','NeST-Com')]
    subjects=[(3,95,'n07860988/ILSVRC2012_val_00010848_n07860988','Pastries'),
              (4,75,'n04204347/ILSVRC2012_val_00018921_n04204347','Shopping cart')]
    mm=S['figure3_mm'];fig=plt.figure(figsize=np.array(mm)/25.4)
    left=1.5;gap=1.2;side=(mm[0]-2*left-5*gap)/6
    placed=[];image_hashes=[]
    for c,(_,label) in enumerate(columns):
        fig.text((left+c*(side+gap)+side/2)/mm[0],.958,label,ha='center',va='center',fontsize=8.3)
    for r,(original_row,idx,sid,subject) in enumerate(subjects):
        bottom=[42,7][r]
        for c,(method,label) in enumerate(columns):
            p=HERE/'inputs/images'/f'{original_row:02d}_{method}.png'
            im=Image.open(p)
            assert im.mode=='RGB' and im.size==(256,256)
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            image_hashes.append(hashlib.sha256(im.tobytes()).hexdigest())
            descriptor=next(v for v in original_manifest['files'] if v['path'].endswith('/'+p.name))
            assert sha==descriptor['sha256'] and p.stat().st_size==descriptor['bytes']
            x=(left+c*(side+gap))/mm[0];y=bottom/mm[1]
            ax=fig.add_axes([x,y,side/mm[0],side/mm[1]])
            # Native pixels are embedded at 256x256 in PDF/SVG, without enhancement.
            ax.imshow(np.asarray(im),interpolation='none');ax.axis('off')
            entry=dict(row=r+1,column=c+1,subject=subject,source_index=idx,source_id=sid,
                method=method,display_name=label,image_path=str(p.relative_to(HERE)).replace('\\','/'),
                image_png_sha256=sha,width_px=256,height_px=256)
            if method!='original':
                v=one(rows,source_index=idx,method=method)
                assert v['source_id']==sid and v['N']=='1024' and v['snr_db']=='10'
                assert v['image_png_sha256']==sha
                assert Path(v['image_path']).name==p.name
                text_label=f"{float(v['psnr_db']):.2f} / {float(v['lpips_alex']):.3f}"
                fig.text(x+side/2/mm[0],y-1.5/mm[1],text_label,
                    ha='center',va='top',fontsize=8.2)
                entry.update({k:v[k] for k in ['version','N','snr_db','noise_seed','noise_namespace',
                    'timing_case','counter','receive_status','actual_profile','float_image_path',
                    'float_image_file_sha256','timing_float_array_sha256','psnr_db','lpips_alex']})
                entry['display_label']=text_label
            else:entry['display_label']='Original'
            placed.append(entry)
    fields=list(dict.fromkeys(k for v in placed for k in v))
    write_csv(out/'figure3_mapping.csv',placed,fields)
    export(fig,out,'figure3',mm)
    # Ensure the exported editable SVG contains exactly the unchanged native pixels.
    import base64,io
    root=ET.parse(out/'figure3.svg').getroot()
    embedded=[]
    for elem in root.findall('.//{http://www.w3.org/2000/svg}image'):
        href=elem.attrib['{http://www.w3.org/1999/xlink}href']
        image=Image.open(io.BytesIO(base64.b64decode(href.split(',',1)[1]))).convert('RGB')
        assert image.size==(256,256)
        embedded.append(hashlib.sha256(image.tobytes()).hexdigest())
    assert sorted(embedded)==sorted(image_hashes)
    AUDIT['figure3']=dict(images=12,originals=2,reconstructions=10,
        native_pixels_equal_in_svg=True,all_scores_from_frozen_manifest=True,
        source_indices=[95,75],native_image_resolution=[256,256],image_display_mm=side,
        effective_native_image_dpi=256/(side/25.4),native_float_scoring_not_recomputed=True)

def tables(out):
    original=(HERE/'inputs/table_i_original.tex').read_bytes()
    (out/'table_i.tex').write_bytes(original)
    txt=original.decode()
    for snr in SNR:
        data=read('n2048_full_paired.csv' if snr==19 else 'n2048_adaptive_paired.csv')
        comparison='NeST_full_m10_minus_ARPC_full_e7' if snr==19 else 'NeST_adaptive_minus_ARPC_adaptive'
        for metric,nd in [('psnr_db',3),('dists',4)]:
            v=one(data,comparison=comparison,snr_db=snr,metric=metric)
            assert v['source_count']=='100';triple(v)
            formatted=[format(Decimal(v[k]),f'.{nd}f') for k in ['mean','ci_low','ci_high']]
            expected=r'\shortstack{$'+formatted[0]+r'$\\[1pt]$['+formatted[1]+', '+formatted[2]+r']$}'
            assert expected in txt,expected
            AUDIT['table_rows'].append(dict(snr_db=snr,metric=metric,comparison=comparison,
                mean=v['mean'],ci_low=v['ci_low'],ci_high=v['ci_high'],display=formatted))
    assert ['0.0054','0.0024','0.0086']==AUDIT['table_rows'][-1]['display']
    AUDIT['table_i_exact_v1_bytes_preserved']=True
    rows=read('timing_summary.csv')
    mapping={'NEST':'NeST-Com','P100000':'Latent JSCC','Swin200000':'SwinJSCC','BPG':'BPG + LDPC','ARPC':'ARPC + LDPC'}
    lines=[r'% Requires booktabs. All values read from the frozen five-method timing summary.',
        r'\begin{table}[t]',r'\centering',r'\caption{Software processing latency (ms/image).}',
        r'\label{tab:software-latency}',r'\begingroup',r'\fontsize{8.5}{10}\selectfont',
        r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{@{}lr rrr@{}}',r'\toprule',
        r'Method & SNR (dB) & TX & RX & Total \\',r'\midrule']
    for j,(method,label) in enumerate(mapping.items()):
        for snr in SNR:
            v=one(rows,method=method,snr_db=snr)
            assert v['sources']=='16' and v['measurements']=='48'
            a,b,c=[float(v[k]) for k in ['tx_ms_mean','rx_ms_mean','total_ms_mean']]
            assert abs(a+b-c)<1e-9
            lines.append(label+' & '+str(snr)+' & '+f'{a:.2f} & {b:.2f} & {c:.2f}'+r' \\')
        if j<4:lines.append(r'\addlinespace[2pt]')
    lines += [r'\bottomrule',r'\end{tabular}',r'\endgroup',r'\par\smallskip',
        r'\begin{minipage}{\columnwidth}\fontsize{8}{9.5}\selectfont',
        r'Means over 16 fixed development images, batch size 1, with three warmups and three measured calls per source and SNR. Total is TX+RX software processing time; air-interface transmission time is excluded. All methods use the same host, CPU affinity and GPU; original method-specific software stacks are retained. Shared-load and timing-boundary details are given in the experimental settings.',
        r'\end{minipage}',r'\end{table}']
    text(out/'timing_table.tex','\n'.join(lines)+'\n')
    AUDIT['timing']=dict(frozen_rows=15,measurement_count_per_row=48,
        source_count_per_row=16,all_total_means_equal_tx_plus_rx=True,new_timing_calls=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    verify_inputs();rc();figure2(args.output);figure3(args.output);tables(args.output)
    text(args.output/'data_validation.json',json.dumps(AUDIT,indent=2)+'\n')
    print(json.dumps({'status':'PASS','figure2_cells':60,'figure3_images':12,
        'table_i_rows':3,'timing_rows':15,'scientific_calls':0}))
if __name__=='__main__':main()
