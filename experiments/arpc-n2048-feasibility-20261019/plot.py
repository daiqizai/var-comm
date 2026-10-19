"""Descriptive cal100 endpoint curve from existing scores; no resampling."""
from pathlib import Path
import argparse,csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STYLE={'width_mm':180,'height_mm':67,'font':'DejaVu Sans','font_pt':8,
       'axis_label_pt':8.5,'panel_pt':9,'line_pt':1.6,'marker_pt':4.5,
       'blue':'#3775BA','range_gray':'#767676','dpi':600}
plt.rcParams.update({'font.family':STYLE['font'],'font.size':8,'axes.labelsize':8.5,
 'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.titlesize':9,'axes.linewidth':.7,
 'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','axes.unicode_minus':True})
def main(root):
    rows=list(csv.DictReader((root/'arpc_endpoint_summary.csv').open()))
    x=np.array([float(r['payload_mean_bits'])/1000 for r in rows])
    low=np.array([float(r['payload_min_bits'])/1000 for r in rows]);high=np.array([float(r['payload_max_bits'])/1000 for r in rows])
    specs=[('psnr_db_mean','(a) PSNR (dB) ↑',(9.2,22.5)),('lpips_alex_mean','(b) LPIPS ↓',(.08,.78)),('dinov2_vitl14_cosine_mean','(c) DINO cosine ↑',(.04,.99))]
    fig,axes=plt.subplots(1,3,figsize=(180/25.4,67/25.4))
    fig.subplots_adjust(left=.062,right=.987,bottom=.22,top=.78,wspace=.32)
    for ax,(metric,title,ylim) in zip(axes,specs):
        y=np.array([float(r[metric]) for r in rows]);ax.set_ylim(*ylim)
        ax.errorbar(x,y,xerr=np.array([x-low,high-x]),fmt='none',ecolor=STYLE['range_gray'],elinewidth=1,capsize=2,zorder=1)
        ax.plot(x,y,'o-',color=STYLE['blue'],lw=1.6,ms=4.5,zorder=3)
        # Close e1/e2 means remain at their true x coordinates. Callouts separate labels.
        offsets=[(20,0),(12,6),(10,6),(3,9),(0,9),(0,9),(-7,8)]
        if metric=='lpips_alex_mean':offsets=[(12,3),(15,0),(12,-4),(9,-6),(0,-12),(0,-12),(-12,9)]
        if metric=='dinov2_vitl14_cosine_mean':offsets=[(8,0),(24,0),(14,5),(4,10),(0,10),(0,10),(-7,8)]
        for i,(xx,yy) in enumerate(zip(x,y)):
            if metric=='dinov2_vitl14_cosine_mean' and i==0:continue
            label='e1/2' if metric=='dinov2_vitl14_cosine_mean' and i==1 else f'e{i+1}'
            dx,dy=offsets[i];ax.annotate(label,(xx,yy),xytext=(dx,dy),textcoords='offset points',fontsize=7.1,ha='center',color='#333333',arrowprops=dict(arrowstyle='-',lw=.45,color='#777777') if i<3 else None)
        ax.set_xlim(-.10,9.6);ax.set_xticks([0,3,6,9]);ax.set_xlabel('Source payload (kbit)');ax.set_title(title,loc='left',pad=7)
        ax.grid(axis='y',color='#dddddd',lw=.5);ax.set_axisbelow(True)
    fig.text(.5,.945,'ARPC: complete native endpoints on fixed cal100',ha='center',va='top',fontsize=9)
    fig.text(.5,.858,'Points: source means   |   Horizontal bars: observed payload min–max',ha='center',fontsize=7.5)
    for ext in ('pdf','svg','png'):fig.savefig(root/f'arpc_payload_quality.{ext}',dpi=600)
    svg=root/'arpc_payload_quality.svg'
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    # Actual-print-size raster preview (150 dpi); no data/image resynthesis.
    fig.savefig(root/'arpc_payload_quality_print_preview.png',dpi=150)
    colors=[line.get_color() for ax in axes for line in ax.lines]
    for ax in axes:
        for line in ax.lines:line.set_color('#333333')
        for collection in ax.collections:collection.set_color('#777777')
    fig.savefig(root/'arpc_payload_quality_grayscale.png',dpi=150)
    plt.close(fig)
    (root/'figure_style.json').write_text(json.dumps(STYLE,indent=2)+'\n')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('output',type=Path);main(a.parse_args().output)
