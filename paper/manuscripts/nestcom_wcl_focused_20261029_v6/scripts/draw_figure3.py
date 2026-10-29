"""Fixed 4 dB native-array displays, no enhancement or scientific rescoring."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
P=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((P/'figure3_metrics.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
names={'original':'Original','WHOLE':'WHOLE','NEST':'NeST-Com','ADA_PREDCLASS_FIXEDFRAME':'Ada (fixed-frame)','ARPC':'ARPC + LDPC','P100000':'Latent JSCC','Swin200000':'SwinJSCC','BPG':'BPG + LDPC'}
main=['original','WHOLE','NEST','ADA_PREDCLASS_FIXEDFRAME','ARPC']
all_methods=['original','WHOLE','NEST','ADA_PREDCLASS_FIXEDFRAME','ARPC','P100000','Swin200000','BPG']
def cell(fig,idx,method,x,y,w,h,label=None):
    if method=='original':
        path=P/f'figures/individual_4db/{idx:04d}_original.png'
        label=label or ('Pastries'if idx==95 else'Dog')
    else:
        r=next(r for r in rows if int(r['source_index'])==idx and r['method']==method)
        path=P/'figures/individual_4db'/Path(r['display_image']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['display_png_sha256']
        label=f"{float(r['lpips_alex']):.3f} / {float(r['dreamsim']):.3f}"
    im=Image.open(path).convert('RGB');assert im.size==(256,256)
    ax=fig.add_axes([x,y,w,h]);ax.imshow(im,interpolation='none');ax.set_axis_off()
    fig.text(x+w/2,y-.012,label,ha='center',va='top',fontsize=8.3)
def save(fig,stem):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in fig.texts:
        b=t.get_window_extent(renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(t.get_text(),b.bounds)
    for ext in ['pdf','svg','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=450,facecolor='white')
    fig.savefig(stem.parent/(stem.name+'_print_size.png'),dpi=150,facecolor='white');plt.close(fig)
def two_rows(methods,size,stem):
    fw,fh=size;fig=plt.figure(figsize=size);left=.018;right=.986;gap=.012;w=(right-left-(len(methods)-1)*gap)/len(methods);h=w*fw/fh
    for c,m in enumerate(methods):fig.text(left+c*(w+gap)+w/2,.974,names[m],ha='center',va='top',fontsize=8.5)
    for r,idx in enumerate([95,21]):
        top=(3.08-r*1.54)/3.35
        for c,m in enumerate(methods):cell(fig,idx,m,left+c*(w+gap),top-h,w,h,('(a) Pastries'if idx==95 else'(b) Dog'))
    save(fig,stem)
two_rows(main,(7.16,3.35),P/'figures/figure3')
two_rows(all_methods,(11.456,3.35),P/'supplemental/N1024_4dB_all_methods')
for idx,title in [(95,'Pastries'),(21,'Dog')]:
    fig=plt.figure(figsize=(7.16,4.40));left=.018;gap=.014;w=(.986-left-3*gap)/4;h=w*7.16/4.40
    for k,m in enumerate(all_methods):
        r,c=divmod(k,4);x=left+c*(w+gap);top=.92-r*.47
        fig.text(x+w/2,top+.025,names[m],ha='center',va='bottom',fontsize=8.5)
        cell(fig,idx,m,x,top-h,w,h,title)
    save(fig,P/f'supplemental/N1024_4dB_{idx:04d}')
(P/'figures/figure3_validation.json').write_text(json.dumps(dict(N=1024,snr_db=4,rows=2,columns=5,methods=main,source_indices=[95,21],full_view=True,reconstruction_count=8,complete_gallery_reconstructions=14,size_inches=[7.16,3.35],all_display_hashes_checked=True,metrics_from_native_arrays=True,metric_order='LPIPS-Alex / DreamSim',new_metric_computation_by_drawing=0),indent=2)+'\n')
