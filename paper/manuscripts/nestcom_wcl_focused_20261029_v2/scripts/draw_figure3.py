"""Full native field of view; fixed examples and unchanged saved PNGs."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
P=Path(__file__).resolve().parents[1];rows=list(csv.DictReader((P/'figure3_metrics.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(7.16,3.35));left=.018;right=.986;gap=.012;width=(right-left-4*gap)/5;height=width*7.16/3.35
methods=['original','NEST','ADA_PREDCLASS_FIXEDFRAME','P100000','ARPC'];names=['Original','NeST-Com','Ada (fixed-frame)','Latent JSCC','ARPC + LDPC']
for c,name in enumerate(names):fig.text(left+c*(width+gap)+width/2,.974,name,ha='center',va='top',fontsize=8.5)
for row,(idx,name)in enumerate(((95,'(a) Pastries'),(21,'(b) Dog'))):
 top=(3.08-row*1.54)/3.35
 for c,method in enumerate(methods):
  if method=='original':path=P/f'figures/individual_n1024/{idx:04d}_original.png';label=name
  else:
   r=next(r for r in rows if r['source_index']==str(idx)and r['method']==method);path=P/r['image_path'].replace('figures/individual/', 'figures/individual_n1024/');assert hashlib.sha256(path.read_bytes()).hexdigest()==r['image_png_sha256'];label=f"{float(r['lpips_alex']):.3f} / {float(r['dreamsim']):.3f}"
  image=Image.open(path).convert('RGB');assert image.size==(256,256)
  x=left+c*(width+gap);ax=fig.add_axes([x,top-height,width,height]);ax.imshow(image,interpolation='none');ax.set_axis_off()
  fig.text(x+width/2,top-height-.035/3.35,label,ha='center',va='top',fontsize=8.3)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for t in fig.texts:
 b=t.get_window_extent(renderer);assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(t.get_text(),b.bounds,fig.bbox.bounds)
for ext in('pdf','svg','png'):fig.savefig(P/'figures'/('figure3.'+ext),dpi=600,facecolor='white')
fig.savefig(P/'figures/figure3_print_size.png',dpi=150,facecolor='white')
(P/'figures/figure3_validation.json').write_text(json.dumps(dict(rows=2,columns=5,source_indices=[95,21],source_count=2,reconstruction_count=8,full_view=True,independent_development_examples=True,no_metric_recomputation=True,size_inches=[7.16,3.35],all_saved_image_hashes_checked=True),indent=2)+'\n')
