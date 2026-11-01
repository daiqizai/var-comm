"""All available methods on the two selected sources, frozen repeat zero."""
from pathlib import Path
import json,hashlib
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
P=Path(__file__).resolve().parents[1]
data=json.loads((P/'inputs/f1/selected_native_mapping.json').read_text())+json.loads((P/'inputs/f1/external_native_mapping.json').read_text())
methods=[('Source','Source'),('WHOLE_final_receiver','WHOLE'),('original_NeST','Quarter'),('final_NeST','NeST-Com'),('Ada_frozen','Ada (adapted)'),('Latent_JSCC','Latent JSCC'),('SwinJSCC','SwinJSCC'),('BPG_LDPC','BPG + LDPC'),('ARPC','ARPC + LDPC')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(9.0,8.8));used=[]
for source_row in [1,2]:
 for j,(method,label)in enumerate(methods):
  v=next(v for v in data if v['row']==source_row and v['method']==method)
  p=P/'figures/individual'/Path(v['display']['path']).name
  assert hashlib.sha256(p.read_bytes()).hexdigest()==v['display']['sha256'];im=Image.open(p);assert im.size==(256,256)
  col=j%5;subrow=j//5;bottom=6.9-(source_row-1)*4.3-subrow*2.05
  ax=fig.add_axes([(.12+col*1.78)/9,bottom/8.8,1.65/9,1.65/8.8]);ax.imshow(im,interpolation='none');ax.axis('off')
  ax.text(.5,1.018,label,ha='center',va='bottom',transform=ax.transAxes,fontsize=9)
  text=v['label']if method=='Source'else f"{v['dists']:.3f} / {v['dreamsim']:.3f}"
  ax.text(.5,-.025,text,ha='center',va='top',transform=ax.transAxes,fontsize=8.5)
  used.append(v)
fig.text(.99,.01,'N = 1024, 7 dB · fixed repeat 0 · DISTS↓ / DreamSim↓',ha='right',fontsize=9)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for t in fig.findobj(matplotlib.text.Text):
 if t.get_visible()and t.get_text():
  b=t.get_window_extent(renderer);assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(t.get_text(),b.bounds)
for ext in ['pdf','svg','png']:fig.savefig(P/'figures'/('figure3_all_methods_supplement.'+ext),dpi=600,facecolor='white')
fig.savefig(P/'figures/figure3_all_methods_print_size.png',dpi=150,facecolor='white')
(P/'validation/figure3_supplement_data_check.json').write_text(json.dumps(dict(source_count=2,columns=9,full_field_of_view=True,all_native_hashes_verified=True,new_scientific_calls=0,visual_review='PENDING'),indent=2)+'\n')
