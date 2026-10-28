"""Preregistered common500 identities, first repeat; display only, no inference."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
P=Path(__file__).resolve().parents[1]
v=json.loads((P/'inputs/figure3_native_mapping.json').read_text())
assert v['selection']['selected_before_reconstruction_quality']
sources=v['selection']['selected'];methods=['Original','WHOLE','NEST','ADA'];names=['Original','WHOLE','NeST-Com','Ada (adapted)']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
w,h=6.,3.30;side=1.32;gap=.16;left=(w-4*side-3*gap)/2;fig=plt.figure(figsize=(w,h),facecolor='white');rows=[]
for col,label in enumerate(names):fig.text((left+col*(side+gap)+side/2)/w,.976,label,ha='center',va='top',fontsize=9)
for ri,s in enumerate(sources):
 top=h-.27-ri*1.51
 for ci,method in enumerate(methods):
  x=left+ci*(side+gap);y=top-side
  if method=='Original':file=f"individual/s{s['source_index']:04d}_Original.png";label=['(a) Image 1','(b) Image 2'][ri]
  else:
   j=next(j for j in v['records']if j['source_id']==s['source_id']and j['method']==method)
   assert j['N']==512 and j['snr_db']==10 and j['noise_seed']==6201
   file=j['display_png'];a,b=j['metrics']['lpips_alex'],j['metrics']['dreamsim'];label=f'{a:.3f} / {b:.3f}'
   rows.append(dict(source_index=s['source_index'],source_id=s['source_id'],preprocessing_id=s['preprocessing_id'],method=method,N=512,snr_db=10,noise_seed=6201,source_status=j['source_status'],fixed_gray=j['fixed_gray'],**j['metrics'],float_image_path=j['image']['path'],file_sha256=j['image']['sha256'],pixel_sha256=j['pixel_sha256'],reference_sha256=j['reference_sha256'],wire_path=j['wire']['path'],wire_sha256=j['wire']['sha256'],display_png=file,display_sha256=hashlib.sha256((P/'figures'/file).read_bytes()).hexdigest()))
  im=Image.open(P/'figures'/file);assert im.size==(256,256)
  ax=fig.add_axes([x/w,y/h,side/w,side/h]);ax.imshow(im,interpolation='none');ax.set_axis_off()
  fig.text((x+side/2)/w,(y-.045)/h,label,ha='center',va='top',fontsize=8.5)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for t in fig.texts:
 b=t.get_window_extent(renderer);assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(t.get_text(),b.bounds)
for ext in['pdf','svg','png']:fig.savefig(P/'figures'/('figure3.'+ext),dpi=600)
fig.savefig(P/'figures/figure3_print_size.png',dpi=150)
with(P/'figure3_metrics.csv').open('w',newline='',encoding='utf8')as f:
 wcsv=csv.DictWriter(f,fieldnames=list(rows[0]));wcsv.writeheader();wcsv.writerows(rows)
(P/'validation/figure3_data_check.json').write_text(json.dumps(dict(size_inches=[w,h],rows=2,columns=4,full_field_of_view=True,metric_source='native float outputs; existing frozen per-frame metrics',new_receiver_calls=0,first_noise_seed=6201,selection_file='inputs/figure3_native_mapping.json',display_metric_pairs=6,all_text_inside_canvas=True),indent=2))
