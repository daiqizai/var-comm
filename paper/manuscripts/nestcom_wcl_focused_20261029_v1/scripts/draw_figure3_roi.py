"""Post-hoc editorial lamp example; identical ROI; existing images and scores only."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
P=Path(__file__).resolve().parents[1]
v=json.loads((P/'inputs/figure3_native_mapping.json').read_text())
sid='n04380533/ILSVRC2012_val_00019316_n04380533'
s=next(x for x in v['selection']['selected'] if x['source_id']==sid)
methods=['Original','WHOLE','NEST','ADA'];names=['Original','WHOLE','NeST-Com','Ada (fixed-frame)']
# ROI chosen on the source image; right/bottom exclusive. No pixel retouching.
roi=(80,0,180,100)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
w,h=6.,3.35;side=1.32;gap=.16;left=(w-4*side-3*gap)/2
fig=plt.figure(figsize=(w,h),facecolor='white');rows=[];identity={}
for ci,(method,name) in enumerate(zip(methods,names)):
 x=left+ci*(side+gap)
 fig.text((x+side/2)/w,.982,name,ha='center',va='top',fontsize=8.8)
 file=f"individual/s{s['source_index']:04d}_{method}.png"
 path=P/'figures'/file;im=Image.open(path).convert('RGB');assert im.size==(256,256)
 identity[method]={'display_png':file,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size':[256,256]}
 y=1.76
 ax=fig.add_axes([x/w,y/h,side/w,side/h]);ax.imshow(im,interpolation='none');ax.set_axis_off()
 ax.add_patch(Rectangle((roi[0]-.5,roi[1]-.5),roi[2]-roi[0],roi[3]-roi[1],fill=False,edgecolor='#56B4E9',linewidth=.85,clip_on=False))
 if method!='Original':
  j=next(j for j in v['records']if j['source_id']==sid and j['method']==method)
  assert j['N']==512 and j['snr_db']==10 and j['noise_seed']==6201 and j['display_png']==file
  label=f"{j['metrics']['psnr_db']:.2f} / {j['metrics']['lpips_alex']:.3f}"
  rows.append(dict(source_index=s['source_index'],source_id=sid,preprocessing_id=s['preprocessing_id'],method=method,N=512,snr_db=10,noise_seed=6201,source_status=j['source_status'],fixed_gray=j['fixed_gray'],**j['metrics'],float_image_path=j['image']['path'],file_sha256=j['image']['sha256'],pixel_sha256=j['pixel_sha256'],reference_sha256=j['reference_sha256'],wire_path=j['wire']['path'],wire_sha256=j['wire']['sha256'],display_png=file,display_sha256=identity[method]['sha256'],metric_region='full_image'))
 else: label='Full image'
 fig.text((x+side/2)/w,(y-.045)/h,label,ha='center',va='top',fontsize=8.4)
 ax=fig.add_axes([x/w,.12/h,side/w,side/h]);ax.imshow(im.crop(roi),interpolation='none');ax.set_axis_off()
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for t in fig.texts:
 b=t.get_window_extent(renderer);assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(t.get_text(),b.bounds)
for ext in ['pdf','svg','png']:fig.savefig(P/'figures'/('figure3.'+ext),dpi=600)
fig.savefig(P/'figures/figure3_print_size.png',dpi=150)
with(P/'figure3_metrics.csv').open('w',newline='',encoding='utf8')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
record=dict(size_inches=[w,h],rows=2,columns=4,top_row='complete 256x256 image',bottom_row='same ROI for all columns',roi_xyxy_exclusive=roi,roi_chosen_on='original source image',selection='post-hoc editorial reuse of the lamp already displayed in the preceding manuscript; not random or newly preregistered',source_id=sid,source_index=s['source_index'],metric_region='full image',metric_source='existing native-float per-frame scores',new_receiver_calls=0,new_metric_calls=0,first_noise_seed=6201,selection_history='inputs/figure3_native_mapping.json',display_metric_pairs=3,images=identity,all_text_inside_canvas=True)
(P/'figures/figure3_roi.json').write_text(json.dumps(record,indent=2)+'\n')
(P/'validation/figure3_data_check.json').write_text(json.dumps(record,indent=2)+'\n')
