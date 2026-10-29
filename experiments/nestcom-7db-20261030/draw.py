"""Full-view 7dB galleries and fixed-source4/7dB views; no image modification."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
W=Path(__file__).resolve().parents[2];D=W/'results/nestcom_7db_20261030_v1/delivery';OLD=W/'results/nestcom_fig3_4db_20261029_v1/delivery';F=D/'figures';F.mkdir(exist_ok=True)
rows={7:list(csv.DictReader((D/'figure3_metrics.csv').open())),4:list(csv.DictReader((OLD/'figure3_metrics.csv').open()))}
for r in rows[4]:
 if r['method']=='ADA_PREDCLASS_FIXEDFRAME':r['method']='ADA'
names={'original':'Original','WHOLE':'WHOLE','NEST':'NeST-Com','ADA':'Ada (fixed-frame)','ARPC':'ARPC + LDPC','P100000':'Latent JSCC','Swin200000':'SwinJSCC','BPG':'BPG + LDPC'}
methods=list(names)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})

def cell(fig,idx,m,snr,x,y,w,h,label):
 root=D if snr==7 else OLD
 if m=='original':path=root/'images'/('%04d_original.png'%idx)
 else:
  r=next(r for r in rows[snr]if int(r['source_index'])==idx and r['method']==m);path=root/r['display_image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['display_png_sha256'];label=f"{float(r['lpips_alex']):.3f} / {float(r['dreamsim']):.3f}"
 im=Image.open(path).convert('RGB');assert im.size==(256,256)
 ax=fig.add_axes([x,y,w,h]);ax.imshow(im,interpolation='none');ax.set_axis_off();fig.text(x+w/2,y-.012,label,ha='center',va='top',fontsize=8.3)
def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for t in fig.texts:
  b=t.get_window_extent(renderer);assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(t.get_text(),b.bounds)
 for ext in ['pdf','svg','png']:fig.savefig(F/(name+'.'+ext),dpi=450,facecolor='white')
 fig.savefig(F/(name+'_print_size.png'),dpi=150,facecolor='white');plt.close(fig)
def rows2(cols,items,name):
 width=7.16 if len(cols)==5 else 11.456;size=(width,3.35);fig=plt.figure(figsize=size);left=.018;gap=.012;w=(.986-left-(len(cols)-1)*gap)/len(cols);h=w*width/3.35
 for c,m in enumerate(cols):fig.text(left+c*(w+gap)+w/2,.974,names[m],ha='center',va='top',fontsize=8.5)
 for row,(idx,snr,label)in enumerate(items):
  for c,m in enumerate(cols):cell(fig,idx,m,snr,left+c*(w+gap),(3.08-row*1.54)/3.35-h,w,h,label)
 save(fig,name)
rows2(methods,[(95,7,'(a) Pastries'),(21,7,'(b) Dog')],'N1024_7dB_all_methods')
rows2(methods[:5],[(95,7,'(a) Pastries'),(21,7,'(b) Dog')],'N1024_7dB_five_columns')
for idx,title in [(95,'Pastries'),(21,'Dog')]:
 rows2(methods,[(idx,4,title+' / 4 dB'),(idx,7,title+' / 7 dB')],f'N1024_4_vs_7dB_{idx:04d}')
 fig=plt.figure(figsize=(7.16,4.4));left=.018;gap=.014;w=(.986-left-3*gap)/4;h=w*7.16/4.4
 for k,m in enumerate(methods):
  r,c=divmod(k,4);x=left+c*(w+gap);top=.92-r*.47;fig.text(x+w/2,top+.025,names[m],ha='center',va='bottom',fontsize=8.5);cell(fig,idx,m,7,x,top-h,w,h,title)
 save(fig,f'N1024_7dB_{idx:04d}')
(F/'validation.json').write_text(json.dumps(dict(snr_db=7,sources=[95,21],all_methods=methods,LPIPS_DreamSim_decimals=3,full_field_256=True,no_crop=True,no_enhancement=True,display_hashes_verified=True,new_metrics_by_drawing=0,visual_review_pending=True),indent=2)+'\n')

