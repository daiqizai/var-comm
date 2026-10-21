"""Render saved full-view PNGs; no inference or scoring."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
R=Path(__file__).resolve().parents[2]
if R.name=='figures':R=R.parent.parent
# Can run from the experiment directory or final figure directory.
if not (R/'results').exists():
    R=next(p for p in Path(__file__).resolve().parents if (p/'results').exists())
P=R/'paper/figures/selected_examples_13db_20261021_v1'
rows=list(csv.DictReader((R/'results/selected_examples_13db_20261021_v1/delivery/display_manifest.csv').open()))
cols=[('original','Original'),('P100000','Latent JSCC'),('Swin200000','SwinJSCC'),('BPG','BPG + LDPC'),('ARPC','ARPC + LDPC'),('NEST','NeST-Com')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
mm=[181.864,148];fig=plt.figure(figsize=np.array(mm)/25.4);side=(mm[0]-3-6.5)/6
for j,(_,label) in enumerate(cols):fig.text((1.5+j*(side+1.3)+side/2)/mm[0],.99,label,ha='center',va='top',fontsize=8.2)
for n,idx in enumerate([21,92,95,75],1):
    bottom=mm[1]-7-side-(n-1)*(side+6.3)
    for j,(m,_) in enumerate(cols):
        p=P/'individual'/f'{n:02d}_{m}.png';im=np.asarray(Image.open(p).convert('RGB'));assert im.shape==(256,256,3)
        x=(1.5+j*(side+1.3))/mm[0];ax=fig.add_axes([x,bottom/mm[1],side/mm[0],side/mm[1]])
        ax.imshow(im,interpolation='none');ax.axis('off');lab='Original'
        if m!='original':
            row=next(v for v in rows if v['method']==m and int(v['source_index'])==idx)
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['image_png_sha256']
            lab=f"{float(row['psnr_db']):.2f} / {float(row['lpips_alex']):.3f}"
        fig.text(x+side/2/mm[0],(bottom-1.2)/mm[1],lab,ha='center',va='top',fontsize=8.1)
for ext in ['pdf','svg','png']:fig.savefig(P/('candidates_13db.'+ext),dpi=600)
svg=P/'candidates_13db.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
im=Image.open(P/'candidates_13db.png');im.thumbnail((1500,1500));im.save(P/'candidates_13db_preview.png')
print('24 full-view panels; native pixels and 20 metric labels verified')
