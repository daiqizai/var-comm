"""Read-only final16 photo delivery; no model calls and no new statistics."""
from pathlib import Path
import csv,hashlib,json,shutil,sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'results/hifi_final_fixed16_20261012'
OUT=ROOT/'paper/figures/hifi_final_fixed16_20261012'
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def main():
 done=read(RESULT/'completion.json');assert done['status']=='COMPLETE_FIXED16_FINAL_HIFI' and read(RESULT/'actual_wait.json')['returncode']==0
 receipts=read(RESULT/'receipts.json');assert sha(RESULT/'receipts.json')==done['receipts_sha256']
 assert len(receipts)==48 and len({(r['method'],r['source_index']) for r in receipts})==48
 reg=read(RESULT/'registration.json');ids=reg['source_indices'];mapping={};bindings={};rows=[]
 folders=['hifi_development_N1024_13dB','hifi_development_N1024_13dB_group02','additional_candidates_20261012/group03','additional_candidates_20261012/group04']
 for j,folder in enumerate(folders,1):
  p=ROOT/'paper/figures'/folder/'selection_and_integrity.json';plan=read(p);a=TMP/f'hifi_final_fixed16_assets_v1/display_seed/group{j:02d}/display_images.npz'
  assert sha(a)==plan['display_images_sha256'];bindings[str(p)]=sha(p);bindings[str(a)]=sha(a)
  with np.load(a,allow_pickle=False) as z:images=z['images'].copy()
  for k,s in enumerate(plan['samples']):
   i=s['source_index'];original=images[k,0];assert hashlib.sha256(original.transpose(2,0,1).tobytes()).hexdigest()==s['preprocessing_id']
   mapping[i]={'Original':original,'NeST-Com (raw)':images[k,plan['array_order'].index('RAW64_PARTIAL_VAR_COMPLETION')]}
   rows.append(dict(source_index=i,method='NeST-Com (raw)',checkpoint='frozen_original',N=1024,snr_db=13,noise_seed=6201,observation_sha256='',image_sha256=hashlib.sha256(mapping[i]['NeST-Com (raw)'].tobytes()).hexdigest(),file=str(a)))
 for r in receipts:
  p=Path(r['png']['path']);assert sha(p)==r['png']['sha256'];a=Path(r['archive']['path']);assert sha(a)==r['archive']['sha256']
  with np.load(a,allow_pickle=False) as z:
   image=z['display'];assert np.array_equal(np.asarray(Image.open(p)),image)
  label={'P100000':'Latent-space JSCC','Swin200000':'SwinJSCC (adapted)','HiFiSwin200000':'HiFi-DiffCom (adapted)'}[r['method']]
  mapping[r['source_index']][label]=image
  bindings[str(p)]=sha(p);bindings[str(a)]=sha(a)
  rows.append(dict(source_index=r['source_index'],method=label,checkpoint=r['checkpoint_sha256'],N=1024,snr_db=13,noise_seed=2001,observation_sha256=r['observation_sha256'],image_sha256=r['reconstruction_sha256'],file=str(a)))
  if r['method']=='HiFiSwin200000':
   base=next(v for v in receipts if v['source_index']==r['source_index'] and v['method']=='Swin200000')
   assert r['paired_Swin_archive']==base['archive'] and r['observation_sha256']==base['observation_sha256'] and r['checkpoint_sha256']==base['checkpoint_sha256']
 assert set(mapping)==set(ids) and all(len(v)==5 for v in mapping.values())
 OUT.mkdir(exist_ok=True,parents=True);native=OUT/'native';native.mkdir(exist_ok=True)
 columns=['Original','SwinJSCC (adapted)','Latent-space JSCC','HiFi-DiffCom (adapted)','NeST-Com (raw)']
 labels=['Original','SwinJSCC\n(adapted)','Latent-space\nJSCC','HiFi-DiffCom\n(adapted)','NeST-Com\n(raw)']
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})
 def draw(indices,stem,pages=None):
  fig,axes=plt.subplots(len(indices),5,figsize=(7,1.38*len(indices)+.53),squeeze=False)
  fig.subplots_adjust(left=.035,right=.997,bottom=.012,top=1-.53/fig.get_figheight(),wspace=.02,hspace=.035)
  for i,index in enumerate(indices):
   for j,key in enumerate(columns):
    ax=axes[i,j];ax.imshow(mapping[index][key],interpolation='none');ax.axis('off')
    if i==0:ax.set_title(labels[j],fontsize=8,pad=5)
    if j==0:ax.text(-.035,.5,f'({chr(97+i)})',transform=ax.transAxes,va='center',ha='right',fontsize=8)
  fig.text(.52,.988,'N = 1024   |   SNR = 13 dB',ha='center',va='top',fontsize=8)
  for ext in ['pdf','svg','png']:fig.savefig(OUT/(stem+'.'+ext),dpi=600)
  if pages:pages.savefig(fig)
  plt.close(fig)
  with Image.open(OUT/(stem+'.png')) as im:im.convert('L').save(OUT/(stem+'_grayscale.png'))
 with PdfPages(OUT/'all16_final_link.pdf') as pages:
  for j in range(4):draw(ids[j*4:(j+1)*4],f'fixed16_page{j+1}',pages)
 draw([21,92,95,75],'selected_four_final_link');draw([21,92,95],'selected_three_final_link')
 for index in ids:
  for j,key in enumerate(columns):Image.fromarray(mapping[index][key]).save(native/f'{index:04d}_{j:02d}.png')
 with (OUT/'plot_data.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 caption='Final-checkpoint qualitative comparison on the original fixed16 development sources at N=1024 and13dB. Latent-space JSCC uses the final100k checkpoint; adapted SwinJSCC and the HiFi forward encoder and confirming decoder all use the final200k checkpoint. Swin and HiFi share exactly the same actual received waveform and decoded paid auxiliary information. HiFi uses the frozen unconditional ImageNet ADM and full registered posterior schedule with sampling seed23. Channel seed2001 is fixed for continuous methods; unchanged digital outputs retain seed6201. There is no best-noise selection or image enhancement. All16 sources are delivered, including failures. Dog, broccoli, pastries and cart were user-selected after viewing earlier examples and do not define a quantitative test population. This compares complete generative reception systems, not isolated VAR versus diffusion backbones. Historical Swin80k/HiFi statistics are separate. Shared-server elapsed times are not a fair speed comparison.'
 (OUT/'captions.tex').write_text(caption+'\n')
 # Reuse the published old16 intervals verbatim in a separate, explicitly historical table.
 src=ROOT/'results/external_comparison_20261004/fixed80k_revision/hifi_fixed16_release/evaluation/metrics_summary.csv'
 historical=[r for r in csv.DictReader(src.open()) if r['N']=='1024' and r['snr_db']=='13' and r['metric'] in ['psnr_db','lpips_alex','dinov2_vitl14_cosine']]
 assert len(historical)==6 and all(r['n_sources']=='16' for r in historical)
 with (OUT/'historical_Swin80k_fixed16.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(historical[0]));w.writeheader();w.writerows(historical)
 paired_src=src.with_name('metrics_paired_intervals.csv')
 paired=[r for r in csv.DictReader(paired_src.open()) if r['N_A']=='1024' and r['snr_A']=='13' and r['metric'] in ['psnr_db','lpips_alex','dinov2_vitl14_cosine']]
 assert len(paired)==3 and all(r['delta']=='A_minus_B' and r['n_sources']=='16' for r in paired)
 with (OUT/'historical_Swin80k_paired.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
 metrics=['psnr_db','lpips_alex','dinov2_vitl14_cosine'];table_rows=[]
 for method,label in [('SwinJSCC_new_shared','SwinJSCC (adapted, 80k)'),('HiFiDiffCom_SwinJSCC','HiFi-DiffCom (adapted, Swin80k)')]:
  values=[next(r for r in historical if r['method']==method and r['metric']==metric) for metric in metrics]
  table_rows.append([label]+[f"{float(r['mean']):.3f}" for r in values])
 table_rows.append(['HiFi minus paired Swin']+[f"{float(next(r for r in paired if r['metric']==m)['mean']):+.3f}" for m in metrics])
 fig,ax=plt.subplots(figsize=(7,1.7));ax.axis('off');fig.subplots_adjust(left=.02,right=.98,top=.81,bottom=.15)
 table=ax.table(cellText=table_rows,colLabels=['Historical receiver','PSNR (dB)','LPIPS','DINOv2-L'],cellLoc='center',loc='center',colWidths=[.49,.17,.17,.17]);table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.65)
 for (i,j),cell in table.get_celld().items():
  cell.set_linewidth(.4);cell.set_edgecolor('#BBBBBB')
  if i==0:cell.set_facecolor('#EEF3F8')
 fig.text(.5,.96,'Historical fixed16 | N = 1024 | 13 dB | Swin80k link',ha='center',va='top',fontsize=9)
 fig.text(.5,.045,'Same received observation within each pair; exploratory sources, not a full-test-set claim.',ha='center',fontsize=7)
 for ext in ['pdf','svg','png']:fig.savefig(OUT/('historical_fixed16_table.'+ext),dpi=600)
 plt.close(fig)
 (OUT/'historical_table_caption.tex').write_text('Historical fixed16 exploratory comparison at N=1024 and13dB. Both receivers use the same Swin80k transmitted signal and actual observation. Means and paired differences are taken from published CSV precision; existing95 percent intervals remain in the accompanying single-method and paired CSVs. Final Swin200k qualitative outputs are not included in this historical table. No new bootstrap or selected-four mean was computed.\n')
 (OUT/'README.md').write_text('# Final-link HiFi qualitative comparison\n\n'+caption+'\n\nAll16 pages: `all16_final_link.pdf`. The selected4 and selected3 panels are display options; all outcomes remain available. Each panel has PDF, editable-text SVG and600dpi PNG. Native pixels remain256x256.\n\n`historical_Swin80k_fixed16.csv` reuses full-precision published statistics and intervals, not new final-link statistics. No scoring or bootstrap was run.\n\nReproduce layouts only: `python experiments/hifi-final-fixed16-20261012/plot_final.py`. Do not rerun the inference entry.\n')
 files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='provenance.json']
 (OUT/'provenance.json').write_text(json.dumps(dict(source_bindings=bindings,completion_sha256=sha(RESULT/'completion.json'),actual_wait_sha256=sha(RESULT/'actual_wait.json'),registration_sha256=sha(RESULT/'registration.json'),script_sha256=sha(__file__),historical_csv_sha256=sha(src),source_indices=ids,user_selected=[21,92,95,75],outputs={str(p.relative_to(OUT)):sha(p) for p in files},new_model_calls=0,new_metric_calls=0,new_bootstrap_calls=0),indent=2))
 print('EXPORTED_ALL16_FINAL_LINK_AND_SELECTED_PANELS',flush=True)
if __name__=='__main__':main()
