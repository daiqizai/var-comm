"""Frozen repeat-zero candidate contact sheets; display-only quantization, no inference."""
from followup_common import *
from PIL import Image,ImageDraw,ImageFont
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];os.sched_setaffinity(0,set(r['cpu_affinity']))
 out=Path(r['output']);out.mkdir(exist_ok=False);(out/'pages').mkdir();beg=time.monotonic()
 try:
  sources=cr(r['sources']);sm={s['source_id']:s for s in sources};frames=cr(r['frames']);methods=['WHOLE_final_receiver','original_NeST','final_NeST','Ada_frozen']
  data={(v['snr_db'],v['source_id'],v['method']):v for v in frames if v['N']==1024 and v['snr_db']in[7,10]and v['channel_repeat']==0 and v['method']in methods};assert len(data)==4000
  index=[];pages=[];fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontpath,15);small=ImageFont.truetype(fontpath,12)
  def render(v,s):
   if v is None:
    with np.load(checked(s['archive']),allow_pickle=False)as z:a=z['pixels'];assert hashlib.sha256(a.tobytes()).hexdigest()==s['preprocessing_id']
    return Image.fromarray(a.transpose(1,2,0))
   if v['image']is None:a=np.full((3,256,256),.5,np.float32)
   else:
    with np.load(checked(v['image']),allow_pickle=False)as z:a=z['image']
   assert asha(a)==v['pixel_sha256'];return Image.fromarray(np.rint(a.transpose(1,2,0)*255).clip(0,255).astype(np.uint8))
  for snr in [7,10]:
   ranked=[]
   for s in sources:
    g={m:data[snr,s['source_id'],m]for m in methods};d=g['final_NeST'];w=g['WHOLE_final_receiver'];a=g['Ada_frozen']
    gains={f'{m}_gain_over_{label}':ref[m]-d[m]for label,ref in [('WHOLE',w),('Ada',a)]for m in ['dists','dreamsim']}
    eligible=all(v['image']is not None for v in g.values())and d['pixel_sha256']!=w['pixel_sha256']
    same_direction=min(gains.values())>0;rank_score=min(gains['dists_gain_over_WHOLE'],gains['dists_gain_over_Ada'])+.5*min(gains['dreamsim_gain_over_WHOLE'],gains['dreamsim_gain_over_Ada'])
    ranked.append((eligible,same_direction,rank_score,s,g,gains))
   ranked.sort(key=lambda x:(-int(x[0]),-int(x[1]),-x[2],x[3]['source_index']))
   for start in range(0,500,8):
    sheet=Image.new('RGB',(1280,40+len(ranked[start:start+8])*290),'white');draw=ImageDraw.Draw(sheet)
    for c,label in enumerate(['Source','WHOLE','Old Quarter','New Token-wise','Ada (adapted)']):draw.text((c*256+5,5),f'{label} | {snr} dB',fill='black',font=font)
    for row,item in enumerate(ranked[start:start+8]):
     eligible,same,score,s,g,gains=item;rank=start+row+1;y=40+row*290
     for c,m in enumerate([None,*methods]):
      v=None if m is None else g[m];sheet.paste(render(v,s),(c*256,y))
      label=f"rank {rank}; source {s['source_index']}"if v is None else f"DISTS {v['dists']:.3f} / DS {v['dreamsim']:.3f}"
      draw.text((c*256+4,y+257),label,fill='black',font=small)
     index.append(dict(snr_db=snr,rank=rank,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],source_pixels_sha256=s['preprocessing_id'],repeat_index=0,success_eligible=eligible,both_metrics_both_comparators=same,rank_score=score,**gains,**{m+'_pixel_sha256':g[m]['pixel_sha256']for m in methods}))
    path=out/'pages'/f'candidates_{snr:02d}db_{start//8+1:02d}.png';sheet.save(path);pages.append(dict(snr_db=snr,rank_first=start+1,rank_last=min(start+8,500),image=pin(path)))
    if time.monotonic()-beg>=r['seconds']:raise BudgetStop('F1 contact output cap')
   print('CONTACT_SNR_COMPLETE',snr,flush=True)
  dst=D/'f1';dst.mkdir(exist_ok=False);csvout(dst/'candidate_index.csv',index);save(dst/'contact_sheet_index.json',pages)
  save(dst/'selection_protocol.json',dict(source_order=[s['source_id']for s in sources],cells=[[1024,7],[1024,10]],repeat_index=0,all500_per_cell=True,ranking='success and different WHOLE/D pixels; then all four DISTS/DreamSim gains positive; descending min(DISTS gains)+0.5 min(DreamSim gains), tie source index',ranking_is_for_illustration_not_statistics=True,top40_per_cell_to_open=True,historical_noise_namespace_unchanged=True,no_quality_inference=True,no_seed_search=True,no_image_enhancement=True,display_quantization='round(native_float*255), PNG only; metrics remain native float',scientific_inputs=r['frames']))
  save(out/'completion.json',dict(status='COMPLETE',sources_per_cell=500,cells=2,pages=len(pages),candidates=1000,new_model_calls=0,new_PHY=0,seconds=time.monotonic()-beg));print('F1_CONTACTS_COMPLETE',len(pages))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
