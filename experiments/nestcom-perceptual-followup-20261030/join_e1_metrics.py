"""Join original metric scalars by full source, event and verified native pixel identity."""
from followup_common import *
def event(v):return(v['source_id'],int(v['snr_db']),int(v['noise_seed']))
def main():
 closed('E1_core_dists');closed('E1_external_dists');closed('E1_external_clip')
 out=L/'E1_join';out.mkdir(exist_ok=False);evidence=[]
 def table(p):evidence.append(pin(p));return rows(p)
 def load(p):evidence.append(pin(p));return read(p)
 sources=load(L/'inventory_core/sources.json');sm={s['source_id']:s for s in sources}
 core=load(L/'inventory_core/logical.json');ext=load(L/'inventory_external/logical.json')
 ds={}
 for role in ['E1_core_dists','E1_external_dists']:ds.update(load(L/role/'scores.json'))
 clip={ (v['method'],*event(v)):v for v in table(B/'clip_i_common500_20261020_v1/score/frame_scores.csv')}
 newclip=load(L/'E1_external_clip/scores.json')
 seven={(v['method'],*event(v)):v for v in table(R/'results/nestcom_7db_common500_20261030_v1/delivery/frame_metrics.csv')}
 oldnames=['P100000','Swin200000','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024']
 supplement={m:{event(v):v for v in table(R/f'results/quality_metrics_supplement_20261014_v1/common500/{m}_frame_scores.csv')}for m in oldnames}
 originals={m:{event(v):v for v in table(R/f'results/baseline_strength_20261011/automatic_followthrough_v1/common500/{m}.csv')}for m in oldnames[:2]}
 originals[oldnames[2]]={event(v):v for v in table(L/'input_evidence/bpg_frame_rows.csv')}
 oldreg=load(R/'results/quality_metrics_supplement_20261014_v1/common500/registration.json');native={(v['method'],*event(v)):v for v in cr(oldreg['frames'])if v['method']in oldnames};evidence.append(oldreg['frames'])
 arpc={}
 for p in [R/'results/arpc_external_baseline_20261019_v1/a3_metrics_v1/actual_pair_scores.csv',R/'results/arpc_common500_extension_20261019_v1/metrics_v1/actual_pair_scores.csv']:
  for v in table(p):
   k=(v['source_id'],v['image_float_sha256'])
   if k in arpc:assert all(arpc[k][m]==v[m]for m in ['psnr_db','lpips_alex','dreamsim'])
   arpc[k]=v
 result=[]
 for src in core+ext:
  v=dict(src);s=sm[v['source_id']];assert int(v['source_index'])==s['source_index']and v['reference_sha256']==s['reference_sha256']
  pk=v['dists_pair_key'];d=ds[pk];assert(d['reference_sha256'],d['pixel_sha256'])==(s['reference_sha256'],v['pixel_sha256']);v['dists']=d['dists']
  if'historical_method'in v:
   method=v['historical_method'];ev=event(v)
   if int(v['snr_db'])==7:
    q=seven[(method,*ev)];c=newclip[pk];assert q['pixel_sha256']==c['pixel_sha256']==v['pixel_sha256']and c['reference_sha256']==s['reference_sha256']and int(q['repeat'])-1==v['channel_repeat']
    v.update({m:q[m]for m in ['psnr_db','lpips_alex','dreamsim']});v['clip_image_cosine']=c['clip_image_cosine'];v['metric_origin']='EXACT_7DB_THREE_PLUS_NEW_TORCH210_CLIP'
   else:
    c=clip[(method,*ev)];assert c['pixel_sha256']==v['pixel_sha256']and c['reference_sha256']==s['reference_sha256'];v['clip_image_cosine']=c['clip_image_cosine']
    if method=='ARPC_ADAPTED':q=arpc[(v['source_id'],v['expected_pixel_sha256'])]
    else:
     su=supplement[method][ev];q=originals[method][ev];f=native[(method,*ev)];assert su['reconstruction_rgb_sha256']==v['pixel_sha256']and su['reference_rgb_sha256']==s['reference_sha256']
     if method=='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':
      proof=f['original_evidence'];assert q['image_sha256']==proof['legacy_image_sha256']and q['reference_sha256']==proof['legacy_reference_sha256']and q['frame_id']==proof['frame_id']
     else:assert q['reference_sha256']==s['reference_sha256']and q['reconstruction_sha256']==v['pixel_sha256']
     q=dict(q,dreamsim=su['dreamsim'])
    v.update({m:q[m]for m in ['psnr_db','lpips_alex','dreamsim']});v['metric_origin']='ORIGINAL_SCALARS_NATIVE_HASH_AND_EVENT_VERIFIED'
  v['image']=image_spec(v['image']);v['N']=int(v['N']);v['snr_db']=int(v['snr_db']);v['source_index']=int(v['source_index']);v['noise_seed']=int(v['noise_seed']);v['channel_repeat']=int(v['channel_repeat'])
  v.update({m:float(v[m])for m in METRICS});assert all(np.isfinite(v[m])for m in METRICS);result.append(v)
 assert len(result)==66000
 save(out/'frames.json',result);save(out/'completion.json',dict(status='COMPLETE',logical=66000,sources=500,metrics=list(METRICS),inputs=evidence,new_models=0,new_PHY=0,new_reconstruction=0,original_four_preserved=True))
 print('E1_ALL_FIVE_JOINED',len(result))
if __name__=='__main__':main()
