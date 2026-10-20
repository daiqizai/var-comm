"""Reuse the registered matched100 bank; never draw new bootstrap indices."""
from common import *
import numpy as np
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
SETTINGS=('original','greedy','low_tau025')
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def main():
 out=O/'delivery';out.mkdir()
 for role in ['three_metrics','clip_metrics']:
  wait=read(L/(role+'_control')/'actual_wait.json')
  assert type(wait['returncode'])==int and wait['returncode']==0 and not wait['timed_out']
  assert not (L/(role+'_control')/'failure.json').exists() and not (L/role/'failure.json').exists()
  done=read(L/role/'completion.json');assert done['unresolved']==0 and done['rows']==592
  checked(done['scores']);checked(done['identity'])
 cfg=read(O/'metric_registration.json');jobs=read(checked(cfg['jobs']))
 new={}
 for role in ['three_metrics','clip_metrics']:
  for v in rows(L/role/'scores.csv'):
   k=v['setting'],v['key'];new.setdefault(k,{})
   new[k].update({m:float(v[m]) for m in METRICS if m in v})
 original=rows(OLD/'a3_metrics_v1/logical_scores.csv')
 clips=rows(B/'clip_i_common500_20261020_v1/score/frame_scores.csv')
 clips={(v['source_id'],int(v['snr_db']),int(v['noise_seed'])):v for v in clips if v['method']=='ARPC_ADAPTED'}
 logical=[]
 for v in original:
  c=clips[(v['source_id'],int(v['snr_db']),int(v['noise_seed']))]
  logical.append({k:v[k] for k in ['source_id','source_index','snr_db','noise_seed','endpoint','receiver_status']}|
   dict(setting='original',state_key=v['key'],psnr_db=float(v['psnr_db']),lpips_alex=float(v['lpips_alex']),
    dreamsim=float(v['dreamsim']),clip_image_cosine=float(c['clip_image_cosine'])))
 for j in jobs:
  values=new[j['setting'],j['key']];assert set(values)==set(METRICS)
  for f in j['frames']:
   logical.append({k:f[k] for k in ['source_id','source_index','snr_db','noise_seed','endpoint','receiver_status']}|
      dict(setting=j['setting'],state_key=j['key'],**values))
 assert len(logical)==2700
 assert len({(v['setting'],v['source_id'],int(v['snr_db']),int(v['noise_seed'])) for v in logical})==2700
 bankmeta=read(OLD/'a3_metrics_v1/bootstrap_identity.json');bank=np.load(checked(bankmeta['indices']),allow_pickle=False)
 ids=bankmeta['source_ids'];assert bank.shape==(10000,100) and len(ids)==100
 assert [s['source_id'] for s in read(O/'metric_sources.json')]==ids
 def ci(x):
  assert x.shape==(100,) and np.isfinite(x).all()
  lo,hi=np.quantile(x[bank].mean(axis=1,dtype=np.float64),[.025,.975],method='linear')
  return dict(mean=float(x.mean(dtype=np.float64)),ci_low=float(lo),ci_high=float(hi))
 grouped={}
 for v in logical:grouped.setdefault((v['setting'],int(v['snr_db']),v['source_id']),[]).append(v)
 means=[];vectors={};summary=[];paired=[]
 for setting in SETTINGS:
  for snr in [4,10,19]:
   for m in METRICS:
    a=[]
    for sid in ids:
     v=grouped[setting,snr,sid];assert len(v)==3 and {int(x['noise_seed']) for x in v}=={6201,6202,6203}
     value=float(np.mean([x[m] for x in v],dtype=np.float64));a.append(value)
     means.append(dict(setting=setting,snr_db=snr,source_id=sid,source_index=v[0]['source_index'],metric=m,source_mean=value,noise_count=3))
    vectors[setting,snr,m]=np.array(a,dtype=np.float64)
    summary.append(dict(setting=setting,snr_db=snr,metric=m,source_count=100,**ci(vectors[setting,snr,m])))
 for a,b in [('greedy','original'),('low_tau025','original'),('greedy','low_tau025')]:
  for snr in [4,10,19]:
   for m in METRICS:
    paired.append(dict(comparison=a+'_minus_'+b,snr_db=snr,metric=m,source_count=100,
      favorable_sign='positive' if m in ['psnr_db','clip_image_cosine'] else 'negative',
      **ci(vectors[a,snr,m]-vectors[b,snr,m])))
 for n,data in [('logical_scores.csv',logical),('source_means.csv',means),('summary.csv',summary),('paired_differences.csv',paired)]:csvsave(out/n,data)
 save(out/'statistics_identity.json',dict(indices=bankmeta['indices'],seed=bankmeta['seed'],source_ids=ids,
  unit='source, first average 3 fixed noises',replicates=10000,new_index_draws=0,
  pairing='same sources and same received channel records within this sensitivity study',
  interval='pointwise percentile95; direct source differences; no multiplicity correction',
  source_pairing_order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
 count={}
 for role in ['rx','three_metrics','clip_metrics']:
  d=read(L/role/'completion.json');count[role]=d.get('counts')
 save(out/'actual_compute_and_reuse.json',dict(original_setting_logical_reused=900,original_setting_unique_reconstructions_reused=296,
  new_logical_setting_frames=1800,new_unique_reconstructions=592,reconstruction_logical_reuse=1208,
  distinct_received_prefixes_reused=296,new_channel_events=0,new_channel_decodes=0,new_source_encodes=0,
  new_captions=0,new_training=0,new_bootstrap_indices=0,original_clip_source_features_reused=100,
  counts=count,no_main_curve_replacement=True,N2048_clip_started=False))
 save(out/'completion.json',dict(status='COMPLETE',settings=list(SETTINGS),logical_setting_frames=2700,
  source_count=100,summary_cells=len(summary),paired_cells=len(paired),
  outputs=[desc(out/n) for n in ['logical_scores.csv','source_means.csv','summary.csv','paired_differences.csv','statistics_identity.json','actual_compute_and_reuse.json']]))
 print(json.dumps(summary));print('paired',json.dumps(paired))
if __name__=='__main__':main()
