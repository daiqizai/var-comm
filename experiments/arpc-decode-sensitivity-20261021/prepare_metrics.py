"""Bind closed native outputs and frozen scorers before any new scoring."""
from common import *
import numpy as np
def ash(a):
 return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def main():
 wait=read(L/'rx_control/actual_wait.json');launch=read(L/'rx_control/launch.json')
 assert wait['returncode']==0 and not wait['timed_out'] and gone(launch['process']) and gone(launch['parent_process'])
 assert not (L/'rx_control/failure.json').exists()
 done=read(L/'rx/completion.json');assert done['counts']==dict(model_load=1,rx_render=592)
 original=read(OLD/'a3_metrics_v1/registration.json')
 config=read(E/'metric_config.json')
 plan=read(checked(original['pair_plan']))['sources']
 outputs={(v['setting'],v['key']):v for v in done['outputs']}
 jobs=[];sources=[]
 for s in plan:
  checked(s['archive'])
  with np.load(s['archive']['path'],allow_pickle=False) as z:x=z['pixels'].astype(np.float32)/np.float32(255)
  assert x.shape==(3,256,256)
  sources.append({k:s[k] for k in ['source_id','source_index','archive']}|{'pixel_sha256':ash(x)})
  for pair in s['pairs']:
   key=Path(pair['image']['path']).stem
   for setting in ['greedy','low_tau025']:
    v=outputs[(setting,key)];p=Path(v['path'])
    with np.load(p,allow_pickle=False) as z:y=z['float_output']
    assert y.dtype==np.float32 and y.shape==x.shape and np.isfinite(y).all()
    assert hashlib.sha256(y.tobytes()).hexdigest()==v['float_sha256']
    jobs.append(dict(setting=setting,key=key,source_id=s['source_id'],source_index=s['source_index'],
      source=s['archive'],reference_sha256=ash(x),image=desc(p),float_sha256=v['float_sha256'],
      pixel_sha256=ash(y),endpoint=v['endpoint'],frames=pair['frames']))
 assert len(jobs)==592 and len(sources)==100
 clipdir=B/'clip_i_common500_20261020_v1/score'
 features={v['pixel_sha256']:v for v in read(clipdir/'feature_index.json')}
 cache=[]
 for s in sources:
  feat=features[s['pixel_sha256']];checked(feat['feature'])
  cache.append(dict(source_id=s['source_id'],pixel_sha256=s['pixel_sha256'],feature=feat['feature']))
 save(O/'metric_jobs.json',jobs);save(O/'metric_sources.json',sources);save(O/'clip_reference_cache.json',cache)
 reg=dict(schema='ARPC_SENSITIVITY_FOUR_FIXED_METRICS_V1',created_unix=time.time(),
  rx_registration=desc(O/'registration.json'),rx_completion=desc(L/'rx/completion.json'),
  rx_wait=desc(L/'rx_control/actual_wait.json'),source_files=[desc(E/n) for n in
   ['score_metrics.py','metrics_control.py','prepare_metrics.py','common.py','metric_config.json']],
  original_four=original['original_four_metrics'],config=config,jobs=desc(O/'metric_jobs.json'),
  sources=desc(O/'metric_sources.json'),clip_cache=desc(O/'clip_reference_cache.json'),
  prior_clip_identity=desc(clipdir/'metric_identity.json'),max_seconds=3600,
  three_limits=dict(load_lpips=1,load_dreamsim=1,dreamsim_reference=100,lpips_pair=592,dreamsim_image=592,psnr=592),
  clip_limits=dict(load_clip=1,clip_image=592,cosine_pair=592),
  zero_calls=['training','source_encoding','caption','channel','reconstruction','other_neural_metrics'],
  original_sampling_scores_reused=True,new_bootstrap_indices=0,automatic_retry=False)
 save(O/'metric_registration.json',reg)
 print(json.dumps(desc(O/'metric_registration.json')))
if __name__=='__main__':main()
