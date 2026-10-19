"""SHA-bound CPU metadata preparation only; no PHY/model/score/selection calls."""
from nc_common import *
from nc_protocol import closed_wait,validate_repair_grid,validate_full_grid
import collections,numpy as np,traceback
REPAIR_SHA='a7d4c76f3b5c386364b1be465acf57fd4eb95bdb1d84ba35a5088bcb99688f15'
REUSE_SHA='1b3575ff3883a63ab4b7b70f108355d07a06e2d610ee55792cbf5d77195603be'
OLDREG=R/'results/t1c_m10_test_20261016_v1/link_registration.json'
def main():
 out=BASE/'repair_preparation_v1';out.mkdir();start=time.monotonic()
 try:
  closed_wait(read(BASE/'audit_control_v2/actual_wait.json'),gone)
  assert sha(BASE/'repair_events.json')==REPAIR_SHA and sha(BASE/'reused_frames.json')==REUSE_SHA
  old=read(OLDREG);assert sha(OLDREG)=='32d148496df2e35912a35d878207a96fbe6d5d20345350276d748365ca3e1ef8'
  sources=checkread(old['sources']);assert len(sources)==600
  current=[dict(x) for x in sources if x['population']=='common500'];assert len(current)==500
  diff=read(BASE/'repair_events.json');reused=read(BASE/'reused_frames.json');rows=[];verified={};rxcache={};noiseproof=[]
  def verify(d):
   if d['path'] not in verified:checked(d);verified[d['path']]=d
   assert verified[d['path']]['sha256']==d['sha256']
  for s in current:
   for k in ('archive','current_streams','old_streams','old_metadata'):verify(s[k])
  for d in diff:
   i=d['source_index'];s=current[i];assert d['source_id']==s['source_id'] and d['tokens_sha256']==s['tokens_sha256'] and d['source_preprocessing_sha256']==s['preprocessing_id']
   for k in ('source_archive','stream_archive','old_physical_frame'):verify(d[k])
   verify(d['old_reconstruction']['image_archive'])
   oldrec=dict(d['old_reconstruction'],old_row=pin(BASE/'repair_events.json'))
   e=dict(population='common500',source_global_index=i,source_index=i,source_id=d['source_id'],snr_db=d['snr_db'],noise_seed=d['noise_seed'],public_frame_counter=d['desired_event']['public_frame_counter'],strategy_family='WHOLE',candidate_id='m7_q2_r2-3' if d['snr_db']==4 else 'm9_q4_r2-3',m=d['actual_m'],K=0,arithmetic_bits=d['arithmetic_bits'],payload_sha256=d['payload_sha256'],profile_id=d['profile_id'],profile_key=d['profile_key'],event_key=d['event_key'],reuse=False,original_logical_pair_unchanged=False,legacy=dict(image=oldrec,metrics=d['old_metrics']),execution_reference=dict(image=oldrec,metrics=d['old_metrics'],physical_frame=d['old_physical_frame']))
   e['frame_index']=len(rows);rows.append(e)
  grid=validate_repair_grid(rows);assert len(reused)==6579;validate_full_grid(rows+reused)
  # Reuse only closed, current frozen H800 RX outputs. No transmitter tokens enter this cache.
  for v in reused:
   verify(v['physical_frame']);rec=v['reconstruction'];verify(rec['image_archive'])
   pkt=read(v['physical_frame']['path']);rx=pkt['actual_RX'];sk=source_key(rx)
   if v['reuse_class']=='EXACT_CROSS_FAMILY_EVENT':
    oldfiles=list((BASE/f'whole_evidence_restored/posthoc_render500_v1/physical_frames/{v["source_index"]:04d}').glob(f'{v["snr_db"]}_{v["noise_seed"]}_*.json'));assert len(oldfiles)==1
    previous=read(oldfiles[0]);a=previous.get('observation_identity',previous);b=pkt.get('observation_identity',pkt.get('event',pkt))
    assert a['noise_sha256']==b['noise_sha256'],('NOISE_MISMATCH',v['source_index'],v['snr_db'],v['noise_seed'])
    noiseproof.append(dict(source_index=v['source_index'],snr_db=v['snr_db'],noise_seed=v['noise_seed'],old=pin(oldfiles[0]),reused=v['physical_frame'],noise_sha256=a['noise_sha256']))
   ev=rec.get('evidence');assert isinstance(ev,dict)
   if 'path' in ev:verify(ev)
   if sk in rxcache:assert rxcache[sk]['reconstruction']['image_sha256']==rec['image_sha256'],('BLOCKED_SAME_RX_DIFFERENT_PIXEL',sk)
   else:rxcache[sk]=dict(source_input_key=sk,reconstruction=rec,validated_unique_fixed_pixel=True,source_truth_used=False,proof_physical_frame=v['physical_frame'])
  assert len(noiseproof)==579
  routes=checkread(old['model_routes'])
  for d in routes['files']+routes['author_files']+[routes['dist']]:verify(d)
  for d in old['static'].values():
   if isinstance(d,dict) and 'path' in d:verify(d)
  save(out/'sources.json',current);save(out/'frames.json',rows);save(out/'receiver_cache.json',rxcache);save(out/'cross_family_noise_proof.json',noiseproof);save(out/'verified_inputs.json',list(verified.values()))
  closure=read(R/'results/t1c_m10_test_20261016_v1/delivery/closure_audit.json');bank=[b for b in closure['bootstrap'] if b['population']=='common500'][0];verify(bank['indices'])
  with Path(bank['indices']['path']).open('rb') as f:idx=np.load(f,allow_pickle=False)
  assert idx.shape==(10000,500) and idx.dtype==np.int64 and asha(idx)==bank['array_sha256']
  save(out/'completion.json',dict(status='PREPARED_METADATA_ONLY_NOT_SCIENTIFIC_COMPLETION',scope=SCOPE,original_registration=pin(OLDREG),audit=pin(R/'results/n1024_consistency_20261019_v1/completion.json'),sources=pin(out/'sources.json'),frames=pin(out/'frames.json'),reused_frames=pin(BASE/'reused_frames.json'),receiver_cache=pin(out/'receiver_cache.json'),cross_family_noise_proof=pin(out/'cross_family_noise_proof.json'),verified_inputs=pin(out/'verified_inputs.json'),exact_grid=grid,closed_RX_cache_entries=len(rxcache),bootstrap=bank,seconds=time.monotonic()-start,new_models=0,new_channel=0,new_scores=0,new_bootstrap_draws=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),new_science=0));raise
if __name__=='__main__':main()
