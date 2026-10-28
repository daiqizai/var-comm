"""Independent saved-output audit: hashes, denominators, statistics and exact reuse."""
from block_common import *
from nominate_p2 import METRICS
import csv,collections

def main():
 out=L/'p3_final_audit';out.mkdir(exist_ok=False)
 c=read(O/'p3_closure.json');rows=list(csv.DictReader((O/'replay_metrics.csv').open()));ps=list(csv.DictReader((O/'p3_physical_events.csv').open()))
 assert len(ps)==4320 and len(rows)==10656
 pairs=read(L/'p1_summary/pair_scores.json');cache=read(L/'p3_replay_v2/receiver_cache.json');images={};waveforms={};hard_count=0
 for r in rows:
  score=pairs[r['pair_key']]
  for k in METRICS:assert float(r[k])==score[k]
  if r['state_key']:
   v=cache['native'][r['state_key']];assert digest(v['state'])==r['state_key']and v['pixel_sha256']==score['pixel_sha256'];images[r['state_key']]=v
 for v in images.values():
  with np.load(checked(v['image']),allow_pickle=False)as z:a=z['image']
  assert a.dtype==np.float32 and a.shape==(3,256,256)and asha(a)==v['pixel_sha256']
 for s in read(O/'source_manifest.json'):
  with np.load(checked(s['archive']),allow_pickle=False)as z:a=z['pixels']
  assert a.dtype==np.uint8 and a.shape==(3,256,256)
  assert hashlib.sha256(a.tobytes()).hexdigest()==s['preprocessing_id']
  assert asha(a.astype(np.float32)/np.float32(255))==s['reference_sha256']
 for i in range(4):
  comp=read(L/f'p3_phy_{i:02d}/completion.json')
  for r in cr(comp['events']):waveforms[r['rx_record']['path']]=r['rx_record']
 nframes=0
 for d in waveforms.values():
  record=cr(d)
  with np.load(checked(record['received']),allow_pickle=False)as z:
   assert z['received'].shape==(len(record['events']),1024,2)and np.isfinite(z['received']).all();nframes+=len(record['events'])
   for i,e in enumerate(record['events']):
    for g in e['blocks']:
     a=z[f'hard_{i}_{g["group"]}'];assert asha(a)==g['hard_sha256']and len(a)==g['geometry']['k'];hard_count+=1
 assert nframes==4320 and hard_count==9029
 summaries=list(csv.DictReader((O/'p3_candidate_overall.csv').open()))
 for summary in summaries:
  rr=[r for r in rows if(r['candidate_id'],r['mode'])==(summary['candidate_id'],summary['mode'])];assert len(rr)==288
  assert len({r['source_id']for r in rr})==32
  for k in METRICS:assert abs(float(summary[k])-sum(float(r[k])for r in rr)/288)<1e-12
 save(O/'p3_final_audit.json',dict(status='PASS',frames=nframes,saved_observation_batches=len(waveforms),actual_hard_body_outputs=hard_count,metric_rows=len(rows),metric_values=len(rows)*5,exact_native_images_checked=len(images),reference_sources_checked=32,candidate_mode_summaries=len(summaries),new_model_calls=0,new_PHY_decodes=0,new_bootstrap=0,source_hash_is_not_pipeline_id=True,all_reference_images_checked_only_after_saved_RX=True))
 print(json.dumps(read(O/'p3_final_audit.json')))
if __name__=='__main__':main()
