"""Source-equal descriptive P3 statistics from saved actual RX, no inference/bootstrap."""
from block_common import *
from nominate_p2 import METRICS
from audit_prepare_p1 import closed
import collections,traceback

def mean(rows,keys=METRICS):return {k:float(np.mean([r[k]for r in rows],dtype=np.float64))for k in keys}

def main():
 out=L/'p3_summary';out.mkdir(exist_ok=False)
 try:
  scope=read(O/'scope.json');nom=read(O/'nomination.json');plan=read(O/'p3_plan.json');sources=cr(scope['source_manifest']);src={s['source_id']:s for s in sources}
  physical=[];receipts=[];totals=collections.Counter()
  for i in range(4):
   role=f'p3_phy_{i:02d}';c=closed(L,role,L/(role+'_control'));physical+=cr(c['events']);receipts.append(dict(completion=pin(L/role/'completion.json'),actual_wait=pin(L/(role+'_control')/'actual_wait.json')))
   for k in ['frames','tx_body_blocks','body_decode_attempts','header_decode_attempts']:totals[k]+=c[k]
  assert len(physical)==totals['frames']==nom['physical_frames']==4320 and totals['tx_body_blocks']==10656
  assert totals['body_decode_attempts']<=scope['bounds']['p3_bodies']and totals['header_decode_attempts']<=scope['bounds']['p3_headers']
  assert len({r['event_id']for r in physical})==len(physical)
  comp=read(L/'p3_replay_v2/completion.json');assert comp['status']=='COMPLETE_EXACT_RX_REPLAY'and comp['missing_count']==0
  replay={r['event_id']:r['result']for r in cr(comp['replay'])};assert set(replay)=={p['event_id']for p in physical}
  scores=read(L/'p1_summary/pair_scores.json');gray={e['source_id']:e['pair_key']for e in read(O/'clean_metric_events.json')if e['state']=='GRAY'}
  candidates={c['candidate_id']:c for c in nom['selected']};rows=[];physical_rows=[];wire={}
  for p in physical:
   sid=p['source_id'];s=src[sid];r=replay[p['event_id']];assert not r['TX_truth_used']and not r['missing']
   if p['rx_record']['path']not in wire:wire[p['rx_record']['path']]=cr(p['rx_record'])
   observation=wire[p['rx_record']['path']]['events'][p['rx_index']]
   modes=('ALL','SEQ','SEL')if candidates[p['candidate_id']]['kind']=='THREE'else('SINGLE',)
   for mode in modes:
    v=r['modes']['SEL'if mode=='SINGLE'else mode]
    if v['state']=='GRAY':key=gray[sid]
    else:key=digest(dict(source_id=sid,reference_sha256=s['reference_sha256'],pixel_sha256=v['pixel_sha256']))
    score=scores.get(key);assert score is not None,'Missing actual displayed source/image metric pair; do not substitute clean expected state'
    assert score['source_id']==sid and score['reference_sha256']==s['reference_sha256']
    rows.append(dict(event_id=p['event_id'],candidate_id=p['candidate_id'],kind=candidates[p['candidate_id']]['kind'],source_id=sid,source_index=p['source_index'],snr_db=p['snr_db'],replicate=p['replicate'],mode=mode,state=v['state'],state_key=v['state_key']or'',pair_key=key,**{k:score[k]for k in METRICS}))
   physical_rows.append(dict(**{k:p[k]for k in ['event_id','candidate_id','source_id','source_index','snr_db','replicate','tx_profile_id','tx_null','tx_fallback','tx_source_bits','tx_information_bits','tx_body_blocks','N','Es','tx_energy','header_accepted_wrong']},body_accepted_wrong_count=len(p['body_accepted_wrong']),rx_profile_id=observation['header'].get('profile_id'),header_status='ACCEPTED'if observation['header']['header_ok']else'REJECTED',C_status=r['source_status'].get('C','HEADER_UNAVAILABLE'),A_status=r['source_status'].get('A','NOT_ENABLED'),B_status=r['source_status'].get('B','NOT_ENABLED'),packet_statuses='|'.join(d['status']for d in observation['blocks']),source_capacity_utilization=p['tx_source_bits']/(p['tx_information_bits']-29*p['tx_body_blocks'])))
  csvsave(O/'p3_physical_events.csv',physical_rows);csvsave(O/'replay_metrics.csv',rows)
  groups=collections.defaultdict(list)
  for r in rows:groups[r['candidate_id'],r['mode'],r['snr_db'],r['source_id']].append(r)
  means=[]
  for (cid,mode,snr,sid),rr in groups.items():
   assert len(rr)==3 and {r['replicate']for r in rr}==set(scope['replicates'])
   means.append(dict(candidate_id=cid,mode=mode,snr_db=snr,source_id=sid,source_index=rr[0]['source_index'],**mean(rr)))
  csvsave(O/'p3_source_means.csv',means)
  summaries=[];overall=[];states=[]
  for cid,c in candidates.items():
   modes=('ALL','SEQ','SEL')if c['kind']=='THREE'else('SINGLE',)
   for mode in modes:
    for snr in scope['actual_snr_db']:
     rr=[r for r in means if (r['candidate_id'],r['mode'],r['snr_db'])==(cid,mode,snr)];assert len(rr)==32
     summaries.append(dict(candidate_id=cid,kind=c['kind'],mode=mode,snr_db=snr,n_sources=32,physical_frames=96,**mean(rr)))
     cc=collections.Counter(r['state']for r in rows if(r['candidate_id'],r['mode'],r['snr_db'])==(cid,mode,snr));assert sum(cc.values())==96
     for state in ('GRAY','C','CA','CB','CAB','SINGLE'):states.append(dict(candidate_id=cid,mode=mode,snr_db=snr,state=state,count=cc[state],denominator=96,fraction=cc[state]/96))
    rr=[r for r in summaries if(r['candidate_id'],r['mode'])==(cid,mode)];assert len(rr)==3
    overall.append(dict(candidate_id=cid,kind=c['kind'],mode=mode,physical_frames=288,n_sources=32,**mean(rr)))
  csvsave(O/'p3_candidate_snr_summary.csv',summaries);csvsave(O/'p3_candidate_overall.csv',overall);csvsave(O/'state_counts.csv',states)
  best=sorted([r for r in overall if r['kind']=='SINGLE'],key=lambda r:(-r['dino_cosine'],r['candidate_id']))[0]
  lookup={(r['candidate_id'],r['mode'],r['snr_db'],r['source_id']):r for r in means};event_lookup={(r['event_id'],r['mode']):r for r in rows}
  paired=[];pair_summary=[];mechanism=[]
  for cid,c in candidates.items():
   if c['kind']!='THREE':continue
   for snr in scope['actual_snr_db']:
    pp=[p for p in physical if(p['candidate_id'],p['snr_db'])==(cid,snr)];assert len(pp)==96
    contribution=[]
    for p in pp:
     result=replay[p['event_id']];sel=event_lookup[p['event_id'],'SEL'];seq=event_lookup[p['event_id'],'SEQ'];st=result['source_status']
     indicator=st.get('C')=='ACCEPTED_CANONICAL_CACHE'and st.get('A')not in ('ACCEPTED_CANONICAL_CACHE','DISABLED_PAID_EMPTY')and st.get('B')=='ACCEPTED_CANONICAL_CACHE'
     assert indicator==(sel['state']=='CB')
     delta={k:sel[k]-seq[k]for k in METRICS}
     if not indicator:assert all(abs(d)<1e-12 for d in delta.values())
     else:assert seq['state']=='C'
     contribution.append(dict(event_id=p['event_id'],indicator=indicator,**delta))
    for metric in METRICS:
     active=[r[metric]for r in contribution if r['indicator']];observed=float(np.mean([r[metric]for r in contribution]));product=len(active)/96*(float(np.mean(active))if active else 0)
     assert abs(observed-product)<1e-12
     mechanism.append(dict(candidate_id=cid,snr_db=snr,metric=metric,C_success_A_fail_B_success=len(active),denominator=96,frequency=len(active)/96,conditional_CB_minus_C=float(np.mean(active))if active else 0,SEL_minus_SEQ=observed,frequency_times_quality=product))
    for sid in src:
     a=lookup[cid,'SEL',snr,sid]
     for contrast,other in [('SEL_minus_SEQ',lookup[cid,'SEQ',snr,sid]),('SEL_minus_best_measured_single',lookup[best['candidate_id'],'SINGLE',snr,sid])]:
      paired.append(dict(candidate_id=cid,contrast=contrast,reference_candidate=cid if contrast=='SEL_minus_SEQ'else best['candidate_id'],snr_db=snr,source_id=sid,source_index=a['source_index'],**{k:a[k]-other[k]for k in METRICS}))
   for contrast in ('SEL_minus_SEQ','SEL_minus_best_measured_single'):
    for snr in [*scope['actual_snr_db'],'overall']:
     rr=[r for r in paired if r['candidate_id']==cid and r['contrast']==contrast and(snr=='overall'or r['snr_db']==snr)]
     assert len(rr)==(96 if snr=='overall'else 32)
     pair_summary.append(dict(candidate_id=cid,contrast=contrast,snr_db=snr,n_sources=32,**mean(rr)))
  csvsave(O/'paired_source_differences.csv',paired);csvsave(O/'paired_mean_differences.csv',pair_summary);csvsave(O/'selective_mechanism_contribution.csv',mechanism)
  best_three=sorted([r for r in overall if r['mode']=='SEL'],key=lambda r:(-r['dino_cosine'],r['candidate_id']))[0]
  qualify=[]
  for r in overall:
   if r['mode']!='SEL':continue
   seq=next(z for z in overall if(z['candidate_id'],z['mode'])==(r['candidate_id'],'SEQ'))
   if r['dino_cosine']>best['dino_cosine']and r['dino_cosine']>seq['dino_cosine']:qualify.append(r['candidate_id'])
  closure=dict(status='COMPLETE_P0_P3',recommendation='GO_RECOMMEND_P4_ONLY_NOT_LAUNCHED'if qualify else'NO_GO_STOP',qualifying_candidates=qualify,best_measured_single=best,best_measured_SEL=best_three,delta_J_best_SEL_minus_best_single=best_three['dino_cosine']-best['dino_cosine'],technical_correctness='PASS; original CPU cache-plan failure preserved; independent v2 fully closed',actual_counts=dict(totals),replay_metric_rows=len(rows),new_P3_source_probability_calls=0,new_P3_native_renders=0,new_P3_metric_inferences=0,exact_prior_RX_decode_cache_hits=comp['unique_reused_decode_keys'],distinct_scored_pairs_reused=len({r['pair_key']for r in rows}),header_accepted_wrong=sum(p['header_accepted_wrong']for p in physical),body_accepted_wrong=sum(len(p['body_accepted_wrong'])for p in physical),replay=pin(L/'p3_replay_v2/completion.json'),receipts=receipts,nomination=pin(O/'nomination.json'),statistical_scope='32 calibration sources;3actual noises per SNR, equal source and0/2/4dB; descriptive source-paired contrasts, no new bootstrap, no held-out/significance claim; best single only among4 nominees',P4_P5='NOT_RUN',new_training=False)
  assert closure['header_accepted_wrong']==0 and closure['body_accepted_wrong']==0
  save(O/'p3_closure.json',closure);save(out/'completion.json',dict(status='COMPLETE_P3_STATISTICS',closure=pin(O/'p3_closure.json'),new_inference=0,new_bootstrap=0))
  print(json.dumps(closure))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
