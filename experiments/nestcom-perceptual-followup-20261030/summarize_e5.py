"""Direct end-to-end wall measurements only, no historical six-core additions."""
from followup_common import *
from collections import defaultdict

def main(plan):
 plan=Path(plan);groups=read(plan/'groups.json');sources=read(plan/'sources.json');ids=[s['source_id']for s in sources];allrows=[];summary=[];per_source=[]
 for gi,g in enumerate(groups):
  joined={}
  for direction in ['tx','rx']:
   role=f'E5_g{gi:02d}_{direction}';closed(role);rr=read(L/role/'rows.json');assert len(rr)==160
   assert {v['source_id']for v in rr}==set(ids)
   for v in rr:
    assert v['group']==g['name']and v['direction']==direction;allrows.append(v)
    joined.setdefault((v['source_id'],v['repetition']),{})[direction]=v
  data=defaultdict(list)
  for (sid,rep),v in joined.items():
   assert set(v)=={'tx','rx'}and v['tx']['phase']==v['rx']['phase']
   if v['tx']['phase']!='measured':continue
   data[sid].append(dict(TX=v['tx']['seconds'],RX=v['rx']['seconds'],total=v['tx']['seconds']+v['rx']['seconds'],success=v['rx']['success']))
  assert set(data)==set(ids)and all(len(v)==3 for v in data.values())
  for sid in ids:
   v=data[sid];per_source.append(dict(group=g['name'],source_id=sid,TX=float(np.mean([x['TX']for x in v])),RX=float(np.mean([x['RX']for x in v])),total=float(np.mean([x['total']for x in v])),successes=sum(x['success']for x in v),measured_calls=3))
  for direction in ['TX','RX','total']:
   vals=[x[direction]for sid in ids for x in data[sid]];ss=[x[direction]for sid in ids for x in data[sid]if x['success']]
   summary.append(dict(group=g['name'],method=g['method'],N=g['N'],snr_db=g['snr_db'],grid=g.get('grid',''),direction=direction,sources=32,measured=96,warmup=64,equal_source_mean_seconds=float(np.mean([np.mean([x[direction]for x in data[sid]])for sid in ids])),call_median_seconds=float(np.median(vals)),call_p95_seconds=float(np.quantile(vals,.95)),successful_RX_calls=len(ss),success_conditional_RX_mean_seconds=float(np.mean(ss))if direction=='RX'and ss else '',all_failed_in_denominator=True))
 out=D/'e5';out.mkdir(exist_ok=False);csvout(out/'call_timings.csv',allrows);csvout(out/'source_means.csv',per_source);csvout(out/'summary.csv',summary)
 save(out/'identity.json',dict(plan=pin(plan/'config.json'),source_order=ids,groups=pin(plan/'groups.json'),directions_measured_directly=True,total_definition='paired TX boundary wall time + RX boundary wall time; channel/noise generation excluded',historical_latency_rows=0,CPU_cores=2,original_math_threads=6,batch=1,all_warmup_and_measured_calls_fresh=True,groups_count=16,matched_token_reuses_same_group='Token_NeST_N1024_S10',source_mean_population=32))
 print('E5_TIMING_DELIVERED',len(allrows),len(summary))
if __name__=='__main__':main(*sys.argv[1:])
