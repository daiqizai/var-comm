"""Frozen common500 regression; source-paired bootstrap, no candidate selection."""
from opt_common import *
from route_statistics_v2 import METRICS,CORE,interval,csvout
from frame_plan import CELLS
import collections

def closed(role):
 w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
 c=read(L/role/'completion.json');assert c['status']=='COMPLETE'and c['ledger']['unresolved']==0
 return c

def main(request):
 r=read(request);freeze=cr(r['final_freeze']);assert freeze['configuration']=='original_receiver_plus_D_token';sources=cr(freeze['common500_sources']);ids=[s['source_id']for s in sources];sm={s['source_id']:s for s in sources}
 assert len(set(ids))==500 and [s['source_index']for s in sources]==list(range(500))
 oldproto=cr(r['bank_identity']);assert oldproto['statistics']['source_order']==ids;bp=oldproto['statistics']['bootstrap'];b=np.load(checked(bp),allow_pickle=False);assert b.shape==(10000,500)
 h=cr(r['historical_frames']);missing=cr(closed('final_historical_missing_clip')['scores']);frames=[]
 for row in h:
  v=dict(row)
  if 'clip_image_cosine'not in v:
   q=missing[v['pair_key']];assert q['reference_sha256']==v['reference_sha256']and q['pixel_sha256']==v['pixel_sha256'];v['clip_image_cosine']=q['clip_image_cosine'];v['clip_origin']='NEW_CLIP_ONLY_ON_HISTORICAL_NATIVE_ARRAY'
  else:v['clip_origin']='EXACT_HISTORICAL_SCALAR'
  v['method']={'NEST':'original_NeST','WHOLE':'WHOLE_final_receiver','ADA':'Ada_frozen'}[v['method']];frames.append(v)
 scores={}
 for mode in ['three','clip']:
  for key,v in cr(closed('final_new_'+mode)['scores']).items():scores.setdefault(key,{}).update(v)
 pairs={v['pair_key']:v for v in read(L/'final_new_metric_plan/plan.json')['pairs']}
 for row in read(L/'final_new_metric_plan/logical.json'):
  v=dict(row);q=scores[v['pair_key']];assert q['source_id']==v['source_id']and q['reference_sha256']==sm[v['source_id']]['reference_sha256']
  v.update({m:q[m]for m in METRICS});v.update(image=pairs[v['pair_key']]['image'],pixel_sha256=q['pixel_sha256'],reference_sha256=q['reference_sha256'],metric_origin='NEW_OR_EXACT_SAME_DEVICE_PAIR_CACHE',clip_origin='FORMAL_TORCH210_CLIP')
  v['method']='final_NeST'if v['variant']=='D_token'else'WHOLE_final_receiver';frames.append(v)
 assert len(frames)==42000;events=set();groups=collections.defaultdict(list)
 for v in frames:
  s=sm[v['source_id']];assert v['source_index']==s['source_index'];assert all(np.isfinite(float(v[m]))for m in METRICS)
  key=(v['method'],v['N'],v['snr_db'],v['source_id'],v['noise_seed']);assert key not in events;events.add(key);groups[key[:-1]].append(v)
 source=[]
 for key,g in groups.items():
  assert len(g)==3 and len({v['channel_repeat']for v in g})==3,key
  source.append(dict(method=key[0],N=key[1],snr_db=key[2],source_id=key[3],source_index=sm[key[3]]['source_index'],channel_events=3,**{m:float(np.mean([v[m]for v in g]))for m in METRICS}))
 assert len(source)==14000;sg=collections.defaultdict(dict)
 for v in source:sg[(v['method'],v['N'],v['snr_db'])][v['source_id']]=v
 arrays={};summary=[];paired=[]
 for key,g in sorted(sg.items()):
  assert set(g)==set(ids)
  for m in METRICS:
   x=np.asarray([g[s][m]for s in ids]);arrays[key+(m,)]=x;mean,lo,hi=interval(x,b);summary.append(dict(method=key[0],N=key[1],snr_db=key[2],metric=m,n_sources=500,logical=1500,mean=mean,ci95_low=lo,ci95_high=hi))
 for cand,ref in [('final_NeST','original_NeST'),('final_NeST','WHOLE_final_receiver'),('final_NeST','Ada_frozen'),('WHOLE_final_receiver','Ada_frozen')]:
  for N,snr in CELLS:
   for m in METRICS:
    mean,lo,hi=interval(arrays[(cand,N,snr,m)]-arrays[(ref,N,snr,m)],b);paired.append(dict(candidate=cand,reference=ref,N=N,snr_db=snr,metric=m,n_sources=500,mean_delta=mean,ci95_low=lo,ci95_high=hi))
 fixed={(v['N'],v['snr_db']):v for v in read(D/'fit_baseline_denominators.json')['values']}
 for m in ['lpips_alex','dreamsim']:
  x=np.mean([(arrays[('final_NeST',*c,m)]-arrays[('original_NeST',*c,m)])/fixed[c][m]for c in CORE],axis=0);mean,lo,hi=interval(x,b);paired.append(dict(candidate='final_NeST',reference='original_NeST',N=1024,snr_db='core_mean',metric='FIT_normalized_'+m,n_sources=500,mean_delta=mean,ci95_low=lo,ci95_high=hi))
 counts=[];fg=collections.defaultdict(list)
 for v in frames:fg[(v['method'],v['N'],v['snr_db'])].append(v)
 for key,g in sorted(fg.items()):
  valid=[v for v in g if 'utilization'in v];ep=collections.Counter(str(v.get('endpoint',v.get('received_prefix','NOT_RECORDED')))for v in g)
  counts.append(dict(method=key[0],N=key[1],snr_db=key[2],logical=len(g),fixed_gray=sum(v['image']is None for v in g),receive_status_counts=json.dumps(dict(collections.Counter(v['receiver_status']for v in g)),sort_keys=True),mean_source_utilization=float(np.mean([v['utilization']for v in valid]))if valid else '',capacity_denominator='Arithmetic source bits / (body information bits minus29); not airtime efficiency',endpoint_counts=json.dumps(dict(ep),sort_keys=True)))
 out=D/r['output'];out.mkdir(exist_ok=False);csvout(out/'frame_metrics.csv',frames);csvout(out/'per_source_metrics.csv',source);csvout(out/'summary.csv',summary);csvout(out/'paired_deltas.csv',paired);csvout(out/'endpoint_and_failure_summary.csv',counts)
 save(out/'statistics_identity.json',dict(status='COMPLETE',registration=pin(request),final_freeze=r['final_freeze'],source_order=ids,bank=bp,bank_source_order_verified=True,common500_is_previously_examined_regression=True,independent_new_test=False,source_then_pair=True,preserved_method_native_noise=True,received_failures_kept=True,historical_H800_numeric_route_preserved=True,new_receiver_PRO_FP32=True,CLIP_new_features_torch210_only=True,no_selection_on_final=True,logical=42000,source_rows=14000,mean_CI_rows=len(summary),paired_CI_rows=len(paired),NEW200='BLOCKED_NO_AUDITABLE_UNUSED_RAW_POOL'))
 print('FINAL_REGRESSION_COMPLETE',len(frames),len(source));print([v for v in paired if v['snr_db']=='core_mean'])
if __name__=='__main__':main(*sys.argv[1:])
