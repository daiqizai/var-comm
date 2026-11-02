"""Pair new frozen N1024 results with unchanged, identity-checked external outputs."""
from study_common import *
import csv,collections,traceback

EXTERNAL=('Latent_JSCC','SwinJSCC','BPG_LDPC','ARPC')

def csvread(path):
 with Path(path).open(newline='',encoding='utf-8-sig')as f:return list(csv.DictReader(f))

def old_inputs():
 sets=read(F/'source_sets.json');ids=sets['common500'];assert len(ids)==len(set(ids))==500
 sources=read(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json');assert [s['source_id']for s in sources]==ids
 sm={s['source_id']:s for s in sources}
 identity=read(F/'e1/statistics_identity.json');assert identity['source_order']==ids
 pool=[v for v in read(FL/'E1_join/frames.json')if int(v['N'])==1024 and v['method']in EXTERNAL]
 assert len(pool)==24000
 groups=collections.defaultdict(list);statuses=collections.Counter()
 for v in pool:
  s=sm[v['source_id']];assert int(v['source_index'])==s['source_index']
  assert v['reference_sha256']==s['reference_sha256'] and v['source_pixels_sha256']==s['preprocessing_id']
  assert all(np.isfinite(float(v[m]))for m in METRICS)
  groups[v['method'],int(v['snr_db']),v['source_id']].append(v)
  statuses[v['method'],str(v['receiver_status']),str(v['native_status'])]+=1
 old_source=[r for r in csvread(F/'e1/source_means.csv')if int(r['N'])==1024 and r['method']in EXTERNAL]
 assert len(old_source)==8000
 for row in old_source:
  vv=groups[row['method'],int(row['snr_db']),row['source_id']]
  assert len(vv)==3 and len({int(v['channel_repeat'])for v in vv})==3
  assert int(row['channel_events'])==3
  for m in METRICS:assert abs(float(row[m])-float(np.mean([float(v[m])for v in vv])))<1e-12
 return ids,sources,pool,old_source,statuses

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']+r['inputs']]
 out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));beg=time.monotonic()
 ledger=Ledger(out/'calls.sqlite',dict(new_paired_bootstrap=80))
 try:
  # This task can only consume a closed formal science chain, never select policies.
  w=read(L/'controllers/calibration_chain/actual_wait.json');assert w['returncode']==0 and w['final_identity']is None
  for name in ['owner','child']:
   old=w[name];now=ident(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']or now['state']=='Z'
  assert read(L/'calibration_chain/completion.json')['status']=='COMPLETE_SCIENCE'
  ids,sources,pool,old_source,statuses=old_inputs();sm={s['source_id']:s for s in sources}
  fresh_sources=read(L/'cal_formal_plan/sources.json');assert [source_only(s)for s in fresh_sources]==[source_only(s)for s in sources]
  ident_new=read(D/'formal/statistics_identity.json');assert ident_new['source_order']==ids
  bank=np.load(checked(ident_new['bank']),allow_pickle=False);assert bank.shape==(10000,500)
  assert ident_new['bank']['sha256']==read(F/'source_sets.json')['bootstrap500']['sha256']
  new_source=csvread(D/'formal/source_means.csv');assert len(new_source)==6000
  combined=[];vectors={}
  for row in old_source+new_source:
   method=row['method'];snr=int(row['snr_db']);sid=row['source_id'];assert int(row['source_index'])==sm[sid]['source_index']
   assert int(row.get('repeats',row.get('channel_events')))==3
   combined.append(dict(method=method,N=1024,snr_db=snr,source_id=sid,source_index=sm[sid]['source_index'],repeats=3,reference_sha256=sm[sid]['reference_sha256'],preprocessing_id=sm[sid]['preprocessing_id'],**{m:float(row[m])for m in METRICS}))
  assert len(combined)==14000 and len({(v['method'],v['snr_db'],v['source_id'])for v in combined})==14000
  for method in (*EXTERNAL,'WHOLE','NeST','Ada'):
   for snr in [4,7,10,19]:
    a={v['source_id']:v for v in combined if v['method']==method and v['snr_db']==snr};assert set(a)==set(ids)
    for m in METRICS:vectors[method,snr,m]=np.array([a[sid][m]for sid in ids])
  summary=[]
  for row in csvread(F/'e1/summary.csv'):
   if int(row['N'])!=1024 or row['method']not in EXTERNAL:continue
   v=vectors[row['method'],int(row['snr_db']),row['metric']];assert abs(float(row['mean'])-v.mean())<1e-12
   summary.append(dict(method=row['method'],N=1024,snr_db=int(row['snr_db']),metric=row['metric'],sources=500,repeats=3,mean=float(row['mean']),ci_low=float(row['ci95_low']),ci_high=float(row['ci95_high']),origin='unchanged_external_native_outputs'))
  for row in csvread(D/'formal/summary.csv'):
   summary.append(dict(method=row['method'],N=1024,snr_db=int(row['snr_db']),metric=row['metric'],sources=500,repeats=3,mean=float(row['mean']),ci_low=float(row['ci_low']),ci_high=float(row['ci_high']),origin='new_unified_calibration'))
  assert len(summary)==140
  paired=csvread(D/'formal/paired_deltas.csv');assert len(paired)==60
  for ref in EXTERNAL:
   for snr in [4,7,10,19]:
    for metric in METRICS:
     assert time.monotonic()-beg<r['seconds']and not(L/'STOP').exists()
     def compute():
      delta=vectors['NeST',snr,metric]-vectors[ref,snr,metric];boot=delta[bank].mean(1)
      return dict(comparison='NeST-'+ref,N=1024,snr_db=snr,metric=metric,sources=500,repeats=3,mean=float(delta.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)))
     paired.append(ledger.call('new_paired_bootstrap',f'{ref}/{snr}/{metric}',compute))
  distributions=[]
  for ref in ('WHOLE','Ada',*EXTERNAL):
   for snr in [4,7,10,19]:
    for metric in METRICS:
     gain=(vectors['NeST',snr,metric]-vectors[ref,snr,metric])*(1 if metric in ['psnr_db','clip_image_cosine']else -1)
     distributions.append(dict(candidate='NeST',reference=ref,N=1024,snr_db=snr,metric=metric,sources=500,better=int((gain>1e-10).sum()),equal=int((abs(gain)<=1e-10).sum()),worse=int((gain<-1e-10).sum()),p05_gain=float(np.quantile(gain,.05)),median_gain=float(np.median(gain)),p95_gain=float(np.quantile(gain,.95))))
  joint=[]
  for snr in [4,7,10,19]:
   d=vectors['WHOLE',snr,'dists']-vectors['NeST',snr,'dists'];a=vectors['WHOLE',snr,'dreamsim']-vectors['NeST',snr,'dreamsim']
   joint.append(dict(snr_db=snr,sources=500,DISTS_better=int((d>1e-10).sum()),DreamSim_better=int((a>1e-10).sum()),both_better=int(((d>1e-10)&(a>1e-10)).sum())))
  dest=D/'external_join';dest.mkdir(exist_ok=False)
  for name,rows in [('source_means',combined),('summary',summary),('paired_deltas',paired),('per_source_improvement',distributions),('joint_perceptual_improvement',joint)]:csvout(dest/(name+'.csv'),rows)
  save(dest/'identity.json',dict(new_formal=pin(D/'formal/statistics_identity.json'),old_formal=pin(F/'e1/statistics_identity.json'),source_order=ids,bank=ident_new['bank'],source_references=[source_only(s)for s in sources],new_direct_paired_CIs=80,reused_formal_paired_CIs=60,reused_mean_CIs=140,old_method_noise_preserved=True,same_noise_across_all_methods=False,new_model_calls=0,new_metric_calls=0,new_PHY_calls=0,old_status_counts=[dict(method=m,status=s,native_status=n,count=c)for(m,s,n),c in sorted(statuses.items())]))
  save(out/'completion.json',dict(status='COMPLETE',ledger=ledger.summary(),seconds=time.monotonic()-beg,summary=pin(dest/'summary.csv'),paired=pin(dest/'paired_deltas.csv')))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise

if __name__=='__main__':
 if sys.argv[1:]==['--check-old-inputs']:
  ids,ss,pool,src,status=old_inputs();print(dict(sources=len(ids),native_events=len(pool),source_means=len(src),status_counts={str(k):v for k,v in status.items()}))
 else:main(*sys.argv[1:])
