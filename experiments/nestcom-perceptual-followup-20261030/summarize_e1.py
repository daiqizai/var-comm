"""All five metrics; average original noise outcomes within each source first."""
from followup_common import *
from collections import defaultdict,Counter
import traceback
def interval(a,bank):
 boots=np.empty(len(bank),np.float64)
 for start in range(0,len(bank),250):boots[start:start+250]=a[bank[start:start+250]].mean(1)
 lo,hi=np.quantile(boots,[.025,.975]);return float(a.mean()),float(lo),float(hi)
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];os.sched_setaffinity(0,set(r['cpu_affinity']))
 out=Path(r['output']);out.mkdir(exist_ok=False);beg=time.monotonic()
 try:
  frames=cr(r['frames']);sources=cr(r['sources']);ids=[s['source_id']for s in sources];proto=cr(r['bank_identity']);assert proto['statistics']['source_order']==ids
  bank=np.load(checked(proto['statistics']['bootstrap']),allow_pickle=False);assert bank.shape==(10000,500)and len(set(ids))==500
  groups=defaultdict(list);events=set();cells=defaultdict(list)
  for v in frames:
   key=(v['method'],v['N'],v['snr_db'],v['source_id']);ev=key+(v['channel_repeat'],);assert ev not in events;events.add(ev);groups[key].append(v);cells[key[:3]].append(v)
  assert len(frames)==66000 and len(cells)==44 and set(map(len,cells.values()))=={1500}
  source=[];bycell=defaultdict(dict)
  for k,g in groups.items():
   assert len(g)==3 and {v['channel_repeat']for v in g}=={0,1,2}
   v=dict(method=k[0],N=k[1],snr_db=k[2],source_id=k[3],source_index=g[0]['source_index'],channel_events=3,**{m:float(np.mean([x[m]for x in g]))for m in METRICS})
   source.append(v);bycell[k[:3]][k[3]]=v
  arrays={};summary=[];paired=[]
  for k,g in sorted(bycell.items()):
   assert set(g)==set(ids)
   for m in METRICS:
    a=np.array([g[s][m]for s in ids]);arrays[(*k,m)]=a;mean,lo,hi=interval(a,bank)
    summary.append(dict(method=k[0],N=k[1],snr_db=k[2],metric=m,n_sources=500,logical=1500,mean=mean,ci95_low=lo,ci95_high=hi))
  comparisons=[('final_NeST',ref)for ref in ['original_NeST','WHOLE_final_receiver','Ada_frozen','ARPC','SwinJSCC','Latent_JSCC','BPG_LDPC']]+[('WHOLE_final_receiver','Ada_frozen')]
  for cand,ref in comparisons:
   for method,N,snr in sorted(bycell):
    if method!=cand or(ref,N,snr)not in bycell:continue
    for m in METRICS:
     mean,lo,hi=interval(arrays[cand,N,snr,m]-arrays[ref,N,snr,m],bank)
     paired.append(dict(candidate=cand,reference=ref,N=N,snr_db=snr,metric=m,n_sources=500,mean_delta=mean,ci95_low=lo,ci95_high=hi))
  assert len(summary)==220 and len(paired)==220
  failures=[dict(method=k[0],N=k[1],snr_db=k[2],logical=len(g),fixed_gray=sum(v['image']is None or v.get('fixed_gray')is True for v in g),receive_status_counts=json.dumps(dict(Counter(v.get('receiver_status')for v in g)),sort_keys=True))for k,g in sorted(cells.items())]
  delivery=D/'e1';delivery.mkdir(exist_ok=False)
  csvout(delivery/'frame_metrics.csv',frames);csvout(delivery/'source_means.csv',source);csvout(delivery/'summary.csv',summary);csvout(delivery/'paired_deltas.csv',paired);csvout(delivery/'failure_counts.csv',failures)
  save(delivery/'statistics_identity.json',dict(source_order=ids,bank=proto['statistics']['bootstrap'],bank_identity=r['bank_identity'],replicates=10000,interval='pointwise95 percentile; no simultaneous multiplicity claim',source_first=True,direct_source_paired=True,original_method_native_noise_preserved=True,previously_examined_regression_not_blind=True,failed_outputs_in_denominator=True,metrics=list(METRICS),original_four_scalars_retained=True,DISTS_new_native_floats=True,logical=66000,source_rows=len(source),mean_CIs=220,paired_CIs=220))
  save(out/'completion.json',dict(status='COMPLETE',delivery=str(delivery),logical=66000,mean_CIs=220,paired_CIs=220,seconds=time.monotonic()-beg,new_model_calls=0,new_PHY=0));print('E1_FIVE_METRIC_DELIVERED',len(source),len(summary),len(paired))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
