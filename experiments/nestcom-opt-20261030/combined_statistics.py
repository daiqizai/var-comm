"""Combine closed metric segments without re-scoring; source-paired FIT/CONFIRM."""
from opt_common import *
from route_statistics_v2 import METRICS,CORE,bank,interval,aggregate,csvout,fit_gate
import collections

def main(request):
 r=read(request);dataset=r['dataset'];stage=r['stage'];assert stage in ('FIT','CONFIRM')
 ids,b,bp=bank(dataset);full=[];evidence=[];seen={}
 for segment in r['segments']:
  role=segment['role'];score_sets=[]
  for mode in ('three','clip'):
   p=L/(role+'_'+mode);w=read(p.with_name(p.name+'_control')/'actual_wait.json')
   assert w['returncode']==0 and ident(w['child']['pid'])is None
   c=read(p/'completion.json');assert c['status']=='COMPLETE'and c['ledger']['unresolved']==0
   score_sets.append(cr(c['scores']));evidence.append(pin(p/'completion.json'))
  rows=[v for v in read(L/(role+'_metric_plan/logical.json'))if ('variants'not in segment or v['variant']in segment['variants'])and('methods'not in segment or v['method']in segment['methods'])]
  frames,_=aggregate(rows,score_sets,ids)
  for v in frames:
   key=tuple(v.get(k)for k in ('method','variant','N','snr_db','source_id','noise_seed','generation_seed'))
   if key in seen:
    old=seen[key];assert all(old[k]==v[k]for k in METRICS)and old['pair_key']==v['pair_key']
   else:seen[key]=v;full.append(v)
 # Reuse aggregate's generation -> channel -> source order on existing scalars.
 scalars={v['pair_key']:{'source_id':v['source_id'],**{m:v[m]for m in METRICS}}for v in full}
 full,source=aggregate(full,[scalars],ids);expected=1 if stage=='FIT'else 3
 assert all(v['channel_events']==expected for v in source)
 groups=collections.defaultdict(dict)
 for v in source:
  k=(v['method'],v['variant'],v['N'],v['snr_db']);assert v['source_id']not in groups[k];groups[k][v['source_id']]=v
 arrays={};means={};summary=[];paired=[];normal=[]
 for k,g in groups.items():
  assert set(g)==set(ids),k;means[k]={}
  for m in METRICS:
   x=np.asarray([g[s][m]for s in ids]);arrays[k+(m,)]=x;mean,lo,hi=interval(x,b);means[k][m]=mean
   summary.append(dict(method=k[0],variant=k[1],N=k[2],snr_db=k[3],metric=m,n_sources=len(ids),mean=mean,ci95_low=lo,ci95_high=hi))
 fitpin=pin(D/'fit_baseline_denominators.json');fixed={(x['N'],x['snr_db']):{m:x[m]for m in METRICS}for x in cr(fitpin)['values']}
 gates={}
 for variant in sorted({k[1]for k in groups}-{'original'}):
  for method in sorted({k[0]for k in groups if k[1]==variant}):
   cells={k[2:]:v for k,v in means.items()if k[:2]==(method,variant)}
   baseline={c:means[(method,'original',*c)]for c in cells}
   for c in cells:
    for m in METRICS:
     x=arrays[(method,variant,*c,m)]-arrays[(method,'original',*c,m)];mean,lo,hi=interval(x,b)
     paired.append(dict(method=method,candidate=variant,reference='original',N=c[0],snr_db=c[1],metric=m,n_sources=len(ids),mean_delta=mean,ci95_low=lo,ci95_high=hi))
   if not all(c in cells for c in CORE):continue
   ratios={m:float(np.mean([cells[c][m]/baseline[c][m]for c in CORE]))for m in ('lpips_alex','dreamsim')}
   normal.append(dict(method=method,variant=variant,J_own_baseline=float(np.mean(list(ratios.values()))),**ratios,selection_population='NEST only'if method=='NEST'else'WHOLE separately; not used to select NeST'))
   if method!='NEST':continue
   den=fixed if stage=='FIT'else baseline
   if stage=='FIT':assert all(baseline[c]==fixed[c]for c in cells),'FIT baseline scalar identity changed'
   gate=fit_gate(cells,den);gate.update(guard_cells_complete=len(cells)==7,confirmation='NOT_RUN'if stage=='FIT'else'COMPLETE')
   cis={}
   for m in ('lpips_alex','dreamsim'):
    delta=np.mean([(arrays[(method,variant,*c,m)]-arrays[(method,'original',*c,m)])/fixed[c][m]for c in CORE],axis=0)
    mean,lo,hi=interval(delta,b);cis[m]=dict(mean=mean,ci95_low=lo,ci95_high=hi)
    paired.append(dict(method=method,candidate=variant,reference='original',N=1024,snr_db='core_mean',metric='FIT_normalized_'+m,n_sources=len(ids),mean_delta=mean,ci95_low=lo,ci95_high=hi))
   gate['core_paired_CI_fixed_FIT_denominator']=cis
   if stage=='CONFIRM':
    gate['confirm_pass']=len(cells)==7 and not gate['guard_violations']and min(v['ci95_high']for v in cis.values())<0 and max(v['ci95_high']for v in cis.values())<=.01
    gate['FIT_J_not_reselected_on_confirmation']=True
   else:gate['FIT_ready_for_confirmation']=gate['fit_pass']and len(cells)==7
   gates[variant]=gate
 out=D/r['output'];out.mkdir(exist_ok=False)
 csvout(out/'frame_metrics.csv',full);csvout(out/'per_source_metrics.csv',source);csvout(out/'summary.csv',summary);csvout(out/'paired_deltas.csv',paired);csvout(out/'own_baseline_normalized.csv',normal)
 save(out/'gate.json',dict(stage=stage,dataset=dataset,gates=gates,bank=bp,baseline_denominators=fitpin,metric_completions=evidence,source_order=ids,registration=pin(request),source_equal_weight=True,gray_in_denominator=True,interpretation='Development screening/confirmation, not a new blind holdout'))
 print(json.dumps(gates,indent=2));return gates

if __name__=='__main__':main(*sys.argv[1:])
