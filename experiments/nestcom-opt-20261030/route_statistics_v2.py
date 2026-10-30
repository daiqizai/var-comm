"""Source-level paired descriptive intervals and the predeclared FIT gate."""
from opt_common import *
import csv,collections,re
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
CORE=((1024,7),(1024,10))

def csvout(path,rows):
 rows=list(rows);assert rows
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with Path(path).open('x',newline='')as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def bank(dataset='fit100'):
 sources=read(L/'inputs'/(dataset+'.json'));ids=[s['source_id']for s in sources]
 p=L/'statistics'/(dataset+'_bank.npz');meta=p.with_suffix('.json')
 if meta.exists():
  v=read(meta);assert v['source_order']==ids and v['seed']==2026103017
  with np.load(checked(v['bank']),allow_pickle=False)as z:b=z['indices'].copy()
 else:
  # Seed and algorithm fixed in protocol_defaults before any candidate scores.
  p.parent.mkdir(exist_ok=True);b=np.random.default_rng(2026103017).integers(0,len(ids),(10000,len(ids)),dtype=np.int64)
  with p.open('xb')as f:np.savez_compressed(f,indices=b)
  save(meta,dict(bank=pin(p),source_order=ids,source_pixels=[s['preprocessing_id']for s in sources],seed=2026103017,resamples=10000,unit='source',algorithm='numpy.default_rng.PCG64.integers.int64',seed_preregistered_in=pin(D/'protocol_defaults.json')))
 assert b.shape==(10000,len(ids))and b.min()>=0 and b.max()<len(ids)
 return ids,b,pin(meta)

def interval(x,b):
 x=np.asarray(x,dtype=np.float64);assert x.ndim==1 and np.isfinite(x).all()
 v=x[b].mean(1);lo,hi=np.quantile(v,[.025,.975]);return float(x.mean()),float(lo),float(hi)

def group_variant(row):
 name=row['variant']
 return re.sub(r'_s\d+$','',name) if name.startswith('B_T')else name

def aggregate(rows,score_sets,ids):
 events=collections.defaultdict(list);full=[]
 for row in rows:
  metrics={}
  for scores in score_sets:
   v=scores[row['pair_key']];assert v['source_id']==row['source_id']
   metrics.update({k:float(v[k])for k in METRICS if k in v})
  assert set(metrics)==set(METRICS)and np.isfinite(list(metrics.values())).all()
  v=dict(row,**metrics);full.append(v)
  key=(row['method'],group_variant(row),row['N'],row['snr_db'],row['source_id'],row['noise_seed'])
  events[key].append(v)
 sources=collections.defaultdict(list)
 for key,group in events.items():
  if key[1].startswith('B_T'):assert len(group)==3 and len({x['variant']for x in group})==3
  else:assert len(group)==1,(key,len(group))
  sources[key[:-1]].append({m:np.mean([x[m]for x in group])for m in METRICS})
 result=[]
 for key,group in sources.items():
  method,variant,N,snr,sid=key;assert sid in ids
  result.append(dict(method=method,variant=variant,N=N,snr_db=snr,source_id=sid,channel_events=len(group),generation_events=3 if variant.startswith('B_T')else 1,**{m:float(np.mean([x[m]for x in group]))for m in METRICS}))
 return full,result

def fit_gate(candidate,baseline):
 core={m:np.mean([candidate[c][m]/baseline[c][m]for c in CORE])for m in ('lpips_alex','dreamsim')}
 score=float(np.mean(list(core.values())))
 guards=[]
 for cell,v in candidate.items():
  old=baseline[cell]
  for m in ('lpips_alex','dreamsim'):
   d=v[m]/old[m]-1
   if d>.01+1e-12:guards.append(dict(cell=cell,metric=m,relative_degradation=float(d)))
  if v['psnr_db']-old['psnr_db']<-.1-1e-12:guards.append(dict(cell=cell,metric='psnr_db',delta_db=v['psnr_db']-old['psnr_db']))
 passed=score<=.995 and max(core.values())<=1+1e-12 and not guards
 return dict(score=score,core_normalized_means={k:float(v)for k,v in core.items()},guard_violations=guards,fit_pass=passed,confirmation='NOT_RUN')

def main(role):
 ids,b,bp=bank();out=D/(role+'_delivery');out.mkdir(exist_ok=False)
 paths=[L/(role+'_three'),L/(role+'_clip')];score_sets=[];evidence=[]
 for p in paths:
  w=read(p.with_name(p.name+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
  c=read(p/'completion.json');assert c['status']=='COMPLETE'and c['ledger']['unresolved']==0
  score_sets.append(cr(c['scores']));evidence.append(pin(p/'completion.json'))
 full,source=aggregate(read(L/(role+'_metric_plan/logical.json')),score_sets,ids)
 groups=collections.defaultdict(dict)
 for row in source:
  key=(row['method'],row['variant'],row['N'],row['snr_db']);assert row['source_id']not in groups[key];groups[key][row['source_id']]=row
 arrays={};means={};summary=[];paired=[]
 for key,v in groups.items():
  assert set(v)==set(ids),key
  means[key]={}
  for m in METRICS:
   x=np.array([v[s][m]for s in ids]);arrays[key+(m,)]=x
   mean,lo,hi=interval(x,b);means[key][m]=mean
   summary.append(dict(method=key[0],variant=key[1],N=key[2],snr_db=key[3],metric=m,n_sources=len(ids),mean=mean,ci95_low=lo,ci95_high=hi))
 for key in groups:
  if key[1]=='original':continue
  old=(key[0],'original',*key[2:])
  for m in METRICS:
   mean,lo,hi=interval(arrays[key+(m,)]-arrays[old+(m,)],b)
   paired.append(dict(method=key[0],candidate=key[1],reference='original',N=key[2],snr_db=key[3],metric=m,n_sources=len(ids),mean_delta=mean,ci95_low=lo,ci95_high=hi))
 baseline={k[2:]:v for k,v in means.items()if k[:2]==('NEST','original')}
 bpfile=D/'fit_baseline_denominators.json';bv=dict(source_bank=bp,values=[dict(N=k[0],snr_db=k[1],**v)for k,v in sorted(baseline.items())])
 if bpfile.exists():
  fixed=read(bpfile);assert fixed['source_bank']==bp
  fv={(x['N'],x['snr_db']):{m:x[m]for m in METRICS}for x in fixed['values']}
  assert all(fv[c]==v for c,v in baseline.items()),'Baseline requires exact metric-pair reuse, not cross-device silent substitution'
  baseline=fv
 else:save(bpfile,bv)
 gates={}
 for variant in sorted({k[1]for k in groups}-{'original'}):
  v={k[2:]:x for k,x in means.items()if k[:2]==('NEST',variant)}
  gates[variant]=fit_gate(v,baseline)
  gates[variant]['guard_cells_complete']=len(v)==7
  gates[variant]['FIT_ready_for_confirmation']=gates[variant]['fit_pass']and len(v)==7
  for m in ('lpips_alex','dreamsim'):
   diffs=np.mean([(arrays[('NEST',variant,*c,m)]-arrays[('NEST','original',*c,m)])/baseline[c][m]for c in CORE],axis=0)
   mean,lo,hi=interval(diffs,b)
   paired.append(dict(method='NEST',candidate=variant,reference='original',N=1024,snr_db='core_mean',metric='normalized_'+m,n_sources=len(ids),mean_delta=mean,ci95_low=lo,ci95_high=hi))
 if role=='fit_A':
  gates['A1']['verification_seed_direction_replicated']=all(gates['A2']['core_normalized_means'][m]<=1 for m in ('lpips_alex','dreamsim'))
  gates['A2']['selection_eligible']=False
 csvout(out/'frame_metrics.csv',full);csvout(out/'per_source_metrics.csv',source);csvout(out/'summary.csv',summary);csvout(out/'paired_deltas.csv',paired)
 save(out/'gate.json',dict(stage='FIT100',gates=gates,baseline_denominators=pin(bpfile),bank=bp,metric_completion=evidence,source_order=ids,interpretation='Development screening; intervals descriptive, not independent final validation'))
 print(json.dumps(gates,indent=2))

if __name__=='__main__':main(*sys.argv[1:])
