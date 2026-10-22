"""Read-only closed scientific inputs; one new source-paired bootstrap bank."""
import sys,pathlib
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/progressive-rd-scaleup-20261022'))
from rdscale_common import *
from rdscale_policy import ARMS,SEEDS
import csv,collections
import numpy as np
METRICS=('mse','psnr_db','lpips_alex','ms_ssim','utility')
CONTRASTS=((ARMS[0],'current_dc'),(ARMS[1],'current_dc'),(ARMS[2],'current_dc'),(ARMS[1],ARMS[0]),(ARMS[2],ARMS[0]),(ARMS[2],ARMS[1]))
def csvread(p):
 with pathlib.Path(p).open(newline='')as f:return list(csv.DictReader(f))
def csvsave(p,rows):
 with pathlib.Path(p).open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def gone(ident):
 cur=process(ident['pid']);return cur is None or cur['start_ticks']!=ident['start_ticks']
def closed_wait(path):
 w=read(path);assert type(w['returncode'])is int and w['returncode']==0 and not any(w.get(k,False)for k in ['timed_out','killed','terminated']) and gone(w['process']),path
 return w
def audit_ledger(d):
 events=[json.loads(x)for x in checked(d).read_text().splitlines()]
 start=collections.Counter((x['kind'],x['key'])for x in events if x['status']=='STARTED');end=collections.Counter((x['kind'],x['key'])for x in events if x['status']=='COMPLETE')
 assert start==end and all(v==1 for v in start.values()) and all(x['status']in('STARTED','COMPLETE')for x in events)
 return collections.Counter(k[0]for k in end)
def interval(v,bank):
 v=np.asarray(v,dtype=np.float64);assert v.shape==(bank.shape[1],)and np.isfinite(v).all()
 lo,hi=np.quantile(v[bank].mean(axis=1),[.025,.975]);return dict(mean=float(v.mean()),ci_low=float(lo),ci_high=float(hi))
def main():
 assert sha(O/'registration.json')=='86119c506d6c82d43d75028ac5810cf12fea372488ff1d33b23bfe8c368ce4ac'
 reg=read(O/'registration.json');regsha=sha(O/'registration.json')
 for d in reg['sources']:checked(d)
 from summary_lineage import verify_routes
 assert sha(O/'unlaunched_training_recovery_preparation_v1.json')=='c2921c3af86a40e9abc3be774aee55209df89616f0af2fb60e354fc1b1aab3a8'
 routes,lineage=verify_routes(L,O,list(reg['jobs']),gone)
 out=O/'delivery_v1';stats=L/'statistics_v1';assert not out.exists()and not stats.exists()
 ids=[x['source_id']for x in reg['validation_sources']];sid_to_index={x['source_id']:x['source_index']for x in reg['validation_sources']};assert len(set(ids))==200
 vc=read(L/'validation_delivery_v1/closure.json');assert vc['status']=='PASS'and vc['registration_sha256']==regsha
 vindex=read(checked(vc['index']));assert [x['source_id']for x in vindex['sources']]==ids
 baseline=csvread(checked(vc['baseline']));assert len(baseline)==1600
 for i in range(2):
  closed_wait(L/'dispatch_v1'/('validation_%d'%i)/'actual_wait.json')
  assert not(L/'validation_cache_v1'/('shard_%02d'%i)/'failure.json').exists()
 for x in baseline:x.update(arm='current_dc',job='current_dc',seed='frozen',utility=float(x['mse'])+.1*float(x['lpips_alex']))
 records=baseline[:];closures=[];initial=[];orders={};counts_total=collections.Counter();actual_samples=0
 for name,job in reg['jobs'].items():
  d=routes[name];p=L/'training_v1'/name;w=closed_wait(d/'actual_wait.json')
  assert not any((x/'failure.json').exists()for x in [d,p])and not(p/'paused.json').exists()
  c=read(p/'completion.json');assert c['registration_sha256']==regsha and c['job']==name and c['seed']==job['seed'] and c['arm']==job['arm']
  assert c['training_updates']==10000 and c['training_samples']==40000 and c['validation_sources']==200 and c['validation_endpoint_rows']==1600
  for k in ['checkpoint','initial_checkpoint','validation','ledger','updates']:checked(c[k])
  cnt=audit_ledger(c['ledger']);assert cnt['decoder_update']==10000 and all(cnt[k]==1600 for k in ['calibration_decoder','calibration_lpips','calibration_ms_ssim'])
  anchor=job['arm']=='multiprefix_full_anchor';extra=10000 if anchor else 0
  assert cnt['anchor_teacher_forward']==cnt['anchor_full_student_forward']==extra
  assert c['teacher_training_forwards']==c['full_student_training_forwards']==extra and c['anchor_training_samples']==(40000 if anchor else 0)
  if anchor:assert c['teacher_sha256']==reg['model_states']['decoder']
  else:assert c['teacher_sha256']is None
  ident=read(p/'identity.json');assert ident['dc_initial_states']==reg['model_states']and ident['conditional']is False and ident['affine_parameters']==576
  initial.append(ident['initial_model_sha256']);updates=read(checked(c['updates']));assert len(updates)==10000
  pairs=[u['pair_indices']for u in updates];assert all(len(v)==4 for v in pairs)
  if job['seed']in orders:assert orders[job['seed']]==pairs
  else:orders[job['seed']]=pairs
  rows=csvread(checked(c['validation']));assert len(rows)==1600
  for x in rows:assert x['job']==name and x['arm']==job['arm']and int(x['seed'])==job['seed']and x['checkpoint_sha256']==c['checkpoint']['sha256']
  records.extend(rows);actual_samples+=c['training_samples'];counts_total.update(cnt)
  pairs_flat=np.asarray(pairs).reshape(-1)
  closures.append(dict(job=name,arm=job['arm'],seed=job['seed'],completion=pin(p/'completion.json'),wait=pin(d/'actual_wait.json'),checkpoint=c['checkpoint'],initial_checkpoint=c['initial_checkpoint'],identity=ident,counts=dict(cnt),unique_source_endpoint_pairs=int(len(np.unique(pairs_flat))),unique_training_sources=int(len(np.unique(pairs_flat//8))),training_seconds=c['training_seconds'],total_seconds=c['total_seconds'],peak_allocated_bytes=c['peak_allocated_bytes']))
 assert len(set(initial))==1 and len(records)==11200 and actual_samples==240000
 lookup={}
 for x in records:
  assert x['source_id']in sid_to_index and int(x['source_index'])==sid_to_index[x['source_id']]and x['split']=='validation'
  key=(x['job'],x['source_id'],int(x['m']),int(x['K']));assert key not in lookup;lookup[key]=x
  assert all(np.isfinite(float(x[k]))for k in METRICS)
 for m,k in reg['endpoints']:
  for sid in ids:
   base=lookup['current_dc',sid,m,k]
   for name in reg['jobs']:assert int(lookup[name,sid,m,k]['terminated_bits'])==int(base['terminated_bits'])
 stats.mkdir();out.mkdir()
 bank=np.random.default_rng(reg['bootstrap']['seed']).integers(0,200,(10000,200),dtype=np.int64)
 with(stats/'validation_bootstrap_indices.npy').open('xb')as f:np.save(f,bank,allow_pickle=False)
 summaries=[];paired=[];endpoint_rates=[]
 for m,k in reg['endpoints']:
  lengths=np.array([float(lookup['current_dc',sid,m,k]['terminated_bits'])for sid in ids]);endpoint_rates.append(dict(m=m,K=k,mean_terminated_bits=float(lengths.mean()),mean_bits_per_pixel=float(lengths.mean()/65536),sources=200))
  for metric in METRICS:
   values={name:np.array([float(lookup[name,sid,m,k][metric])for sid in ids])for name in ['current_dc',*reg['jobs']]}
   for name,v in values.items():
    job=reg['jobs'].get(name,dict(arm='current_dc',seed='frozen'))
    summaries.append(dict(aggregation='single_seed'if name!='current_dc'else'frozen_baseline',job=name,arm=job['arm'],seed=job['seed'],m=m,K=k,metric=metric,**interval(v,bank),sources=200,unit='source',subset='historical_calibration_new_validation200'))
   averaged={arm:np.mean([values[next(n for n,j in reg['jobs'].items()if j['arm']==arm and j['seed']==seed)]for seed in SEEDS],axis=0)for arm in ARMS};averaged['current_dc']=values['current_dc']
   for arm in ARMS:
    summaries.append(dict(aggregation='mean_across_two_seeds',job=arm,arm=arm,seed='mean_of_2',m=m,K=k,metric=metric,**interval(averaged[arm],bank),sources=200,unit='source',subset='historical_calibration_new_validation200'))
   for seed in [*SEEDS,'mean_of_2']:
    byarm=averaged if seed=='mean_of_2'else dict(current_dc=values['current_dc'],**{j['arm']:values[n]for n,j in reg['jobs'].items()if j['seed']==seed})
    for a,b in CONTRASTS:
     delta=byarm[a]-byarm[b]
     paired.append(dict(comparison=a+'_minus_'+b,seed=seed,m=m,K=k,metric=metric,**interval(delta,bank),sources=200,unit='source-paired',subset='historical_calibration_new_validation200',training_seed_uncertainty_estimated=False))
 assert len(summaries)==400 and len(paired)==720
 keys=['source_id','source_index','split','job','arm','seed','m','K','terminated_bits']+list(METRICS)
 csvsave(out/'source_metrics.csv',[{k:x[k]for k in keys}for x in records]);csvsave(out/'summary.csv',summaries);csvsave(out/'paired_differences.csv',paired);csvsave(out/'endpoint_rates.csv',endpoint_rates)
 save(out/'closure_audit.json',dict(status='PASS',registration=pin(O/'registration.json'),execution_lineage=lineage,original_metadata_admission_failure=reg['resource_admission_revision'],validation_closure=pin(L/'validation_delivery_v1/closure.json'),jobs=closures,shared_initial_state=initial[0],identical_pair_order_within_seed=True,main_training_samples=actual_samples,extra_full_anchor_samples=80000,training_counts=dict(counts_total),new_validation_cache_rows=1600,new_trained_validation_rows=9600,baseline_rows_reused=1600,bootstrap=pin(stats/'validation_bootstrap_indices.npy'),bootstrap_seed=reg['bootstrap']['seed'],source_order=ids,scientific_subset='Historical calibration population, held out from this decoder training and old pilot100. No blind/common500/wireless claim.',source_CI_does_not_estimate_training_seed_distribution=True,encoder_updates=0,VAR_updates=0,tokenizer_calls=0,PHY_calls=0))
 save(out/'manifest.json',dict(files=[pin(p)for p in sorted(out.iterdir())if p.is_file()]))
 print(json.dumps(dict(closure=pin(out/'closure_audit.json'),manifest=pin(out/'manifest.json'),summary_rows=len(summaries),paired_rows=len(paired))))
if __name__=='__main__':main()
