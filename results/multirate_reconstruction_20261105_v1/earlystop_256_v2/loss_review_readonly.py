"""Read existing rank logs only; never load a model or change a running role."""
from pathlib import Path
from collections import defaultdict,Counter
import json,math,statistics,time,os,sys,hashlib
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
L=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/multirate_reconstruction_20261105_v1')
os.sched_setaffinity(0,{20,22})
sys.path.insert(0,str(R/'experiments/multirate-reconstruction-20261105'))
from collect_snapshot import collect
snapshot=collect();active=snapshot['active_training_role'];progress=snapshot['training']['progress.json']
cutoff=(progress['step']//100)*100
windows=[('40k',40001,41000),('50k',50001,51000),('60k',60001,61000),('70k',70001,71000),('late_5000_to_4001',cutoff-4999,cutoff-4000),('late_4000_to_3001',cutoff-3999,cutoff-3000),('late_3000_to_2001',cutoff-2999,cutoff-2000),('late_2000_to_1001',cutoff-1999,cutoff-1000),('latest_1000',cutoff-999,cutoff)]
data={};health={};files=[];truncated=[]
def group():return {'n':0,'records':0,'sums':defaultdict(float),'values':defaultdict(list),'endpoints':Counter(),'by_mode':defaultdict(Counter),'steps':set(),'grad':[],'lr':[],'rank_steps':Counter()}
for label,lo,hi in windows:
 data[label]={'range':[lo,hi],'full':group(),'budgeted':group()}
for folder in [L/active]:
 for path in sorted(folder.glob('updates_rank*.jsonl')):
  rank=int(path.stem.split('rank')[1]);n=0;bad=0;first=last=None;size=path.stat().st_size
  with path.open() as f:
   while f.tell()<size:
    line=f.readline()
    if not line:break
    try:row=json.loads(line)
    except json.JSONDecodeError:truncated.append(str(path));break
    step=row['step'];first=first or step;last=step;n+=1
    numeric=[row['grad_norm'],row['lr'],row['seconds']]
    for rec in row['records']:
     numeric.extend(rec[k] for k in ['mse','lpips_vgg','soft_bpp','vq'])
    if not all(math.isfinite(v) for v in numeric):bad+=1
    for label,lo,hi in windows:
     if not lo<=step<=hi:continue
     recs=row['records'];full=recs[0]['full'];g=data[label]['full' if full else 'budgeted']
     examples=iter(row['examples'])
     for rec in recs:
      ep=rec['endpoints'];count=len(ep);g['n']+=count;g['records']+=1
      for k in ['mse','lpips_vgg','soft_bpp','vq']:g['sums'][k]+=float(rec[k])*count;g['values'][k].append(float(rec[k]))
      g['endpoints'].update(ep)
      for e in ep:g['by_mode'][str(next(examples)['mode'])].update([str(e)])
     g['steps'].add(step);g['rank_steps'][rank]+=1
     if rank==0:g['grad'].append(row['grad_norm']);g['lr'].append(row['lr'])
  health[str(path.relative_to(L))]={'rows':n,'first_step':first,'last_step':last,'nonfinite_logged_rows':bad}
  files.append({'path':str(path),'bytes_at_start':size})
def quantile(vals,p):
 vals=sorted(vals)
 return vals[min(len(vals)-1,int(p*(len(vals)-1)))] if vals else None
summaries={}
for label,a in data.items():
 out={'range':a['range']}
 for arm in ['full','budgeted']:
  g=a[arm];n=g['n'];count=sum(g['endpoints'].values());grad=g['grad']
  out[arm]={'source_exposures':n,'updates':len(g['steps']),'rank_steps':dict(g['rank_steps']),'means':{k:v/n for k,v in g['sums'].items()} if n else {},
   'min_max_logged_microbatch':{k:[min(v),max(v)] for k,v in g['values'].items()},
   'endpoint_histogram':dict(sorted(g['endpoints'].items())),'endpoint_by_mode':dict(g['by_mode']),
   'null_fraction':g['endpoints'][0]/count if count else None,
   'grad_norm_pre_clip':{'median':statistics.median(grad),'p95':quantile(grad,.95),'max':max(grad),'fraction_above_1':sum(v>1 for v in grad)/len(grad)} if grad else None,
   'lr_range':[min(g['lr']),max(g['lr'])] if g['lr'] else None}
 summaries[label]=out
stderr=L/'controllers'/active/'stderr.log'
err=stderr.read_text() if stderr.exists() else ''
bound_mismatches=[p for p,h in snapshot['source_bindings'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
rootdirs=[p.name for p in L.iterdir() if p.is_dir()]
result={'time':time.time(),'head':snapshot['head'],'active_role':active,'progress':progress,'checkpoint':snapshot['training']['latest_checkpoint.json'],
 'active':snapshot['ledger'][active],'old_four_paused':snapshot['old_P2_stop'] and snapshot['old_pause']=='PAUSED_BY_USER_INCOMPLETE',
 'bound_source_mismatches':bound_mismatches,'all_log_health':health,'trailing_partial_records':truncated,'windows':summaries,
 'stderr':{'bytes':len(err.encode()),'traceback':err.count('Traceback'),'nonfinite_mentions':err.lower().count('nonfinite'),'warning_lines':sorted(set(v for v in err.splitlines() if 'Warning:' in v))},
 'task_directories':rootdirs,'new_quality_artifacts':[str(p.relative_to(R)) for p in (R/'results/multirate_reconstruction_20261105_v1').rglob('*') if p.is_file() and any(s in p.name for s in ['summary','scores','metrics','paired'])],
 'limitations':['Loss statistics are training-batch averages, not held-out codec quality.','Total scalar loss and excess-budget term are not logged separately.','soft_bpp is a differentiable proxy, not an actual terminated packet length.','Earlier windows use different data snapshots and source batches.'],
 'new_model_calls':0,'new_training_updates':0,'log_inputs':files}
print(json.dumps(result))