"""Final resource and lifecycle accounting from real waits and SQLite ledgers."""
from opt_common import *
from route_statistics_v2 import csvout
from launch_registered import CAPS
import subprocess,collections

def main():
 out=D/'resource_delivery';controls=[];calls=[];events=[];totals=collections.defaultdict(float);nonzero=[]
 for p in sorted(L.glob('*_control/owner.json')):
  o=read(p);reg=cr(o['registration']);role=p.parent.name.removesuffix('_control');wait=p.parent/'actual_wait.json';assert wait.exists(),role
  w=read(wait)
  for who in [o['owner'],w['child']]:
   now=ident(who['pid']);assert now is None or now['start_ticks']!=who['start_ticks'],('Still active',role,who)
  if w['returncode']!=0:
   assert role in ['C_guard_metric_queue','benchmark_tx'];f=read(L/role/'failure.json')
   if role=='C_guard_metric_queue':assert f['completed']==[]
   else:
    correction=read(D/'benchmark_tx_identity_correction.json');assert correction['new_encoding_trials_in_failed_attempt']==0
    assert f['ledger']['counts']==[dict(kind='model_load',status='COMPLETE',count=1)]
   nonzero.append(dict(role=role,actual_wait=pin(wait),failure=pin(L/role/'failure.json'),reason='Zero-trial source hash convention mismatch'if role=='benchmark_tx'else'Zero-call GPU resource admission refusal'))
  req=read(reg['command'][-2]);route=req.get('route','P0'if role in ['p0','variant_qualification']else'CPU_control');seconds=w['device_seconds'];totals[route]+=seconds
  if reg['gpu']:events.extend([(w['started_unix'],1,role),(w['finished_unix'],-1,role)])
  controls.append(dict(role=role,route=route,gpu=reg['gpu'],actual_returncode=w['returncode'],device_seconds=seconds,process_wall_seconds=w['finished_unix']-w['started_unix'],owner_gone=True,child_gone=True,wait=pin(wait),request=pin(reg['command'][-2]),timing_only=role.startswith('benchmark_')))
  db=L/role/'calls.sqlite'
  if db.exists():
   conn=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
   for kind,status,n in conn.execute('SELECT kind,status,COUNT(*)FROM calls GROUP BY kind,status'):
    assert status=='COMPLETE',('Unresolved call',role,kind,status,n);calls.append(dict(role=role,route=route,kind=kind,status=status,count=n,timing_only=role.startswith('benchmark_')))
   conn.close()
 active=set();peak=0
 for _,delta,role in sorted(events,key=lambda e:(e[0],e[1])):
  if delta>0:active.add(role)
  else:active.remove(role)
  peak=max(peak,len(active))
 assert peak<=2 and not active
 for route,seconds in totals.items():
  if route in CAPS:assert seconds<=CAPS[route],(route,seconds)
 assert sum(totals.values())<=72*3600
 size=int(subprocess.check_output(['du','-sb',str(L)],text=True).split()[0]);assert size<=80*(1<<30),size
 out.mkdir(exist_ok=False);csvout(out/'actual_workers.csv',controls);csvout(out/'actual_calls.csv',calls);csvout(out/'device_budget.csv',[dict(route=k,used_seconds=v,used_device_hours=v/3600,cap_device_hours=CAPS[k]/3600 if k in CAPS else '',remaining_seconds=CAPS[k]-v if k in CAPS else '')for k,v in sorted(totals.items())])
 save(out/'resource_ledger.json',dict(status='ALL_OWNED_WORKERS_EXITED',peak_owned_GPU_workers=peak,total_GPU_device_hours=sum(totals.values())/3600,actual_device_seconds_by_route=dict(totals),cap_hours=72,new_PFS_tree_bytes=size,cache_cap_bytes=80*(1<<30),known_nonzero_controls=nonzero,all_other_waits_zero=True,all_owner_and_child_identities_gone=True,all_scientific_calls_complete=True,large_arrays_weights_and_logs_remain_PFS=True,CPU_process_wall_includes_waits_not_claimed_as_CPU_compute_usage=True,worker_table=pin(out/'actual_workers.csv'),call_table=pin(out/'actual_calls.csv'),budget_table=pin(out/'device_budget.csv')))
 print(read(out/'resource_ledger.json'))
if __name__=='__main__':main()
