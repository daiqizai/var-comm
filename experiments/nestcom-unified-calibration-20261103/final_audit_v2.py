"""Read-only scientific closure audit; write a new, separate delivery receipt."""
from study_common import *
import csv,sqlite3,collections,traceback


def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']+r['inputs']]
 out=Path(r['output']);out.mkdir(exist_ok=False);beg=time.monotonic();os.sched_setaffinity(0,set(r['cpu_affinity']))
 try:
  workers=[];controllers=[];totals={True:0.,False:0.};calls=collections.Counter();failure_roles=[]
  for c in sorted(L.glob('*_control')):
   role=c.name[:-8]
   if role=='final_audit_v2':continue
   o=read(c/'owner.json');reg=cr(o['registration']);w=read(c/'actual_wait.json')
   assert w['final_identity']is None and not w['timed_out'],role
   live={}
   for key in ['owner','child']:
    v=w[key];now=ident(v['pid']);assert now is None or now['start_ticks']!=v['start_ticks']or now['state']=='Z',(role,key,now)
    live[key]=now
   for v in reg['code']:checked(v)
   totals[bool(reg['gpu'])]+=w['elapsed_seconds']
   completion=read(L/role/'completion.json')if(L/role/'completion.json').exists()else None
   ledger=[]
   for path in sorted((L/role).glob('*.sqlite'))if(L/role).exists()else[]:
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    names={v[0]for v in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'calls'not in names:db.close();continue
    rows=[dict(kind=k,status=s,count=n)for k,s,n in db.execute('SELECT kind,status,COUNT(*)FROM calls GROUP BY kind,status')];db.close()
    assert all(v['status']=='COMPLETE'for v in rows),(role,rows)
    for v in rows:calls[v['kind']]+=v['count']
    ledger.append(dict(file=pin(path),counts=rows))
   if role=='compiled_quality_restore':
    assert w['returncode']==1 and completion is None and not ledger
    log=(c/'worker.log').read_text();assert "ModuleNotFoundError: No module named 'n512_phy'"in log
    failure_roles.append(role)
   elif role=='final_audit':
    assert w['returncode']==1 and completion is None and not ledger
    failure=read(L/role/'failure.json')
    assert "('low_budget/summary.csv', 32)" in failure['traceback']
    failure_roles.append(role)
   else:
    assert w['returncode']==0,role;closed(role)
   workers.append(dict(role=role,registration=o['registration'],actual_wait=pin(c/'actual_wait.json'),returncode=w['returncode'],elapsed_seconds=w['elapsed_seconds'],gpu=reg['gpu'],processes_gone=True,ledger=ledger,completion=pin(L/role/'completion.json')if completion else None))
  assert failure_roles==['compiled_quality_restore','final_audit']
  for c in sorted((L/'controllers').iterdir()):
   if not c.is_dir():continue
   w=read(c/'actual_wait.json');assert w['returncode']==0 and w['final_identity']is None and not w['timed_out']
   for key in ['owner','child']:
    v=w[key];now=ident(v['pid']);assert now is None or now['start_ticks']!=v['start_ticks']or now['state']=='Z'
   controllers.append(dict(role=c.name,actual_wait=pin(c/'actual_wait.json'),elapsed_seconds=w['elapsed_seconds'],processes_gone=True,accounting='Waiting/dispatch controller; separate from scientific process budget.'))
  protocol=read(D/'protocol.json');assert totals[True]<protocol['budget']['GPU_device_seconds']and totals[False]<protocol['budget']['CPU_process_seconds']
  assert time.time()<protocol['deadline_unix']
  def count(p):
   with p.open(newline='')as f:return sum(1 for _ in csv.DictReader(f))
  expected={'formal/frame_metrics.csv':18000,'formal/source_means.csv':6000,'formal/summary.csv':60,'formal/paired_deltas.csv':60,'external_join/summary.csv':140,'external_join/paired_deltas.csv':140,'low_budget/summary.csv':32,'low_budget/paired_deltas.csv':15,'compiled32/paired_deltas.csv':15}
  files={}
  for rel,n in expected.items():assert count(D/rel)==n,(rel,count(D/rel));files[rel]=dict(**pin(D/rel),rows=n)
  with (D/'low_budget/summary.csv').open(newline='')as f: low=list(csv.DictReader(f))
  assert len([v for v in low if v['metric']in METRICS])==20
  assert len([v for v in low if v['metric']not in METRICS])==12
  assert {v['metric']for v in low if v['metric']not in METRICS}=={'utilization','source_bits','actual_length_comparisons'}
  report=dict(status='COMPLETE_SCIENTIFIC_AUDIT',protocol=pin(D/'protocol.json'),frozen_policy=pin(D/'frozen_policies.json'),workers=workers,controllers=controllers,known_preserved_failed_role=failure_roles,successful_ledger_calls=dict(calls),scientific_worker_GPU_hours=totals[True]/3600,scientific_worker_CPU_hours=totals[False]/3600,resource_limits=protocol['budget'],deadline_unix=protocol['deadline_unix'],audit_process_accounted_in_own_actual_wait=True,logical_denominator_checks=files,automatic_retry=False,new_model_calls=0,new_PHY=0,manuscript_publication='PENDING')
  save(out/'scientific_audit.json',report);save(D/'scientific_closure_audit.json',report)
  save(out/'completion.json',dict(status='COMPLETE',seconds=time.monotonic()-beg,report=pin(D/'scientific_closure_audit.json')))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])
