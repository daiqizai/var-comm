"""Explicit zero-science admission failure lineage; unknown failures remain fatal."""
import pathlib,json,hashlib
NAMES=('multiprefix_full_anchor_s2026102201','multiprefix_unconditional_s2026102202')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def pin(p):p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(d):
 p=pathlib.Path(d['path']);assert p.is_file()and not p.is_symlink()and p.stat().st_size==d['bytes']and sha(p)==d['sha256'];return p

def verify_routes(L,O,jobs,gone):
 auth=read(O/'unlaunched_training_recovery_preparation_v1.json');assert auth['jobs']==list(NAMES)and auth['prior_scientific_attempts']==0 and auth['extra_scientific_budget']==0
 for d in auth['sources']:checked(d)
 for n in NAMES:
  d=L/'dispatch_v1'/n;assert pin(d/'failure.json')==auth['failures'][n]
  assert not(d/'launch.json').exists()and not(d/'actual_wait.json').exists()
  f=read(d/'failure.json');assert "'97'"in f['error']and 'admitted(int(fields[1]),int(fields[2]))'in f['traceback']
 for path in [L/'initial_control_v1',L/'outer_dispatch_v1',L/'recovery_initial_control_v1',L/'recovery_outer_dispatch_v1']:
  assert gone(read(path/'launch.json')['process'])
 old=read(L/'outer_dispatch_v1/actual_wait.json');assert type(old['returncode'])is int and old['returncode']==1 and not any(old.get(k,False)for k in ['timed_out','terminated','killed'])and gone(old['process'])
 oc=read(L/'dispatch_v1/completion.json');assert oc['all_exit0']is False and len(oc['results'])==len(jobs)
 for n,r in zip(jobs,oc['results']):
  if n in NAMES:assert r==dict(error=read(L/'dispatch_v1'/n/'failure.json')['error'])
  else:assert type(r['returncode'])is int and r['returncode']==0
 assert read(L/'dispatch_v1/failure.json')['error']=="AssertionError('A bounded training job failed; preserve and diagnose')"
 assert read(L/'outer_dispatch_v1/failure.json')['error']=="RuntimeError('Owner exit 1')"
 rc=read(L/'recovery_outer_dispatch_v1/actual_wait.json');assert type(rc['returncode'])is int and rc['returncode']==0 and not any(rc.get(k,False)for k in ['timed_out','terminated','killed'])and gone(rc['process'])
 assert read(L/'unlaunched_training_recovery_v1/completion.json')['all_exit0']is True
 for d in [L/'initial_control_v1',L/'recovery_initial_control_v1',L/'recovery_outer_dispatch_v1',L/'unlaunched_training_recovery_v1']:assert not(d/'failure.json').exists()
 routes={n:L/('unlaunched_training_recovery_v1'if n in NAMES else'dispatch_v1')/n for n in jobs}
 return routes,dict(authorization=pin(O/'unlaunched_training_recovery_preparation_v1.json'),original_outer_actual_wait=pin(L/'outer_dispatch_v1/actual_wait.json'),original_owner_failure=pin(L/'dispatch_v1/failure.json'),original_supervisor_failure=pin(L/'outer_dispatch_v1/failure.json'),recovery_outer_actual_wait=pin(L/'recovery_outer_dispatch_v1/actual_wait.json'),zero_science_prelaunch_failures=auth['failures'],original_all_exit0=False,recovery_route_closure='Original four successful jobs plus two first scientific launches under original registration')
