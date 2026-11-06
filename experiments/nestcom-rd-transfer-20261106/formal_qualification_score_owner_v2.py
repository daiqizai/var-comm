"""Only the unexecuted qualification metrics; reuse successful TX/RX bytes."""
import argparse,json,time
from pathlib import Path
from runtime import *
setup()
from owned_process import run_role
p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
r=json.loads(Path(a.request).read_text());base=Path(r['output']);old_paused()
for stage in ['tx','rx']:
 q=json.loads((base/'roles'/stage/'actual_wait.json').read_text());assert q['returncode']==0 and q['child_gone']
failed=json.loads((base/'roles/score/actual_wait.json').read_text());assert failed['returncode']==1 and failed['child_gone']
assert not (base/'score/progress.json').exists() and not list((base/'score').glob('*_started.json'))
assert 'cannot import name \'SourcePacket\'' in (base/'roles/score/stderr.log').read_text()
bindings=dict(r['bindings']);bindings[a.request]=sha(a.request)
e=env();e['CUDA_VISIBLE_DEVICES']=str(r['gpu'])
run_role(base/'roles','score_v2',[PY,str(S/'formal_worker_v2.py'),'--request',a.request,'--stage','score'],
    e,r['cpus'],[r['gpu']],min(1800,r['deadline']-time.time()),bindings)
write(base/'complete.json',dict(passed=True,time=time.time(),qualification=True,
    reuse=['tx','rx'],new_role='score_v2',original_score_import_failure_retained=True))
