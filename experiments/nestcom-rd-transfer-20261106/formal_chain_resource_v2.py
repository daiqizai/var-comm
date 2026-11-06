"""Sole bounded post-5000 owner; never retries a failed role."""
import json,os,subprocess,time,traceback
from pathlib import Path
from runtime import *
setup()
from owned_process import proc_identity,run_role

F=L/'formal500_v1'

def admitted(gpus):
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy={ln.split(',')[0].strip() for ln in apps.splitlines() if ln.strip()}
    for line in lines.splitlines():
        i,u,m,p=[x.strip() for x in line.split(',')]
        if int(i) in gpus:assert u not in busy and int(m)<100 and int(p)==0,dict(gpu=i,apps=apps)
    return dict(inventory=lines,apps=apps,time=time.time())

def main():
    r=json.loads((F/'protocol_resource_v2.json').read_text());old_paused()
    for p,h in r['bindings'].items():assert sha(p)==h,p
    control=F/'controller';control.mkdir(exist_ok=False);write(control/'owner.json',dict(**proc_identity(os.getpid()),started=time.time()))
    try:
        prior=json.loads((L/'roles/training_continue_v1/actual_wait.json').read_text())
        assert prior['returncode']==0 and prior['child_gone']
        assert json.loads((L/'continuation_v1/segment_complete.json').read_text())['step']==5000
        write(control/'qualification_admission.json',admitted([1]))
        cmd=[PY,str(S/'formal_owner_v2.py'),'--request',str(F/'requests_resource_v2/qualification.json')]
        p=subprocess.Popen(cmd,stdout=(control/'qualification.out').open('xb'),stderr=(control/'qualification.err').open('xb'),env=env())
        write(control/'qualification_child.json',proc_identity(p.pid));start=time.time();rc=p.wait(timeout=3700)
        write(control/'qualification_wait.json',dict(returncode=rc,seconds=time.time()-start,child_gone=not Path('/proc',str(p.pid)).exists()))
        assert rc==0,'Qualification failed; no test launch and no automatic retry'
        assert json.loads((F/'qualification/complete.json').read_text())['passed']
        write(control/'formal_admission.json',admitted(list(range(1,8))))
        children=[]
        for shard in range(7):
            p=subprocess.Popen([PY,str(S/'formal_owner_v2.py'),'--request',str(F/f'requests_resource_v2/shard{shard}.json')],
                stdout=(control/f'shard{shard}.out').open('xb'),stderr=(control/f'shard{shard}.err').open('xb'),env=env())
            children.append((shard,p,time.time()));write(control/f'shard{shard}_child.json',proc_identity(p.pid))
        failed=[]
        first=children[0]
        rc=first[1].wait(timeout=max(1,r['deadline']-time.time()))
        if rc==0:
            write(control/'shard7_admission.json',admitted([1]))
            p=subprocess.Popen([PY,str(S/'formal_owner_v2.py'),'--request',str(F/'requests_resource_v2/shard7.json')],stdout=(control/'shard7.out').open('xb'),stderr=(control/'shard7.err').open('xb'),env=env())
            children.append((7,p,time.time()));write(control/'shard7_child.json',proc_identity(p.pid))
        else:failed.append(7)
        for shard,p,start in children:
            rc=p.wait(timeout=max(1,r['deadline']-time.time()))
            write(control/f'shard{shard}_wait.json',dict(returncode=rc,seconds=time.time()-start,child_gone=not Path('/proc',str(p.pid)).exists()))
            if rc:failed.append(shard)
        assert not failed,dict(failed_shards=failed,no_retry=True)
        (F/'analysis_roles').mkdir(exist_ok=False)
        run_role(F/'analysis_roles','summary',[PY,str(S/'formal_summary_v1.py')],env(),[24,26],[],900,r['bindings'])
        write(control/'complete.json',dict(passed=True,time=time.time(),old_jobs_paused=True))
    except BaseException:
        (control/'failure.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
