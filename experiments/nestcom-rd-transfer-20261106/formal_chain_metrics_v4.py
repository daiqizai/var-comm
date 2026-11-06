"""Finite shard queue on currently idle devices; immutable same-device pairs."""
import json,os,subprocess,time,traceback
from pathlib import Path
from runtime import *
setup()
from owned_process import proc_identity,run_role
F=L/'formal500_v1'

def idle_devices():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy={ln.split(',')[0].strip() for ln in apps.splitlines() if ln.strip()};free=[]
    for line in lines.splitlines():
        i,u,m,p=[x.strip() for x in line.split(',')]
        if u not in busy and int(m)<100 and int(p)==0:free.append(int(i))
    return free,dict(inventory=lines,apps=apps,time=time.time())

def main():
    protocol=F/'protocol_metrics_v4.json';r=json.loads(protocol.read_text());old_paused()
    for p,h in r['bindings'].items():assert sha(p)==h,p
    control=F/'controller_metrics_v4';control.mkdir(exist_ok=False);write(control/'owner.json',dict(**proc_identity(os.getpid()),started=time.time()))
    def start(label,gpu,snapshot):
        q=json.loads((F/'requests'/(label+'.json')).read_text());q['gpu']=gpu
        q['bindings']=dict(r['bindings']);q['bindings'][str(protocol)]=sha(protocol)
        request=F/'actual_requests_metrics_v4'/(label+'.json');assert not request.exists();write(request,q)
        write(control/(label+'_admission.json'),dict(gpu=gpu,request_sha256=sha(request),**snapshot))
        p=subprocess.Popen([PY,str(S/('formal_qualification_score_owner_v2.py' if label=='qualification' else 'formal_owner_v3.py')),'--request',str(request)],
            stdout=(control/(label+'.out')).open('xb'),stderr=(control/(label+'.err')).open('xb'),env=env())
        write(control/(label+'_child.json'),dict(gpu=gpu,**proc_identity(p.pid)));return p,time.time(),gpu
    try:
        a=json.loads((L/'roles/training_continue_v1/actual_wait.json').read_text())
        assert a['returncode']==0 and a['child_gone']
        assert json.loads((L/'continuation_v1/segment_complete.json').read_text())['step']==5000
        while True:
            assert time.time()<r['deadline']-60
            free,snap=idle_devices()
            if free:break
            time.sleep(5)
        p,begin,gpu=start('qualification',free[0],snap);rc=p.wait(timeout=min(3700,r['deadline']-time.time()))
        write(control/'qualification_wait.json',dict(returncode=rc,seconds=time.time()-begin,child_gone=not Path('/proc',str(p.pid)).exists()))
        assert rc==0,'Qualification failed; no formal test or automatic retry'
        assert json.loads((F/'qualification/complete.json').read_text())['passed']
        pending=[f'shard{i}' for i in range(8)];running={};failed=[]
        while pending or running:
            assert time.time()<r['deadline']
            for label,(p,begin,gpu) in list(running.items()):
                rc=p.poll()
                if rc is not None:
                    p.wait();write(control/(label+'_wait.json'),dict(returncode=rc,seconds=time.time()-begin,child_gone=not Path('/proc',str(p.pid)).exists()))
                    if rc:failed.append(label)
                    del running[label]
            free,snap=idle_devices();occupied={v[2] for v in running.values()}
            for gpu in free:
                if not pending:break
                if gpu in occupied:continue
                label=pending.pop(0);running[label]=start(label,gpu,snap)
            write(control/'queue.json',dict(time=time.time(),pending=pending,running={k:dict(pid=v[0].pid,gpu=v[2]) for k,v in running.items()},failed=failed))
            if pending or running:time.sleep(3)
        assert not failed,dict(failed=failed,no_automatic_retry=True)
        (F/'analysis_roles').mkdir(exist_ok=False)
        run_role(F/'analysis_roles','summary',[PY,str(S/'formal_summary_v2.py')],env(),[24,26],[],900,r['bindings'])
        write(control/'complete.json',dict(passed=True,time=time.time(),old_jobs_paused=True))
    except BaseException:
        (control/'failure.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
