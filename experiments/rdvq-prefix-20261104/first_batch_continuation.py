"""Finite, identity-bound P1 -> P2 handoff. No retry and no P3/P4.

This is a process for this registered experiment, not a recurring scheduler.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
from owned_process import run_role,proc_identity

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)

def main():
    rp=Path(sys.argv[1]);r=json.loads(rp.read_text());L=Path(r['root']);S=Path(__file__).resolve().parent;O=Path(r['output']);O.mkdir(exist_ok=False)
    write(O/'owner.json',dict(**proc_identity(os.getpid()),started=time.time()));write(O/'registration.json',r)
    for p,h in r['bindings'].items():assert sha(p)==h,p
    def await_actual(path,deadline):
        p=Path(path)
        while not p.exists():
            if time.time()>deadline:raise TimeoutError('Original finite wait deadline exhausted: '+str(p))
            if (L/'STOP_P2').exists():raise RuntimeError('User stop present; no new phase admitted')
            time.sleep(30)
        value=json.loads(p.read_text());assert value['child_gone'];return value
    def helper(name):
        for p,h in r['bindings'].items():assert sha(p)==h,p
        w=run_role(O,name,[sys.executable,'-u',str(S/(name+'.py'))],dict(os.environ,CUDA_VISIBLE_DEVICES=''),[16,17],[],180,r['bindings']);assert w['returncode']==0
    try:
        wait=await_actual(L/'p1_source_chain_owner_control_v1/actual_wait.json',r['source_wait_deadline']);assert wait['returncode']==0
        assert json.loads((L/'p1_source_gate.json').read_text())['passed']
        data=json.loads((L/'data_union_verified_v1/actual_wait.json').read_text());assert data['returncode']==0 and data['child_gone']
        assert json.loads((L/'data_union_verified_v1/complete.json').read_text())['passed']
        helper('start_p1_control_analysis')
        helper('prepare_p2_registration')
        helper('start_p2_registered')
        write(O/'handoff.json',dict(P1_source_actual_exit=0,P2_first_launch=json.loads((L/'p2_launch.json').read_text()),P1_control_CPU_started=True,time=time.time()))
        # These were explicitly registered first launches, never retries.
        deadline=json.loads((L/'p2_registration_v1/queue_request.json').read_text())['absolute_deadline']+6*3600
        control=await_actual(L/'p1_control_analysis_v1/actual_wait.json',time.time()+4200)
        write(O/'control_analysis_outcome.json',control)
        # A failed control reference does not erase valid source-training work.
        study=await_actual(L/'p2_owner_control_v1/study/actual_wait.json',deadline)
        write(O/'p2_outer_outcome.json',study)
        req=L/'p2_registration_v1/analysis_request.json';a=json.loads(req.read_text());bindings={str(req):sha(req),str(S/'analyze_matched.py'):sha(S/'analyze_matched.py')}
        for arm in a['arms']:
            root=Path(arm['output_root'])
            if (root/'complete.json').exists():
                bindings[str(root/'complete.json')]=sha(root/'complete.json')
                for ev in json.loads((root/'complete.json').read_text())['evaluations']:
                    if ev['split']=='test':bindings[ev['scores']]=sha(ev['scores'])
        result=run_role(L,'p2_matched_analysis_v1',[r['cpu_python'],'-u',str(S/'analyze_matched.py'),'--request',str(req)],dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),[16,17],[],3600,bindings)
        analysis=json.loads((L/'p2_matched_analysis_v1/complete.json').read_text())
        write(O/'complete.json',dict(passed=analysis['passed'] and control['returncode']==0,analysis=analysis,review_and_publication_pending=True,automatic_P3_P4=False))
    except BaseException:
        (O/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
