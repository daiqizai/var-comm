"""Only actual received wires enter native RX; post-exit truth is never a repair."""
import argparse,time
from stage_support import *
RUN=L/'rx_v1';OUT=O/'rx_v1';REG=OUT/'registration.json'
def prepare():
    from source_host_v1 import require_admission,ADMISSION
    require_admission();q=qualification();link_review=closed('link');source_review=closed('source')
    assert not RUN.exists()and not OUT.exists()
    pc=read(O/'link_v1/completion.json');assert pc['logical_frames']==3600 and pc['unresolved']==0
    for d in pc['outputs']:verify(d)
    jobsfile=L/'link_v1/rx/actual_source_jobs.json';jobs=read(jobsfile)
    from rx_cache import load
    match,extend_cache=load()
    oldstates=PO/'a2_clean_v1/clean_states.json';states=read(oldstates)['states'];assert len(states)==700
    inputs=[link_review,source_review,desc(ADMISSION),desc(O/'link_v1/registration.json'),desc(O/'link_v1/completion.json'),desc(O/'link_v1/logical_frames.json'),desc(jobsfile),desc(oldstates)]
    for name in ('a2_clean_v1','a2b_rx_v1','a3_rx_v1'):
        rp=PO/name/'root_closure_review.json';r=read(rp);assert r['all_processes_gone'] and r['unresolved']==0
        for d in r['controls']+r['outputs']:verify(d)
        inputs.append(desc(rp))
        if name!='a2_clean_v1':
            p=PO/name/'actual_states.json';states=extend_cache(states,read(p));inputs.append(desc(p))
    clean=read(PO/'a2_clean_v1/registration.json');old=read(PO/'a3_rx_v1/registration.json')
    for d in clean['source_files']+old['source_files']+clean['weights']:verify(d)
    original=read(PL/'a2_clean_v1/public.json');cfg=dict(read(L/'source_v1/public.json'))
    skip={'schema','run','output','resource','cuda_admission','new_job_count'}
    assert {k:v for k,v in cfg.items()if k not in skip}=={k:v for k,v in original.items()if k not in skip}
    for k in ('start_free_bytes','running_free_bytes','allocator_max_bytes','threads','interop_threads'):assert cfg['resource'][k]==original['resource'][k]
    for j in jobs:verify(j['input']);assert j['path']==j['input']['path']
    for s in states:verify(s['wire']);verify(s['image'])
    reused,new=match(jobs,states)
    gpu=cfg['resource']['gpu'];now=next(g for g in gpus()if g['uuid']==gpu['uuid']);assert now['free_mib']>=65536 and now['utilization']<=25
    RUN.mkdir();OUT.mkdir();wires=RUN/'wire_inputs';wires.mkdir();received=[]
    for j in new:
        p=wires/(j['key']+'.arc');raw=Path(j['path']).read_bytes()
        with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        pin=desc(p);assert all(pin[k]==j['input'][k]for k in ('bytes','sha256'))
        received.append({'key':j['key'],'path':str(p),'input':pin,'paid_profile':j['paid_profile'],'generation_seed':0})
    cfg.update(run=str(RUN),output=str(OUT),schema='ARPC_COMMON500_REMAINING400_ACTUAL_RX_V1',new_job_count=len(new))
    put(wires/'jobs.json',received);put(RUN/'public.json',cfg);put(OUT/'reused_states.json',reused)
    inputs+=q+[desc(p)for p in (PL/'a2_clean_v1/public.json',L/'source_v1/public.json',PO/'a2_clean_v1/registration.json',PO/'a3_rx_v1/registration.json',wires/'jobs.json',RUN/'public.json',OUT/'reused_states.json')]
    code=[desc(E/n)for n in ('common.py','stage_support.py','execute_stage_v1.py','rx_control.py','rx_native.py','rx_cache.py','rx_audit.py','source_host_v1.py')]
    reg={'schema':'ARPC_COMMON500_REMAINING400_ACTUAL_RX_REGISTRATION_V1','host':os.uname().nodename,'created_unix':time.time(),'source_files':old['source_files']+code,'weights':clean['weights'],'inputs':inputs,'public':desc(RUN/'public.json'),'new_jobs':len(new),'reused_jobs':len(reused),'actual_unique_jobs':len(jobs),'new_rx_state_cap':3600,'startup_free_mib':65536,'phase_wall_seconds':15000,'stage_seconds':{'rx':14400,'audit':600},'cache_identity':'Exact received full wire SHA/bytes, paid endpoint, seed0, shared native model/source/numerical configuration; Python ABI and GPU architecture checked by explicit host admission.','truth_correction':False,'native_python':'/usr/bin/python3'}
    put(REG,reg);print(json.dumps(desc(REG)))
def run(expected):
    reg=read(REG);cfg=read(RUN/'public.json');env=dict(os.environ,PYTHONPATH=str(PL/'environment_v1')+':'+str(PE),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(cfg['cuda_admission']['visible_index']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    jobs=[('rx',['/usr/bin/python3','-B',str(E/'rx_native.py'),'rx','--public',str(RUN/'public.json')]),('audit',['/usr/bin/python3','-B',str(E/'rx_audit.py'),'--expected-registration',expected])]
    run_stages('rx',expected,jobs,env,reg['stage_seconds'],gpu=cfg['resource']['gpu'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','run']);p.add_argument('--expected-registration');a=p.parse_args();prepare()if a.role=='prepare'else run(a.expected_registration)
