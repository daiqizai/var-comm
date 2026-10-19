"""Finite extension controls; cross-host closure uses saved actual-wait evidence."""
import fcntl,time,traceback,subprocess
from common import *
import a1_owner as owner
def closed(stage):
    p=O/(stage+'_v1')/'root_closure_review.json';d=read(p)
    assert d['status']=='PASS' and d['all_processes_gone'] and d['unresolved']==0
    for x in d['evidence']:verify(x)
    return desc(p)
def qualification():
    q=L/'downstream_qualification_v1';d=read(q/'completion.json')
    assert d['status']=='PASS' and d['science_calls']==0
    check_wait(q/'tests.actual_wait.json',live=False)
    for x in d['sources']:verify(x)
    return [desc(q/'completion.json'),desc(q/'tests.actual_wait.json')]
def run_stages(stage,expected,jobs,env,seconds,gpu=None):
    run=L/(stage+'_v1');out=O/(stage+'_v1');reg=read(out/'registration.json')
    assert desc(out/'registration.json')['sha256']==expected and reg['host']==os.uname().nodename
    for d in reg['source_files']+reg['inputs']+reg.get('weights',[]):verify(d)
    assert not(run/'owner.launch.json').exists() and not(run/'failure.json').exists()
    lock=(PL/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if gpu:
        now=next(g for g in gpus()if g['uuid']==gpu['uuid'])
        assert now['free_mib']>=reg['startup_free_mib'] and now['utilization']<=25
        glock=(PL/f"gpu_{gpu['uuid']}.lock").open('a+');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    put(run/'owner.launch.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'registration_sha256':expected,'started_unix':started})
    owner.RUN=run
    try:
        for name,argv in jobs:
            remain=reg['phase_wall_seconds']-(time.time()-started);assert remain>0
            if name=='audit':
                for previous,_ in jobs[:jobs.index((name,argv))]:check_wait(run/(previous+'.actual_wait.json'))
            owner.run_stage(name,argv,dict(env,CUDA_VISIBLE_DEVICES='') if name=='audit'else env,min(seconds[name],remain))
        put(run/'completion.json',{'status':'PASS','output':desc(out/('score_completion.json'if stage=='metrics'else'completion.json')),'seconds':time.time()-started})
    except BaseException:put(run/'failure.json',{'error':traceback.format_exc(),'automatic_retry':False});raise
def close(stage):
    """Called on the execution host only, after the finite supervisor exits."""
    run=L/(stage+'_v1');out=O/(stage+'_v1');control=L/(stage+'_dispatch_control_v1')
    assert not(out/'root_closure_review.json').exists()
    reg=read(out/'registration.json');assert reg['host']==os.uname().nodename
    check_wait(control/'child.actual_wait.json')
    assert gone(read(control/'supervisor.json')['process']) and gone(read(run/'owner.launch.json')['process'])
    waits=sorted(run.glob('*.actual_wait.json'));assert waits
    for p in waits:check_wait(p)
    assert not(control/'failure.json').exists() and not(run/'failure.json').exists()
    final=out/('score_completion.json'if stage=='metrics'else'completion.json');d=read(final);assert d['unresolved']==0
    for pin in d['outputs']:verify(pin)
    evidence=[desc(p)for p in [out/'registration.json',final,run/'completion.json',run/'owner.launch.json',control/'child.actual_wait.json',control/'completion.json',control/'supervisor.json']+waits]+d['outputs']
    put(out/'root_closure_review.json',{'status':'PASS','reviewer':'root','host':os.uname().nodename,'observed_unix':time.time(),'all_processes_gone':True,'unresolved':0,'evidence':evidence,'scientific_calls_repeated':0})
    print(json.dumps(desc(out/'root_closure_review.json')))
