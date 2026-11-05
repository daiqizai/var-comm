"""Evaluation-only handoff after the explicitly authorized early training stop.

The original train-to-150k chain is retired, not modified or restarted. All
packet workers and test inputs are unchanged. This module cannot launch training.
"""
from pathlib import Path
import argparse,json,signal,sys,time,traceback
from runtime import L,OLD,PY,sha,write
import posttrain_chain_v1 as engine

STOP=False
def stop(*args):
    global STOP
    STOP=True

def boundary(protocol):
    root=Path(protocol['root'])
    if STOP or (root/'STOP_EVALUATION').exists():
        raise InterruptedError('evaluation stop retained')
    assert (OLD/'STOP_P2').exists(),'old P2 pause marker missing'
    assert (L/'STOP_MAIN').exists() and sha(L/'STOP_MAIN')==protocol['training_stop_sha256'],'training pause changed'
    if time.time()>protocol['deadline']:
        raise TimeoutError('finite evaluation deadline reached')
    for path,h in protocol['bindings'].items():
        assert sha(path)==h,path
    for folder in (L/'controllers').iterdir():
        if folder.name.startswith('training') or folder.name=='posttrain_chain_v1':
            for name in ['owner.json','child.json']:
                p=folder/name
                assert not (p.exists() and engine.live(engine.read(p))),f'prior compute role live: {p}'

def admit(protocol):
    assert protocol['max_new_training_segments']==0
    assert protocol['training_state']['schedule_completed'] is False
    assert protocol['training_state']['updates_512']==0
    boundary(protocol)
    cp=protocol['evaluation_checkpoint']
    assert cp['step']==protocol['training_state']['stopped_step']<100000
    assert sha(cp['path'])==cp['sha256']
    for name in ['training_world8_v1','posttrain_chain_v1']:
        wait=engine.read(L/'controllers'/name/'actual_wait.json')
        assert wait['child_gone'] and wait['returncode']==(0 if name.startswith('training') else 1)
    failure=engine.read(L/'posttrain_v1/failure.json')
    assert failure['error']=="InterruptedError('user/main stop retained; no continuation')"
    assert not list((L/'posttrain_v1/roles').glob('*')),'old test calls already exist'
    return cp

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    protocol=engine.read(a.protocol);root=Path(protocol['root'])
    checkpoint=admit(protocol)
    if a.check_only:
        print(json.dumps(dict(passed=True,step=checkpoint['step'],training_stopped=True,training_launches=0,model_calls=0)))
        return
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    assert not (root/'started.json').exists()
    write(root/'started.json',dict(time=time.time(),protocol_sha256=sha(a.protocol),training_resume_authorized=False))
    write(root/'final_checkpoint.json',checkpoint)
    # The reusable helpers call this module's stricter, evaluation-only boundary.
    # This changes only this new process; no source or previous process is edited.
    engine.boundary=boundary
    try:
        for size in [256,512]:
            try:engine.evaluate_size(protocol,checkpoint,size)
            except (InterruptedError,TimeoutError):raise
            except Exception as e:
                write(root/f'resolution_{size}_failure.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        source=Path(__file__).resolve().parent
        job=engine.launch_owner(protocol,'summary',[PY,str(source/'earlystop_summary_v1.py'),'--protocol',a.protocol],[],[20,22],3600,protocol['bindings'])
        assert engine.monitor([job],protocol)==[0]
        write(root/'status.json',dict(phase='finished',time=time.time(),scientific_result=engine.read(root/'delivery/complete.json')))
    except BaseException as e:
        write(root/'failure.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,training_resumed=False))
        raise

if __name__=='__main__':main()
