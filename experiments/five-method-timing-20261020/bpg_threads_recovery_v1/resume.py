"""Explicit BPG resource recovery, then the previously unstarted ARPC stages."""
import os
import sys
from pathlib import Path

E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E/'library_recovery_v1'))
import adapter as previous
from contract import *
import native

HERE = E/'bpg_threads_recovery_v1'
CONTROL = T/'bpg_threads_recovery_v1'
REG = O/'bpg_threads_recovery_v1/registration.json'
OLD_REG = previous.OLD_REG
PREVIOUS = '86ff233c504830f347aa4d3914945f6c58566ad148a9fa509610e0764ecbfec6'
SHIM = T/'bpg_threads_qualification_v1/affinity_cpu_count.so'
SHIM_SHA = '953375c8aa9c3ddd8f3313f99425ee2fe33a08b838492b0fee965dd04211494d'
gone = previous.gone


def stages():
    return [(method,role) for method in ('BPG','ARPC') for role in ('tx','phy','rx')]


def stage_dir(method,role):
    if (method,role)==('BPG','tx'):
        return T/'BPG/tx_threads_recovery_v1'
    return previous.stage_dir(method,role)


def bind(expected):
    assert sha(REG)==expected
    rec=read(REG)
    r,_=previous.bind(PREVIOUS)
    for item in rec['sources']+rec['evidence']:
        checked(item)
    assert sha(SHIM)==SHIM_SHA
    assert rec['remaining_stages']==[list(s) for s in stages()]
    return r,rec


def preflight(expected):
    r,rec=bind(expected)
    for root in [T,previous.CONTROL]:
        for rel in ['dispatch/launch.json','dispatch/supervisor.json','control/launch.json']:
            assert gone(read(root/rel)['process'])
        assert read(root/'dispatch/actual_wait.json')['returncode']==1
    failure=read(T/'BPG/tx/failure.json')
    assert failure['completed']==0 and 'TimeoutExpired' in failure['traceback']
    assert len(list((T/'BPG/tx/cases').glob('*/reservation.json')))==1
    assert not list((T/'BPG/tx/cases').glob('*/complete.json'))
    for m,role in [('NEST','tx'),('NEST','phy'),('NEST','rx'),
                   ('P100000','tx'),('P100000','rx'),('Swin200000','tx'),('Swin200000','rx')]:
        root=T if (m,role) in [('NEST','tx'),('NEST','phy')] else previous.CONTROL
        wait=read(root/'control'/(m+'_'+role)/'actual_wait.json')
        assert wait['returncode']==0 and not wait['timed_out'] and gone(wait['child'])
        c=read(previous.stage_dir(m,role)/'completion.json')
        assert c['calls']==288 and c['formal']==144 and not c['blocked_reads']
    for m,role in stages():
        assert not stage_dir(m,role).exists()
    return r,rec


def codec_environment(argv):
    # Only the bpgenc child sees the resource adapter. bpgdec, Python, models and
    # the rest of the host retain their original environment.
    if argv[0]!=str(T/'assets/bpgenc'):
        return None
    assert not os.environ.get('LD_PRELOAD')
    assert sorted(os.sched_getaffinity(0))==list(range(6))
    return dict(os.environ,LD_PRELOAD=str(SHIM))


def adapted_worker():
    w=previous.adapted_worker()
    import bpg
    bpg.codec_environment=codec_environment
    bpg.run=native.replace_function(bpg,'run',[
        ('subprocess.run(argv, capture_output=True, timeout=120)',
         'subprocess.run(argv, capture_output=True, timeout=120, env=codec_environment(argv))')])
    w.main=native.replace_function(w,'main',[
        ('out = T/a.method/a.role',
         "out = T/a.method/('tx_threads_recovery_v1' if (a.method,a.role)==('BPG','tx') else a.role)"),
        ("a.method+'_'+a.role",
         "a.method+'_'+a.role+('_threads_recovery_v1' if (a.method,a.role)==('BPG','tx') else '')"),
        ("p=T/a.method/'tx/cases'/key/'output.json'",
         "p=T/a.method/('tx_threads_recovery_v1/cases' if a.method=='BPG' else 'tx/cases')/key/'output.json'")])
    return w


def worker(expected,method,role):
    bind(expected)
    assert (method,role) in stages()
    sys.argv=[str(HERE/'resume.py'),method,role,'--registration',OLD_REG]
    adapted_worker().main()


def owner(expected):
    m=native.load(E/'library_recovery_v1/owner.py','_previous_library_owner')
    m.__dict__.update(bind=bind,CONTROL=CONTROL,HERE=HERE,stage_dir=stage_dir,
                      stages=stages,preflight=preflight)
    text=Path(m.__file__).read_text()
    start=text.index('    r, rec = bind(expected)')
    end=text.index("    out = CONTROL/'control'",start)
    guard=text[start:end]
    run=native.replace_function(m,'run',[
        (guard,'    r, rec = preflight(expected)\n'),
        ("[python, '-B', str(HERE/'adapter.py'), expected, method, role]",
         "[python, '-B', str(HERE/'resume.py'), 'worker', expected, method, role]"),
        ("reused_completed_stages=['NEST_tx','NEST_phy']",
         "reused_completed_stages=['NEST_tx','NEST_phy','NEST_rx','P100000_tx','P100000_rx','Swin200000_tx','Swin200000_rx']"),
        ('failed_initial_rx_cases=0, seconds=',
         'failed_initial_rx_cases=0, failed_bpg_incomplete_encoder_attempts=1, seconds=')])
    run(expected)


def dispatch(expected):
    m=native.load(E/'library_recovery_v1/dispatch.py','_previous_library_dispatch')
    m.__dict__.update(bind=bind,CONTROL=CONTROL,HERE=HERE,stages=stages)
    main=native.replace_function(m,'main',[
        ("[CPY, '-B', str(HERE/'owner.py'), expected]",
         "[CPY, '-B', str(HERE/'resume.py'), 'owner', expected]")])
    return main(expected)


if __name__=='__main__':
    role=sys.argv[1]
    if role=='worker':
        worker(*sys.argv[2:])
    elif role=='owner':
        owner(sys.argv[2])
    elif role=='dispatch':
        sys.exit(dispatch(sys.argv[2]))
    else:
        raise ValueError(role)
