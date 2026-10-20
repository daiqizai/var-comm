"""Explicit recovery evidence routing around the unchanged timing statistics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'bpg_threads_recovery_v1'))
import resume as a
import native
original = native.load(a.E/'summarize.py', '_original_timing_summary')

EXPECTED_RECOVERY = 'f513248d68146729ced3ef9dc7b591acf1a697c3931acab0d9345ad06fe16dbb'


def recovery_closure():
    r, rec = a.bind(EXPECTED_RECOVERY)
    d = a.CONTROL/'dispatch'
    outer = a.read(d/'actual_wait.json')
    assert type(outer['returncode']) is int and outer['returncode']==0 and not outer['timed_out']
    for p in [d/'supervisor.json', d/'launch.json', a.CONTROL/'control/launch.json']:
        assert a.gone(a.read(p)['process'])
    assert not (a.CONTROL/'control/failure.json').exists()
    c = a.read(a.CONTROL/'control/completion.json')
    assert c['formal_calls']==720 and c['failed_initial_rx_cases']==0
    # Never represent the original failed process as successful.
    assert a.read(a.T/'dispatch/actual_wait.json')['returncode']==1
    assert a.read(a.T/'control/NEST_rx/actual_wait.json')['returncode']==1
    assert a.read(a.T/'NEST/rx/failure.json')['completed']==0
    assert not (a.T/'NEST/rx/cases').exists()
    assert a.read(a.previous.CONTROL/'dispatch/actual_wait.json')['returncode']==1
    assert a.read(a.T/'BPG/tx/failure.json')['completed']==0
    assert not list((a.T/'BPG/tx/cases').glob('*/complete.json'))
    return dict(registration=a.pin(a.REG), original_failed_outer=a.pin(a.T/'dispatch/actual_wait.json'),
        original_failed_receiver=a.pin(a.T/'NEST/rx/failure.json'),
        library_recovery=a.pin(a.previous.REG),
        failed_bpg_attempt=a.pin(a.T/'BPG/tx/failure.json'),
        successful_continuation_wait=a.pin(d/'actual_wait.json'),
        incomplete_failed_encoder_attempts=1, failed_completed_cases=0, replayed_successful_calls=0,
        reused_completed_stages=['NEST_tx','NEST_phy','NEST_rx','P100000_tx','P100000_rx','Swin200000_tx','Swin200000_rx'],
        notes='Frozen library-path admission, output routing and child-only x265 CPU-count query adaptation. Original codec binary/math/presets/search and statistics unchanged.')


def wait_path(method, role):
    root = (a.T if (method,role) in [('NEST','tx'),('NEST','phy')]
            else a.CONTROL if method in ('BPG','ARPC') else a.previous.CONTROL)
    return root/'control'/(method+'_'+role)/'actual_wait.json'


def main():
    original.recovery_closure = recovery_closure
    original.stage_dir = a.stage_dir
    original.wait_path = wait_path
    guard = """outer=read(T/'dispatch/actual_wait.json')
    assert type(outer['returncode']) is int and outer['returncode']==0 and not outer['timed_out']
    for d in (T/'dispatch/launch.json',T/'control/launch.json'):
        ident=read(d)['process']
        now=process(ident['pid'])
        assert now is None or now['start_ticks']!=ident['start_ticks']
    assert not (T/'control/failure.json').exists()
    assert read(T/'control/completion.json')['formal_calls']==720"""
    run = native.replace_function(original, 'main', [
        (guard, 'recovery = recovery_closure()'),
        ('d=T/method/role', 'd=stage_dir(method,role)'),
        ("wait=read(T/'control'/(method+'_'+role)/'actual_wait.json')",
         'wait=read(wait_path(method,role))'),
        ("protocol=r['timing'],registration=pin(O/'registration.json'),completed_inputs=bindings,",
         "protocol=r['timing'],registration=pin(O/'registration.json'),completed_inputs=bindings,recovery=recovery,"),
        ("text=['# Current five-method fixed16 N1024 timing','',",
         "text=['# Current five-method fixed16 N1024 timing','',"
         "'One initial receiver import failed before any case or model construction. One BPG encoder attempt later stalled and produced no completed case. Both failures are retained. Explicit library-path and child-only CPU-count continuations reused completed stages without replaying successful calls. The unchanged x265 binary now detects the allocated six CPUs; the 256px qualification reports six pool workers and two frame threads.','',")])
    run(a.OLD_REG)


if __name__ == '__main__':
    main()
