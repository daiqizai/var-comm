"""Continue only unmeasured stages after a zero-case import failure."""
import os
import fcntl
import signal
import subprocess
import sys
import time
import traceback
from adapter import *


def run(expected):
    r, rec = bind(expected)
    for path in ['dispatch/launch.json', 'dispatch/supervisor.json', 'control/launch.json',
                 'control/NEST_tx/launch.json', 'control/NEST_phy/launch.json',
                 'control/NEST_rx/launch.json']:
        assert gone(read(T/path)['process'])
    assert read(T/'dispatch/actual_wait.json')['returncode'] == 1
    assert read(T/'NEST/rx/failure.json')['completed'] == 0
    assert not (T/'NEST/rx/cases').exists()
    assert not (T/'NEST/rx/setup.json').exists()
    for role in ['tx', 'phy']:
        assert read(T/'control'/('NEST_'+role)/'actual_wait.json')['returncode'] == 0
        c = read(T/'NEST'/role/'completion.json')
        assert c['calls'] == 288 and c['formal'] == 144
    for method, role in stages():
        assert not stage_dir(method, role).exists()
    out = CONTROL/'control'
    out.mkdir()
    lock = (T/'timing.lock').open('r+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(out/'launch.json', dict(process=process(os.getpid()),
        parent_process=process(os.getppid()), argv=sys.argv, recovery_registration_sha256=expected))
    start = time.monotonic()
    try:
        for method, role in stages():
            assert not (T/'STOP').exists()
            label = method+'_'+role
            d = out/label
            d.mkdir()
            python = CPY if role=='phy' or method=='BPG' else '/usr/bin/python3' if method=='ARPC' else PY
            argv = [python, '-B', str(HERE/'adapter.py'), expected, method, role]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if role=='phy' or method=='BPG' else str(r['resource']['gpu_index']),
                CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6',
                NUMEXPR_NUM_THREADS='6', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            env.pop('PYTHONPATH', None)
            if method=='ARPC' and role!='phy':
                env['PYTHONPATH'] = r['arpc_public']['environment']+':'+str(E)
            before = time.monotonic()
            with (d/'stdout.log').open('xb') as stdout, (d/'stderr.log').open('xb') as stderr:
                p = subprocess.Popen(argv, env=env, cwd=E, stdout=stdout, stderr=stderr, start_new_session=True)
                ident = process(p.pid)
                save(d/'launch.json', dict(process=ident, parent_process=process(os.getpid()), argv=argv,
                    environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONNOUSERSITE']},
                    recovery_registration_sha256=expected))
                timed_out = False
                try:
                    code = p.wait(timeout=r['stage_wall_seconds'][role]+300)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        code = p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        code = p.wait(timeout=30)
            save(d/'actual_wait.json', dict(returncode=code, seconds=time.monotonic()-before,
                child=ident, timed_out=timed_out, child_gone=gone(ident)))
            assert code == 0 and not timed_out, (label, code)
            target = stage_dir(method, role)
            assert not (target/'failure.json').exists()
            c = read(target/'completion.json')
            assert c['registration_sha256']==OLD_REG and c['calls']==288 and c['formal']==144
        save(out/'completion.json', dict(status='COMPLETE_WITH_EXPLICIT_LIBRARY_RECOVERY',
            original_registration_sha256=OLD_REG, recovery_registration_sha256=expected,
            full_calls=1440, formal_calls=720, reused_completed_stages=['NEST_tx','NEST_phy'],
            failed_initial_rx_cases=0, seconds=time.monotonic()-start))
    except BaseException:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    run(sys.argv[1])
