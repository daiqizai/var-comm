"""Recover only twenty missing RX pixels from immutable timing observations."""
import argparse
import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
EOLD = R/'experiments/five-method-timing-20261020'
sys.path.insert(0, str(EOLD))
from contract import checked, read, save, pin, sha, process, binding, PY, CPY, B, RT
import native

E = R/'experiments/selected-examples-10db-20261020'
O = R/'results/selected_examples_10db_20261020_v1'
T = B/'selected_examples_10db_20261020_v1'
OLD = 'd4e7a6bbad058fff98d0c061515c11c562af5d8cc59be5c9e249351363964f31'
METHODS = ('NEST', 'P100000', 'Swin200000', 'BPG', 'ARPC')
INDICES = (21, 92, 95, 75)


def validate(reg):
    assert reg['timing_registration']['sha256'] == OLD
    assert reg['indices'] == list(INDICES) and reg['snr_db'] == 10
    assert len(reg['cases']) == 20
    assert {(c['method'], c['source_index']) for c in reg['cases']} == {
        (m, i) for m in METHODS for i in INDICES}
    for c in reg['cases']:
        assert c['key'] == f"s10_i{c['source_index']:04d}_r3"
        assert c['counter'] == (100+c['source_index'])*3
    assert reg['limits'] == dict(rx_cases=20, tx=0, channel_events=0,
                                quality_pairs=20, training=0, bootstrap=0)


def bind(expected):
    assert sha(O/'registration.json') == expected
    reg = read(O/'registration.json')
    validate(reg)
    for d in reg['new_sources']:
        checked(d)
    return reg


def setup(out):
    os.sched_setaffinity(0, set(range(6)))
    cache = out/'runtime_cache'
    (cache/'tmp').mkdir(parents=True)
    os.environ.update(TMPDIR=str(cache/'tmp'),
        TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
        TRITON_CACHE_DIR=str(cache/'triton'), HF_HOME=str(cache/'hf'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    tempfile.tempdir = str(cache/'tmp')


def receiver(expected, method):
    reg = bind(expected)
    r = binding(OLD)
    rows = [{k: c[k] for k in ('key', 'source_index', 'counter', 'received_input')}
            for c in reg['cases'] if c['method'] == method]
    assert [x['source_index'] for x in rows] == list(INDICES)
    # Output hashes and original pixels are never arguments to a receiver.
    del reg
    out = T/method
    out.mkdir()
    setup(out)
    save(out/'launch.json', dict(process=process(os.getpid()),
        parent_process=process(os.getppid()), argv=sys.argv, registration=expected))
    done = []
    try:
        sys.path.insert(0, str(EOLD/'library_recovery_v1'))
        from adapter import adapted_worker
        w = adapted_worker()
        save(out/'resource_before.json', w.resource(r, initial=True))
        for row in rows:
            checked(row['received_input'])
        blocked = w.firewall(r, method, out)
        import numpy as np
        t = None
        ctx = None
        calls = native.Calls()
        if method != 'BPG':
            import torch, ctypes
            torch.cuda.set_device(0)
            lib = ctypes.CDLL('libcuda.so.1')
            device = ctypes.c_int()
            assert lib.cuDeviceGet(ctypes.byref(device), 0) == 0
            buf = ctypes.create_string_buffer(32)
            assert lib.cuDeviceGetPCIBusId(buf, 32, device) == 0
            assert buf.value.decode().lower().lstrip('0') == r['resource']['gpu_pci'].lower().lstrip('0')
            save(out/'device.json', dict(PCI=buf.value.decode(),
                CUDA_UUID=str(torch.cuda.get_device_properties(0).uuid),
                visible=os.environ['CUDA_VISIBLE_DEVICES']))
        if method in ('P100000', 'Swin200000'):
            ctx, calls = native.continuous(r, out, method)
            t = ctx.t
        elif method == 'NEST':
            ctx = native.nest_codec(r, out)
            t, calls = ctx[0].torch, ctx[3]
        elif method == 'ARPC':
            ctx = native.arpc(r, out, 'rx')
            t = ctx.torch
        else:
            import bpg
            # Failure evidence belongs to this new run; codec math is unchanged.
            bpg.workspace = contextlib.contextmanager(native.replace_function(
                bpg, 'workspace', [("T/'BPG/failed_codec_evidence'/path.name",
                "Path("+repr(str(out/'failed_codec_evidence'))+")/path.name")]))
            bpg.legacy()
            r['_tmpfs'] = '/dev/shm/selected_examples_10db_20261020_v1_BPG'
            Path(r['_tmpfs']).mkdir()
        save(out/'runtime.json', dict(python=sys.version,
            torch=None if t is None else t.__version__,
            cuda=None if t is None else t.version.cuda,
            threads=None if t is None else t.get_num_threads(),
            interop_threads=None if t is None else t.get_num_interop_threads(),
            cpu_affinity=sorted(os.sched_getaffinity(0))))
        (out/'cases').mkdir()
        for row in rows:
            case = out/'cases'/row['key']
            case.mkdir()
            assert not (T/'STOP').exists()
            save(case/'reservation.json', dict(row, process=process(os.getpid())))
            meter = w.Clock(t)
            if method in ('P100000', 'Swin200000'):
                with np.load(row['received_input']['path'], allow_pickle=False) as z:
                    observed = z['observed'].copy()
                image, actual = ctx.receive(observed, 10, row['counter'], meter, row['key'])
                status = actual['status']
            else:
                actual = read(row['received_input']['path'])
                image, status = w.native_rx(actual, r, method, ctx, meter)
            assert image.dtype == np.float32 and image.shape == (3,256,256)
            assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
            with (case/'image.npz').open('xb') as f:
                np.savez(f, image=image)
                f.flush()
                os.fsync(f.fileno())
            result = dict(key=row['key'], source_index=row['source_index'],
                status=status, image=pin(case/'image.npz'),
                image_sha256=w.array_sha(image), source_truth_used=False,
                model_calls=calls.counts.copy(), received_input=row['received_input'])
            save(case/'complete.json', result)
            done.append(result)
            print(method, len(done), '/ 4', row['key'], flush=True)
        assert len(done) == 4 and not blocked
        save(out/'completion.json', dict(status='COMPLETE', results=done,
            blocked_reads=blocked, resource_after=w.resource(r),
            source_truth_used=False, tx_calls=0, channel_events=0))
    except BaseException:
        save(out/'failure.json', dict(traceback=traceback.format_exc(),
            completed=len(done), automatic_retry=False))
        raise


def owner(expected):
    reg = bind(expected)
    r = binding(OLD)
    out = T/'control'
    out.mkdir()
    save(out/'launch.json', dict(process=process(os.getpid()),
        parent_process=process(os.getppid()), argv=sys.argv, registration=expected))
    lock = (B/'five_method_timing_20261020_v1/timing.lock').open('r+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        for d in r['original_bindings']:
            checked(d)
        for c in reg['cases']:
            checked(c['received_input'])
        save(out/'preflight.json', dict(status='PASS', original_bindings=len(r['original_bindings']),
            received_inputs=20, training=0, scientific_calls=0))
        for method in METHODS:
            stage = out/method
            stage.mkdir()
            python = CPY if method == 'BPG' else '/usr/bin/python3' if method == 'ARPC' else PY
            argv = [python, '-B', str(E/'execute.py'), 'receiver',
                    '--registration', expected, '--method', method]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if method == 'BPG' else '1',
                CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONNOUSERSITE='1',
                PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6',
                OPENBLAS_NUM_THREADS='6', NUMEXPR_NUM_THREADS='6')
            env.pop('PYTHONPATH', None)
            if method == 'ARPC':
                env['PYTHONPATH'] = r['arpc_public']['environment']+':'+str(EOLD)
            start = time.monotonic()
            with (stage/'stdout.log').open('xb') as stdout, (stage/'stderr.log').open('xb') as stderr:
                p = subprocess.Popen(argv, env=env, cwd=E, stdout=stdout,
                    stderr=stderr, start_new_session=True)
                ident = process(p.pid)
                save(stage/'launch.json', dict(process=ident, parent_process=process(os.getpid()),
                    argv=argv, environment={k:env.get(k) for k in
                    ('CUDA_VISIBLE_DEVICES','PYTHONPATH','OMP_NUM_THREADS','CUDA_DEVICE_ORDER')}))
                timed_out = killed = False
                try:
                    code = p.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        code = p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        killed = True
                        os.killpg(p.pid, signal.SIGKILL)
                        code = p.wait(timeout=30)
            save(stage/'actual_wait.json', dict(returncode=code, child=ident,
                seconds=time.monotonic()-start, timed_out=timed_out, killed=killed,
                child_gone=process(p.pid) is None))
            assert code == 0 and not timed_out
            c = read(T/method/'completion.json')
            assert len(c['results']) == 4 and not c['blocked_reads']
            for result in c['results']:
                expected_case = next(v for v in reg['cases'] if
                    v['method'] == method and v['key'] == result['key'])
                assert result['image_sha256'] == expected_case['expected_image_sha256'], result
                assert result['status'] == expected_case['expected_status'], result
            save(stage/'hash_review.json', dict(status='PASS', exact_timing_hash_matches=4))
        save(out/'completion.json', dict(status='TWENTY_RX_PIXELS_EXACT_TIMING_HASH_MATCH',
            rx_reconstructions=20, reused_received_inputs=20, tx=0, channel_events=0))
    except BaseException:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def supervise(expected):
    bind(expected)
    d = T/'dispatch'
    d.mkdir()
    argv = [CPY, '-B', str(E/'execute.py'), 'owner', '--registration', expected]
    start = time.monotonic()
    with (d/'stdout.log').open('xb') as out, (d/'stderr.log').open('xb') as err:
        p = subprocess.Popen(argv, stdout=out, stderr=err, cwd=E, start_new_session=True)
        ident = process(p.pid)
        save(d/'launch.json', dict(process=ident, parent_process=process(os.getpid()), argv=argv))
        timed_out = killed = False
        try:
            code = p.wait(timeout=9300)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                code = p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                killed = True
                os.killpg(p.pid, signal.SIGKILL)
                code = p.wait(timeout=30)
    save(d/'actual_wait.json', dict(returncode=code, child=ident, seconds=time.monotonic()-start,
        timed_out=timed_out, killed=killed, child_gone=process(p.pid) is None))
    assert code == 0 and not timed_out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('role', choices=['owner', 'receiver', 'supervise'])
    ap.add_argument('--registration', required=True)
    ap.add_argument('--method', choices=METHODS)
    a = ap.parse_args()
    if a.role == 'owner':
        owner(a.registration)
    elif a.role == 'supervise':
        supervise(a.registration)
    else:
        receiver(a.registration, a.method)
