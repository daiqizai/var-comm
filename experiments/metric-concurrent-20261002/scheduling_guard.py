"""Admit only an identity-bound peer to the unchanged GPU resource guard.

This changes scheduling permission alone. Unknown GPU users and the original
thermal checks remain active; exclusive timing never admits another process.
"""
from __future__ import annotations
import atexit
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time

NAME = 'METRIC-CONCURRENT-R4-20261002'
ROLES = ('metrics', 'm2')


class SchedulingPause(RuntimeError):
    """Operational admission wait, never a scientific or numerical failure."""


@contextmanager
def admission_lock(out):
    import fcntl
    with (Path(out)/'admission.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def timing_pending(root):
    root = Path(root)
    request = root/'outputs'/NAME/'exclusive_timing_requested.json'
    done = root/'outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002/m2_timing_complete.json'
    return request.exists() and not (done.exists() and read(done).get('status') == 'M2_TIMING_COMPLETE')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def process(pid, proc_root=Path('/proc')):
    path = Path(proc_root) / str(int(pid))
    try: raw = (path / 'stat').read_text()
    except FileNotFoundError: return None
    fields = raw[raw.rfind(') ') + 2:].split()
    if len(fields) < 20: raise RuntimeError('Incomplete process identity')
    try: cmd = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except FileNotFoundError: cmd = ''
    return dict(pid=int(pid), start_ticks=fields[19], state=fields[0], ppid=int(fields[1]), command=cmd)


def identity(value):
    if (not isinstance(value, dict) or isinstance(value.get('pid'), bool)
            or int(value.get('pid', 0)) <= 1 or not str(value.get('start_ticks', '')).isdigit()):
        raise RuntimeError('Invalid process identity')
    return dict(pid=int(value['pid']), start_ticks=str(value['start_ticks']))


def alive(expected, actual):
    return actual is not None and actual.get('state') != 'Z' and identity(actual) == identity(expected)


def sources(here):
    here = Path(here).resolve()
    paths = sorted(p for p in here.iterdir() if p.suffix in ('.py', '.md'))
    if not paths or any(p.is_symlink() or not p.is_file() for p in paths):
        raise RuntimeError('Invalid runtime source inventory')
    return {str(p): sha(p) for p in paths}


def verify(mapping):
    if not isinstance(mapping, dict) or not mapping: raise RuntimeError('Missing frozen bindings')
    for path, digest in mapping.items():
        if len(str(digest)) != 64 or sha(path) != digest: raise RuntimeError('Frozen file changed: ' + path)


def validate_admission(root, path, here=None):
    root, path = Path(root).resolve(), Path(path).resolve()
    out = root / 'outputs' / NAME
    here = Path(here or Path(__file__).parent).resolve()
    if path != out / 'admission.json' or here != out / 'runtime':
        raise RuntimeError('Scheduling admission must use the registered separate runtime')
    value = read(path)
    if (value.get('status') != 'REGISTERED_SCHEDULING_ADMISSION'
            or value.get('training_updates') != 0 or value.get('policy_selection_updates') != 0
            or value.get('allowed_roles') != list(ROLES) or value.get('exclusive_stages') != ['timing']
            or value.get('source_bindings') != sources(here)):
        raise RuntimeError('Scheduling admission scope/source differs')
    verify(value['original_guard_bindings'])
    return value


def owner_path(out, role):
    return Path(out) / ('controller_launch.json' if role == 'metrics' else 'm2_controller_launch.json')


def approved_peer(out, admission, admission_sha, role, reader=process):
    """Return one live registered peer, or None. Never approve by PID alone."""
    peer_role = 'm2' if role == 'metrics' else 'metrics'
    path = Path(out) / f'active_{peer_role}.json'
    if not path.exists(): return None
    lease = read(path)
    if lease.get('status') != 'ACTIVE': return None
    if (lease.get('role') != peer_role or lease.get('admission_sha256') != admission_sha
            or lease.get('source_bindings') != admission['source_bindings']):
        raise RuntimeError('Peer scheduling lease is unregistered')
    owner = read(owner_path(out, peer_role))
    if (owner.get('admission_sha256') != admission_sha or owner.get('source_bindings') != admission['source_bindings']
            or identity(owner) != identity(lease['owner'])):
        raise RuntimeError('Peer owner lineage differs')
    actual_owner, actual = reader(owner['pid']), reader(lease['pid'])
    if not alive(owner, actual_owner): return None
    expected_owner = 'controller.py' if peer_role == 'metrics' else 'm2_controller.py'
    expected_worker = 'concurrent_runner.py' if peer_role == 'metrics' else 'scheduled_process.py'
    runtime = (Path(out) / 'runtime').as_posix() + '/'
    if runtime + expected_owner not in actual_owner.get('command', '').replace('\\', '/'):
        raise RuntimeError('Peer owner command differs')
    if lease.get('exclusive') is True: return None
    if not alive(lease, actual):
        # The driver may briefly list a just-exited registered CUDA process.
        # Never forgive a live process whose PID was reused by another identity.
        if actual is not None and (identity(actual) != identity(lease) or actual.get('state') != 'Z'):
            return None
        stamp = lease.get('cleanup_started_time')
        if isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or not math.isfinite(stamp): return None
        elapsed = time.time()-stamp
        return identity(lease) if 0 <= elapsed < 5 else None
    if not actual_owner.get('command') or not actual.get('command'):
        again_owner,again = reader(owner['pid']),reader(lease['pid'])
        if not alive(owner,again_owner) or not alive(lease,again): return None
        if not again_owner.get('command') or not again.get('command'): return None
        actual_owner,actual = again_owner,again
    if (runtime + expected_owner not in actual_owner.get('command', '').replace('\\', '/')
            or runtime + expected_worker not in actual.get('command', '').replace('\\', '/')
            or actual.get('ppid') != int(owner['pid'])):
        raise RuntimeError('Peer process ancestry/command differs')
    return identity(lease)


def filtered_foreign(original, out, admission, admission_path, role, *, exclusive=False, reader=process):
    expected_sha = sha(admission_path)
    def guarded():
        if sha(admission_path) != expected_sha: raise RuntimeError('Scheduling admission changed')
        foreign = original()
        if not isinstance(foreign, list) or any(not isinstance(p, dict) or isinstance(p.get('pid'), bool)
                or not isinstance(p.get('pid'), int) or p['pid'] <= 1 for p in foreign):
            raise RuntimeError('Unexpected original GPU resource response')
        peer = None if exclusive else approved_peer(out, admission, expected_sha, role, reader)
        return [p for p in foreign if peer is None or p['pid'] != peer['pid']]
    return guarded


def install_admission(root, admission_path, role='metrics', exclusive=False):
    if role not in ROLES: raise RuntimeError('Unknown scheduling role')
    root, path = Path(root).resolve(), Path(admission_path).resolve()
    out = root / 'outputs' / NAME
    admission = validate_admission(root, path)
    digest = sha(path)
    owner = read(owner_path(out, role)); me = process(os.getpid())
    if (me is None or me['ppid'] != int(owner['pid']) or not alive(owner, process(owner['pid']))
            or owner.get('admission_sha256') != digest or owner.get('source_bindings') != admission['source_bindings']):
        raise RuntimeError('Scheduling worker is not a child of its registered live owner')
    lease_path = out / f'active_{role}.json'
    lease = dict(status='ACTIVE', **identity(me), role=role, owner=identity(owner),
        admission_sha256=digest, source_bindings=admission['source_bindings'], exclusive=bool(exclusive), time=time.time())
    with admission_lock(out):
        if role == 'metrics' and timing_pending(root):
            raise SchedulingPause('Original timing requires exclusive GPU ownership')
        if lease_path.exists():
            prior = read(lease_path)
            if prior.get('status') == 'ACTIVE' and alive(prior, process(prior['pid'])) and identity(prior) != identity(me):
                raise RuntimeError('Another scheduling worker for this role is active')
        write(lease_path,lease)
    runtime = importlib.import_module('latent_enhancement.runtime')
    origin = str(Path(runtime.__file__).resolve())
    if admission['original_guard_bindings'].get(origin) != sha(origin):
        raise RuntimeError('Resource guard module origin differs from frozen admission')
    if getattr(runtime.foreign_gpu_processes, '_registered_scheduling_guard', False):
        raise RuntimeError('Scheduling resource guard already installed')
    guarded = filtered_foreign(runtime.foreign_gpu_processes, out, admission, path, role, exclusive=exclusive)
    guarded._registered_scheduling_guard = True
    runtime.foreign_gpu_processes = guarded
    lineage = out / 'guard_lineage' / f'{role}_{me["pid"]}_{me["start_ticks"]}.json'
    write(lineage, dict(**lease, original_guard_bindings=admission['original_guard_bindings'],
        only_registered_peer_filter=True, unknown_gpu_users_blocked=True, thermal_guard_unchanged=True,
        inference_and_metric_functions_unchanged=True, training_updates=0, policy_selection_updates=0))
    def release():
        if lease_path.exists() and identity(read(lease_path)) == identity(me):
            # Python atexit precedes CUDA teardown. Keep this identity admitted
            # until /proc proves actual exit/Z/PID reuse, avoiding a false peer.
            write(lease_path, dict(lease, cleanup_started_time=time.time()))
    atexit.register(release)
    return dict(admission_path=str(path), admission_sha256=digest,
        guard_source_bindings=admission['source_bindings'], lineage_path=str(lineage))
