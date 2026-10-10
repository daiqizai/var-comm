"""Versioned, low-occupancy shared-host metadata/CPU-engineering entry only.

No installation, model/PHY execution, GPU allocation or numerical qualification.
An allowlisted engineering check is one CPU child; arbitrary commands are refused.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SCHEMA = 'LEO_SHARED_PREFLIGHT_CPU_ENGINEERING_V1'
PERSONAL_BASE = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu')
GPU_UUID = 'GPU-bf8340be-bd73-7413-62d5-0537be7f30cb'
EP = 'experiments/entropy-partial-mismatch-20261010/scripts/'
OLD = 'experiments/wcl-evidence-closure-20261009/scripts/'
# Exact locally reviewed engineering code, not a user-supplied command label.
BINDINGS = {
    "experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py": "12cad3e495fc91d8f6819b895674df08e312da3833291019b8966a2211ec8c75",
    "experiments/entropy-partial-mismatch-20261010/scripts/ep_plan.py": "7f354687caa3271ce4375f598f4d9e3ff6086df28d8be478c80835ab4ccd7862",
    "experiments/entropy-partial-mismatch-20261010/scripts/ep_source_codec.py": "65a9f599bef2205c15f864f40ec07f27d8c9d03ed2d318838706f29dd1b3cb4e",
    "experiments/entropy-partial-mismatch-20261010/scripts/test_ep_phy.py": "2619805128b3b552dd6846c1abe9e18ec0718fcf6c16e73d8ee1ca16c036bec1",
    "experiments/entropy-partial-mismatch-20261010/scripts/test_ep_plan.py": "8bca1c415fe3014152f291cd7feef993c9e21a070323ac845be104184fe8a5a6",
    "experiments/entropy-partial-mismatch-20261010/scripts/test_ep_source_codec.py": "2f4bc483a43df43edd875d1dc1e4134b68b86809a5c77e3d9be4f0cbaf4dd7d2",
    "experiments/wcl-evidence-closure-20261009/scripts/t1_codec_runtime.py": "245f0bc9a67cffbcc08e36d7942c5e555b144f400a860940f04da62a012b1d13",
    "experiments/wcl-evidence-closure-20261009/scripts/t1_entropy_core.py": "7cefad21134ac64c5eab5541afe4e188fdeaf343f374d8ea3eac966ff51901c9",
    "experiments/wcl-evidence-closure-20261009/scripts/t1_phy.py": "47f75430235c98432a513dbbcb492556d4e1977f3631f1806e648f558b1b583e",
    "src/var_comm/entropy.py": "f3adf956fba05d60e687472fa153e0bf684119ef6c8885a75fe4ea21962b2b2b",
    "src/var_comm/whole_entropy.py": "5d08d2f8c218f0bdf5b6d2a8ca7b3aa729baa75e274531afbfd8f31c95b24a5b"
}
CHECK_FILES = {
    'plan': [EP+'test_ep_plan.py', EP+'ep_plan.py'],
    'phy_format': [EP+'test_ep_phy.py', EP+'ep_phy.py', EP+'ep_plan.py', OLD+'t1_phy.py'],
    'source_fake_cdf': [EP+'test_ep_source_codec.py', EP+'ep_source_codec.py', EP+'ep_plan.py',
                        OLD+'t1_codec_runtime.py', OLD+'t1_entropy_core.py',
                        'src/var_comm/entropy.py', 'src/var_comm/whole_entropy.py'],
}
CACHE_NAMES = {
    'TMPDIR': 'tmp', 'TEMP': 'tmp', 'TMP': 'tmp',
    'XDG_CACHE_HOME': 'cache/xdg', 'HF_HOME': 'cache/huggingface',
    'HUGGINGFACE_HUB_CACHE': 'cache/huggingface/hub', 'TORCH_HOME': 'cache/torch',
    'MPLCONFIGDIR': 'cache/matplotlib', 'NUMBA_CACHE_DIR': 'cache/numba',
    'PIP_CACHE_DIR': 'cache/pip', 'UV_CACHE_DIR': 'cache/uv',
    'CUDA_CACHE_PATH': 'cache/cuda', 'TRITON_CACHE_DIR': 'cache/triton',
    'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions',
    'PYTHONPYCACHEPREFIX': 'cache/pycache',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def inside(path, base=PERSONAL_BASE):
    """Check lexical and resolved paths; no traversal or symlink escape."""
    p, b = Path(path), Path(base).absolute()
    require(p.is_absolute() and '..' not in p.parts, 'Absolute personal path without traversal required')
    require(p.is_relative_to(b) and p.resolve().is_relative_to(b.resolve()),
            'Path escapes the registered personal base: '+str(p))
    return p


def interpreter_path(value, base=PERSONAL_BASE):
    # Do not resolve the executable symlink: that would discard venv identity.
    p = Path(os.path.abspath(os.path.expanduser(value)))
    inside(p.parent, base)
    require(p.is_file() and p.parent.name == 'bin', 'Explicit personal venv/bin/python required')
    require((p.parent.parent/'pyvenv.cfg').is_file(), 'Personal Python venv identity required')
    return p


@contextlib.contextmanager
def owner_lock(path):
    """Locks only our own personal control file; never another user's locks."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if p.stat().st_size == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def parse_gpu(text, expected=GPU_UUID):
    rows = list(csv.reader(io.StringIO(text.strip())))
    require(len(rows) == 1 and len(rows[0]) == 6, 'Exactly one queried GPU record required')
    uuid, index, name, total, free, util = [x.strip() for x in rows[0]]
    require(uuid == expected, 'Selected GPU UUID differs; no automatic alternate device')
    total, free, util = float(total), float(free), float(util)
    require(all(math.isfinite(x) for x in (total, free, util)) and
            0 <= free <= total and 0 <= util <= 100, 'Invalid GPU resource reading')
    return dict(uuid=uuid, index=int(index), name=name, total_MiB=total, free_MiB=free,
                utilization_percent=util, shared_device=True, exclusivity_verified=False)


def cpu_snapshot(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    allowed = sorted(os.sched_getaffinity(0))
    quota, period = (cgroup/'cpu.max').read_text().split()
    quota_cores = None if quota == 'max' else int(quota)/int(period)
    effective = min(len(allowed), quota_cores) if quota_cores is not None else len(allowed)
    meminfo = {}
    for line in (proc/'meminfo').read_text().splitlines():
        k, v = line.split(':', 1)
        meminfo[k] = int(v.split()[0])*1024
    current = int((cgroup/'memory.current').read_text())
    maximum = (cgroup/'memory.max').read_text().strip()
    available = meminfo['MemAvailable']
    if maximum != 'max':
        available = min(available, max(0, int(maximum)-current))
    return dict(allowed_cpus=allowed, cgroup_quota_cores=quota_cores,
                effective_cpu_cores=effective, load1=float((proc/'loadavg').read_text().split()[0]),
                available_memory_bytes=available, cgroup_memory_current_bytes=current,
                cgroup_memory_max_bytes=None if maximum == 'max' else int(maximum))


def collect_snapshot(base=PERSONAL_BASE, gpu_uuid=GPU_UUID, runner=subprocess.run):
    require(gpu_uuid == GPU_UUID, 'Only the specifically registered GPU UUID is admitted')
    started = time.time()
    def counters():
        fields = Path('/proc/stat').read_text().splitlines()[0].split()
        require(fields[0] == 'cpu', 'Host CPU sample unavailable')
        values = [int(x) for x in fields[1:9]]  # Guest counters already included.
        return sum(values), values[3], values[4]
    before = counters()
    query = ['/usr/bin/nvidia-smi', '--id='+gpu_uuid,
             '--query-gpu=uuid,index,name,memory.total,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits']
    completed = runner(query, check=True, capture_output=True, text=True, timeout=10)
    if time.time()-started < .25:
        time.sleep(.25-(time.time()-started))
    after = counters()
    ticks = after[0]-before[0]
    require(ticks > 0, 'No advancing CPU utilization sample')
    cpu = cpu_snapshot()
    cpu.update(host_busy_percent=100*(ticks-(after[1]-before[1])-(after[2]-before[2]))/ticks,
               host_iowait_percent=100*(after[2]-before[2])/ticks,
               utilization_scope='host-wide sampled counters; loadavg retained as context')
    return dict(started_unix=started, completed_unix=time.time(),
                gpu=parse_gpu(completed.stdout, gpu_uuid), cpu=cpu,
                personal_filesystem_free_bytes=shutil.disk_usage(base).free,
                personal_quota_verified=False, gpu_query_argv=query,
                note='Fresh sampled metadata, not an allocation, peak-memory measurement or numerical qualification')


def check_resources(snapshot, threads, now=None):
    require(type(threads) is int and 2 <= threads <= 4, 'Two to four CPU threads only')
    now = time.time() if now is None else now
    require(0 <= now-snapshot['started_unix'] <= 15 and
            snapshot['started_unix'] <= snapshot['completed_unix'] <= now,
            'Resource snapshot stale or clock inconsistent')
    g, c = snapshot['gpu'], snapshot['cpu']
    require(g['uuid'] == GPU_UUID, 'Wrong GPU resource snapshot')
    # These commands never allocate on a GPU. Another user's GPU load does not
    # forbid CPU checks. This estimate never authorizes GPU qualification.
    gpu_headroom = g['free_MiB'] >= 20*1024 and g['utilization_percent'] <= 90
    require(c['effective_cpu_cores'] >= threads and len(c['allowed_cpus']) >= threads,
            'Insufficient allowed CPU capacity')
    require(math.isfinite(c['host_busy_percent']) and 0 <= c['host_busy_percent'] <= 95,
            'Sampled CPU pressure above bounded preparation threshold')
    require(c['available_memory_bytes'] >= 2*(1 << 30), 'Less than2GiB memory headroom')
    require(snapshot['personal_filesystem_free_bytes'] >= 10*(1 << 30),
            'Less than10GiB filesystem headroom; personal quota still needs separate confirmation')
    return dict(cpu_affinity=c['allowed_cpus'][:threads], threads=threads, workers=1,
                GPU_allocations_allowed=False, science_calls_allowed=False,
                sampled_GPU_headroom_estimate=gpu_headroom,
                GPU_headroom_does_not_block_CPU_only_checks=True,
                resource_thresholds_are_admission_estimates_not_peak_or_isolation_guarantees=True)


def controlled_environment(base, work, threads):
    env = dict(os.environ)
    for name in ('PYTHONHOME', 'PYTHONPATH', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV'):
        env.pop(name, None)
    controlled = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                  'PYTHONNOUSERSITE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        controlled[name] = str(threads)
    for name, suffix in CACHE_NAMES.items():
        p = inside(Path(work)/suffix, base)
        p.mkdir(parents=True, exist_ok=True)
        controlled[name] = str(p)
    env.update(controlled)
    return env, controlled


def bind_check(project, kind, base=PERSONAL_BASE):
    require(kind in CHECK_FILES, 'Only versioned CPU engineering checks, not arbitrary commands')
    project = inside(project, base)
    require(project.is_dir(), 'Existing personal project required')
    pins = {}
    for rel in CHECK_FILES[kind]:
        p = inside(project/rel, base)
        require(p.is_file() and not p.is_symlink(), 'Regular admitted engineering file required')
        require(sha(p) == BINDINGS[rel], 'Engineering code changed; prepare a new reviewed version: '+rel)
        pins[str(p)] = BINDINGS[rel]
    return project/CHECK_FILES[kind][0], pins


def child_limits(cpus, seconds, threads):
    def apply():
        import resource
        os.sched_setaffinity(0, set(cpus))  # Only the newly owned child.
        resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(seconds*threads), math.ceil(seconds*threads)))
        os.nice(10)  # Child only; no host/global scheduler changes.
    return apply


def wait_owned(child, timeout):
    """Stop/wait only the Popen child this invocation created. Never retry."""
    reason = None
    try:
        code = child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        reason = 'ENGINEERING_DEADLINE_EXCEEDED'
        child.terminate()
        try:
            code = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill(); code = child.wait()
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        raise
    return dict(actual_child_waited=True, child_exit_code=code, stop_reason=reason,
                success=code == 0 and reason is None, automatic_retry=False)


def execute(args):
    require(sys.platform.startswith('linux'), 'Actual entry requires Linux; local tests use injected metadata only')
    base = PERSONAL_BASE
    require(base.is_dir(), 'Registered personal base must already exist')
    out = inside(Path(args.out), base)
    require(out != base and not out.exists(), 'Fresh personal output directory required')
    require(2 <= args.threads <= 4 and 1 <= args.timeout <= 600, 'Bounded threads/time only')
    with owner_lock(inside(base/'controls/leo_shared_preflight_v1/owner.lock', base)):
        require(not out.exists(), 'Run directory already claimed')
        out.mkdir(parents=True)
        save(out/'intent.json', dict(schema=SCHEMA, kind=args.command, owner_pid=os.getpid(),
             started_unix=time.time(), tool_sha256=sha(__file__), gpu_uuid=GPU_UUID,
             threads=args.threads, workers=1, max_seconds=args.timeout,
             science_calls_allowed=False, qualification=False, automatic_successor=False))
        child = None
        try:
            snapshot = collect_snapshot(base)
            admission = check_resources(snapshot, args.threads)
            save(out/'resource_snapshot.json', snapshot)
            save(out/'admission.json', admission)
            if args.command == 'preflight':
                result = dict(status='PASS_METADATA_PREFLIGHT_ONLY', schema=SCHEMA,
                              child_started=False, science_calls=0, numerical_qualification=False)
            else:
                executable = interpreter_path(args.python, base)
                script, pins = bind_check(Path(args.project_root), args.check, base)
                env, controlled = controlled_environment(base, out, args.threads)
                env['VIRTUAL_ENV'] = str(executable.parent.parent)
                save(out/'engineering_inputs.json', dict(check=args.check, inputs=pins,
                     python=str(executable), python_binary_sha256=sha(executable), controlled_environment=controlled))
                # A changed input/headroom between preparation and launch stops.
                script, current = bind_check(Path(args.project_root), args.check, base)
                require(current == pins, 'Engineering bindings changed before launch')
                latest = collect_snapshot(base)
                admission = check_resources(latest, args.threads)
                save(out/'launch_resource_snapshot.json', latest)
                argv = [str(executable), '-B', '-u', str(script)]
                with (out/'child.log').open('x', encoding='utf-8') as log:
                    child = subprocess.Popen(argv, cwd=args.project_root, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        preexec_fn=child_limits(admission['cpu_affinity'], args.timeout, args.threads))
                    save(out/'child_started.json', dict(pid=child.pid, argv=argv,
                         started_unix=time.time(), cpu_affinity=admission['cpu_affinity'],
                         CUDA_VISIBLE_DEVICES='', original_selected_gpu_uuid=GPU_UUID))
                    waited = wait_owned(child, args.timeout)
                save(out/'actual_child_wait.json', waited)
                require(waited['success'], 'Engineering child failed/OOM/timed out; stop and retain evidence')
                require(all(sha(p) == s for p, s in pins.items()), 'Engineering source changed during execution')
                result = dict(status='PASS_ALLOWLISTED_CPU_ENGINEERING_ONLY', schema=SCHEMA,
                    check=args.check, **waited, science_calls=0, GPU_allocations=0,
                    numerical_qualification=False, production_owner_replacement=False)
            result.update(completed_unix=time.time(), automatic_successor=False,
                          memory_isolation_guaranteed=False, GPU_exclusive=False)
            save(out/'completion.json', result)
            return result
        except BaseException as error:
            # Includes failure while writing the launch receipt; no orphan child.
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            save(out/'failure.json', dict(schema=SCHEMA, status='STOPPED_NO_AUTOMATIC_RETRY',
                 exception_type=type(error).__name__, error=str(error), stopped_unix=time.time(),
                 child_started=child is not None, child_exit_code=None if child is None else child.returncode,
                 science_calls_allowed=False, concurrency_expansion=False))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'run-engineering'):
        s = subs.add_parser(name)
        s.add_argument('--out', required=True)
        s.add_argument('--threads', type=int, choices=(2, 3, 4), default=2)
        s.add_argument('--timeout', type=int, default=120)
        if name == 'run-engineering':
            s.add_argument('--python', required=True)
            s.add_argument('--project-root', required=True)
            s.add_argument('--check', choices=tuple(CHECK_FILES), required=True)
    result = execute(p.parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
