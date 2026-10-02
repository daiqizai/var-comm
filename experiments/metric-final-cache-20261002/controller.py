"""CPU guardian for a sealed cache; never signal the scientific M2 queue."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

NAME = 'METRIC-FINAL-CACHE-20261002'
PRIOR = 'METRIC-CONCURRENT-R4-20261002'
METRICS = 'UNIFIED-METRICS-20261002'
PARENT = 'SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def sources(folder):
    paths = sorted(p for p in Path(folder).resolve().iterdir() if p.suffix in ('.py', '.md'))
    if not paths or any(not p.is_file() or p.is_symlink() for p in paths):
        raise RuntimeError('Invalid frozen runtime inventory')
    return {str(p): sha(p) for p in paths}


def verify(mapping):
    if not isinstance(mapping, dict) or not mapping: raise RuntimeError('Missing frozen bindings')
    for path, digest in mapping.items():
        if not re.fullmatch('[0-9a-f]{64}', str(digest)) or sha(path) != digest:
            raise RuntimeError('Frozen file changed: ' + path)


def identity(value):
    if (not isinstance(value, dict) or type(value.get('pid')) is not int or value['pid'] <= 1
            or not str(value.get('start_ticks', '')).isdigit()):
        raise RuntimeError('Invalid process identity')
    return dict(pid=value['pid'], start_ticks=str(value['start_ticks']))


def process(pid):
    folder = Path('/proc') / str(int(pid))
    try:
        raw = (folder / 'stat').read_text(); values = raw[raw.rfind(') ') + 2:].split()
        command = (folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (FileNotFoundError, ProcessLookupError): return None
    return dict(pid=int(pid), start_ticks=values[19], state=values[0], ppid=int(values[1]), command=command)


def alive(expected, actual):
    return actual is not None and actual.get('state') != 'Z' and identity(actual) == identity(expected)


def lock(path):
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a')
    try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException: stream.close(); raise
    return stream


def published(record):
    if (record.get('status') != 'PUSHED' or record.get('checks') != 'PASS'
            or not re.fullmatch('[0-9a-f]{40}', str(record.get('commit', '')))
            or record.get('commit') != record.get('remote_commit')):
        raise RuntimeError('Checked normal publication required')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value); return value


def retire_idle(expected, reader, recheck, sleep=time.sleep, now=time.time):
    """Stop, recheck the genuine idle owner, then retire this one pidfd only."""
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support required')
    fd = os.pidfd_open(expected['pid'], 0); stopped = False; terminated = False
    try:
        if not alive(expected, reader(expected['pid'])): raise RuntimeError('Nominated owner identity changed')
        recheck()
        signal.pidfd_send_signal(fd, signal.SIGSTOP, None, 0); stopped = True
        deadline = now() + 10
        while True:
            actual = reader(expected['pid'])
            if not alive(expected, actual): raise RuntimeError('Idle owner exited during handoff')
            if actual['state'] in ('T', 't'): break
            if now() >= deadline: raise RuntimeError('Idle owner did not stop')
            sleep(.05)
        recheck()  # A child or final-publication transition invalidates retirement.
        signal.pidfd_send_signal(fd, signal.SIGTERM, None, 0); terminated = True
        try: signal.pidfd_send_signal(fd, signal.SIGCONT, None, 0)
        except ProcessLookupError: pass
    finally:
        if stopped and not terminated:
            try: signal.pidfd_send_signal(fd, signal.SIGCONT, None, 0)
            except ProcessLookupError: pass
        os.close(fd)


class Controller:
    def __init__(self, root, here=None, *, reader=process, sleep=time.sleep, now=time.time,
                 lock_fn=lock, spawn=subprocess.Popen, retire_fn=retire_idle, pid=None, publisher=None):
        self.root = Path(root).resolve(); self.here = Path(here or Path(__file__).parent).resolve()
        self.out = self.root / 'outputs' / NAME; self.prior = self.root / 'outputs' / PRIOR
        self.metrics = self.root / 'outputs' / METRICS; self.parent = self.root / 'outputs' / PARENT
        self.original = self.root / 'experiments/unified-metrics-20261002'
        self.reader, self.sleep, self.now = reader, sleep, now
        self.lock_fn, self.spawn, self.retire_fn = lock_fn, spawn, retire_fn
        self.pid = os.getpid() if pid is None else pid; self.publisher = publisher
        self.owns = False; self.worker = None; self.owner_lock = None

    def status(self, state, **fields):
        value = dict(status=state, pid=self.pid, time=self.now(), worker_pid=None)
        value.update(fields)
        write(self.out / 'controller_status.json', value)
        if self.owns: write(self.metrics / 'supervisor_status.json', value)

    def initialize(self):
        self.bound = sources(self.here); self.oldbound = sources(self.prior / 'runtime')
        self.original_bound = sources(self.original)
        self.hand = read(self.out / 'handoff.json'); self.old = identity(self.hand['previous_owner'])
        if self.old['pid'] == self.pid or self.hand.get('new_source_bindings') != self.bound:
            raise RuntimeError('New guardian handoff/source differs')
        regpath = self.prior / 'runtime_registration.json'; launchpath = self.prior / 'controller_launch.json'
        previous = read(regpath); previous_launch = read(launchpath)
        if (previous.get('runtime_source_bindings') != self.oldbound
                or previous_launch.get('source_bindings') != self.oldbound
                or identity(previous_launch) != self.old
                or previous_launch.get('runtime_registration_sha256') != sha(regpath)
                or sha(regpath) != self.hand.get('previous_runtime_registration_sha256')
                or sha(launchpath) != self.hand.get('previous_launch_sha256')
                or previous.get('original_metric_source_bindings') != self.original_bound):
            raise RuntimeError('Frozen R4 owner/source lineage differs')
        original_reg = self.metrics / 'supervisor_registration.json'
        if (read(original_reg).get('source_bindings') != self.original_bound
                or previous.get('original_registration_sha256') != sha(original_reg)):
            raise RuntimeError('Original metric registration differs')
        self.regpath = self.out / 'runtime_registration.json'
        self.registration = dict(status='REGISTERED_FINAL_CACHE_GUARDIAN', runtime_source_bindings=self.bound,
            source_bindings=self.bound, original_metric_source_bindings=self.original_bound,
            previous_runtime_source_bindings=self.oldbound, previous_owner=self.old,
            previous_runtime_registration_sha256=sha(regpath), previous_launch_sha256=sha(launchpath),
            handoff_sha256=sha(self.out / 'handoff.json'), original_registration_sha256=sha(original_reg),
            training_updates=0, policy_selection_updates=0, scientific_m2_process_unchanged=True,
            scope='verified_cache_assembly_then_original_analysis_and_publication', stop_after_scope=True)
        qualification = self.out / 'real_cached_row_qualification.json'
        if qualification.exists():
            self.registration['real_cached_row_qualification_bindings'] = {str(qualification): sha(qualification)}
        if self.regpath.exists():
            if read(self.regpath) != self.registration: raise RuntimeError('Guardian registration changed on resume')
        else: write(self.regpath, self.registration)
        prior_status = self.out / 'controller_status.json'
        if prior_status.exists():
            prior = read(prior_status)
            if prior.get('worker_pid') is not None:
                expected = dict(pid=prior['worker_pid'], start_ticks=prior.get('worker_start_ticks'))
                if alive(expected, self.reader(expected['pid'])): raise RuntimeError('An orphan final worker remains active')
        assets = read(self.metrics / 'assets_complete.json')
        if assets.get('status') != 'ASSETS_READY' or assets.get('manifest_sha256') != sha(self.metrics / 'modelmanifest.json'):
            raise RuntimeError('Original metric assets are not sealed')
        self.python = assets['environment']
        if self.publisher is None:
            self.publisher = load_module('_final_cache_prior_publisher', self.prior / 'runtime/publish_extension.py')
        self.me = identity(self.reader(self.pid))
        launch = dict(status='DETACHED_CPU_GUARDIAN', **self.me, source_bindings=self.bound,
            runtime_registration_sha256=sha(self.regpath), command=[sys.executable, '-u', str(self.here / 'controller.py'), '--root', str(self.root)])
        history = self.out / 'launch_history' / f'{self.pid}.json'
        launch['history_path'] = str(history)
        if history.exists() and read(history) != launch: raise RuntimeError('Guardian launch history changed')
        write(history, launch)
        write(self.out / 'controller_launch.json', launch)
        self.verify()

    def verify(self):
        verify(self.bound); verify(self.oldbound); verify(self.original_bound)
        if self.registration.get('real_cached_row_qualification_bindings'):
            verify(self.registration['real_cached_row_qualification_bindings'])
        if (sources(self.here) != self.bound or sources(self.prior / 'runtime') != self.oldbound
                or read(self.regpath) != self.registration
                or sha(self.out / 'handoff.json') != self.registration['handoff_sha256']
                or sha(self.metrics / 'supervisor_registration.json') != self.registration['original_registration_sha256']
                or sha(self.prior / 'runtime_registration.json') != self.registration['previous_runtime_registration_sha256']
                or sha(self.prior / 'controller_launch.json') != self.registration['previous_launch_sha256']):
            raise RuntimeError('Frozen guardian/previous/original registration changed')

    def idle_ready(self):
        state = read(self.prior / 'controller_status.json')
        if state.get('status') == 'FAILED': raise RuntimeError('R4 metric controller failed')
        snap_path, status_path = self.prior / 'partial_completion.json', self.prior / 'status.json'
        if not snap_path.exists() or not status_path.exists(): return False
        snap, status = read(snap_path), read(status_path)
        return (state.get('status') == 'WAITING_FOR_ORIGINAL_COMPLETION' and state.get('pid') == self.old['pid']
            and state.get('worker_pid') is None and snap.get('sources_complete') == 100
            and type(snap.get('sources_complete')) is int and status.get('status') == 'SEALED_STUDIES_COMPLETE')

    def idle_proof(self):
        if not self.idle_ready(): raise RuntimeError('R4 must have 100 sealed sources and a genuinely idle owner')
        live = self.publisher.process_snapshot()
        peers = self.scientific_peers(live)
        # This exemption belongs only to the cache/idle check. The unchanged
        # publication gate later requires these scientific processes to exit.
        metric_processes = {pid: p for pid, p in live.items() if pid not in peers}
        bound, _ = self.publisher.concurrent_gate(self.prior, self.prior / 'runtime',
            self.prior / 'concurrent_registration.json', self.prior / 'partial_completion.json', metric_processes,
            controller_pid=self.old['pid'], controller_ticks=self.old['start_ticks'])
        if bound != self.oldbound: raise RuntimeError('R4 full cache/source differs')
        launch = read(self.metrics / 'supervisor_launch.json'); state = read(self.metrics / 'supervisor_status.json')
        prior_state = read(self.prior / 'controller_status.json')
        if identity(launch) != self.old or any(state.get(k) != prior_state.get(k) for k in ('pid', 'status', 'worker_pid')):
            raise RuntimeError('Idle R4 owner does not own the original metric queue')
        paths = [self.prior / n for n in ('concurrent_registration.json', 'partial_completion.json',
            'controller_launch.json', 'controller_status.json', 'runtime_registration.json', 'admission.json', 'partial_launch.json', 'status.json')]
        return {str(p): sha(p) for p in paths}

    def scientific_peers(self, processes):
        """Admit only the frozen R4 M2 owner and its registered current child."""
        launch_path = self.prior / 'm2_controller_launch.json'
        owner = read(launch_path); actual = processes.get(owner['pid'])
        if not alive(owner, dict(actual, pid=owner['pid']) if actual is not None else None): return set()
        admission_path = self.prior / 'admission.json'; admission = read(admission_path)
        original = self.root / 'experiments/metric-speed-20261002/controller.py'
        handoff_path = self.prior / 'm2_handoff.json'
        parent_launch = read(self.parent / 'supervisor_launch.json'); state = read(self.parent / 'supervisor_status.json')
        command = actual.get('command', '').replace('\\', '/')
        registered_command = [str(v).replace('\\', '/') for v in owner.get('command', [])]
        if (owner.get('source_bindings') != self.oldbound or admission.get('source_bindings') != self.oldbound
                or owner.get('admission_sha256') != sha(admission_path)
                or owner.get('handoff_sha256') != sha(handoff_path)
                or owner.get('scientific_controller') != str(original)
                or owner.get('scientific_controller_sha256') != sha(original)
                or admission.get('original_execution_bindings', {}).get(str(original)) != sha(original)
                or identity(parent_launch) != identity(owner) or state.get('pid') != owner['pid']
                or parent_launch.get('scheduling_extension_launch') != str(launch_path)
                or parent_launch.get('scheduling_extension_launch_sha256') != sha(launch_path)
                or parent_launch.get('scheduling_only') is not True
                or shlex.split(command) != registered_command
                or registered_command[2:] != [(self.prior / 'runtime/m2_controller.py').as_posix(), '--root', self.root.as_posix()]):
            raise RuntimeError('Live R4 scientific owner identity/source/command lineage differs')
        peers = {owner['pid']}; worker_pid = state.get('worker_pid')
        worker = processes.get(worker_pid)
        if worker is None or worker.get('state') == 'Z': return peers
        entry = (self.prior / 'runtime/scheduled_process.py').as_posix()
        worker_command = shlex.split(worker.get('command', '').replace('\\', '/'))
        if entry not in worker_command: return peers  # Unchanged CPU jobs are not runtime exclusions.
        lease = read(self.prior / 'active_m2.json')
        expected = dict(pid=worker_pid, start_ticks=str(state.get('worker_start_ticks', '')))
        enriched_worker = self.reader(worker_pid)
        stage = {'m2_calibration': 'calibration', 'm2_evaluation': 'evaluation', 'm2_actual': 'actual', 'm2_timing': 'timing'}.get(state.get('stage'))
        if (lease.get('status') != 'ACTIVE' or lease.get('role') != 'm2'
                or identity(lease) != identity(expected) or identity(lease.get('owner')) != identity(owner)
                or str(worker.get('start_ticks')) != expected['start_ticks']
                or not alive(expected, enriched_worker) or enriched_worker.get('ppid') != owner['pid']
                or enriched_worker.get('command', '').replace('\\', '/') != worker.get('command', '').replace('\\', '/')
                or lease.get('source_bindings') != self.oldbound or lease.get('admission_sha256') != sha(admission_path)
                or stage is None or lease.get('exclusive') is not (stage == 'timing')
                or worker_command[2:] != [Path(entry).as_posix(), '--root', self.root.as_posix(), '--admission', admission_path.as_posix(), '--stage', stage]):
            raise RuntimeError('Live R4 scientific worker identity/source/command lineage differs')
        peers.add(worker_pid); return peers

    def takeover(self):
        path = self.out / 'r4_retirement.json'
        if not path.exists():
            while not self.idle_ready():
                self.verify(); self.status('WAITING_FOR_R4_FULL_CACHE', r4_owner=self.old)
                if not alive(self.old, self.reader(self.old['pid'])): raise RuntimeError('R4 owner exited before the registered idle boundary')
                self.sleep(10)
            proof = self.idle_proof()
            heartbeat = str(self.prior / 'controller_status.json')
            stable = {p: h for p, h in proof.items() if p != heartbeat}
            def recheck():
                nonlocal proof
                self.verify()
                current = self.idle_proof()
                if {p: h for p, h in current.items() if p != heartbeat} != stable:
                    raise RuntimeError('Idle cache/owner changed during retirement')
                actual = self.reader(self.old['pid'])
                if actual is not None and actual.get('state') in ('T', 't'): proof = current
            self.retire_fn(self.old, self.reader, recheck, self.sleep, self.now)
            deadline = self.now() + 30
            while alive(self.old, self.reader(self.old['pid'])):
                if self.now() >= deadline: raise RuntimeError('Idle R4 metric owner did not exit')
                self.sleep(.1)
            verify(proof)
            archive = self.out / 'retired_r4_owner'; archive.mkdir(parents=True, exist_ok=True)
            for source in list(proof) + [str(self.metrics / n) for n in ('supervisor_launch.json', 'supervisor_status.json')]:
                target = archive / (('metrics_' if Path(source).parent == self.metrics else 'r4_') + Path(source).name)
                data = Path(source).read_bytes()
                if target.exists() and target.read_bytes() != data: raise RuntimeError('Retired owner evidence differs')
                target.write_bytes(data)
            write(path, dict(status='R4_IDLE_METRIC_OWNER_RETIRED', oldowner=self.old, proof_bindings=proof,
                original_scientific_owner_untouched=True, sources_complete=100, runtime_registration_sha256=sha(self.regpath)))
        retirement = read(path); verify(retirement['proof_bindings'])
        if retirement.get('status') != 'R4_IDLE_METRIC_OWNER_RETIRED' or retirement.get('oldowner') != self.old:
            raise RuntimeError('R4 retirement receipt differs')
        if alive(self.old, self.reader(self.old['pid'])): raise RuntimeError('Retired metric owner is still alive')
        self.owner_lock = self.lock_fn(self.metrics / 'supervisor.lock')
        previous = read(self.metrics / 'supervisor_launch.json')
        if identity(previous) != self.old and alive(previous, self.reader(previous['pid'])):
            raise RuntimeError('A different metric queue owner remains alive')
        launch = read(self.out / 'controller_launch.json')
        write(self.metrics / 'supervisor_launch.json', dict(launch, source_bindings=self.original_bound,
            runtime_source_bindings=self.bound, final_cache_extension=True))
        self.owns = True; self.status('FINAL_CACHE_OWNER_READY')

    def parent_ready(self):
        path = self.parent / 'completion.json'
        if not path.exists(): return False
        if read(path).get('status') != 'AUTHORIZED_TWO_METHODS_COMPLETE': raise RuntimeError('Unexpected scientific completion')
        for folder, names in ((self.parent, ('supervisor_launch.json',)),
                (self.prior, ('m2_controller_launch.json', 'partial_launch.json', 'controller_launch.json'))):
            for name in names:
                launch = read(folder / name)
                if alive(launch, self.reader(launch['pid'])): return False
        self.publisher.parent_gate(self.root, self.publisher.process_snapshot())
        marker = (self.prior / 'runtime').as_posix() + '/'
        if any(p.get('state') != 'Z' and marker in p.get('command', '').replace('\\', '/')
               for p in self.publisher.process_snapshot().values()): return False
        return True

    def environment(self, cpu):
        env = dict(os.environ, OMP_NUM_THREADS='2' if cpu else '6', OPENBLAS_NUM_THREADS='2')
        env['PYTHONPATH'] = os.pathsep.join(str(self.root / p) for p in (
            'experiments/rx-posterior-step1-20260929', 'src', 'experiments/var-latent-enhancement-20260917/src',
            'experiments/var-latent-enhancement-20260917/phase_b/src', 'experiments/var-latent-enhancement-20260917/evaluation/src',
            'experiments/var-latent-enhancement-20260917/followup/src', 'experiments/var-latent-enhancement-20260917/research/src',
            'experiments/var-latent-enhancement-20260917/mechanisms/src', 'experiments/var-short-prefix-hybrid-20260923/src',
            'experiments/token_channel_efficiency_20260923/src'))
        if cpu: env.update(CUDA_VISIBLE_DEVICES='', MKL_NUM_THREADS='2')
        return env

    def job(self, name, script, args=(), cpu=True):
        self.verify(); log = self.out / f'{name}_{time.time_ns()}.log'
        with log.open('a') as stream:
            worker = self.spawn([self.python, '-u', str(script), *map(str, args)], cwd=self.root,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, env=self.environment(cpu))
            self.worker = worker; actual = self.reader(worker.pid)
            if actual is None: raise RuntimeError('Cannot bind final child identity')
            self.worker_identity = identity(actual)
            self.status('RUNNING', stage=name, worker_pid=worker.pid, worker_start_ticks=actual['start_ticks'], log=str(log))
            code = worker.wait(); self.worker = None
        self.verify()
        if code != 0: raise RuntimeError(f'{name} failed ({code}): {log}')

    def final(self):
        while not self.parent_ready():
            self.verify(); self.status('WAITING_FOR_ORIGINAL_COMPLETION'); self.sleep(15)
        prior_pub = self.prior / 'extension_source_publication.json'
        if not prior_pub.exists() or read(prior_pub).get('status') != 'PUSHED':
            self.job('publish_r4_extension', self.prior / 'runtime/publish_extension.py', ['--root', self.root])
        published(read(prior_pub)); verify(read(prior_pub)['source_bindings'])
        if read(prior_pub).get('runtime_source_bindings') != self.oldbound: raise RuntimeError('R4 published inventory differs')
        new_pub = self.out / 'extension_source_publication.json'
        if not new_pub.exists() or read(new_pub).get('status') != 'PUSHED':
            self.job('publish_final_cache_extension', self.here / 'publish_extension.py',
                ['--root', self.root, '--controller-pid', self.pid, '--controller-start-ticks', self.me['start_ticks']])
        published(read(new_pub)); verify(read(new_pub)['source_bindings'])
        if read(new_pub).get('runtime_source_bindings') != self.bound: raise RuntimeError('Final cache publication source differs')
        old_pub = self.metrics / 'source_publication.json'
        if not old_pub.exists() or read(old_pub).get('status') != 'PUSHED':
            self.job('publish_original_metric_source', self.original / 'publish.py', ['--phase', 'source'])
        published(read(old_pub)); verify(read(old_pub)['source_bindings'])
        if read(old_pub)['source_bindings'] != self.original_bound: raise RuntimeError('Original metric publication source differs')
        completion = self.metrics / 'scoring_completion.json'
        if not completion.exists():
            while True:
                self.verify()
                gpu = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
                if not [x for x in gpu.splitlines() if x.strip().isdigit()]: break
                self.status('WAITING_FOR_FREE_GPU'); self.sleep(10)
            self.job('assemble_verified_cache_and_replay_uncached', self.here / 'final_runner.py',
                ['--root', self.root, '--cache-dir', self.prior], cpu=False)
        scoring = read(completion); expected = dict(self.original_bound, **self.oldbound, **self.bound)
        if (scoring.get('status') != 'COMPLETE' or scoring.get('synthetic') is not False
                or scoring.get('training_updates') != 0 or scoring.get('policy_selection_updates') != 0
                or scoring.get('parity_passed') is not True or scoring.get('source_bindings') != expected):
            raise RuntimeError('Final assembled metric coverage/parity/source differs')
        for field in ('inputs', 'outputs'): verify(scoring[field])
        result = self.root / 'results/unified_metrics_20261002'; analysis_path = result / 'metrics_analysis_completion.json'
        if not analysis_path.exists(): self.job('original_paired_analysis', self.original / 'analysis.py', ['--results-dir', result])
        analysis = read(analysis_path)
        if analysis.get('status') != 'COMPLETE': raise RuntimeError('Metric analysis incomplete')
        for field in ('inputs', 'outputs'): verify(analysis[field])
        final_path = self.metrics / 'completion.json'
        if not final_path.exists(): self.job('original_result_publication', self.original / 'publish.py', ['--phase', 'results'])
        final = read(final_path)
        if final.get('status') != 'UNIFIED_METRICS_COMPLETE': raise RuntimeError('Metric final delivery incomplete')
        published(final['publication']); self.verify()
        self.status('COMPLETE', stopped_after_authorized_scope=True)
        write(self.out / 'controller_completion.json', dict(status='COMPLETE', **self.me,
            metrics_completion_sha256=sha(final_path), source_bindings=self.bound,
            training_updates=0, policy_selection_updates=0, stopped_after_authorized_scope=True))

    def run(self):
        own = self.lock_fn(self.out / 'controller.lock')
        try:
            self.initialize(); self.takeover(); self.final()
        except BaseException as error:
            self.status('FAILED', error=str(error), worker_pid=None if self.worker is None else self.worker.pid,
                worker_start_ticks=None if self.worker is None else getattr(self, 'worker_identity', {}).get('start_ticks'))
            raise
        finally:
            if self.owner_lock is not None: self.owner_lock.close()
            own.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    Controller(parser.parse_args().root).run()


if __name__ == '__main__': main()
