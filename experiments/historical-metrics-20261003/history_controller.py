"""Serial, publication-gated continuation for completed historical methods.

This controller never sends signals to an earlier owner or invokes an old
scientific main. It launches only this immutable bundle's scoring/analysis and
publication entries, after checked original delivery and process exit.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from history_common import read, write, sha, verify, identity, source_bindings, proc, alive

NAME = 'HISTORICAL-METRICS-20261003'
NUMERIC = 'METRIC-NUMERIC-RECOVERY-20261003'


class NotReady(RuntimeError):
    pass


def published(value):
    if (value.get('status') != 'PUSHED' or value.get('checks') != 'PASS'
            or not re.fullmatch('[0-9a-f]{40}', str(value.get('commit', '')))
            or value.get('remote_commit') != value['commit']):
        raise RuntimeError('Checked normal push and exact remote SHA required')


def process_snapshot():
    directory = Path('/proc')
    if not directory.is_dir():
        raise RuntimeError('Linux PID/start-tick evidence is required')
    result = {}
    for path in directory.iterdir():
        if path.name.isdecimal():
            try:
                record = proc(int(path.name))
            except PermissionError:
                record = {'pid': int(path.name), 'unreadable': True}
            if record is not None:
                result[int(path.name)] = record
    return result


def require_exited(record, processes, label):
    if (type(record.get('pid')) is not int or record['pid'] <= 1
            or not str(record.get('start_ticks', '')).isdigit()):
        raise RuntimeError('Missing exact process identity: ' + label)
    actual = processes.get(record['pid'])
    if actual and actual.get('unreadable'):
        raise RuntimeError('Unable to inspect registered owner: ' + label)
    if (actual and actual.get('state') != 'Z'
            and str(actual.get('start_ticks')) == str(record['start_ticks'])):
        raise NotReady('Waiting for original owner exit: ' + label)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent_gate(root, processes=None, *, base_gate=None):
    """Validate actual numeric publication, then the already published full gate."""
    root = Path(root).resolve()
    processes = process_snapshot() if processes is None else processes
    folder = root/'outputs'/NUMERIC
    paths = [folder/n for n in ('publication.json', 'publication_status.json', 'delivery_launch.json')]
    if not all(p.is_file() for p in paths):
        raise NotReady('Waiting for original numeric recovery publication')
    publication, status, launch = map(read, paths)
    if publication.get('status') != 'PUSHED' or status.get('status') != 'COMPLETE':
        raise NotReady('Waiting for completed numeric normal push')
    published(publication)
    if (status.get('publication') != publication or status.get('stop') is not True
            or status.get('pid') != launch.get('pid')):
        raise RuntimeError('Numeric completion/publication/owner identity differs')
    for field in ('published_files', 'inputs', 'source_bindings'):
        if not publication.get(field):
            raise RuntimeError('Numeric publication lacks ' + field)
        verify(publication[field])
    require_exited(launch, processes, 'numeric delivery')
    # Reuse the exact already published, CPU-only parent gate. It checks unified
    # completion, both scientific supervisors, final guardian, R4 owners and
    # remaining old workers. Never invoke that publisher's main function.
    previous = root/'outputs/M2-JSON-RECOVERY-20261002/publication.json'
    helper = root/'experiments/m2-json-recovery-20261002/publish_when_complete.py'
    record = read(previous); published(record)
    if record.get('source_bindings', {}).get(str(helper)) != sha(helper):
        raise RuntimeError('Original full completion gate is not published/bound')
    if base_gate is None:
        base = load_module(helper, '_historical_original_completion_gate')
        try:
            completed = base.completion_gate(root, processes)
        except base.NotReady as error:
            raise NotReady(str(error)) from error
    else:
        completed = base_gate(root, processes)  # CPU fixture injection only.
    verify(completed['inputs'])
    unified = root/'outputs/UNIFIED-METRICS-20261002/completion.json'
    done = read(unified)
    if done.get('status') != 'UNIFIED_METRICS_COMPLETE':
        raise NotReady('Waiting for original unified metrics completion')
    published(done['publication'])
    allpaths = [*paths, previous, helper, unified]
    return {'status': 'ORIGINAL_DELIVERY_PUSHED_AND_ALL_OWNERS_EXITED',
            'inputs': {**completed['inputs'], **{str(p): sha(p) for p in allpaths}},
            'commits': list(dict.fromkeys([*completed['commits'], publication['commit'], record['commit']])),
            'original_processes_signalled': False}


def gpu_pids():
    value = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], text=True)
    values = [s.strip() for s in value.splitlines() if s.strip()]
    if any(not s.isdecimal() for s in values):
        raise RuntimeError('Unreadable GPU ownership query')
    return [int(s) for s in values]


def validate_queue(root, runtime, path):
    root, runtime, path = Path(root).resolve(), Path(runtime).resolve(), Path(path).resolve()
    queue = read(path)
    if (queue.get('status') != 'REGISTERED' or queue.get('training_updates') != 0
            or queue.get('policy_selection_updates') != 0):
        raise RuntimeError('Immutable zero-training historical queue required')
    if source_bindings(runtime) != queue.get('source_bindings'):
        raise RuntimeError('Historical runtime inventory changed')
    for field in ('source_bindings', 'shared_metric_bindings', 'input_proof_bindings'):
        if not isinstance(queue.get(field), dict) or not queue[field]:
            raise RuntimeError('Explicit queue proof map required: ' + field)
        verify(queue[field])
    coverage = queue.get('coverage_manifest', {})
    if set(coverage) != {'path', 'sha256'} or sha(coverage['path']) != coverage['sha256']:
        raise RuntimeError('Immutable historical coverage manifest required')
    jobs = queue.get('jobs')
    if not isinstance(jobs, list) or not jobs or len({j.get('study') for j in jobs}) != len(jobs):
        raise RuntimeError('Nonempty unique historical study roster required')
    for job in jobs:
        if (set(job) != {'study', 'adapter'} or not re.fullmatch('[A-Z][A-Z0-9_]*', job['study'])
                or not re.fullmatch('historical_[a-z0-9_]+', job['adapter'])):
            raise RuntimeError('Unsafe or unreviewed historical job')
        adapter = runtime/(job['adapter']+'.py')
        if str(adapter) not in queue['source_bindings']:
            raise RuntimeError('Adapter is not in the frozen runtime')
    if 'execution_groups' in queue:
        groups = queue['execution_groups']
        if (set(groups) != {'required', 'analysis_supplements'} or not groups['required']
                or groups['required'] + groups['analysis_supplements'] != [j['study'] for j in jobs]):
            raise RuntimeError('All required studies must precede the authorized analysis supplements')
    if not isinstance(queue.get('contrasts'), list):
        raise RuntimeError('Explicit predeclared contrasts required, possibly empty')
    original = root/'experiments/unified-metrics-20261002'
    for name in ('runner.py', 'replay.py', 'metric_models.py', 'batch_speed.py', 'publish.py'):
        if str(original/name) not in queue['shared_metric_bindings']:
            raise RuntimeError('Original shared evaluator/helper is not registered: ' + name)
    return queue


def lock(path):
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close(); raise
    return handle


class Controller:
    def __init__(self, root, queue_path, *, runtime=None, poll_seconds=30, once=False):
        self.root = Path(root).resolve(); self.here = Path(runtime or Path(__file__).parent).resolve()
        self.queue_path = Path(queue_path).resolve(); self.out = self.root/'outputs'/NAME
        self.result = self.root/'results/historical_metrics_20261003'
        self.poll_seconds, self.once = poll_seconds, once
        self.queue = validate_queue(self.root, self.here, self.queue_path)
        self.queue_sha = sha(self.queue_path); self.worker = None
        self.out.mkdir(parents=True, exist_ok=True)

    def status(self, state, **extra):
        write(self.out/'controller_status.json', dict(status=state, pid=os.getpid(),
              queue_registration_sha256=self.queue_sha, time=time.time(), **extra))

    def verify(self):
        if sha(self.queue_path) != self.queue_sha:
            raise RuntimeError('Queue registration changed')
        validate_queue(self.root, self.here, self.queue_path)

    def wait_gate(self):
        while True:
            self.verify()
            try:
                gate = parent_gate(self.root)
                pids = gpu_pids()
                if pids:
                    raise NotReady('Waiting for free GPU; current compute PIDs: '+str(pids))
                return gate
            except NotReady as error:
                self.status('WAITING_FOR_ORIGINAL_DELIVERY_AND_FREE_GPU', reason=str(error))
                if self.once:
                    return None
                time.sleep(self.poll_seconds)

    def job(self, name, script, args=(), *, cpu=True):
        self.verify()
        command = [sys.executable, str(self.here/script), *map(str, args)]
        env = dict(os.environ)
        if cpu:
            env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        logpath = self.out/(name+'.log')
        with logpath.open('a', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT)
            self.worker = child
            who = proc(child.pid)
            if who is None:
                child.wait()
                raise RuntimeError('Cannot register child process identity: '+name)
            launch = dict(pid=child.pid, start_ticks=who['start_ticks'], command=command,
                          job=name, queue_registration_sha256=self.queue_sha, time=time.time())
            write(self.out/'launches'/f'{name}_{time.time_ns()}.json', launch)
            while child.poll() is None:
                self.status('RUNNING', job=name, worker=launch, log=str(logpath))
                time.sleep(min(15, self.poll_seconds))
            code = child.returncode; self.worker = None
            if code:
                raise RuntimeError(f'Historical job failed ({code}): {name}; inspect {logpath}')
        self.verify()

    def run(self):
        own = lock(self.out/'controller.lock')
        try:
            # A dead controller must not cause duplicate work while its child
            # remains alive; all previous child identities are explicit.
            for path in (self.out/'launches').glob('*.json'):
                require_exited(read(path), process_snapshot(), 'previous historical child')
            who = proc(os.getpid())
            if who is None:
                raise RuntimeError('Cannot identify historical controller')
            launch = dict(pid=os.getpid(), start_ticks=who['start_ticks'],
                          queue_registration_sha256=self.queue_sha, source_bindings=self.queue['source_bindings'])
            write(self.out/'controller_launch.json', launch)
            checks = self.out/'cpu_checks.json'
            if checks.exists():
                saved = read(checks)
                if saved.get('status') != 'PASS' or saved.get('source_bindings') != self.queue['source_bindings']:
                    raise RuntimeError('Historical CPU qualification differs')
            else:
                tests = sorted(self.here.glob('test_*.py'))
                if not tests:
                    raise RuntimeError('Missing historical CPU tests')
                for test in tests:
                    self.job(test.stem, test.name)
                write(checks, dict(status='PASS', synthetic_tests=True, scientific_result=False,
                    source_bindings=self.queue['source_bindings'], tests=[p.name for p in tests]))
            gate = self.wait_gate()
            if gate is None:
                return None
            write(self.out/'parent_gate.json', gate)
            self.job('source_publication', 'history_publish.py',
                ['--root', self.root, '--queue-registration', self.queue_path, '--phase', 'source'])
            for job in self.queue['jobs']:
                if self.wait_gate() is None:
                    return None
                complete = self.out/job['study']/'completion.json'
                if complete.exists():
                    value = read(complete)
                    if value.get('status') != 'HISTORICAL_STUDY_METRICS_COMPLETE' or value.get('parity_passed') is not True:
                        raise RuntimeError('Existing historical completion is invalid')
                    verify(value['inputs']); verify(value['outputs'])
                else:
                    self.job(job['study'], 'history_score.py', ['--root', self.root, '--adapter', job['adapter'],
                        '--study', job['study'], '--queue-registration', self.queue_path], cpu=False)
                groups = self.queue.get('execution_groups')
                if groups and job['study'] == groups['required'][-1]:
                    proofs = {}
                    for study in groups['required']:
                        p = self.out/study/'completion.json'
                        value = read(p)
                        if value.get('status') != 'HISTORICAL_STUDY_METRICS_COMPLETE' or value.get('parity_passed') is not True:
                            raise RuntimeError('Required study incomplete before optional phase')
                        verify(value['inputs']); verify(value['outputs'])
                        proofs[str(p)] = sha(p)
                    write(self.out/'required_studies_completion.json', dict(status='REQUIRED_STUDIES_COMPLETE',
                        inputs=proofs, queue_registration_sha256=self.queue_sha, synthetic=False,
                        next_phase='authorized analysis supplements', time=time.time()))
            self.job('analysis', 'history_analysis.py', ['--root', self.root, '--queue-registration', self.queue_path])
            self.job('result_publication', 'history_publish.py',
                ['--root', self.root, '--queue-registration', self.queue_path, '--phase', 'results'])
            final = read(self.out/'results_publication.json'); published(final)
            write(self.out/'completion.json', dict(**launch, status='HISTORICAL_METRICS_COMPLETE', publication=final,
                training_updates=0, policy_selection_updates=0, stop=True))
            self.status('COMPLETE', stopped_after_registered_scope=True)
            return final
        except BaseException as error:
            self.status('FAILED', error=str(error), worker_pid=self.worker.pid if self.worker else None)
            raise
        finally:
            own.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--queue-registration', required=True)
    p.add_argument('--poll-seconds', type=int, default=30); p.add_argument('--once', action='store_true')
    a = p.parse_args()
    if not 1 <= a.poll_seconds <= 60:
        p.error('poll-seconds must be 1..60')
    Controller(a.root, a.queue_registration, poll_seconds=a.poll_seconds, once=a.once).run()


if __name__ == '__main__':
    main()
