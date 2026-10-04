"""Measure source-parallel M1 execution, then retain the qualified faster mode.

The original science, checkpoint binding and publication paths are unchanged.
This owner is launched only after the previous reconstruction owner has been
explicitly retired at a source boundary. It never signals an external GPU user.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA = 'M1_GPU_COHORT_V1'
BENCHMARK_SOURCES = list(range(6))
MIN_SPEEDUP = 1.15
STOP = False


class StopRequested(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def seal(path, value):
    if Path(path).exists():
        require(read(path) == value, 'Immutable scheduling receipt changed: ' + str(path))
    else:
        write(path, value)


def verify(bindings):
    for path, expected in bindings.items():
        require(sha(path) == expected, 'Bound scheduling/scientific source changed: ' + str(path))


def process_identity(pid):
    folder = Path('/proc') / str(pid)
    stat = (folder / 'stat').read_text()
    tail = stat[stat.rfind(')') + 2:].split()
    require(tail[0] != 'Z', 'Process is a zombie: ' + str(pid))
    uid = next(line.split()[1] for line in (folder / 'status').read_text().splitlines() if line.startswith('Uid:'))
    argv = (folder / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    require(argv and argv[0], 'Process command is empty')
    return dict(pid=int(pid), start_ticks=tail[19], uid=int(uid), argv=argv)


def partition(indices, count):
    require(type(count) is int and count in (1, 2, 4), 'Only 1/2/4 workers are registered')
    require(indices and len(indices) == len(set(indices)) and all(type(i) is int and 0 <= i < 1000 for i in indices),
            'Unique registered calibration source indices required')
    return [indices[offset::count] for offset in range(count) if indices[offset::count]]


def checkpoint_rows(path, binding, index):
    cp = read(path)
    payload = {k: v for k, v in cp.items() if k != 'payload_sha256'}
    require(cp.get('payload_sha256') == identity(payload) and cp.get('binding') == binding
            and cp.get('source_index') == index, 'Original source checkpoint seal/binding differs')
    rows = cp.get('rows', [])
    require(len(rows) == 2070 and all(row.get('source_index') == index and row.get('N') == 2048 for row in rows),
            'Original source checkpoint is not a full N2048 grid')
    keys = [(r['phy_family'], r['snr_db'], r['action_id'], r['noise_seed']) for r in rows]
    require(len(set(keys)) == 2070, 'Duplicated original calibration events')
    return identity(rows)


def summarize_trial(worker_count, completions, references, elapsed, samples):
    require(len(completions) == worker_count, 'Missing worker completion')
    seen = {}
    for done in completions:
        require(done.get('status') == 'M1_SOURCE_WORKER_COMPLETE', 'Source worker did not complete')
        for key, digest in done['rows_sha256_by_source'].items():
            index = int(key)
            require(index not in seen and references.get(index) == digest, 'Benchmark row parity failed or duplicate source')
            seen[index] = digest
    require(set(seen) == set(BENCHMARK_SOURCES), 'Benchmark must cover the same six original sources')
    start = min(float(d['compute_started_unix']) for d in completions)
    end = max(float(d['compute_finished_unix']) for d in completions)
    steady = end - start
    require(steady > 0 and elapsed >= steady, 'Nonpositive or inconsistent benchmark timing')
    return dict(status='EXACT_FULL_ROWS_PARITY_PASS', workers=worker_count, sources=6,
                rows=12420, rows_sha256_by_source={str(k): seen[k] for k in sorted(seen)},
                steady_seconds=steady, sources_per_second=6 / steady,
                end_to_end_seconds=elapsed, startup_and_exit_seconds=max(0., elapsed - steady),
                peak_gpu_used_mib=max((s['used_mib'] for s in samples), default=0.),
                peak_temperature=max((s['temperature'] for s in samples), default=0.),
                mean_gpu_utilization=sum(s['utilization'] for s in samples) / len(samples) if samples else 0.,
                worker_peak_cuda_reserved_bytes=[d['peak_cuda_reserved_bytes'] for d in completions])


def choose_mode(trials, remaining):
    require(trials and trials[0]['workers'] == 1 and remaining >= 0, 'Serial benchmark is required')
    baseline = trials[0]
    require(all(t['status'] == 'EXACT_FULL_ROWS_PARITY_PASS' for t in trials), 'Unqualified trial cannot select execution mode')
    qualified = [t for t in trials if t['sources_per_second'] / baseline['sources_per_second'] >= MIN_SPEEDUP]
    # Startup cost is included: a mode that helps throughput may not help a tiny tail.
    serial_eta = baseline['startup_and_exit_seconds'] + remaining / baseline['sources_per_second']
    qualified = [t for t in qualified if t['startup_and_exit_seconds'] + remaining / t['sources_per_second'] < serial_eta]
    chosen = min(qualified, key=lambda t: (t['startup_and_exit_seconds'] + remaining / t['sources_per_second'], t['workers'])) if qualified else baseline
    return dict(status='EXECUTION_MODE_FROZEN', selected_workers=chosen['workers'],
                speedup=chosen['sources_per_second'] / baseline['sources_per_second'], minimum_speedup=MIN_SPEEDUP,
                remaining_sources=remaining, serial_remaining_estimate_seconds=serial_eta,
                selected_remaining_estimate_seconds=chosen['startup_and_exit_seconds'] + remaining / chosen['sources_per_second'],
                scientific_functions_changed=False, benchmark_sources=BENCHMARK_SOURCES,
                selection_is_engineering_only=True, reason='qualified_total_time_improvement' if chosen['workers'] > 1 else 'resume_original_serial')


class Coordinator:
    def __init__(self, config):
        self.config = config
        self.root = Path(config['root']).resolve()
        self.out = Path(config['out']).resolve()
        self.original = Path(config['original_out']).resolve()
        self.code = Path(config['original_code']).resolve()
        self.here = Path(__file__).resolve().parent
        self.python = config['native_python']
        self.children = []
        self.departed_gpu = {}
        self.original_lock = None
        self.self_identity = process_identity(os.getpid())
        self.registration = read(self.original / 'calibrate_registration.json')
        self.binding = identity(self.registration)
        self.registration_sha = sha(self.original / 'calibrate_registration.json')
        require(config['expected_calibration_registration_sha256'] == self.registration_sha, 'Calibration registration changed')
        require(config.get('max_workers', 2) in (2, 4), 'Maximum worker count must be 2 or 4')
        require(float(config.get('benchmark_budget_seconds', 900)) <= 900, 'Benchmark budget cannot exceed fifteen minutes')

    def check(self):
        if STOP or (self.out / 'STOP').exists() or (self.original / 'STOP').exists():
            raise StopRequested('Requested source-safe stop')
        verify(self.config['source_bindings'])
        require(sha(self.original / 'calibrate_registration.json') == self.registration_sha, 'Original registration changed')

    def status(self, state, **fields):
        write(self.out / 'status.json', dict(status=state, owner=self.self_identity, time=time.time(), **fields))

    def acquire_original(self):
        import fcntl
        require(self.original_lock is None, 'Original execution lock is already held')
        handle = (self.original / 'execution.lock').open('a+')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            handle.close()
            raise
        self.original_lock = handle

    def release_original(self):
        if self.original_lock is not None:
            self.original_lock.close()
            self.original_lock = None

    def gpu(self, admitted=()):
        raw = subprocess.check_output(['nvidia-smi', '--id=0',
            '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,clocks_event_reasons.hw_thermal_slowdown',
            '--format=csv,noheader,nounits'], text=True).strip().split(',')
        require(len(raw) == 5, 'Unexpected GPU resource response')
        used, total, util, temp = map(float, raw[:4])
        active = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                                         '--format=csv,noheader,nounits'], text=True).strip().splitlines()
        allowed = {r['pid']: r for r in admitted}
        for line in active:
            require(line.strip().isdigit(), 'Invalid GPU process listing')
            pid = int(line.strip())
            departed = self.departed_gpu.get(pid)
            grace = departed is not None and 0 <= time.monotonic() - departed['exited_at'] < 5
            expected = allowed.get(pid, departed['identity'] if grace else None)
            require(expected is not None, 'Unknown GPU process appeared: ' + str(pid))
            folder = Path('/proc') / str(pid)
            if folder.exists():
                stat = (folder / 'stat').read_text(); tail = stat[stat.rfind(')') + 2:].split()
                if tail[0] == 'Z':
                    uid = next(line.split()[1] for line in (folder / 'status').read_text().splitlines() if line.startswith('Uid:'))
                    require(grace and tail[19] == expected['start_ticks'] and int(uid) == expected['uid'],
                            'Departing GPU process identity changed')
                else:
                    require(process_identity(pid) == expected, 'Admitted GPU process identity changed')
            else:
                require(grace, 'Admitted GPU process disappeared without owned exit receipt')
        sample = dict(time=time.time(), used_mib=used, total_mib=total, utilization=util, temperature=temp,
                      thermal_slowdown=raw[4].strip().lower() == 'active', compute_pids=[int(line.strip()) for line in active])
        require(temp < 86 and not sample['thermal_slowdown'], 'GPU thermal limit reached; stop cohort')
        require(total - used >= float(self.config.get('gpu_reserve_mib', 4096)), 'GPU memory reserve exhausted; stop cohort')
        return sample

    def gpu_idle(self):
        """Allow only the brief driver teardown of an identified, exited child."""
        while True:
            self.check()
            sample = self.gpu()
            if not sample['compute_pids']:
                return sample
            time.sleep(.5)

    def environment(self):
        env = os.environ.copy()
        env.update(self.config['native_environment'])
        env.update(PYTHONDONTWRITEBYTECODE='1', VAR_COMM_ROOT=str(self.root))
        return env

    def stop_children(self):
        # Every entry is a Popen child of this owner. External PIDs are never signalled.
        live = [p for p in self.children if p.poll() is None]
        for p in live:
            p.terminate()
        deadline = time.monotonic() + 120
        while any(p.poll() is None for p in live) and time.monotonic() < deadline:
            time.sleep(1)
        for p in live:
            if p.poll() is None:
                p.kill()
        for p in live:
            if p.poll() is None:
                p.wait(timeout=10)

    def completed_sources(self):
        found = []
        for index in range(1000):
            path = self.original / 'calibrate/source_checkpoints' / ('%04d.json' % index)
            if path.exists():
                checkpoint_rows(path, self.binding, index)
                found.append(index)
        return found

    def cohort(self, stage, count, indices):
        self.check()
        self.gpu_idle()
        folder = self.out / (('benchmark_w%d' % count) if stage == 'benchmark' else 'production')
        require(not folder.exists(), 'Existing cohort evidence requires explicit review: ' + str(folder))
        folder.mkdir(parents=True)
        manifest = folder / 'cohort_manifest.json'
        batches = partition(indices, count)
        children, records, configs = [], [], []
        started = time.monotonic()
        try:
            for worker_id, assigned in enumerate(batches):
                output = folder / ('worker%d' % worker_id)
                cfg = dict(root=str(self.root), original_out=str(self.original), original_code=str(self.code),
                    protocol=self.config['protocol'], cohort_manifest=str(manifest), worker_id=worker_id,
                    source_indices=assigned, stage=stage, scratch_out=str(output / 'scratch'), output_dir=str(output),
                    stop_file=str(self.out / 'STOP'), source_bindings=self.config['source_bindings'],
                    expected_calibration_registration_sha256=self.registration_sha)
                cfgpath = folder / ('worker%d_config.json' % worker_id)
                seal(cfgpath, cfg)
                argv = [str(self.python), '-B', str(self.here / 'source_worker.py'), '--config', str(cfgpath)]
                with (folder / ('worker%d.log' % worker_id)).open('ab') as log:
                    child = subprocess.Popen(argv, cwd=self.root, env=self.environment(), stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.children.append(child)
                children.append(child)
                record = process_identity(child.pid)
                records.append(dict(worker_id=worker_id, source_indices=assigned, **record))
                configs.append(cfg)
            seal(manifest, dict(schema=SCHEMA, status='READY', owner=self.self_identity, workers=records, stage=stage))
            gpu_records = [{k: r[k] for k in ('pid', 'start_ticks', 'uid', 'argv')} for r in records]
            samples = []
            while any(p.poll() is None for p in children):
                self.check()
                require(all(p.poll() in (None, 0) for p in children), 'Source worker failed or paused; inspect cohort logs')
                for process, record in zip(children, gpu_records):
                    if process.poll() is not None and process.pid not in self.departed_gpu:
                        self.departed_gpu[process.pid] = dict(identity=record, exited_at=time.monotonic())
                samples.append(self.gpu([r for p, r in zip(children, gpu_records) if p.poll() is None]))
                self.status('RUNNING_' + stage.upper(), workers=len(children), cohort=str(folder), gpu=samples[-1])
                time.sleep(2)
            require(all(p.returncode == 0 for p in children), 'Source worker failed')
            for process, record in zip(children, gpu_records):
                self.departed_gpu.setdefault(process.pid, dict(identity=record, exited_at=time.monotonic()))
            elapsed = time.monotonic() - started
            completions = []
            for cfg in configs:
                donepath = Path(cfg['output_dir']) / 'completion.json'
                done = read(donepath)
                require(done.get('status') == 'M1_SOURCE_WORKER_COMPLETE'
                        and done.get('registration_sha256') == self.registration_sha
                        and done.get('source_bindings') == self.config['source_bindings'], 'Worker completion identity differs')
                require(done.get('stage') == stage and done.get('synthetic') is False
                        and done.get('development_read') is False and done.get('training_updates') == 0
                        and done.get('batch_size') == 1 and done.get('calibration_binding') == self.binding
                        and done.get('cohort_manifest_sha256') == sha(manifest), 'Worker completion scientific scope differs')
                require(done.get('source_indices') == cfg['source_indices']
                        and done.get('physical_frames') == 2070 * len(cfg['source_indices']), 'Worker completion frame count differs')
                if stage == 'benchmark':
                    require(done.get('complete_row_parity_verified') is True, 'Worker did not verify exact original rows')
                require(set(map(int, done['rows_sha256_by_source'])) == set(cfg['source_indices']), 'Worker source assignment differs')
                require(isinstance(done.get('outputs'), dict) and bool(done['outputs']), 'Worker outputs are unsealed')
                verify(done['outputs'])
                completions.append(done)
            receipt = dict(status='COHORT_COMPLETE', stage=stage, worker_count=len(children), source_indices=indices,
                           manifest_sha256=sha(manifest), completion_sha256={str(Path(c['output_dir']) / 'completion.json'):
                           sha(Path(c['output_dir']) / 'completion.json') for c in configs},
                           elapsed_seconds=elapsed, gpu_samples=samples)
            seal(folder / 'completion.json', receipt)
            return completions, elapsed, samples
        except BaseException:
            self.stop_children()
            raise

    def original_stage(self, stage):
        require(self.original_lock is None, 'Release coordinator execution lock before original CLI acquires it')
        self.check()
        self.gpu_idle()
        args = [str(self.python), '-B', str(self.code / 'm1_performance.py'), '--root', str(self.root),
                '--protocol', self.config['protocol'], '--output', str(self.original), '--stage', stage]
        self.status('ORIGINAL_' + stage.upper())
        with (self.out / ('original_%s.log' % stage)).open('ab') as log:
            process = subprocess.Popen(args, cwd=self.root, env=self.environment(), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(process)
        record = process_identity(process.pid)
        while process.poll() is None:
            self.check()
            time.sleep(2)
        require(process.returncode == 0, 'Original stage failed: ' + stage)
        self.departed_gpu[process.pid] = dict(identity=record, exited_at=time.monotonic())
        receipt = read(self.original / (stage + '_completion.json'))
        require(receipt.get('status') == 'COMPLETE' and receipt.get('registration_sha256') == sha(self.original / 'registration.json'),
                'Original completion registration differs: ' + stage)
        verify(receipt['outputs'])

    def run(self):
        self.check()
        require(not (self.original / 'owner_failure.json').exists(), 'Existing original failure needs review')
        self.acquire_original()
        try:
            completed = self.completed_sources()
            require(set(BENCHMARK_SOURCES).issubset(completed), 'First six completed calibration sources are required for parity')
            references = {i: checkpoint_rows(self.original / 'calibrate/source_checkpoints' / ('%04d.json' % i), self.binding, i)
                          for i in BENCHMARK_SOURCES}
            seal(self.out / 'handoff_snapshot.json', dict(status='ORIGINAL_SOURCE_BOUNDARY_SNAPSHOT',
                completed_source_indices=completed, registration_sha256=self.registration_sha,
                benchmark_reference_rows={str(k): v for k, v in references.items()},
                old_sources_untouched=True, source_bindings=self.config['source_bindings']))
            trials = []
            benchmark_started = time.monotonic()
            for count in (1, 2, 4):
                if count > self.config.get('max_workers', 2):
                    break
                if count == 2 and time.monotonic() - benchmark_started + trials[0]['end_to_end_seconds'] > self.config.get('benchmark_budget_seconds', 900):
                    # Use the serial end-to-end measurement as a conservative
                    # next-trial allowance. Keep the qualified serial fallback
                    # when unexpectedly slow startup would consume the budget.
                    break
                if count == 4:
                    previous = trials[-1]
                    if previous['sources_per_second'] / trials[0]['sources_per_second'] >= MIN_SPEEDUP and previous['mean_gpu_utilization'] >= 80:
                        break
                    expected = previous['end_to_end_seconds']
                    if time.monotonic() - benchmark_started + expected > self.config.get('benchmark_budget_seconds', 900):
                        break
                    if 2 * previous['peak_gpu_used_mib'] + self.config.get('gpu_reserve_mib', 4096) > self.gpu()['total_mib']:
                        break
                done, elapsed, samples = self.cohort('benchmark', count, BENCHMARK_SOURCES)
                result = summarize_trial(count, done, references, elapsed, samples)
                seal(self.out / ('benchmark_w%d/summary.json' % count), result)
                trials.append(result)
            remaining = [i for i in range(1000) if i not in set(completed)]
            decision = choose_mode(trials, len(remaining))
            decision['trials'] = trials
            seal(self.out / 'benchmark_decision.json', decision)
            self.status('BENCHMARK_DECIDED', **{k: v for k, v in decision.items() if k != 'status'})
            if decision['selected_workers'] > 1 and remaining:
                self.cohort('production', decision['selected_workers'], remaining)
                require(self.completed_sources() == list(range(1000)), 'Parallel calibration source coverage is incomplete')
        finally:
            self.release_original()
        # Original aggregation reads all cached sources in the original source
        # order. If parallelism was rejected it computes only the missing tail.
        self.original_stage('calibrate')
        self.original_stage('development')
        seal(self.original / 'owner_completion.json', dict(status='RECONSTRUCTIONS_COMPLETE', metrics='separate registered delivery owner'))
        seal(self.out / 'completion.json', dict(status='ORIGINAL_RECONSTRUCTIONS_COMPLETE',
             calibration_completion_sha256=sha(self.original / 'calibrate_completion.json'),
             development_completion_sha256=sha(self.original / 'development_completion.json'),
             benchmark_decision_sha256=sha(self.out / 'benchmark_decision.json'), delivery_path_unchanged=True))
        self.status('RECONSTRUCTIONS_COMPLETE_DELIVERY_CONTINUES')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    out = Path(config['out']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / 'owner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (out / 'failure.json').exists(), 'Earlier multiworker failure needs explicit review')
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    owner = Coordinator(config)
    try:
        owner.run()
    except BaseException as error:
        owner.stop_children()
        owner.release_original()
        failure = dict(status='PAUSED_REQUIRES_REVIEW' if isinstance(error, StopRequested) else 'FAILED_REQUIRES_REVIEW',
                       error=repr(error), time=time.time(), owner=owner.self_identity, automatic_retry=False)
        write(out / 'failure.json', failure)
        if not isinstance(error, StopRequested):
            write(owner.original / 'owner_failure.json', dict(error=repr(error), automatic_retry=False, multiworker_failure=str(out / 'failure.json')))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
