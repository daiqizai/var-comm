"""Wait for the sealed initial H batch to exit, then run one bound CPU batch.

This wrapper does not decode, load image models, signal the initial owner, or
change the frozen stage owner. Future completion files are verified through the
already bound initial registration and its complete receipt chain.
"""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

DEADLINE = 1791564605.9549868


def load_owner(path):
    spec = importlib.util.spec_from_file_location('registered_h_stage_owner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exited(expected, reader):
    """Only absence or a proven later PID incarnation establishes old exit.

    A zombie is not reaped and continues to block this launch. Permission and
    malformed identity errors are not interpreted as process absence.
    """
    actual = reader(int(expected['pid']))
    if actual is None:
        return True
    if int(actual['pid']) != int(expected['pid']):
        raise RuntimeError('Process reader returned a different PID')
    if int(actual['start_ticks']) != int(expected['start_ticks']):
        return True
    if int(actual['uid']) != int(expected['uid']):
        raise RuntimeError('Initial process UID changed without a new start time')
    if actual.get('state') not in ('Z', 'X') and actual['argv'] != expected['argv']:
        raise RuntimeError('Initial process command changed')
    return False


def verify_batch(api, config, registration, expected_uid, state_reader):
    """Read-only closure validation of every stage, job, log and science output."""
    r = api.read(registration); rsha = api.sha(registration)
    c = api.read(config); api.validate_config(c, r, api.sha(config))
    base = Path(c['owner_out']); done_path = base / 'completion.json'
    api.require(not (base / 'failure.json').exists(), 'Owner failure blocks continuation')
    done = api.read(done_path)
    api.require(done.get('status') == 'REGISTERED_H_STAGE_BATCH_COMPLETE'
                and done.get('registration_sha256') == rsha
                and done.get('allowed_stage_ids') == r['allowed_stage_ids']
                and done.get('qualification_passed') is True,
                'Initial batch completion identity/status differs')
    for key in ('future_stages_started', 'H_full_delivery_claimed', 'C_started', 'holdout_started'):
        api.require(done.get(key) is False, 'Unexpected continuation claim: ' + key)
    api.require(done.get('budget', {}).get('unresolved') == 0, 'Batch did not close its packet ledger')
    completed = done['completed']
    api.require([x['stage'] for x in completed] == [s['id'] for s in c['stages']],
                'Batch is missing or reordering registered stages')
    bindings = {str(done_path): api.sha(done_path)}; child_identities = []
    for stage, link in zip(c['stages'], completed):
        directory = base / 'stages' / stage['id']; path = directory / 'completion.json'
        api.require(link['completion'] == str(path) and link['sha256'] == api.sha(path), 'Stage receipt changed')
        bindings[str(path)] = api.sha(path); sd = api.read(path)
        api.require(sd.get('status') == 'REGISTERED_STAGE_COMPLETE' and sd.get('stage') == stage['id']
                    and sd.get('registration_sha256') == rsha and sd.get('budget', {}).get('unresolved') == 0,
                    'Stage did not complete cleanly')
        jobs = {j['id']: j for j in stage['jobs']}
        api.require(len(sd['jobs']) == len(jobs) and {e['job_id'] for e in sd['jobs']} == set(jobs),
                    'Stage job receipt coverage differs')
        for event in sd['jobs']:
            job = jobs[event['job_id']]; evidence = directory / 'workers' / job['id']
            ep = evidence / 'exit_receipt.json'; lp = evidence / 'launch.json'
            api.require(api.read(ep) == event and event['exit_code'] == 0
                        and event['registration_sha256'] == rsha, 'Job did not exit successfully')
            ident = event['identity']; launch = api.read(lp)
            api.require(int(ident['uid']) == int(expected_uid) and ident['argv'] == job['argv']
                        and launch.get('identity') == ident and launch.get('registration_sha256') == rsha,
                        'Job launch/exit identity differs')
            api.require(exited(ident, state_reader), 'Completed job is still present; wait for reap')
            log = evidence / 'worker.log'
            api.require(api.sha(log) == event['closed_log_sha256'], 'Closed worker log changed')
            cp = Path(job['completion'])
            api.require(str(cp) == event['completion'] and api.sha(cp) == event['completion_sha256']
                        and not (Path(job['out']) / 'failure.json').exists(), 'Scientific completion failed/changed')
            science_reg = Path(job.get('receipt_registration', registration))
            science = api.receipt(cp, job['accepted_statuses'], api.sha(science_reg),
                                  job.get('receipt_expect'), job['out'])
            api.require(not science.get('development_used', False) and not science.get('holdout_used', False),
                        'The initial bridge unexpectedly read development/holdout')
            for file in (ep, lp, log, cp): bindings[str(file)] = api.sha(file)
            bindings.update(science['outputs']); child_identities.append(ident)
    api.verify(r['source_bindings']); api.verify(r['input_bindings'])
    return dict(status='INITIAL_BATCH_AND_ALL_CHILDREN_CLOSED', completion_sha256=api.sha(done_path),
                bindings=bindings, child_identities=child_identities)


class Waiter:
    def __init__(self, config):
        # Import the owner only after locating its exact registered source. The
        # source is verified before any owner instance or scientific command.
        import json, hashlib
        self.config_path = Path(config).resolve()
        cfg = json.loads(self.config_path.read_text(encoding='utf-8-sig'))
        reg = json.loads(Path(cfg['registration']).read_text(encoding='utf-8-sig'))
        source = str(Path(cfg['stage_owner']).resolve())
        if reg['source_bindings'].get(source) != hashlib.sha256(Path(source).read_bytes()).hexdigest():
            raise RuntimeError('Frozen owner source is not bound')
        self.a = load_owner(source); a = self.a; self.cfg = cfg; self.reg = reg
        self.regsha = a.sha(cfg['registration']); self.out = a.absolute(cfg['out'])
        self.waiter_out = a.within(cfg['waiter_out'], self.out)
        a.require(cfg.get('schema') == 'H_COARSE_WAIT_V1' and cfg['poll_seconds'] == 10
                  and cfg['deadline_unix'] == DEADLINE, 'Waiter protocol/deadline differs')
        a.require(reg['status'] == 'H_EXECUTION_REVISION_REGISTERED' and reg['branch'] == 'H', 'Unregistered CPU batch')
        a.require(reg['source_bindings'].get(str(Path(__file__).resolve())) == a.sha(__file__), 'Waiter source unbound')
        a.verify(reg['source_bindings']); a.verify(reg['input_bindings'])
        self.initial = cfg['initial']; self.next = a.read(cfg['next_owner_config'])
        for p in (self.config_path, cfg['next_owner_config'], *[self.initial[k] for k in ('registration', 'config', 'launch')]):
            a.require(reg['input_bindings'].get(str(p)) == a.sha(p), 'Waiter dependency unbound: ' + str(p))
        a.validate_config(self.next, reg, a.sha(cfg['next_owner_config']))
        a.require(self.next['registration'] == cfg['registration'] and self.next['root'] == cfg['root']
                  and self.next['out'] == cfg['out'] and reg['allowed_stage_ids'] == ['coarse', 'report']
                  and all(s['resource'] == 'cpu' for s in self.next['stages']), 'Only coarse plus CPU table merge may run')
        for stage in self.next['stages']:
            expected_phase = 'worker' if stage['id'] == 'coarse' else 'merge'
            a.require(len(stage['jobs']) == (2 if expected_phase == 'worker' else 1), 'Coarse/merge worker coverage differs')
            for job in stage['jobs']:
                argv = job['argv']; entry = a.command_entry(argv)
                a.require(entry.name == 'h_coarse_driver.py' and argv.count('--stage') == 1
                          and argv[argv.index('--stage') + 1] == expected_phase, 'Only the bound coarse worker/merge entry may run')
        a.require(a.absolute(cfg['root']) == a.absolute(self.next['root'])
                  and a.absolute(cfg['python'], resolve=False).is_file(), 'Bad root/interpreter')
        self.old = a.read(self.initial['config']); self.oldreg = a.read(self.initial['registration'])
        a.validate_config(self.old, self.oldreg, a.sha(self.initial['config']))
        a.require(self.oldreg['allowed_stage_ids'] == ['qualification', 'source', 'clean_quality']
                  and self.old['owner_out'] == self.initial['owner_out']
                  and self.old['out'] == cfg['out'], 'Wrong prerequisite initial H batch')
        for key in ('budget_path', 'budget_registration', 'budget_registration_sha256', 'phase_limits'):
            a.require(self.old[key] == self.next[key], 'Shared budget changed: ' + key)
        self.oldlaunch = a.read(self.initial['launch']); self.oldid = self.oldlaunch['identity']
        expected = ['-B', source, '--config', self.initial['config']]
        a.require(self.oldlaunch['registration_sha256'] == a.sha(self.initial['registration'])
                  and self.oldlaunch['owner_config_sha256'] == a.sha(self.initial['config'])
                  and self.oldlaunch['argv'] == self.oldid['argv'] and self.oldid['argv'][1:] == expected
                  and a.absolute(self.oldid['argv'][0], resolve=False).is_file(), 'Initial launch is not the bound owner')
        a.require(self.waiter_out != Path(self.next['owner_out'])
                  and self.waiter_out != Path(self.initial['owner_out']), 'Waiter evidence overlaps an owner')
        self.child = None; self.log = None; self.child_identity = None; self.stop = False
        self.started = time.time(); self.mono = time.monotonic()

    def wall(self):
        return max(time.time(), self.started + time.monotonic() - self.mono)

    def guard(self):
        a = self.a
        a.require(not self.stop and not (self.waiter_out / 'STOP').exists(), 'Waiter STOP requested')
        a.require(self.wall() < self.cfg['deadline_unix'], 'Original 96-hour deadline reached')
        a.require(not (self.out / 'STOP').exists(), 'H STOP blocks continuation')
        a.require(not (Path(self.initial['owner_out']) / 'failure.json').exists(), 'Initial owner failed')
        for stage in self.old['stages']:
            for job in stage['jobs']:
                a.require(not (Path(job['out']) / 'failure.json').exists(), 'Initial science stage failed')

    def wait_gate(self):
        a = self.a; cp = Path(self.initial['owner_out']) / 'completion.json'
        while True:
            self.guard(); gone = exited(self.oldid, a.raw_process_state)
            if gone:
                a.require(cp.exists(), 'Initial owner exited without a complete batch receipt')
                gate = verify_batch(a, self.initial['config'], self.initial['registration'],
                                    self.oldid['uid'], a.raw_process_state)
                a.verify(self.reg['source_bindings']); a.verify(self.reg['input_bindings'])
                a.budget_snapshot(self.next['budget_path'], self.next['budget_registration_sha256'],
                                  self.next['phase_limits'], quiescent=True)
                self.guard()
                a.save(self.waiter_out / 'release_gate.json', dict(**gate, initial_identity=self.oldid,
                       registration_sha256=self.regsha, time=time.time()), exclusive=True)
                return
            a.save(self.waiter_out / 'status.json', dict(status='WAITING_INITIAL_H_BATCH_EXIT',
                   initial_identity=self.oldid, completion_present=cp.exists(), time=time.time(),
                   registration_sha256=self.regsha, scientific_children_started=False))
            time.sleep(self.cfg['poll_seconds'])

    def run(self):
        a = self.a
        try:
            self.wait_gate(); self.guard()
            argv = [self.cfg['python'], '-B', self.cfg['stage_owner'], '--config', self.cfg['next_owner_config']]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
            self.log = (self.waiter_out / 'owner.log').open('xb')
            self.child = subprocess.Popen(argv, cwd=self.cfg['root'], env=env, stdin=subprocess.DEVNULL,
                                          stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            # Own the child before identity capture/evidence writing can fail.
            self.child_identity = dict(pid=self.child.pid, identity_capture='PENDING')
            self.child_identity = a.identity(self.child.pid)
            a.require(self.child_identity['argv'] == argv and self.child_identity['uid'] == os.getuid(), 'New owner identity differs')
            a.save(self.waiter_out / 'owner_launch.json', dict(identity=self.child_identity, argv=argv,
                   registration_sha256=self.regsha, time=time.time()), exclusive=True)
            a.save(self.waiter_out / 'status.json', dict(status='REGISTERED_CPU_BATCH_RUNNING',
                   child_identity=self.child_identity, registration_sha256=self.regsha, time=time.time(),
                   GPU_used=False, scientific_children_started=True))
            while self.child.poll() is None:
                self.guard(); a.check_live(self.child, self.child_identity)
                time.sleep(self.cfg['poll_seconds'])
            code = self.child.wait(); self.log.close(); self.log = None
            a.save(self.waiter_out / 'owner_exit.json', dict(identity=self.child_identity, exit_code=code,
                   closed_log_sha256=a.sha(self.waiter_out / 'owner.log'), registration_sha256=self.regsha,
                   time=time.time()), exclusive=True)
            a.require(code == 0, 'CPU owner failed; preserve original failure')
            final = Path(self.next['owner_out']) / 'completion.json'
            done = a.read(final)
            a.require(done.get('status') == 'REGISTERED_H_STAGE_BATCH_COMPLETE'
                      and done.get('registration_sha256') == self.regsha
                      and not (Path(self.next['owner_out']) / 'failure.json').exists(), 'CPU batch did not complete')
            closure = verify_batch(a, self.cfg['next_owner_config'], self.cfg['registration'],
                                   self.child_identity['uid'], a.raw_process_state)
            a.save(self.waiter_out / 'completion.json', dict(status='H_COARSE_WAIT_AND_CPU_BATCH_COMPLETE',
                   registration_sha256=self.regsha, owner_completion=str(final), owner_completion_sha256=a.sha(final),
                   output_bindings=closure['bindings'], real_decodes_in_waiter=0, GPU_used=False, time=time.time()), exclusive=True)
        except BaseException:
            detail = traceback.format_exc()
            a.save(self.waiter_out / 'failure.json', dict(status='FAILED_PRESERVE_NO_RETRY', traceback=detail,
                   registration_sha256=self.regsha, child_identity=self.child_identity, time=time.time()), exclusive=True)
            if self.child is not None and self.child.poll() is None:
                # Stop only our new owner, using its registered safe-boundary file.
                stop = self.out / 'STOP'
                try:
                    with stop.open('x', encoding='utf-8') as f: f.write('Registered coarse waiter failed; safe drain\n')
                except FileExistsError: pass
                self.child.wait()
            if self.log is not None: self.log.close(); self.log = None
            if self.child is not None and not (self.waiter_out / 'owner_exit.json').exists():
                a.save(self.waiter_out / 'owner_exit.json', dict(identity=self.child_identity,
                       exit_code=self.child.wait(), closed_log_sha256=a.sha(self.waiter_out / 'owner.log'),
                       registration_sha256=self.regsha, safe_drain_after_failure=True, time=time.time()), exclusive=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    waiter = Waiter(parser.parse_args().config); a = waiter.a
    waiter.waiter_out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (waiter.waiter_out / 'waiter.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.require(not any((waiter.waiter_out / n).exists() for n in ('identity.json', 'failure.json', 'completion.json')),
                  'Existing waiter attempt requires explicit recovery')
        os.setpriority(os.PRIO_PROCESS, 0, 15)
        a.save(waiter.waiter_out / 'identity.json', dict(**a.identity(os.getpid()), registration_sha256=waiter.regsha,
               config_sha256=a.sha(waiter.config_path), time=time.time(), nice=15), exclusive=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(waiter, 'stop', True))
        waiter.run()


if __name__ == '__main__': main()
