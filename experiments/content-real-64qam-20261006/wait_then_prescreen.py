"""After the complete coarse waiter exits, run one registered CPU prescreen.

No decoder or visual model is loaded here. The frozen owner and prior waiter
are read-only dependencies. Refinement requests are terminal outputs, not jobs.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

DEADLINE = 1791564605.9549868
PRESCREEN_STATUSES = {'H_PRESCREEN_COMPLETE_REFINEMENT_REQUIRED', 'H_PRESCREEN_COMPLETE_FINAL'}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def verify_coarse_closed(a, prior_api, prior, launch, identity, state_reader):
    """Validate future receipts through the previously bound execution chain."""
    base = prior.waiter_out; regsha = prior.regsha
    a.require(prior_api.exited(identity, state_reader), 'Coarse waiter has not been reaped')
    a.require(not (base / 'failure.json').exists() and not (base / 'STOP').exists(), 'Coarse waiter failed/stopped')
    cp = base / 'completion.json'; done = a.read(cp)
    a.require(done.get('status') == 'H_COARSE_WAIT_AND_CPU_BATCH_COMPLETE'
              and done.get('registration_sha256') == regsha
              and done.get('real_decodes_in_waiter') == 0 and done.get('GPU_used') is False,
              'Coarse waiter completion identity/scope differs')
    lp, ep, log = base / 'owner_launch.json', base / 'owner_exit.json', base / 'owner.log'
    launched, ended = a.read(lp), a.read(ep); ownerid = launched['identity']
    argv = [prior.cfg['python'], '-B', prior.cfg['stage_owner'], '--config', prior.cfg['next_owner_config']]
    a.require(launched.get('registration_sha256') == regsha and launched.get('argv') == argv
              and ownerid['argv'] == argv and int(ownerid['uid']) == int(identity['uid'])
              and ended.get('registration_sha256') == regsha and ended.get('identity') == ownerid
              and ended.get('exit_code') == 0 and ended.get('closed_log_sha256') == a.sha(log),
              'Coarse owner launch/closed exit/log differs')
    a.require(prior_api.exited(ownerid, state_reader), 'Coarse owner has not been reaped')
    ownercp = Path(prior.next['owner_out']) / 'completion.json'
    a.require(done.get('owner_completion') == str(ownercp)
              and done.get('owner_completion_sha256') == a.sha(ownercp), 'Coarse owner completion changed')
    closure = prior_api.verify_batch(a, prior.cfg['next_owner_config'], prior.cfg['registration'],
                                    identity['uid'], state_reader)
    a.require(done.get('output_bindings') == closure['bindings'], 'Coarse waiter output closure is incomplete/changed')
    # Also close the earlier source/clean batch, including its original owner.
    a.require(prior_api.exited(prior.oldid, state_reader), 'Initial H owner remains present')
    original = prior_api.verify_batch(a, prior.initial['config'], prior.initial['registration'],
                                     identity['uid'], state_reader)
    gp = base / 'release_gate.json'; gate = a.read(gp)
    a.require(gate.get('registration_sha256') == regsha and gate.get('initial_identity') == prior.oldid
              and gate.get('bindings') == original['bindings']
              and gate.get('completion_sha256') == original['completion_sha256'], 'Initial release gate changed')
    a.verify(prior.reg['source_bindings']); a.verify(prior.reg['input_bindings'])
    result = dict(closure['bindings']); result.update(original['bindings'])
    for path in (cp, lp, ep, log, gp): result[str(path)] = a.sha(path)
    return dict(status='COARSE_WAITER_OWNER_AND_CHILDREN_CLOSED', bindings=result,
                coarse_waiter_identity=identity, coarse_owner_identity=ownerid,
                child_identities=closure['child_identities'] + original['child_identities'])


class Waiter:
    def __init__(self, config):
        self.config_path = Path(config).resolve()
        self.cfg = json.loads(self.config_path.read_text(encoding='utf-8-sig')); c = self.cfg
        self.reg = json.loads(Path(c['registration']).read_text(encoding='utf-8-sig')); r = self.reg
        for key in ('stage_owner', 'wait_then_coarse'):
            path = str(Path(c[key]).resolve())
            if r['source_bindings'].get(path) != hashlib.sha256(Path(path).read_bytes()).hexdigest():
                raise RuntimeError('Frozen dependency source unbound: ' + key)
        self.a = module(c['stage_owner'], 'prescreen_frozen_owner'); a = self.a
        a.require(r.get('source_bindings', {}).get(str(Path(__file__).resolve())) == a.sha(__file__), 'Waiter source unbound')
        a.verify(r['source_bindings']); a.verify(r['input_bindings'])
        self.p = module(c['wait_then_coarse'], 'prescreen_frozen_coarse_waiter')
        self.regsha = a.sha(c['registration']); self.out = a.absolute(c['out'])
        self.waiter_out = a.within(c['waiter_out'], self.out)
        a.require(c.get('schema') == 'H_PRESCREEN_WAIT_V1' and c['poll_seconds'] == 10
                  and c['deadline_unix'] == DEADLINE, 'Waiter protocol/deadline differs')
        a.require(r.get('status') == 'H_EXECUTION_REVISION_REGISTERED' and r.get('branch') == 'H', 'Prescreen not registered')
        dep = c['coarse_wait']; self.prior = self.p.Waiter(dep['config'])
        self.prior_launch = a.read(dep['launch']); stamp = a.read(dep['identity']); self.priorid = self.prior_launch['identity']
        expected = [self.prior_launch['argv'][0], '-B', c['wait_then_coarse'], '--config', dep['config']]
        a.require(self.prior_launch['argv'] == expected and self.priorid['argv'] == expected
                  and self.prior_launch.get('registration_sha256') == self.prior.regsha
                  and self.prior_launch.get('config_sha256') == a.sha(dep['config'])
                  and a.same_identity(stamp, self.priorid)
                  and stamp.get('registration_sha256') == self.prior.regsha
                  and stamp.get('config_sha256') == a.sha(dep['config']), 'Coarse waiter identity/launch differs')
        a.absolute(expected[0], resolve=False)
        a.require(a.absolute(dep['identity']) == self.prior.waiter_out / 'identity.json', 'Wrong coarse waiter identity evidence')
        self.next = a.read(c['next_owner_config'])
        for path in (self.config_path, c['next_owner_config'], dep['config'], dep['launch'], dep['identity'],
                     self.prior.cfg['registration'], self.prior.cfg['next_owner_config']):
            a.require(r['input_bindings'].get(str(path)) == a.sha(path), 'Waiter dependency unbound: ' + str(path))
        a.validate_config(self.next, r, a.sha(c['next_owner_config']))
        a.require(self.next['registration'] == c['registration'] and self.next['root'] == c['root']
                  and self.next['out'] == c['out'] and self.prior.cfg['root'] == c['root']
                  and self.prior.cfg['out'] == c['out'] and r['allowed_stage_ids'] == ['freeze'], 'Only one H freeze stage may run')
        stage = self.next['stages'][0]
        a.require(stage['resource'] == 'cpu' and len(stage['jobs']) == 1, 'Prescreen must be one CPU worker')
        job = stage['jobs'][0]; argv = job['argv']; entry = a.command_entry(argv)
        offset = 2 if argv[1] == '-B' else 1
        a.require(entry.name == 'h_prescreen.py' and argv[offset + 1:] == ['--config', argv[-1]], 'Only h_prescreen --config may run')
        self.science_config = a.absolute(argv[-1]); self.science = a.read(self.science_config)
        a.require(r['input_bindings'].get(str(self.science_config)) == a.sha(self.science_config)
                  and self.science.get('phase') == 'coarse' and self.science['registration'] == c['registration']
                  and self.science['out'] == job['out'] and set(job['accepted_statuses']) == PRESCREEN_STATUSES,
                  'Prescreen phase/config/statuses differ')
        self.job = job
        for key in ('budget_path', 'budget_registration', 'budget_registration_sha256', 'phase_limits'):
            a.require(self.prior.next[key] == self.next[key], 'Independent H budget changed: ' + key)
        a.require(a.absolute(c['python'], resolve=False).is_file(), 'Registered interpreter absent')
        for other in (self.prior.waiter_out, Path(self.prior.next['owner_out']), Path(self.next['owner_out'])):
            a.require(self.waiter_out != other and self.waiter_out not in other.parents
                      and other not in self.waiter_out.parents, 'Waiter evidence overlaps an existing owner/waiter')
        self.child = None; self.log = None; self.child_identity = None; self.stop = False
        self.started = time.time(); self.mono = time.monotonic(); self.before = None

    def wall(self): return max(time.time(), self.started + time.monotonic() - self.mono)

    def guard(self):
        a = self.a
        a.require(not self.stop and not (self.waiter_out / 'STOP').exists(), 'Prescreen waiter STOP requested')
        a.require(self.wall() < DEADLINE, 'Original96-hour deadline reached')
        a.require(not (self.out / 'STOP').exists(), 'H STOP blocks prescreen')
        for base in (self.prior.waiter_out, Path(self.prior.next['owner_out']), Path(self.prior.initial['owner_out'])):
            a.require(not (base / 'failure.json').exists() and not (base / 'STOP').exists(), 'Prerequisite failed/stopped: ' + str(base))
        for cfg in (self.prior.old, self.prior.next):
            for stage in cfg['stages']:
                for job in stage['jobs']:
                    a.require(not (Path(job['out']) / 'failure.json').exists(), 'Prerequisite science stage failed')

    def snapshot(self):
        return self.a.budget_snapshot(self.next['budget_path'], self.next['budget_registration_sha256'],
                                      self.next['phase_limits'], quiescent=True)

    def wait_gate(self):
        a = self.a; cp = self.prior.waiter_out / 'completion.json'
        while True:
            self.guard()
            if self.p.exited(self.priorid, a.raw_process_state):
                a.require(cp.exists(), 'Coarse waiter exited without completion; no retry')
                gate = verify_coarse_closed(a, self.p, self.prior, self.prior_launch, self.priorid, a.raw_process_state)
                a.verify(self.reg['source_bindings']); a.verify(self.reg['input_bindings'])
                self.before = self.snapshot(); self.guard()
                a.save(self.waiter_out / 'release_gate.json', dict(**gate, budget=self.before,
                       registration_sha256=self.regsha, time=time.time()), exclusive=True)
                return
            a.save(self.waiter_out / 'status.json', dict(status='WAITING_COARSE_WAITER_AND_OWNER_EXIT',
                   coarse_waiter_identity=self.priorid, completion_present=cp.exists(),
                   scientific_children_started=False, registration_sha256=self.regsha, time=time.time()))
            time.sleep(10)

    def run(self):
        a = self.a; c = self.cfg
        try:
            self.wait_gate(); self.guard()
            argv = [c['python'], '-B', c['stage_owner'], '--config', c['next_owner_config']]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
            self.log = (self.waiter_out / 'owner.log').open('xb')
            self.child = subprocess.Popen(argv, cwd=c['root'], env=env, stdin=subprocess.DEVNULL,
                                          stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            self.child_identity = dict(pid=self.child.pid, identity_capture='PENDING')
            self.child_identity = a.identity(self.child.pid)
            a.require(self.child_identity['argv'] == argv and int(self.child_identity['uid']) == os.getuid(), 'New owner identity differs')
            a.save(self.waiter_out / 'owner_launch.json', dict(identity=self.child_identity, argv=argv,
                   registration_sha256=self.regsha, time=time.time()), exclusive=True)
            a.save(self.waiter_out / 'status.json', dict(status='REGISTERED_CPU_PRESCREEN_RUNNING',
                   child_identity=self.child_identity, registration_sha256=self.regsha, GPU_used=False, time=time.time()))
            while self.child.poll() is None:
                self.guard(); a.check_live(self.child, self.child_identity); time.sleep(10)
            code = self.child.wait(); self.log.close(); self.log = None
            a.save(self.waiter_out / 'owner_exit.json', dict(identity=self.child_identity, exit_code=code,
                   closed_log_sha256=a.sha(self.waiter_out / 'owner.log'), registration_sha256=self.regsha,
                   time=time.time()), exclusive=True)
            a.require(code == 0, 'Prescreen owner failed; no retry')
            closure = self.p.verify_batch(a, c['next_owner_config'], c['registration'], self.child_identity['uid'], a.raw_process_state)
            a.require(self.p.exited(self.child_identity, a.raw_process_state), 'Prescreen owner was not reaped')
            final = a.receipt(self.job['completion'], PRESCREEN_STATUSES, self.regsha,
                              {'new_packet_decodes':0, 'new_visual_inference':0}, self.job['out'])
            need = final['status'] == 'H_PRESCREEN_COMPLETE_REFINEMENT_REQUIRED'
            a.require(final.get('ready_for_real_calibration') is (not need), 'Prescreen readiness contradicts status')
            a.require(self.before == self.snapshot(), 'CPU prescreen changed the packet ledger')
            a.save(self.waiter_out / 'completion.json', dict(status='H_COARSE_WAIT_AND_PRESCREEN_COMPLETE',
                   registration_sha256=self.regsha, output_bindings=closure['bindings'],
                   prescreen_status=final['status'], refinement_requested=need,
                   refinement_started=False, ready_for_real_calibration=not need,
                   new_packet_decodes=0, new_visual_inference=0, GPU_used=False,
                   H_full_delivery_claimed=False, C_started=False, holdout_started=False, time=time.time()), exclusive=True)
        except BaseException:
            a.save(self.waiter_out / 'failure.json', dict(status='FAILED_PRESERVE_NO_RETRY', traceback=traceback.format_exc(),
                   registration_sha256=self.regsha, child_identity=self.child_identity, time=time.time()), exclusive=True)
            if self.child is not None and self.child.poll() is None:
                try:
                    with (self.out / 'STOP').open('x', encoding='utf-8') as f: f.write('Prescreen waiter failed; safe drain owned batch\n')
                except FileExistsError: pass
                self.child.wait()
            if self.log is not None: self.log.close(); self.log = None
            if self.child is not None and not (self.waiter_out / 'owner_exit.json').exists():
                a.save(self.waiter_out / 'owner_exit.json', dict(identity=self.child_identity, exit_code=self.child.wait(),
                       closed_log_sha256=a.sha(self.waiter_out / 'owner.log'), registration_sha256=self.regsha,
                       safe_drain_after_failure=True, time=time.time()), exclusive=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    waiter = Waiter(parser.parse_args().config); a = waiter.a
    waiter.waiter_out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (waiter.waiter_out / 'waiter.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.require(not any((waiter.waiter_out / n).exists() for n in ('identity.json', 'failure.json', 'completion.json')),
                  'Existing prescreen waiter attempt requires explicit recovery')
        os.setpriority(os.PRIO_PROCESS, 0, 15)
        a.save(waiter.waiter_out / 'identity.json', dict(**a.identity(os.getpid()), registration_sha256=waiter.regsha,
               config_sha256=a.sha(waiter.config_path), time=time.time(), nice=15), exclusive=True)
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: setattr(waiter, 'stop', True))
        waiter.run()


if __name__ == '__main__': main()
