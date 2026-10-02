"""Own only the metric queue; observe the scientific queue without signalling it."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from scheduling_guard import read, write, sha, sources, verify, validate_admission, identity, alive, process

NAME = 'METRIC-CONCURRENT-R4-20261002'
METRICS = 'UNIFIED-METRICS-20261002'
PARENT = 'SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'
PERFORMANCE_CAP = 1.75
WAITING = {'WAITING_FOR_ORIGINAL_EXPERIMENTS','WAITING_FOR_ORIGINAL_REPAIR','WAITING_FOR_ORIGINAL_EXIT'}


def lock(path):
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a')
    try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException: f.close(); raise
    return f


def pushed(value):
    commit = value.get('commit','')
    if (value.get('status') != 'PUSHED' or value.get('checks') != 'PASS'
            or value.get('remote_commit') != commit or len(commit) != 40
            or any(x not in '0123456789abcdef' for x in commit)):
        raise RuntimeError('Verified normal publication required')


def retire(expected, reader=process):
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support required')
    fd = os.pidfd_open(expected['pid'],0)
    try:
        p = reader(expected['pid'])
        if not alive(expected,p) or p['state'] not in ('T','t'):
            raise RuntimeError('The nominated idle metric supervisor must be identity-checked and paused')
        signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
        try: signal.pidfd_send_signal(fd,signal.SIGCONT,None,0)
        except ProcessLookupError: pass
    finally: os.close(fd)


class Slowdown:
    """Three consecutive measured source intervals, excluding model loading."""
    def __init__(self, baseline):
        self.stage = baseline['stage']; self.seconds = float(baseline['seconds_per_source'])
        if self.seconds <= 0 or int(baseline['sample_count']) < 3: raise RuntimeError('Invalid M2 baseline')
        self.previous = None; self.previous_scoring = False; self.bad = 0; self.samples = []; self.intervals = 0

    def observe(self, point, scoring):
        if not point: return False
        key = (point['stage'],int(point['sources']),float(point['time']))
        if self.previous == key: return False
        previous = self.previous; previous_scoring = self.previous_scoring
        self.previous = key; self.previous_scoring = scoring
        if previous is None or previous[0] != key[0] or key[1] <= previous[1]:
            self.bad = 0; self.samples = []; return False
        duration = (key[2]-previous[2])/(key[1]-previous[1])
        if duration <= 0: raise RuntimeError('Nonmonotonic M2 progress time')
        if key[0] != self.stage:
            if scoring: return False  # A new stage needs an exclusive baseline first.
            self.samples.append(duration)
            if len(self.samples) >= 3:
                self.stage = key[0]; self.seconds = sorted(self.samples[-3:])[1]; self.bad = 0
            return False
        self.intervals += int(scoring and previous_scoring)
        self.bad = self.bad+1 if scoring and previous_scoring and duration > PERFORMANCE_CAP*self.seconds else 0
        return self.bad >= 3


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi','--id=0',
        '--query-gpu=memory.used,memory.total,temperature.gpu,clocks_event_reasons.hw_thermal_slowdown',
        '--format=csv,noheader,nounits'],text=True).strip().split(',')
    if len(raw) != 4: raise RuntimeError('Unexpected GPU resource response')
    used,total,temp = map(float,raw[:3])
    return dict(used_mib=used,total_mib=total,free_mib=total-used,temperature=temp,
                thermal_slowdown=raw[3].strip().lower() == 'active')


def resource_reason(value):
    if value['free_mib'] < 6144 or value['used_mib'] > .75*value['total_mib']:
        return 'COMBINED_GPU_MEMORY_RESERVE'
    if value['temperature'] >= 82 or value['thermal_slowdown']:
        return 'THERMAL_HEADROOM'
    return None


class Controller:
    def __init__(self, root, here=None, *, reader=process, retire_fn=retire, lock_fn=lock,
                 spawn=subprocess.Popen, sleep=time.sleep, now=time.time, gpu=gpu_snapshot, pid=None):
        self.root = Path(root).resolve(); self.out = self.root/'outputs'/NAME
        self.here = Path(here or Path(__file__).parent).resolve()
        self.metrics = self.root/'outputs'/METRICS; self.parent = self.root/'outputs'/PARENT
        self.oldsource = self.root/'experiments/unified-metrics-20261002'
        self.reader,self.retire_fn,self.lock_fn,self.spawn = reader,retire_fn,lock_fn,spawn
        self.sleep,self.now,self.gpu = sleep,now,gpu; self.pid = os.getpid() if pid is None else pid
        self.owns = False; self.worker = None; self.owner_lock = None
        self.suspended_stage = None
        self.peer_observed = False

    def status(self,state,**fields):
        value = dict(status=state,pid=self.pid,time=self.now(),**fields)
        write(self.out/'controller_status.json',value)
        if self.owns: write(self.metrics/'supervisor_status.json',value)

    def initialize(self):
        self.admission_path = self.out/'admission.json'
        self.admission = validate_admission(self.root,self.admission_path,self.here)
        verify(self.admission['original_execution_bindings'])
        self.bound = sources(self.here); self.hand = read(self.out/'handoff.json')
        self.old = identity(self.hand['oldsupervisor'])
        if self.old['pid'] == self.pid or self.hand.get('new_source_bindings') != self.bound:
            raise RuntimeError('Metric handoff identity/source differs')
        self.original_bound = sources(self.oldsource)
        self.assets = read(self.metrics/'assets_complete.json'); self.python = self.assets['environment']
        manifest = self.metrics/'modelmanifest.json'; qpath = self.metrics/'models_qualification.json'; q = read(qpath)
        if (self.assets.get('status') != 'ASSETS_READY' or self.assets.get('manifest_sha256') != sha(manifest)
                or q.get('status') != 'REAL_MODEL_WEIGHTS_QUALIFICATION_PASS'
                or q.get('modelmanifest_sha256') != sha(manifest)):
            raise RuntimeError('Six-metric assets/qualification required')
        verify(q['source_bindings'])
        regpath = self.metrics/'supervisor_registration.json'; oldreg = read(regpath)
        if oldreg.get('source_bindings') != self.original_bound:
            raise RuntimeError('Original metric supervisor sources differ')
        base = dict(status='REGISTERED',runtime_source_bindings=self.bound,
            source_bindings=self.bound,original_metric_source_bindings=self.original_bound,
            admission_sha256=sha(self.admission_path),handoff_sha256=sha(self.out/'handoff.json'),
            original_registration_sha256=sha(regpath),training_updates=0,policy_selection_updates=0,
            original_m2_source_or_selection_changed=False,baseline=self.hand['m2_baseline'],
            scheduling_policy=dict(objective='combined_completion_time_provisional_trial',
                provisional_m2_slowdown_cap=PERFORMANCE_CAP,consecutive_intervals=3,
                memory_thermal_and_exclusive_timing_unchanged=True,net_total_speedup_not_yet_established=True))
        self.registration_path = self.out/'runtime_registration.json'
        previous = self.hand.get('previous_extension')
        if previous is not None:
            self.validate_previous(previous,require_current=not self.registration_path.exists())
            base['previous_extension'] = previous
        if self.registration_path.exists():
            if read(self.registration_path) != base: raise RuntimeError('Concurrent runtime resume binding changed')
        else:
            launch = read(self.metrics/'supervisor_launch.json'); status = read(self.metrics/'supervisor_status.json')
            if previous is None:
                if (identity(launch) != self.old or sha(self.metrics/'supervisor_launch.json') != self.hand['oldlaunch_sha256']
                        or status.get('pid') != self.old['pid'] or status.get('status') not in WAITING
                        or status.get('worker_pid') is not None):
                    raise RuntimeError('Only the nominated idle original metric owner can be replaced')
                p = self.reader(self.old['pid'])
                if not alive(self.old,p) or p['state'] not in ('T','t'):
                    raise RuntimeError('Operator must first pause the identity-checked idle metric owner')
                if (self.oldsource/'supervisor.py').as_posix() not in p.get('command','').replace('\\','/'):
                    raise RuntimeError('The nominated idle metric process runs a different command')
            archive = self.out/'retired_metrics_supervisor'
            archive.mkdir(parents=True,exist_ok=True)
            for name in ('supervisor_launch.json','supervisor_status.json','supervisor_registration.json'):
                target = archive/name; data = (self.metrics/name).read_bytes()
                if target.exists() and target.read_bytes() != data: raise RuntimeError('Archived metric owner evidence differs')
                target.write_bytes(data)
            if previous is not None:
                for source in previous['bindings']:
                    target = archive/('previous_extension_' + Path(source).name)
                    data = Path(source).read_bytes()
                    if target.exists() and target.read_bytes() != data: raise RuntimeError('Prior extension archive differs')
                    target.write_bytes(data)
            write(self.registration_path,base)
        self.registration = base; self.slowdown = Slowdown(base['baseline']); self.verify()

    def validate_previous(self,previous,require_current,expected_owner=None,visited=None):
        prior = Path(previous['out']).resolve()
        visited = set() if visited is None else visited
        predecessor_names = ('METRIC-CONCURRENT-R3-20261002','METRIC-CONCURRENT-R2-20261002','METRIC-CONCURRENT-20261002')
        if len(visited) >= len(predecessor_names): raise RuntimeError('Unexpected scheduling ancestry depth')
        allowed = self.root/'outputs'/predecessor_names[len(visited)]
        if prior != allowed or prior in visited:
            raise RuntimeError('Only the explicitly nominated R3/R2/R1 ancestry can be replaced')
        visited.add(prior)
        expected_owner = self.old if expected_owner is None else identity(expected_owner)
        required = ('runtime_registration.json','admission.json','handoff.json','controller_launch.json',
                    'controller_status.json','partial_launch.json')
        mapping = previous['bindings']
        if not {str(prior/n) for n in required}.issubset(mapping):
            raise RuntimeError('Previous extension evidence is incomplete')
        verify(mapping)
        reg,adm,hand,launch,status,worker = [read(prior/n) for n in required]
        bound = reg['runtime_source_bindings']
        idle_r3 = (len(visited) == 1 and status.get('status') == 'WAITING_FOR_CONCURRENT_ADMISSION'
            and status.get('reason') == 'M2_SLOWDOWN_OVER_20_PERCENT_THREE_INTERVALS' and status.get('worker_pid') is None)
        accepted_state = idle_r3 if len(visited) == 1 else status.get('status') == 'FAILED'
        if (sources(prior/'runtime') != bound or adm.get('source_bindings') != bound
                or reg.get('admission_sha256') != sha(prior/'admission.json')
                or reg.get('handoff_sha256') != sha(prior/'handoff.json')
                or hand.get('new_source_bindings') != bound
                or launch.get('source_bindings') != bound or launch.get('runtime_registration_sha256') != sha(prior/'runtime_registration.json')
                or identity(launch) != expected_owner or not accepted_state or status.get('pid') != expected_owner['pid']
                or reg.get('original_metric_source_bindings') != self.original_bound
                or reg.get('original_registration_sha256') != sha(self.metrics/'supervisor_registration.json')
                or worker.get('source_bindings') != bound or worker.get('runtime_registration_sha256') != sha(prior/'runtime_registration.json')
                or identity(worker.get('owner')) != expected_owner):
            raise RuntimeError('Previous scheduling extension provenance/state differs')
        verify(bound)
        if alive(launch,self.reader(launch['pid'])) or alive(worker,self.reader(worker['pid'])):
            raise RuntimeError('Previous scheduling owner or latest worker remains active')
        if status.get('worker_pid') is not None and (status['worker_pid'] != worker['pid']
                or str(status.get('worker_start_ticks')) != str(worker['start_ticks'])):
            raise RuntimeError('Failed status identifies an unaccounted worker')
        if require_current:
            current = read(self.metrics/'supervisor_launch.json'); current_status = read(self.metrics/'supervisor_status.json')
            if (identity(current) != self.old or sha(self.metrics/'supervisor_launch.json') != self.hand['oldlaunch_sha256']
                    or current.get('runtime_source_bindings') != bound or current.get('source_bindings') != self.original_bound
                    or current_status != status):
                raise RuntimeError('Current metric owner is not the checked idle R3 extension')
        ancestor = reg.get('previous_extension')
        if ancestor is not None:
            ancestor_launch = read(Path(ancestor['out'])/'controller_launch.json')
            self.validate_previous(ancestor,False,expected_owner=identity(ancestor_launch),visited=visited)

    def verify(self):
        if (sources(self.here) != self.bound or sources(self.oldsource) != self.original_bound
                or sha(self.admission_path) != self.registration['admission_sha256']
                or sha(self.out/'handoff.json') != self.registration['handoff_sha256']
                or read(self.registration_path) != self.registration
                or sha(self.metrics/'supervisor_registration.json') != self.registration['original_registration_sha256']):
            raise RuntimeError('Frozen scheduling/source registration changed')
        if self.registration.get('previous_extension') is not None:
            verify(self.registration['previous_extension']['bindings'])

    def takeover(self):
        retired = self.out/'retirement_complete.json'
        if retired.exists():
            if read(retired).get('oldsupervisor') != self.old or alive(self.old,self.reader(self.old['pid'])):
                raise RuntimeError('Metric retirement receipt differs')
        else:
            if alive(self.old,self.reader(self.old['pid'])): self.retire_fn(self.old,self.reader)
            deadline = self.now()+30
            while alive(self.old,self.reader(self.old['pid'])):
                if self.now() > deadline: raise RuntimeError('Retired idle metric owner did not exit')
                self.sleep(1)
            write(retired,dict(status='IDLE_METRIC_SUPERVISOR_RETIRED',oldsupervisor=self.old,
                runtime_registration_sha256=sha(self.registration_path)))
        self.owner_lock = self.lock_fn(self.metrics/'supervisor.lock')
        previous = read(self.metrics/'supervisor_launch.json')
        if identity(previous) != self.old:
            if alive(previous,self.reader(previous['pid'])): raise RuntimeError('Previous metric replacement still active')
            state = read(self.metrics/'supervisor_status.json')
            if state.get('worker_pid'):
                actual = self.reader(state['worker_pid'])
                ticks = state.get('worker_start_ticks')
                if actual is not None and actual['state'] != 'Z' and (ticks is None or str(actual['start_ticks']) == str(ticks)):
                    raise RuntimeError('Orphan metric worker is still active')
        me = self.reader(self.pid)
        if me is None or me['state'] == 'Z': raise RuntimeError('Cannot bind new metric owner')
        self.me = identity(me)
        self.launch = dict(**self.me,command=[sys.executable,'-u',str(self.here/'controller.py'),'--root',str(self.root)],
            admission_sha256=sha(self.admission_path),source_bindings=self.bound,
            runtime_registration_sha256=sha(self.registration_path),retired_original=self.old,time=self.now())
        write(self.out/'controller_launch.json',self.launch)
        write(self.metrics/'supervisor_launch.json',dict(self.launch,source_bindings=self.original_bound,
            runtime_source_bindings=self.bound,scheduling_extension=True))
        self.owns = True; self.status('CONCURRENT_OWNER_READY',worker_pid=None)

    def progress(self):
        parent = read(self.parent/'supervisor_status.json'); worker = parent.get('worker_pid')
        candidates = []
        for path in self.parent.glob('m2_*_status.json'):
            row = read(path)
            if row.get('pid') == worker and isinstance(row.get('sources'),int):
                if row.get('stage') == 'm2_gate':
                    row = dict(row,stage='m2_gate_reuse' if row['sources'] < 200 else 'm2_gate_new')
                candidates.append(row)
        return parent,max(candidates,key=lambda x:x['time']) if candidates else None

    def parent_done(self):
        path = self.parent/'completion.json'
        if not path.exists(): return False
        done = read(path); status = read(self.parent/'supervisor_status.json'); launch = read(self.parent/'supervisor_launch.json')
        if done.get('status') != 'AUTHORIZED_TWO_METHODS_COMPLETE' or status.get('status') != 'COMPLETE': return False
        if done.get('synthetic') is not False or done.get('training_updates') != 0 or done.get('stop') is not True:
            raise RuntimeError('Scientific final scope differs')
        pushed(done['publication']); verify(done['publication']['source_bindings'])
        if alive(launch,self.reader(launch['pid'])): return False
        m2launch = self.out/'m2_controller_launch.json'
        if m2launch.exists() and alive(read(m2launch),self.reader(read(m2launch)['pid'])): return False
        return True

    def pause(self,reason):
        write(self.out/'pause_requested.json',dict(status='PAUSE_REQUESTED',reason=reason,time=self.now(),
            controller_pid=self.pid,controller_start_ticks=self.me['start_ticks'],admission_sha256=sha(self.admission_path)))

    def launch_worker(self,pilot=False):
        marker = self.out/'pause_requested.json'
        if marker.exists(): marker.unlink()  # Only this scheduler clears its operational request.
        argv = [self.python,'-u',str(self.here/'concurrent_runner.py'),'--root',str(self.root),
            '--cache-dir',str(self.out),'--admission',str(self.admission_path),'--gpu-memory-fraction','0.45']
        previous_cache = Path(self.hand['previous_cache_dir']).resolve()
        if previous_cache != self.root/'outputs/METRIC-CONCURRENT-R3-20261002':
            raise RuntimeError('Only the checked R3 sealed cache can be inherited')
        argv += ['--previous-cache-dir',str(previous_cache)]
        if pilot: argv += ['--max-sources','1']
        log = self.out/f'concurrent_{time.time_ns()}.log'; self.stream = log.open('a')
        self.worker = self.spawn(argv,cwd=self.root,stdin=subprocess.DEVNULL,stdout=self.stream,
            stderr=subprocess.STDOUT,env=self.environment())
        actual = self.reader(self.worker.pid)
        if actual is None: raise RuntimeError('Cannot bind concurrent child identity')
        launch = dict(**identity(actual),command=argv,source_bindings=self.bound,
            runtime_registration_sha256=sha(self.registration_path),owner=self.me,time=self.now())
        previous = self.out/'partial_launch.json'
        if previous.exists(): write(self.out/'launch_history'/f'{read(previous)["pid"]}.json',read(previous))
        write(previous,launch); self.worker_identity = identity(actual)
        self.status('CONCURRENT_PILOT' if pilot else 'CONCURRENT_SCORING',worker_pid=actual['pid'],
            worker_start_ticks=actual['start_ticks'],log=str(log))

    def environment(self,cpu=False):
        env = dict(os.environ,OMP_NUM_THREADS='2' if cpu else '6',OPENBLAS_NUM_THREADS='2')
        env['PYTHONPATH'] = os.pathsep.join(str(self.root/p) for p in (
            'experiments/rx-posterior-step1-20260929','src',
            'experiments/var-latent-enhancement-20260917/src',
            'experiments/var-latent-enhancement-20260917/phase_b/src',
            'experiments/var-latent-enhancement-20260917/evaluation/src',
            'experiments/var-latent-enhancement-20260917/followup/src',
            'experiments/var-latent-enhancement-20260917/research/src',
            'experiments/var-latent-enhancement-20260917/mechanisms/src',
            'experiments/var-short-prefix-hybrid-20260923/src',
            'experiments/token_channel_efficiency_20260923/src'))
        if cpu: env.update(CUDA_VISIBLE_DEVICES='',MKL_NUM_THREADS='2')
        return env

    def stage_admission_reason(self,point,done,slow):
        if slow and point is not None: self.suspended_stage = point['stage']
        if (done or (point is not None and self.suspended_stage is not None
                and point['stage'] != self.suspended_stage and point['stage'] == self.slowdown.stage)):
            self.suspended_stage = None
        if self.suspended_stage is not None and not done:
            return 'M2_SLOWDOWN_OVER_75_PERCENT_THREE_INTERVALS'
        floor = self.hand.get('m2_resume_floor',{})
        if point is not None and point['stage'] == floor.get('stage') and point['sources'] <= int(floor.get('sources',-1)) and not done:
            return 'M2_RESUME_REPLAYING_OLD_CHECKPOINTS'
        if point is None and not done: return 'WAITING_FOR_ACTIVE_M2_SOURCE_PROGRESS'
        if point and point['stage'] == 'm2_gate_reuse' and not done: return 'GATE_FIRST_200_REUSES_SCREEN_RESULTS'
        if point is not None and point['stage'] != self.slowdown.stage and not done:
            return 'NEW_M2_STAGE_NEEDS_EXCLUSIVE_BASELINE'
        return None

    def concurrent(self):
        paused = False; pilot = not (self.out/'partial_completion.json').exists(); last_wait = 0
        while True:
            self.verify(); done = self.parent_done(); parent,point = self.progress()
            status = read(self.out/'status.json') if (self.out/'status.json').exists() else {}
            scoring = self.worker is not None and status.get('pid') == self.worker.pid and status.get('status') == 'SCORING_SEALED_STUDIES'
            lease = self.out/'active_m2.json'
            if scoring and lease.exists():
                peer = read(lease)
                self.peer_observed |= peer.get('status') == 'ACTIVE' and alive(peer,self.reader(peer['pid']))
            slow = self.slowdown.observe(point,scoring)
            timing = (self.out/'exclusive_timing_requested.json').exists() and not (self.parent/'m2_timing_complete.json').exists()
            resource = None if timing and self.worker is None else resource_reason(self.gpu())
            stage_reason = self.stage_admission_reason(point,done,slow)
            snapshot = self.out/'partial_completion.json'
            at_least_one = snapshot.exists() and read(snapshot).get('sources_complete',0) >= 1
            reason = ('ORIGINAL_PIPELINE_DELIVERED' if done and at_least_one else 'ORIGINAL_EXCLUSIVE_TIMING' if timing and not done else
                'ORIGINAL_PIPELINE_RESOURCE_WAIT' if parent.get('status') in ('FAILED','WAITING_FOR_SAFE_RESOURCE') else
                stage_reason or resource)
            if reason and self.worker is not None and not paused:
                self.pause(reason); paused = True; last_wait = self.now()
                self.status('PAUSING_CONCURRENT_WORKER',reason=reason,worker_pid=self.worker.pid,
                    worker_start_ticks=self.worker_identity['start_ticks'])
            if self.worker is not None:
                code = self.worker.poll()
                if code is not None:
                    self.stream.close(); self.worker = None
                    if code == 75:
                        paused = True; last_wait = self.now()
                        self.status('CONCURRENT_PAUSED',reason=status.get('reason'),worker_pid=None)
                    elif code != 0: raise RuntimeError('Concurrent metric worker failed: ' + str(code))
                    elif status.get('status') not in ('PILOT_COMPLETE','SEALED_STUDIES_COMPLETE'):
                        raise RuntimeError('Metric child exited without a completed pilot/cache receipt')
                    else:
                        if pilot:
                            write(self.out/'pilot_completion.json',dict(status='METRIC_REPLAY_PILOT_PASS',
                                original_m2_active=not done,parallel_claim=self.peer_observed and self.slowdown.intervals > 0,
                                original_parity_passed=True,concurrent_metric_scalar_qualification=True,
                                measured_m2_intervals=self.slowdown.intervals,slowdown_baseline_seconds=self.slowdown.seconds,
                                slowdown_limit=PERFORMANCE_CAP,consecutive_limit=3,training_updates=0,policy_selection_updates=0,
                                total_completion_speedup_claimed=False,
                                worker=read(self.out/'partial_launch.json'),time=self.now()))
                        pilot = False; paused = False
                        self.status(status['status'],worker_pid=None)
            at_least_one = snapshot.exists() and read(snapshot).get('sources_complete',0) >= 1
            if done and self.worker is None and at_least_one: return
            fully_scored = status.get('status') == 'SEALED_STUDIES_COMPLETE'
            safe = reason is None or (done and resource is None)
            if self.worker is None and not fully_scored and safe and (not paused or self.now()-last_wait >= 60):
                self.launch_worker(pilot=pilot or (done and not at_least_one)); paused = False
            elif self.worker is None:
                self.status('WAITING_FOR_CONCURRENT_ADMISSION' if not fully_scored else 'WAITING_FOR_ORIGINAL_COMPLETION',
                    reason=reason,worker_pid=None,parent_stage=parent.get('stage'))
            self.sleep(10)

    def run_final_job(self,name,script,args=(),cpu=True):
        self.verify(); log = self.out/f'{name}_{time.time_ns()}.log'
        with log.open('a') as stream:
            worker = self.spawn([self.python,'-u',str(script),*map(str,args)],cwd=self.root,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=self.environment(cpu))
            actual = self.reader(worker.pid)
            self.worker = worker; self.worker_identity = identity(actual) if actual is not None else dict(pid=worker.pid,start_ticks='0')
            self.status('RUNNING',stage=name,worker_pid=worker.pid,
                worker_start_ticks=None if actual is None else actual['start_ticks'],log=str(log))
            code = worker.wait()
            self.worker = None
        self.verify()
        if code != 0: raise RuntimeError(f'{name} failed ({code}): {log}')

    def final(self):
        if not self.parent_done(): raise RuntimeError('Scientific completion/publication/exit required')
        source = self.out/'extension_source_publication.json'
        if not source.exists() or read(source).get('status') != 'PUSHED':
            self.run_final_job('publish_concurrent_extension',self.here/'publish_extension.py',
                ['--root',self.root,'--controller-pid',self.pid,'--controller-start-ticks',self.me['start_ticks']])
        pushed(read(source)); verify(read(source)['published_files'])
        if read(source).get('runtime_source_bindings') != self.bound:
            raise RuntimeError('Published extension source inventory differs')
        original_pub = self.metrics/'source_publication.json'
        if not original_pub.exists() or read(original_pub).get('status') != 'PUSHED':
            self.run_final_job('publish_metric_source',self.oldsource/'publish.py',['--phase','source'])
        pushed(read(original_pub)); verify(read(original_pub)['source_bindings'])
        completion = self.metrics/'scoring_completion.json'
        if not completion.exists():
            self.run_final_job('replay_and_score_with_cache',self.here/'final_runner.py',
                ['--root',self.root,'--cache-dir',self.out],cpu=False)
        scored = read(completion)
        expected = dict(self.original_bound,**self.bound)
        if (scored.get('status') != 'COMPLETE' or scored.get('synthetic') is not False
                or scored.get('training_updates') != 0 or scored.get('policy_selection_updates') != 0
                or scored.get('parity_passed') is not True or scored.get('source_bindings') != expected):
            raise RuntimeError('Final metric coverage/parity/source receipt differs')
        for field in ('inputs','outputs'): verify(scored[field])
        result = self.root/'results/unified_metrics_20261002'
        if not (result/'metrics_analysis_completion.json').exists():
            self.run_final_job('paired_analysis',self.oldsource/'analysis.py',['--results-dir',result])
        analysis = read(result/'metrics_analysis_completion.json')
        if analysis.get('status') != 'COMPLETE': raise RuntimeError('Final analysis incomplete')
        for field in ('inputs','outputs'): verify(analysis[field])
        if not (self.metrics/'completion.json').exists():
            self.run_final_job('publish_metric_results',self.oldsource/'publish.py',['--phase','results'])
        final = read(self.metrics/'completion.json')
        if final.get('status') != 'UNIFIED_METRICS_COMPLETE': raise RuntimeError('Final metric completion invalid')
        pushed(final['publication'])
        self.status('COMPLETE',worker_pid=None,stopped_after_authorized_scope=True)
        write(self.out/'controller_completion.json',dict(status='COMPLETE',**self.me,
            metrics_completion_sha256=sha(self.metrics/'completion.json'),source_bindings=self.bound,
            training_updates=0,policy_selection_updates=0,stopped_after_authorized_scope=True))

    def run(self):
        own = self.lock_fn(self.out/'controller.lock')
        try:
            self.initialize(); self.takeover(); self.concurrent(); self.final()
        except BaseException as error:
            self.status('FAILED',error=str(error),worker_pid=None if self.worker is None else self.worker.pid,
                worker_start_ticks=None if self.worker is None else self.worker_identity['start_ticks'])
            if self.worker is not None: self.pause('CONTROLLER_FAILURE')
            raise
        finally:
            if self.owner_lock is not None: self.owner_lock.close()
            own.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root',required=True)
    Controller(parser.parse_args().root).run()


if __name__ == '__main__': main()
