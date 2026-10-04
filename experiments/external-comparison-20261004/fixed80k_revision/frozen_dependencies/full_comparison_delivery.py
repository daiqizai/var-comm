"""Isolated, failure-stopping continuation after the six-stage external run."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import full_comparison_common as c


class Waiting(RuntimeError):pass
class Paused(RuntimeError):pass


def upstream_gate(config,reader=c.process):
    p=c.locations(config['root']);folder=p['out']/'controller';config_path=Path(config['external_controller_config'])
    c.require(c.sha(config_path)==config['bindings'][str(config_path)],'External controller config changed')
    if (folder/'failure.json').exists():raise RuntimeError('External controller failed; review required')
    path=folder/'completion.json'
    if not path.exists():
        state=c.read(folder/'status.json') if (folder/'status.json').exists() else {}
        if state.get('status')=='PAUSED':raise Paused('External controller is paused')
        raise Waiting('Waiting for complete external training and evaluation')
    done=c.read(path)
    c.require(done.get('status')=='EXTERNAL_TRAINING_AND_EVALUATION_COMPLETE'
        and done.get('config_sha256')==c.sha(config_path)
        and set(done.get('stages',{}))=={'qualification','hifi_preflight','training','hifi_qualification','reconstruct','score'},
        'Complete six-stage external evaluation required')
    launch=c.read(folder/'controller_launch.json')
    c.require(launch.get('config_sha256')==c.sha(config_path),'External owner/config identity differs')
    owners=[launch,*[c.read(x) for x in sorted((folder/'launches').glob('*.json'))]]
    for owner in owners:
        if c.live(owner,reader):raise Waiting('External completion exists; waiting for its exact owner/worker exit')
    bindings={str(path):c.sha(path),str(config_path):c.sha(config_path)}
    for proof in done['stages'].values():
        c.verify({proof['path']:proof['sha256']})
        c.require(c.read(proof['path']).get('status')==proof['status'],'External stage receipt differs')
        bindings[proof['path']]=proof['sha256']
    # Snapshot process records as values. Mutable dispatch/status/launch files
    # are never permanent scientific inputs.
    return dict(status='EXTERNAL_SIX_STAGES_COMPLETE_AND_OWNERS_EXITED',bindings=bindings,
        owners=owners,external_config_sha256=c.sha(config_path),original_processes_signalled=False)


def external_delivery_gate(config,reader=c.process):
    p=c.locations(config['root']);folder=p['out']/'delivery'
    if (folder/'failure.json').exists():raise RuntimeError('Earlier external report/publication failed; review required')
    path=folder/'completion.json';launch_path=folder/'launch.json'
    if not path.exists() or not launch_path.exists():raise Waiting('Waiting for earlier external report and normal publication')
    done=c.read(path);launch=c.read(launch_path)
    c.require(done.get('status')=='EXTERNAL_BASELINES_REPORTED_AND_PUSHED'
        and done.get('full_plan_complete') is False,'Earlier external delivery receipt differs')
    if c.live(launch,reader):raise Waiting('Waiting for earlier CPU report/publisher owner exit')
    publication_path=p['out']/'results_publication.json';publication=c.read(publication_path)
    c.require(publication.get('status')=='PUSHED' and publication.get('checks')=='PASS'
        and publication.get('phase')=='results' and publication.get('commit')==publication.get('remote_commit')==done.get('commit'),
        'Earlier external results lack a checked normal publication')
    return dict(status='EARLIER_EXTERNAL_DELIVERY_PUSHED_AND_EXITED',owner=launch,
        bindings={str(path):c.sha(path),str(publication_path):c.sha(publication_path)})


def failure_paths(config,stage):
    p=c.locations(config['root'])
    if stage=='own_controls':return [p['own']/(s+'_failure.json') for s in c.OWN_STAGES]
    if stage=='score':return [p['own']/'score_failure.json']
    if stage.startswith('cost_'):return [p['cost']/(stage[5:]+'_failure.json')]
    if stage=='full_report':return [p['final']/'failure.json']
    return [p['delivery']/'publication_failure.json']


def status_paths(config,stage):
    p=c.locations(config['root'])
    if stage=='own_controls':return [p['own']/(s+'_status.json') for s in c.OWN_STAGES]
    if stage=='score':return [p['own']/'score_status.json']
    if stage.startswith('cost_'):return [p['cost']/(stage[5:]+'_status.json')]
    if stage=='full_report':return [p['final']/'status.json']
    return []


def pause_ready(config,stage,launch):
    if stage=='publish':return False  # A Git transaction must finish normally.
    for path in status_paths(config,stage):
        if not path.exists() or path.stat().st_mtime_ns<launch['started_ns']:continue
        state=c.read(path)
        if state.get('pid')!=launch['pid']:continue
        if stage=='own_controls':
            # Frozen Runner.run installs handlers before its first stage status.
            if state.get('stage') in c.OWN_STAGES:return True
        elif state.get('launch_id')==launch['launch_id'] and state.get('safe_pause_handler_installed') is True:return True
    return False


def was_paused(config,stage,launch):
    for path in status_paths(config,stage):
        if not path.exists() or path.stat().st_mtime_ns<launch['started_ns']:continue
        state=c.read(path)
        if state.get('status')=='PAUSED' and state.get('pid')==launch['pid'] and (
            stage=='own_controls' or state.get('launch_id')==launch['launch_id']):return True
    return False


class Delivery:
    def __init__(self,config_path):
        self.config_path=Path(config_path).resolve();self.config=c.validate_config(c.read(self.config_path))
        self.config_sha=c.sha(self.config_path);self.paths=c.locations(self.config['root']);self.out=self.paths['delivery']
        self.stop=False;self.child=None

    def verify(self):
        c.require(c.sha(self.config_path)==self.config_sha,'Full comparison configuration changed')
        c.verify(self.config['bindings'])

    def status(self,status,**fields):
        c.write(self.out/'status.json',dict(status=status,pid=os.getpid(),config_sha256=self.config_sha,
            updated=time.time(),safe_pause_handler_installed=True,**fields))

    def wait_upstream(self,active=False):
        while True:
            if self.stop and not active:raise Paused('Stopped while waiting for external completion')
            try:
                value=upstream_gate(self.config)
                path=self.out/'upstream_admission.json';c.seal(path,value);return
            except Waiting as error:
                c.require(not active,'An active child has lost its completed upstream admission')
                self.status('WAITING_FOR_EXTERNAL_COMPLETION_AND_EXIT',reason=str(error))
                time.sleep(self.config['poll_seconds'])

    def existing(self,stage):
        path=self.out/'launches'/(stage+'_current.json')
        if not path.exists():return None
        value=c.read(path);command,environment,gpu=c.stage_spec(self.config,stage,value['launch_id'])
        c.require(value.get('config_sha256')==self.config_sha and value.get('stage')==stage
            and value.get('command')==command and value.get('environment')==environment and value.get('gpu')==gpu,
            'Existing worker differs from frozen command/environment')
        original=self.out/'launches'/(value['launch_id']+'.json')
        c.require(original.is_file() and c.read(original)==value,'Worker has no matching immutable launch record')
        return value

    def observe(self,stage,launch,child=None):
        forwarded=False
        while c.live(launch):
            if child is not None and child.poll() is not None:break
            if self.stop and not forwarded and pause_ready(self.config,stage,launch):
                # Exact PID and kernel start tick were checked immediately above.
                os.kill(launch['pid'],signal.SIGTERM);forwarded=True
            self.status('RUNNING' if not self.stop else 'WAITING_FOR_SAFE_PAUSE',stage=stage,worker=launch,
                adopted=child is None,signal_forwarded=forwarded)
            time.sleep(self.config['poll_seconds'])
        code=child.wait() if child is not None else None
        if code==75:raise Paused('Worker paused at its registered safe boundary: '+stage)
        if code not in (None,0):raise RuntimeError('Worker failed; no automatic retry: '+stage+' exit '+str(code))
        for path in failure_paths(self.config,stage):c.require(not path.exists(),'Stage needs review: '+str(path))
        proof=c.stage_proof(self.config,stage,missing_ok=True)
        if proof is None:
            if was_paused(self.config,stage,launch):
                raise Paused('Adopted worker paused: '+stage)
            raise RuntimeError('Worker exited without a complete bound receipt: '+stage)
        c.seal(self.out/'stages'/(stage+'.json'),dict(status='COMPLETE',stage=stage,
            config_sha256=self.config_sha,launch_id=launch['launch_id'],proof=proof))
        self.child=None

    def wait_gpu(self):
        while True:
            if self.stop:raise Paused('Stopped before next GPU stage')
            owners=c.gpu_pids()
            if not owners:return
            self.status('WAITING_FOR_EXCLUSIVE_GPU',gpu_owners=owners)
            time.sleep(self.config['poll_seconds'])

    def run_stage(self,stage):
        self.verify()
        for path in failure_paths(self.config,stage):c.require(not path.exists(),'Prior stage failure requires review: '+str(path))
        launch=self.existing(stage)
        if launch and c.live(launch):
            self.observe(stage,launch);return
        proof=c.stage_proof(self.config,stage,missing_ok=True)
        if proof:
            self.status('REUSING_COMPLETE_STAGE',stage=stage);return
        if launch:c.require(was_paused(self.config,stage,launch),'Previous worker exited without completion or exact safe-pause proof; review required')
        if self.stop:raise Paused('Stopped before launching '+stage)
        if stage=='publish':
            while True:
                try:
                    c.seal(self.out/'earlier_delivery_admission.json',external_delivery_gate(self.config));break
                except Waiting as error:
                    if self.stop:raise Paused('Stopped while waiting for earlier publication')
                    self.status('WAITING_FOR_EARLIER_PUBLICATION',reason=str(error));time.sleep(self.config['poll_seconds'])
        token=stage+'_'+uuid.uuid4().hex;command,environment,gpu=c.stage_spec(self.config,stage,token)
        if gpu:self.wait_gpu()
        logpath=self.out/'logs'/(token+'.log');logpath.parent.mkdir(parents=True,exist_ok=True)
        started_ns=time.time_ns()
        with logpath.open('ab',buffering=0) as stream:
            child=subprocess.Popen(command,cwd=self.paths['root'],env=dict(os.environ,**environment),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.child=child;who=c.process(child.pid)
        c.require(who is not None,'Cannot register actual child process')
        launch=dict(pid=child.pid,start_ticks=who['start_ticks'],stage=stage,launch_id=token,
            config_sha256=self.config_sha,command=command,environment=environment,gpu=gpu,
            log=str(logpath),started_ns=started_ns)
        c.seal(self.out/'launches'/(token+'.json'),launch);c.write(self.out/'launches'/(stage+'_current.json'),launch)
        self.observe(stage,launch,child)

    def run(self):
        import fcntl
        self.out.mkdir(parents=True,exist_ok=True)
        with (self.out/'run.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            c.require(not (self.out/'failure.json').exists(),'Prior full comparison failure needs review')
            if (self.out/'completion.json').exists():
                done=c.read(self.out/'completion.json')
                c.require(done.get('status')=='FULL_MATCHED_COMPARISON_REPORTED_AND_PUSHED'
                    and done.get('config_sha256')==self.config_sha and done.get('full_plan_complete') is True
                    and done.get('publication_sha256')==c.sha(self.out/'publication.json'),'Existing complete delivery differs')
                from publish_full_comparison import publish
                published=publish(self.config_path)
                c.require(done.get('commit')==published['commit'],'Completed public commit differs')
                self.status('COMPLETE',full_plan_complete=True,reused=True);return 0
            owner=c.process(os.getpid());c.require(owner is not None,'Controller process identity missing')
            record=dict(pid=os.getpid(),start_ticks=owner['start_ticks'],config_sha256=self.config_sha)
            c.seal(self.out/'controller_launches'/(str(owner['pid'])+'_'+owner['start_ticks']+'.json'),record)
            c.write(self.out/'controller_launch.json',record)
            signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True))
            signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
            try:
                active=[stage for stage in c.STAGES if (value:=self.existing(stage)) and c.live(value)]
                c.require(len(active)<=1,'Multiple prior full-comparison workers remain alive')
                self.wait_upstream(active=bool(active))
                self.run_stages(active)
                publication=c.read(self.out/'publication.json')
                done=dict(status='FULL_MATCHED_COMPARISON_REPORTED_AND_PUSHED',full_plan_complete=True,
                    commit=publication['commit'],publication_sha256=c.sha(self.out/'publication.json'),
                    stop=True,**record)
                c.seal(self.out/'completion.json',done);self.status('COMPLETE',full_plan_complete=True);return 0
            except Paused as error:
                self.status('PAUSED',reason=str(error),resume_same_configuration=True);return 75
            except BaseException as error:
                failure=dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),config_sha256=self.config_sha,
                    automatic_retry=False,surviving_child_pid=self.child.pid if self.child else None)
                if not (self.out/'failure.json').exists():c.write(self.out/'failure.json',failure)
                self.status('FAILED_REQUIRES_REVIEW',error=repr(error));raise

    def run_stages(self,active):
        for stage in c.STAGES:
            # A resumed controller with a stop request still reaches its
            # existing child, so it can safely pause that owner.
            if self.stop and active and stage!=active[0]:continue
            self.run_stage(stage)
            if self.stop:raise Paused('Stopped after safe completion of '+stage)


def register(root,config_path=None):
    p=c.locations(root);path=Path(config_path).resolve() if config_path else p['delivery']/'config.json'
    c.require(path==p['delivery']/'config.json','Use the dedicated immutable full-delivery config path')
    config=c.read(path) if path.exists() else c.default_config(root)
    bindings=c.required_sources(config)
    if 'bindings' in config:c.require(config['bindings']==bindings,'Existing registration sources changed')
    config['bindings']=bindings;c.validate_config(config);c.seal(path,config)
    print('REGISTERED_FULL_COMPARISON',path,c.sha(path),flush=True);return path


def detach(config_path):
    import fcntl
    path=Path(config_path).resolve();config=c.validate_config(c.read(path));out=Path(config['output'])
    out.mkdir(parents=True,exist_ok=True)
    with (out/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c.require(not (out/'failure.json').exists(),'Failed full comparison requires review')
        for name in ('dispatch.json','controller_launch.json'):
            if (out/name).exists():
                old=c.read(out/name);c.require(old['config_sha256']==c.sha(path),'Existing controller config differs')
                if c.live(old):print('ALREADY_RUNNING',old['pid'],flush=True);return
        log=out/'controller.log';command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(path)]
        with log.open('ab',buffering=0) as stream:
            child=subprocess.Popen(command,cwd=config['root'],stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        who=c.process(child.pid);c.require(who is not None,'Controller exited before dispatch registration')
        record=dict(pid=child.pid,start_ticks=who['start_ticks'],command=command,config_sha256=c.sha(path),log=str(log))
        c.seal(out/'dispatches'/(str(child.pid)+'_'+who['start_ticks']+'.json'),record);c.write(out/'dispatch.json',record)
        print('DETACHED_FULL_COMPARISON',child.pid,who['start_ticks'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root');parser.add_argument('--config',type=Path)
    parser.add_argument('--register',action='store_true');parser.add_argument('--detach',action='store_true')
    args=parser.parse_args()
    if args.register:register(args.root,args.config)
    elif args.detach:detach(args.config)
    else:raise SystemExit(Delivery(args.config).run())
