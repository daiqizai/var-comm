"""Durable fixed16 paired evaluation and checked publication, then STOP."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import hifi_fixed16_delivery_common as c

class Paused(RuntimeError):pass

def status_path(config,stage):
    return c.locations(config['root'])['evaluation']/(stage+'_status.json') if stage!='publish' else None

def failure_path(config,stage):
    p=c.locations(config['root'])
    return p['child']/'publication_failure.json' if stage=='publish' else p['evaluation']/(stage+'_failure.json')

def exact_status(config,stage,launch):
    path=status_path(config,stage)
    if path is None or not path.exists() or path.stat().st_mtime_ns<launch['started_ns']:return None
    value=c.read(path)
    return value if value.get('pid')==launch['pid'] and value.get('launch_id')==launch['launch_id'] else None

def safe_pause(config,stage,launch):
    value=exact_status(config,stage,launch);return bool(value and value.get('safe_pause_handler_installed') is True)

def was_paused(config,stage,launch):
    value=exact_status(config,stage,launch);return bool(value and value.get('status')=='PAUSED')

def signal_owned(launch):
    c.require(c.live(launch),'Pause process identity changed')
    if hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'):
        fd=os.pidfd_open(launch['pid'])
        try:c.require(c.live(launch),'Pause identity changed');signal.pidfd_send_signal(fd,signal.SIGTERM)
        finally:os.close(fd)
    else:c.require(c.live(launch),'Pause identity changed');os.kill(launch['pid'],signal.SIGTERM)

def manual_launch(config,digest):
    path=config.get('manual_reconstruct')
    if not path:return None
    c.require(config['bindings'].get(path)==c.sha(path),'Manual fixed16 launch was not frozen')
    value=c.read(path);command,environment,gpu=c.stage_spec(config,'reconstruct',value['launch_id'])
    c.require(value.get('command')==command and value.get('environment')==environment and value.get('stage')=='reconstruct',
        'Manual reconstruction command/environment differs')
    c.require(value.get('config_sha256',value.get('science_config_sha256'))==c.sha(config['science_config']),
        'Manual reconstruction scientific configuration differs')
    started=value.get('started_ns')
    if started is None:started=int(value['started_unix']*1e9)
    return dict(value,config_sha256=digest,started_ns=started,gpu=gpu,manual_launch_path=path)

def register(root):
    p=c.locations(root);parent=c.read(p['parent_config']);c.verify(parent['bindings'])
    config=dict(version=c.VERSION,root=str(p['root']),output=str(p['delivery']),science_config=str(p['child']/'config.json'),
        parent_config=str(p['parent_config']),parent_config_sha256=c.sha(p['parent_config']),
        pause_evidence=str(p['child']/'pause_verified.json'),pause_admission=str(p['delivery']/'pause_admission.json'),
        selected_step=80000,training_resume_allowed=False,resume_full_queue_allowed=False,stop_after_publication=True,
        sources=16,physical_frames=64,rows=128,source_indices=list(c.FIXED),snrs=[7,13],noise_seeds=[2001],budgets=[1024,2048],poll_seconds=15)
    c.frozen_parent(config)
    owners=[c.read(launch) for launch,_ in c.old_queue_paths(config)]
    c.seal(config['pause_admission'],dict(status='FULL_QUEUES_SAFELY_PAUSED_FOR_FIXED16',owners=owners,
        pause_evidence_sha256=c.sha(config['pause_evidence']),training_resume_allowed=False,full_queue_resume_allowed=False))
    c.paused_queues(config)
    required=(*c.OWN_FILES,'hifi16_eval.py','hifi_fixed16_report.py','hifi_fixed16_publish.py')
    c.require(all((p['child']/name).is_file() for name in required),'All reviewed stage sources must exist before registration')
    science=c.read(config['science_config'])
    c.require(science.get('methods')==list(c.METHODS) and science.get('source_indices')==list(c.FIXED)
        and science.get('source_count')==16 and science.get('physical_frames')==64 and science.get('rows')==128
        and science.get('total_N')==[1024,2048] and science.get('snrs')==[7,13] and science.get('noise_seeds')==[2001]
        and science.get('full_comparison_complete') is False and science.get('full_queue_resume_allowed') is False,
        'Scientific configuration exceeds or differs from the requested 64-frame tier')
    c.verify(science['fixed16_source_bindings'])
    files=[*p['child'].glob('*.py'),*p['child'].glob('*.md'),p['child']/'config.json',p['child']/'pause_verified.json',
        p['child']/'user_request.json',Path(config['pause_admission']),p['parent_config']]
    manual=p['child']/'reconstruct_launch.json'
    if manual.exists():config['manual_reconstruct']=str(manual);files.append(manual)
    config['bindings']={**parent['bindings'],**science['fixed16_source_bindings'],**{str(path):c.sha(path) for path in files}}
    c.validate(config);path=p['delivery']/'config.json';c.seal(path,config)
    if manual.exists():manual_launch(config,c.sha(path))
    print('REGISTERED_FIXED16_DELIVERY_STOP_AFTER_PUBLICATION',path,c.sha(path),flush=True);return path

class Controller:
    def __init__(self,path):
        self.path=Path(path).resolve();self.config=c.validate(c.read(path));self.digest=c.sha(path)
        self.out=Path(self.config['output']);self.stop=False
    def status(self,status,**values):
        c.write(self.out/'status.json',dict(status=status,pid=os.getpid(),config_sha256=self.digest,
            safe_pause_handler_installed=True,training_resumed=False,full_queue_resumed=False,updated=time.time(),**values))
    def existing(self,stage):
        path=self.out/'launches'/(stage+'_current.json')
        if not path.exists():return manual_launch(self.config,self.digest) if stage=='reconstruct' else None
        value=c.read(path);command,environment,gpu=c.stage_spec(self.config,stage,value['launch_id'])
        c.require(value.get('config_sha256')==self.digest and value.get('command')==command
            and value.get('environment')==environment and value.get('gpu')==gpu,'Existing worker differs from frozen launch')
        original=self.out/'launches'/(value['launch_id']+'.json')
        c.require(original.is_file() and c.read(original)==value,'Immutable worker launch missing');return value
    def observe(self,stage,launch,child=None):
        signalled=False
        while c.live(launch):
            if child is not None and child.poll() is not None:break
            if self.stop and not signalled and safe_pause(self.config,stage,launch):signal_owned(launch);signalled=True
            self.status('WAITING_FOR_SAFE_PAUSE' if self.stop else 'RUNNING',stage=stage,worker=launch)
            time.sleep(self.config['poll_seconds'])
        code=child.wait() if child is not None else None
        if code==75 or was_paused(self.config,stage,launch):raise Paused('Worker paused at a safe boundary: '+stage)
        c.require(code in (0,None),'Worker failed; no automatic retry: '+stage+' exit '+str(code))
        c.require(not failure_path(self.config,stage).exists(),'Worker failure requires review: '+stage)
        proof=c.stage_proof(self.config,stage)
        c.seal(self.out/'stages'/(stage+'.json'),dict(status='COMPLETE',stage=stage,proof=proof,config_sha256=self.digest))
    def stage(self,stage):
        c.validate(self.config);c.require(c.sha(self.path)==self.digest,'Fixed16 delivery configuration changed')
        c.require(not failure_path(self.config,stage).exists(),'Previous stage failure requires review: '+stage)
        launch=self.existing(stage)
        if launch and c.live(launch):self.observe(stage,launch);return
        if c.stage_proof(self.config,stage,True):return
        if launch:c.require(was_paused(self.config,stage,launch),'Exited worker lacks complete or safe-pause proof; review required')
        if self.stop:raise Paused('Stopped before '+stage)
        token=stage+'_'+uuid.uuid4().hex;command,environment,gpu=c.stage_spec(self.config,stage,token)
        if gpu:
            while c.gpu_pids():
                if self.stop:raise Paused('Stopped waiting for exclusive GPU')
                self.status('WAITING_FOR_EXCLUSIVE_GPU',stage=stage);time.sleep(self.config['poll_seconds'])
        started=time.time_ns();logpath=self.out/'logs'/(token+'.log');logpath.parent.mkdir(parents=True,exist_ok=True)
        with logpath.open('ab') as log:
            child=subprocess.Popen(command,cwd=self.config['root'],env=dict(os.environ,**environment),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owner=c.process(child.pid);c.require(owner is not None,'Worker exited before exact identity capture')
        launch=dict(owner,stage=stage,launch_id=token,command=command,environment=environment,gpu=gpu,
            started_ns=started,log=str(logpath),config_sha256=self.digest)
        c.seal(self.out/'launches'/(token+'.json'),launch);c.write(self.out/'launches'/(stage+'_current.json'),launch)
        self.observe(stage,launch,child)
    def run_stages(self,active):
        for stage in c.STAGES:
            if self.stop and active and stage!=active[0]:continue
            self.stage(stage)
            if self.stop:raise Paused('Stopped after safe stage boundary: '+stage)
    def run(self):
        import fcntl
        self.out.mkdir(parents=True,exist_ok=True)
        with (self.out/'run.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            c.require(not (self.out/'failure.json').exists(),'Previous fixed16 delivery failure requires review')
            signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True));signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
            owner=dict(c.process(os.getpid()),config_sha256=self.digest)
            c.seal(self.out/'controller_launches'/(str(owner['pid'])+'_'+owner['start_ticks']+'.json'),owner)
            c.write(self.out/'controller_launch.json',owner)
            try:
                active=[stage for stage in c.STAGES if (launch:=self.existing(stage)) and c.live(launch)]
                c.require(len(active)<=1,'Multiple fixed16 stage owners remain alive')
                self.run_stages(active);proof=c.stage_proof(self.config,'publish');publication=c.read(proof['path'])
                c.paused_queues(self.config)
                c.seal(self.out/'completion.json',dict(status='HIFI_FIXED16_REPORTED_PUSHED_AND_STOPPED',
                    sources=16,physical_frames=64,rows=128,selected_step=80000,training_resumed=False,full_queue_resumed=False,
                    stop=True,full_comparison_complete=False,config_sha256=self.digest,commit=publication['commit'],publication=proof))
                self.status('COMPLETE_STOPPED',commit=publication['commit'],stop=True,full_comparison_complete=False);return 0
            except Paused as error:self.status('PAUSED',reason=str(error));return 75
            except BaseException as error:
                c.write(self.out/'failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False,
                    config_sha256=self.digest,time=time.time()));self.status('FAILED_REQUIRES_REVIEW',error=repr(error));raise

def detach(path):
    import fcntl
    path=Path(path).resolve();config=c.validate(c.read(path));out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c.require(not (out/'failure.json').exists(),'Review previous failure before restarting')
        for name in ('dispatch.json','controller_launch.json'):
            if (out/name).exists():c.require(not c.live(c.read(out/name)),'A fixed16 controller is already alive')
        parent=c.read(config['parent_config']);command=[parent['report_python'],'-B','-u',str(Path(__file__).resolve()),'--config',str(path)]
        with (out/'controller.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=config['root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        who=c.process(child.pid);c.require(who is not None,'Detached controller exited before identity capture')
        record=dict(who,config_sha256=c.sha(path),command=command)
        c.seal(out/'dispatches'/(str(child.pid)+'_'+record['start_ticks']+'.json'),record);c.write(out/'dispatch.json',record)
        print('DETACHED_FIXED16_DELIVERY',record,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root');parser.add_argument('--config',type=Path)
    parser.add_argument('--register',action='store_true');parser.add_argument('--detach',action='store_true');args=parser.parse_args()
    if args.register:register(args.root)
    elif args.detach:detach(args.config)
    else:raise SystemExit(Controller(args.config).run())
