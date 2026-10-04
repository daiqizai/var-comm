"""Publish real Swin results first, then resume the unchanged paired HiFi run."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import swin_early_delivery_common as c

class Paused(RuntimeError):pass

def status_paths(config,stage):
    p=c.locations(config['root'])
    if stage in ('reconstruct','score','report'):return [p['evaluation']/(stage+'_status.json')]
    if stage=='resume_hifi':return [p['parent_delivery']/'status.json']
    return []

def failure_paths(config,stage):
    p=c.locations(config['root'])
    if stage in ('reconstruct','score','report'):return [p['evaluation']/(stage+'_failure.json')]
    if stage=='publish':return [p['child']/'publication_failure.json']
    return [p['delivery']/'hifi_resume_failure.json',p['parent_delivery']/'failure.json']

def exact_status(config,stage,launch):
    for path in status_paths(config,stage):
        if not path.exists() or path.stat().st_mtime_ns<launch['started_ns']:continue
        value=c.read(path)
        if value.get('pid')!=launch['pid']:continue
        if stage=='resume_hifi':
            if value.get('config_sha256')==config['parent_config_sha256']:return value
        elif value.get('launch_id')==launch['launch_id']:return value
    return None

def safe_pause(config,stage,launch):
    value=exact_status(config,stage,launch)
    return bool(value and value.get('safe_pause_handler_installed') is True)

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
    c.require(config['bindings'].get(path)==c.sha(path),'Manual Swin launch was not frozen')
    value=c.read(path);command,environment,gpu=c.stage_spec(config,'reconstruct',value['launch_id'])
    c.require(value.get('command')==command and value.get('environment')==environment and value.get('stage')=='reconstruct',
        'Manual Swin command or environment differs')
    c.require(value.get('config_sha256',value.get('science_config_sha256'))==c.sha(config['science_config']),
        'Manual Swin science configuration differs')
    started=value.get('started_ns')
    if started is None:started=int(value['started_unix']*1e9)
    return dict(value,config_sha256=digest,started_ns=started,gpu=gpu,manual_launch_path=path)

def register(root):
    p=c.locations(root);parent_path=p['parent_config'];parent=c.read(parent_path)
    c.verify(parent['bindings']);c.require((p['child']/'config.json').is_file(),'Register paused Swin science first')
    config=dict(version=c.VERSION,root=str(p['root']),output=str(p['delivery']),science_config=str(p['child']/'config.json'),
        parent_config=str(parent_path),parent_config_sha256=c.sha(parent_path),pause_evidence=str(p['child']/'pause_verified.json'),
        pause_admission=str(p['delivery']/'pause_admission.json'),selected_step=80000,training_resume_allowed=False,
        resume_unchanged_hifi_after_publication=True,poll_seconds=15)
    common,_=c.parent_modules(config);common.validate_config(parent);common.assert_original_paused(parent)
    snapshot=c.read(config['pause_evidence'])
    owners=[c.read(p['parent']/'reconstruct_launch.json')]
    for name in ('controller_launch.json','dispatch.json'):
        path=p['parent_delivery']/name
        if path.exists():owners.append(c.read(path))
    for path in sorted((p['parent_delivery']/'launches').glob('*.json')):owners.append(c.read(path))
    unique={(owner['pid'],str(owner['start_ticks'])):owner for owner in owners}
    admission=dict(status='PAIRED_FIXED80K_SAFE_PAUSE_ADMITTED',parent_config_sha256=c.sha(parent_path),
        owners=list(unique.values()),completed_physical_frames=snapshot['completed_physical_frames'],
        original_manual_launch_sha256=c.sha(p['parent']/'reconstruct_launch.json'),
        pause_evidence_sha256=c.sha(config['pause_evidence']),training_resumed=False)
    c.seal(config['pause_admission'],admission);c.snapshot_gate(config)
    required=(*c.OWN_FILES,'swin_early.py','swin_early_common.py','swin_early_score.py','test_swin_early.py',
        'swin_early_report.py','swin_early_publish.py')
    c.require(all((p['child']/name).is_file() for name in required),'All reviewed stages must exist before delivery registration')
    files=[*p['child'].glob('*.py'),*p['child'].glob('*.md'),p['child']/'config.json',p['child']/'pause_verified.json',
        p['child']/'user_request.json',Path(config['pause_admission']),parent_path]
    manual=p['child']/'reconstruct_launch.json'
    if manual.exists():config['manual_reconstruct']=str(manual);files.append(manual)
    science=c.read(config['science_config'])
    c.require(science.get('methods')==['SwinJSCC_new_shared'],'Exactly the registered Swin method required')
    c.verify(science['early_source_bindings'])
    config['bindings']={**parent['bindings'],**science['early_source_bindings'],**{str(path):c.sha(path) for path in files}}
    c.validate(config);path=p['delivery']/'config.json';c.seal(path,config)
    if manual.exists():manual_launch(config,c.sha(path))
    print('REGISTERED_SWIN_EARLY_DELIVERY',path,c.sha(path),flush=True);return path

class Controller:
    def __init__(self,path):
        self.path=Path(path).resolve();self.config=c.validate(c.read(path));self.digest=c.sha(path)
        self.p=c.locations(self.config['root']);self.out=self.p['delivery'];self.stop=False
    def status(self,status,**values):
        c.write(self.out/'status.json',dict(status=status,pid=os.getpid(),config_sha256=self.digest,
            safe_pause_handler_installed=True,training_resumed=False,updated=time.time(),**values))
    def existing(self,stage):
        path=self.out/'launches'/(stage+'_current.json')
        if not path.exists():return manual_launch(self.config,self.digest) if stage=='reconstruct' else None
        value=c.read(path);command,environment,gpu=c.stage_spec(self.config,stage,value['launch_id'])
        c.require(value.get('config_sha256')==self.digest and value.get('command')==command
            and value.get('environment')==environment and value.get('gpu')==gpu,'Existing worker launch differs')
        original=self.out/'launches'/(value['launch_id']+'.json')
        c.require(original.is_file() and c.read(original)==value,'Immutable launch record missing')
        return value
    def observe(self,stage,launch,child=None):
        sent=False
        while c.live(launch):
            if child is not None and child.poll() is not None:break
            if self.stop and not sent and safe_pause(self.config,stage,launch):signal_owned(launch);sent=True
            publication=c.receipt_path(self.config,'publish')
            published=publication.exists() and c.read(publication).get('status')=='PUSHED'
            self.status('WAITING_FOR_SAFE_PAUSE' if self.stop else 'RUNNING',stage=stage,worker=launch,
                swin_release_pushed=published)
            time.sleep(self.config['poll_seconds'])
        code=child.wait() if child is not None else None
        if code==75 or was_paused(self.config,stage,launch):raise Paused('Worker safely paused: '+stage)
        c.require(code in (0,None),'Worker failed; no automatic retry: '+stage+' exit '+str(code))
        c.require(not any(path.exists() for path in failure_paths(self.config,stage)),'Worker failure requires review: '+stage)
        proof=c.stage_proof(self.config,stage)
        c.seal(self.out/'stages'/(stage+'.json'),dict(status='COMPLETE',stage=stage,proof=proof,config_sha256=self.digest))
    def stage(self,stage):
        c.validate(self.config);c.require(c.sha(self.path)==self.digest,'Early delivery config changed')
        c.require(not any(path.exists() for path in failure_paths(self.config,stage)),'Prior failure requires review: '+stage)
        launch=self.existing(stage)
        if launch and c.live(launch):self.observe(stage,launch);return
        if c.stage_proof(self.config,stage,True):return
        if launch:c.require(was_paused(self.config,stage,launch),'Previous owner exited without complete or safe-pause proof')
        if self.stop:raise Paused('Stopped before '+stage)
        if stage=='resume_hifi':self.release_receipt()
        token=stage+'_'+uuid.uuid4().hex;command,environment,gpu=c.stage_spec(self.config,stage,token)
        if gpu:
            while c.gpu_pids():
                if self.stop:raise Paused('Stopped waiting for exclusive GPU')
                self.status('WAITING_FOR_EXCLUSIVE_GPU',stage=stage);time.sleep(self.config['poll_seconds'])
        started=time.time_ns();logpath=self.out/'logs'/(token+'.log');logpath.parent.mkdir(parents=True,exist_ok=True)
        with logpath.open('ab') as log:
            child=subprocess.Popen(command,cwd=self.config['root'],env=dict(os.environ,**environment),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owner=c.process(child.pid);c.require(owner is not None,'Worker exited before identity capture')
        launch=dict(owner,stage=stage,launch_id=token,command=command,environment=environment,gpu=gpu,
            started_ns=started,log=str(logpath),config_sha256=self.digest)
        c.seal(self.out/'launches'/(token+'.json'),launch);c.write(self.out/'launches'/(stage+'_current.json'),launch)
        self.observe(stage,launch,child)
    def release_receipt(self):
        proof=c.stage_proof(self.config,'publish');publication=c.read(proof['path'])
        result=dict(status='SWIN_EARLY_RELEASE_REPORTED_AND_PUSHED',sources=100,rows=1800,selected_step=80000,
            full_comparison_complete=False,training_resumed=False,commit=publication['commit'],
            config_sha256=self.digest,publication_sha256=proof['sha256'],
            next_authorized_step='Resume unchanged paired HiFi from preserved exact physical frames')
        c.seal(self.out/'swin_release_completion.json',result);return result
    def run_stages(self,active):
        for stage in c.STAGES:
            if self.stop and active and stage!=active[0]:continue
            self.stage(stage)
            if stage=='publish':self.release_receipt()
            if self.stop:raise Paused('Stopped after safe stage boundary: '+stage)
    def run(self):
        import fcntl
        self.out.mkdir(parents=True,exist_ok=True)
        with (self.out/'run.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            c.require(not (self.out/'failure.json').exists(),'Previous delivery failure requires review')
            signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True));signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
            owner=dict(c.process(os.getpid()),config_sha256=self.digest)
            c.seal(self.out/'controller_launches'/(str(owner['pid'])+'_'+owner['start_ticks']+'.json'),owner)
            c.write(self.out/'controller_launch.json',owner)
            try:
                active=[stage for stage in c.STAGES if (launch:=self.existing(stage)) and c.live(launch)]
                c.require(len(active)<=1,'Multiple early-release stage owners remain alive')
                self.run_stages(active)
                proof=c.stage_proof(self.config,'resume_hifi')
                c.seal(self.out/'completion.json',dict(status='SWIN_EARLY_RELEASE_AND_UNCHANGED_PARENT_DELIVERED',
                    training_resumed=False,selected_step=80000,full_comparison_complete=True,
                    config_sha256=self.digest,early_commit=self.release_receipt()['commit'],parent_completion=proof))
                self.status('COMPLETE');return 0
            except Paused as error:self.status('PAUSED',reason=str(error));return 75
            except BaseException as error:
                c.write(self.out/'failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False,
                    config_sha256=self.digest,time=time.time()));self.status('FAILED_REQUIRES_REVIEW',error=repr(error));raise

def detach(path):
    import fcntl
    path=Path(path).resolve();config=c.validate(c.read(path));out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c.require(not (out/'failure.json').exists(),'Review failure before resuming')
        for name in ('dispatch.json','controller_launch.json'):
            if (out/name).exists():c.require(not c.live(c.read(out/name)),'An early-release controller is already alive')
        parent=c.read(config['parent_config']);command=[parent['report_python'],'-B','-u',str(Path(__file__).resolve()),'--config',str(path)]
        with (out/'controller.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=config['root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        record=dict(c.process(child.pid),config_sha256=c.sha(path),command=command)
        c.seal(out/'dispatches'/(str(child.pid)+'_'+record['start_ticks']+'.json'),record);c.write(out/'dispatch.json',record)
        print('DETACHED_SWIN_EARLY_DELIVERY',record,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root');parser.add_argument('--config',type=Path)
    parser.add_argument('--register',action='store_true');parser.add_argument('--detach',action='store_true');args=parser.parse_args()
    if args.register:register(args.root)
    elif args.detach:detach(args.config)
    else:raise SystemExit(Controller(args.config).run())
