"""Durable exact-80k evaluation and one final checked normal publication."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import fixed80k_delivery_common as c

class Paused(RuntimeError):pass

def register(root):
    p=c.locations(root);original_path=p['out']/'full_delivery/config.json';original=c.read(original_path)
    c.verify(original['bindings']);evaluation=p['revision']/'external_eval_config.json'
    config={**original,'version':c.VERSION,'output':str(p['delivery']),'requested_step':80000,
        'training_resume_allowed':False,'full_plan_scope':'original_five_method_comparison_with_user_fixed_80k_external_checkpoint',
        'pause_evidence':str(p['revision']/'pause_verified.json'),'external_eval_config':str(evaluation),
        'original_full_config':str(original_path),'poll_seconds':15}
    paths=list(p['revision'].glob('*.py'))+list(p['revision'].glob('*.md'))+[original_path,evaluation,p['revision']/'pause_verified.json',
        p['revision']/'user_request.json',p['revision']/'fixed80k_registration.json']
    manual=p['revision']/'reconstruct_launch.json'
    if manual.exists():paths.append(manual)
    config['bindings']={**original['bindings'],**{str(v):c.sha(v) for v in paths}}
    c.validate_config(config);c.assert_original_paused(config)
    path=p['delivery']/'config.json';c.seal(path,config);return path

def stage_kind(stage):
    if stage in ('hifi_qualification','reconstruct','cost_external'):return 'swin',True
    if stage in ('external_score','score'):return 'metric',True
    if stage in ('own_controls','cost_own'):return 'native',True
    return 'report',False

def manual_reconstruction(config):
    p=c.locations(config['root']);path=p['revision']/'reconstruct_launch.json'
    if not path.exists():return None
    c.require(config['bindings'].get(str(path))==c.sha(path),'Manual reconstruction was not registered')
    record=c.read(path);adapter=p['revision']/'fixed80k_adapter.py'
    command=[config['swin_python'],'-B','-u',str(adapter),'--root',config['root'],'--stage','reconstruct','--launch-id',record['launch_id']]
    environment=dict(config['swin_environment'],PYTHONDONTWRITEBYTECODE='1')
    c.require(record['command']==command and record['environment']==environment and record['stage']=='reconstruct'
        and record['selected_checkpoint_step']==80000 and record['adapter_sha256']==c.sha(adapter)
        and record['evaluation_config_sha256']==c.sha(config['external_eval_config']),'Manual reconstruction launch differs')
    current=c.process(record['pid'])
    if current and current['start_ticks']==str(record['start_ticks']):
        c.require(current['command']==command,'Active manual process command differs')
    return dict(record,started_ns=int(record['started_unix']*1e9),config_sha256=c.sha(Path(config['output'])/'config.json'))

def failure_paths(config,stage):
    p=c.locations(config['root']);ev=c.read(config['external_eval_config']);out=Path(ev['output'])
    if stage in ('reconstruct','external_score','external_report'):
        return [out/({'external_score':'score','external_report':'report'}.get(stage,stage)+'_failure.json')]
    if stage=='publish':return [p['delivery']/'publication_failure.json']
    if stage=='hifi_qualification':return []
    c.original_modules(config)
    from full_comparison_delivery import failure_paths as original
    return original(config,stage)

def status_paths(config,stage):
    p=c.locations(config['root']);ev=c.read(config['external_eval_config']);out=Path(ev['output'])
    if stage in ('reconstruct','external_score'):return [out/('reconstruct_status.json' if stage=='reconstruct' else 'score_status.json')]
    if stage=='external_report':return [out/'report_status.json']
    if stage=='own_controls':return [p['own']/(s+'_status.json') for s in ('qualify','screen','calibrate','development','export-n1024')]
    if stage=='score':return [p['own']/'score_status.json']
    if stage.startswith('cost_'):return [p['cost']/(stage[5:]+'_status.json')]
    if stage=='full_report':return [p['final']/'status.json']
    return []

def safe_to_pause(config,stage,launch):
    for path in status_paths(config,stage):
        if not path.exists() or path.stat().st_mtime_ns<launch['started_ns']:continue
        status=c.read(path)
        if status.get('pid')==launch['pid'] and (stage=='own_controls' or status.get('launch_id')==launch['launch_id']
            and status.get('safe_pause_handler_installed') is True):return True
    return False

def paused(config,stage,launch):
    return any(path.exists() and c.read(path).get('status')=='PAUSED' and c.read(path).get('pid')==launch['pid']
        for path in status_paths(config,stage))

def run_worker(config_path,stage,token):
    config=c.validate_config(c.read(config_path));c.assert_original_paused(config)
    root=config['root'];runtime=Path(config['runtime']);sys.path.insert(0,str(runtime))
    if stage in ('hifi_qualification','reconstruct','external_score'):
        import fixed80k_adapter as adapter
        sys.argv=[str(Path(adapter.__file__)),'--root',root,'--stage',
            {'hifi_qualification':'qualify','reconstruct':'reconstruct','external_score':'score'}[stage],'--launch-id',token]
        adapter.main();return
    if stage=='publish':
        import fixed80k_publish
        fixed80k_publish.publish(config_path);return
    if stage in ('external_report','full_report'):
        from fixed80k_report_adapter import install_reports
        external,own=install_reports(root)
        module=external if stage=='external_report' else own
        args=['--config',config['external_eval_config'],'--fixed-examples',config['fixed_examples'],'--launch-id',token]
        if stage=='full_report':args+=['--receiver-cost',str(c.locations(root)['cost']/'completion.json')]
    else:
        from fixed80k_adapter import install
        install(root)
    if stage in ('external_report','full_report'):pass
    elif stage=='own_controls':
        import own_controls as module
        args=['--root',root,'--stage','all','--protocol',str(runtime/'own_controls_protocol.json')]
    elif stage=='score':
        import own_controls_score as module
        args=['--root',root,'--launch-id',token]
    elif stage.startswith('cost_'):
        import own_controls_cost as module
        import own_controls_cost_common as cost
        original=cost.code_bindings
        cost.code_bindings=lambda:{**original(),str(Path(__file__).resolve()):c.sha(__file__),
            str(c.HERE/'fixed80k_adapter.py'):c.sha(c.HERE/'fixed80k_adapter.py')}
        args=['--root',root,'--config',config['external_eval_config'],'--stage',stage[5:],'--launch-id',token]
    else:raise RuntimeError('Unregistered worker stage')
    sys.argv=[module.__file__,*args]
    code=module.main()
    if code:raise SystemExit(code)

class Controller:
    def __init__(self,path):
        self.path=Path(path).resolve();self.config=c.validate_config(c.read(path));self.digest=c.sha(path)
        self.out=Path(self.config['output']);self.stop=False
    def status(self,status,**fields):
        c.write(self.out/'status.json',dict(status=status,pid=os.getpid(),config_sha256=self.digest,
            safe_pause_handler_installed=True,updated=time.time(),selected_step=80000,training_resumed=False,**fields))
    def observe(self,stage,launch,child=None):
        sent=False
        while c.live(launch):
            if child is not None and child.poll() is not None:break
            if self.stop and not sent and safe_to_pause(self.config,stage,launch):
                if hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'):
                    fd=os.pidfd_open(launch['pid'])
                    try:
                        c.require(c.live(launch),'Pause target identity changed')
                        signal.pidfd_send_signal(fd,signal.SIGTERM)
                    finally:os.close(fd)
                else:
                    c.require(c.live(launch),'Pause target identity changed')
                    os.kill(launch['pid'],signal.SIGTERM)
                sent=True
            self.status('WAITING_FOR_SAFE_PAUSE' if self.stop else 'RUNNING',stage=stage,worker=launch)
            time.sleep(self.config['poll_seconds'])
        code=child.wait() if child is not None else None
        if code==75 or paused(self.config,stage,launch):raise Paused('Worker safely paused: '+stage)
        c.require(code in (0,None),'Worker failed; review required: '+stage+' exit '+str(code))
        c.require(not any(path.exists() for path in failure_paths(self.config,stage)),'Worker failure receipt requires review: '+stage)
        proof=c.stage_proof(self.config,stage)
        c.seal(self.out/'stages'/(stage+'.json'),dict(stage=stage,status='COMPLETE',proof=proof,
            config_sha256=self.digest,launch_id=launch['launch_id']))
    def stage(self,stage):
        c.validate_config(self.config);c.require(c.sha(self.path)==self.digest,'Controller configuration changed')
        c.assert_original_paused(self.config)
        c.require(not any(path.exists() for path in failure_paths(self.config,stage)),'Previous failure requires review: '+stage)
        if self.stop:raise Paused('Stopped before '+stage)
        if stage=='reconstruct':
            manual=manual_reconstruction(self.config)
            if manual:
                self.observe(stage,manual);return
        current=self.out/'launches'/(stage+'_current.json')
        if current.exists():
            launch=c.read(current);c.require(launch['config_sha256']==self.digest,'Prior worker configuration differs')
            if c.live(launch):self.observe(stage,launch);return
            if c.stage_proof(self.config,stage,True):return
            c.require(paused(self.config,stage,launch),'Exited worker lacks safe pause or completion; review required')
        elif c.stage_proof(self.config,stage,True):return
        kind,gpu=stage_kind(stage)
        if gpu:
            while c.original_modules(self.config).gpu_pids():
                if self.stop:raise Paused('Stopped waiting for exclusive GPU')
                self.status('WAITING_FOR_EXCLUSIVE_GPU',stage=stage);time.sleep(self.config['poll_seconds'])
            # A reviewed manual full reconstruction may finish while this
            # controller is waiting for its GPU owner to exit. Reuse its real
            # bound receipt; never run the same 1,800 frames again.
            if c.stage_proof(self.config,stage,True):return
        token=stage+'_'+uuid.uuid4().hex
        command=[self.config[kind+'_python'],'-B','-u',str(Path(__file__).resolve()),'--config',str(self.path),
            '--worker',stage,'--launch-id',token]
        environment=dict(self.config[kind+'_environment']) if gpu else dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
        environment['PYTHONDONTWRITEBYTECODE']='1';started=time.time_ns()
        with (self.out/(stage+'.log')).open('ab') as log:
            proc=subprocess.Popen(command,cwd=self.config['root'],env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owner=c.process(proc.pid);c.require(owner is not None,'Worker exited before identity capture')
        launch=dict(owner,stage=stage,launch_id=token,config_sha256=self.digest,started_ns=started,command=command,
            environment=environment,gpu=gpu)
        c.seal(self.out/'launches'/(token+'.json'),launch);c.write(current,launch);self.observe(stage,launch,proc)
    def run(self):
        import fcntl
        self.out.mkdir(parents=True,exist_ok=True)
        with (self.out/'run.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            c.require(not (self.out/'failure.json').exists(),'Prior controller failure requires review')
            signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True));signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
            launch=dict(c.process(os.getpid()),config_sha256=self.digest)
            c.seal(self.out/'controller_launches'/(str(os.getpid())+'_'+launch['start_ticks']+'.json'),launch)
            c.write(self.out/'controller_launch.json',launch)
            try:
                for stage in c.STAGES:self.stage(stage)
                c.final_table_gate(self.config);publication=c.read(self.out/'publication.json')
                done=dict(status='USER_FIXED80K_MATCHED_EVALUATION_REPORTED_AND_PUSHED',selected_step=80000,
                    actual_training_pause_step=81551,original_training_finished=False,training_resumed=False,
                    full_plan_complete=True,config_sha256=self.digest,commit=publication['commit'],
                    publication_sha256=c.sha(self.out/'publication.json'))
                c.seal(self.out/'completion.json',done);self.status('COMPLETE',commit=publication['commit']);return 0
            except Paused as error:self.status('PAUSED',reason=str(error));return 75
            except BaseException as error:
                c.write(self.out/'failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False,
                    config_sha256=self.digest,time=time.time()));self.status('FAILED_REQUIRES_REVIEW',error=repr(error));raise

def detach(path):
    config=c.validate_config(c.read(path));out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    for name in ('dispatch.json','controller_launch.json'):
        if (out/name).exists():c.require(not c.live(c.read(out/name)),'Existing revision controller is alive')
    with (out/'controller.log').open('ab') as log:
        proc=subprocess.Popen([config['report_python'],'-B','-u',str(Path(__file__).resolve()),'--config',str(path)],
            cwd=config['root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    owner=c.process(proc.pid);c.require(owner is not None,'Detached revision controller exited early')
    record=dict(owner,config_sha256=c.sha(path));c.write(out/'dispatch.json',record);print(record,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root');parser.add_argument('--config',type=Path)
    parser.add_argument('--register',action='store_true');parser.add_argument('--detach',action='store_true')
    parser.add_argument('--worker',choices=c.STAGES);parser.add_argument('--launch-id')
    args=parser.parse_args()
    if args.register:print(register(args.root));return
    c.require(args.config is not None,'Explicit revision configuration required')
    if args.worker:run_worker(args.config,args.worker,args.launch_id);return
    if args.detach:detach(args.config);return
    raise SystemExit(Controller(args.config).run())

if __name__=='__main__':main()
