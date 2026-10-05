"""Bounded registered H-stage batches; no implied C or holdout continuation.

Stage commands are supplied by a sealed execution revision. Only implemented,
bound entry points can run. An H qualification pause writes a priority handoff
request for the separately registered main-system calibration; no placeholder
command or old queue is started here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import math
import re
import signal
import sqlite3
import subprocess
import time
import traceback

QUALIFICATION_START = 1791219005.9549868
QUALIFICATION_DEADLINE = QUALIFICATION_START + 96*3600
STAGE_IDS = frozenset(('qualification', 'source', 'coarse', 'clean_quality', 'initial_true200',
    'refine', 'whole_calibration', 'partial_calibration', 'freeze', 'development',
    'render', 'timing', 'report'))


def require(test, message):
    if not test:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def save(path, value, exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    require(not exclusive or not path.exists(),'Prior evidence exists: '+str(path))
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)


def absolute(path, resolve=True):
    path=Path(path);require(path.is_absolute(),'Absolute path required: '+str(path))
    return path.resolve() if resolve else path


def within(path, directory):
    path,directory=absolute(path),absolute(directory)
    require(path==directory or directory in path.parents,'Output path escapes H: '+str(path))
    return path


def verify(bindings):
    require(isinstance(bindings,dict),'Bindings must be a path/SHA map')
    for name,digest in bindings.items():
        require(absolute(name).is_file() and sha(name)==digest,'Immutable SHA changed: '+name)


def identity(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    require(fields[0] not in ('Z','X'),'Process is no longer live')
    return dict(pid=int(pid),start_ticks=int(fields[19]),uid=p.stat().st_uid,
                argv=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))


def same_identity(a,b):
    return all(int(a[k])==int(b[k]) for k in ('pid','start_ticks','uid')) and a['argv']==b['argv']


def check_live(proc, expected, reader=identity):
    if proc.poll() is not None:return False
    try:current=reader(proc.pid)
    except (OSError,RuntimeError):
        if proc.poll() is not None:return False
        raise
    require(same_identity(current,expected),'Live child identity changed')
    return True


def raw_process_state(pid):
    """Read identity even for an exited task; absence is distinct from permission errors."""
    p=Path('/proc')/str(pid)
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),state=fields[0],start_ticks=int(fields[19]),uid=p.stat().st_uid,
                    argv=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except FileNotFoundError:return None


def wait_gpu_idle(query, reaped, state_reader=raw_process_state, timeout=30,
                  clock=time.monotonic, sleep=time.sleep):
    """Wait only for NVML entries of exact owned/reaped workers to disappear.

    A foreign PID, a live owned PID, or PID reuse fails immediately. No stale
    entry is treated as available: successful admission still requires an empty
    actual NVML response before its bounded deadline.
    """
    known={int(r['identity']['pid']):r for r in reaped}
    began=clock();polls=0;stale=set()
    while True:
        remaining=timeout-(clock()-began)
        require(remaining>0,'NVML did not clear owned/reaped worker within30seconds')
        pids=set(query(min(5,remaining)))
        require(clock()-began<=timeout,'GPU query exceeded the30second admission deadline')
        if not pids:
            return dict(status='GPU_IDLE_CONFIRMED',elapsed_seconds=clock()-began,
                        stale_polls=polls,previously_reaped_stale_pids=sorted(stale))
        for pid in pids:
            record=known.get(pid)
            require(record is not None and record.get('wait_completed') is True,
                    'Foreign/unproven GPU compute PID: '+str(pid))
            expected=record['identity'];current=state_reader(pid)
            if current is not None:
                require(current.get('state') in ('Z','X') and all(int(current[k])==int(expected[k])
                        for k in ('pid','start_ticks','uid')),'Live or reused GPU PID: '+str(pid))
                if current.get('argv') not in ([],['']):
                    require(current['argv']==expected['argv'],'Terminated GPU command identity changed')
            stale.add(pid)
        polls+=1
        remaining=timeout-(clock()-began)
        require(remaining>0,'NVML did not clear owned/reaped worker within30seconds')
        sleep(min(1,remaining))


def budget_snapshot(path, registration_sha, limits, quiescent=False):
    path=Path(path)
    if not path.exists():return dict(created=False,charged=0,phase_charged={p:0 for p in limits},unresolved=0)
    connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=60)
    try:
        connection.execute('PRAGMA query_only=ON');connection.execute('BEGIN')
        meta=dict(connection.execute('SELECT key,value FROM meta'))
        require(meta['branch']=='H' and meta['registration_sha256']==registration_sha,
                'Wrong independent H budget identity')
        require(json.loads(meta['limits'])==limits and int(meta['total_cap'])==200000,'Budget phase quotas changed')
        counts=dict(connection.execute('SELECT phase,charged FROM counters'))
        rows=list(connection.execute('SELECT phase,status,count(*) FROM events GROUP BY phase,status'))
        observed={p:sum(n for phase,status,n in rows if phase==p) for p in limits}
        require(counts==observed and sum(counts.values())==sum(row[2] for row in rows),'Budget counters disagree')
        require(all(0<=counts[p]<=limits[p] for p in counts) and sum(counts.values())<=200000,'Budget cap exceeded')
        unresolved=sum(n for _,status,n in rows if status!='COMPLETE')
        if quiescent:require(unresolved==0,'Charged FAILED/RESERVED calls remain; independent diagnosis required')
        return dict(created=True,charged=sum(counts.values()),phase_charged=counts,unresolved=unresolved,
                    failed=sum(n for _,status,n in rows if status=='FAILED'),development_remaining=limits['development']-counts['development'])
    finally:connection.close()


def command_entry(argv):
    require(isinstance(argv,list) and all(isinstance(x,str) for x in argv),'An exact argv is required')
    require(len(argv)>=2 and absolute(argv[0],resolve=False).is_file(),'Missing registered interpreter')
    index=2 if argv[1]=='-B' else 1
    require(len(argv)>index and not argv[index].startswith('-'),'Shell snippets/-c/module shortcuts are prohibited')
    entry=absolute(argv[index]);require(entry.suffix=='.py','A bound Python entry point is required')
    return entry


def receipt(path, expected_statuses, registration_sha, expected=None, directory=None):
    value=read(path)
    require(value.get('status') in expected_statuses,'Unexpected completion status: '+str(path))
    require(value.get('registration_sha256')==registration_sha,'Wrong completion registration')
    for key,wanted in (expected or {}).items():require(value.get(key)==wanted,'Completion field differs: '+key)
    require(value.get('holdout_read',False) is False and value.get('training_updates',0)==0,'Out-of-scope completion')
    outputs=value.get('outputs');require(isinstance(outputs,dict) and outputs,'Scientific outputs must be sealed')
    for name in outputs:
        p=absolute(name)
        if directory is not None:within(p,directory)
        require(p.name not in ('worker.log','status.json','launch.json','exit_receipt.json','failure.json'),
                'Scientific completion binds mutable execution metadata')
    verify(outputs)
    for field in ('input_bindings','source_bindings'):
        if field in value:verify(value[field])
    return value


def validate_config(cfg, reg, config_sha):
    require(cfg.get('branch')=='H' and reg.get('branch')=='H','Only the independently registered H batch is supported')
    require(reg.get('status')=='H_EXECUTION_REVISION_REGISTERED','Execution revision is not registered')
    require(reg.get('owner_config_sha256')==config_sha,'Owner configuration differs from registration')
    ids=[s['id'] for s in cfg['stages']]
    require(ids and len(ids)==len(set(ids)) and set(ids)<=STAGE_IDS,'Unknown or duplicate H stages')
    require(ids==reg.get('allowed_stage_ids'),'Only implemented stages in this frozen revision may run')
    require(cfg['qualification_started_unix']==QUALIFICATION_START and cfg['qualification_deadline_unix']==QUALIFICATION_DEADLINE,
            'Original96-hour qualification clock changed')
    limits=cfg['phase_limits']
    require(limits==reg['phase_limits'] and sum(limits.values())==200000,'H quotas must total exactly200000')
    require(limits.get('development')==13200 and limits.get('engineering_reserve')==19360,
            'Frozen development/recovery reserves changed')
    require(all(type(n) is int and n>=0 for n in limits.values()),'Invalid phase quotas')
    require(len(cfg['cpu_affinities'])==2 and all(len(set(x))==2 for x in cfg['cpu_affinities'])
            and set(cfg['cpu_affinities'][0]).isdisjoint(cfg['cpu_affinities'][1]),'Two disjoint two-core CPU slots required')
    require(type(cfg['gpu_threads']) is int and cfg['gpu_threads']>0 and len(set(cfg['gpu_affinity']))==cfg['gpu_threads'],
            'Frozen visual CPU runtime must be explicit')
    require(math.isfinite(cfg['overall_deadline_unix']) and cfg['overall_deadline_unix']>=cfg['qualification_started_unix'],
            'A finite overall deadline at/after execution is required')
    seen=set();job_outputs=set()
    for stage in cfg['stages']:
        require(stage['resource'] in ('cpu','gpu'),'Unknown resource class')
        require(set(stage.get('requires',[]))<=seen,'Dependency is missing or out of order')
        require(math.isfinite(stage.get('max_seconds',0)) and stage.get('max_seconds',0)>0,'Each stage must have a finite positive work limit')
        require(1<=len(stage['jobs'])<=(1 if stage['resource']=='gpu' else 2),'Concurrent worker limit exceeded')
        jobids=[j['id'] for j in stage['jobs']]
        require(len(set(jobids))==len(jobids) and all(isinstance(j,str) and re.fullmatch('[a-z0-9_-]+',j) for j in jobids),
                'Duplicate or unsafe job IDs')
        for job in stage['jobs']:
            entry=command_entry(job['argv'])
            require(reg['source_bindings'].get(str(entry))==sha(entry),'Stage entry is not bound: '+str(entry))
            require(absolute(job['cwd'])==absolute(cfg['root']),'Stage cwd differs from root')
            jobout=within(job['out'],cfg['out']);within(job['completion'],job['out'])
            ownerout=within(cfg['owner_out'],cfg['out'])
            require(jobout not in job_outputs and jobout!=ownerout and ownerout not in jobout.parents
                    and jobout not in ownerout.parents,'Job output collides with execution evidence or another job')
            job_outputs.add(jobout)
            require(job.get('accepted_statuses'),'Explicit scientific completion statuses are required')
            absolute(job.get('receipt_registration',cfg['registration']))
        seen.add(stage['id'])


class Owner:
    def __init__(self, config):
        self.config_path=absolute(config);self.cfg=read(config)
        self.root=absolute(self.cfg['root']);self.out=absolute(self.cfg['out'])
        self.owner_out=within(self.cfg['owner_out'],self.out)
        self.regpath=absolute(self.cfg['registration']);self.reg=read(self.regpath);self.regsha=sha(self.regpath)
        validate_config(self.cfg,self.reg,sha(self.config_path))
        require(self.reg['source_bindings'].get(str(Path(__file__).resolve()))==sha(__file__),'Running owner source is not bound')
        verify(self.reg['source_bindings']);verify(self.reg['input_bindings'])
        gate=self.cfg['S1_gate'];require(self.reg['input_bindings'].get(gate['path'])==gate['sha256'] and sha(gate['path'])==gate['sha256'],'S1 gate is not bound')
        done=read(gate['path']);require(done.get('status')==gate['status']=='S1_BRIDGE_COMPLETE_AWAIT_NEXT_DECISION','S1 bridge is not complete')
        verify(done['outputs'])
        budget_registration=absolute(self.cfg['budget_registration'])
        require(self.reg['input_bindings'].get(str(budget_registration))==sha(budget_registration)
                ==self.cfg['budget_registration_sha256'],'Independent budget registration is not bound')
        budget=read(budget_registration)
        require(budget.get('status')=='FROZEN' and budget.get('schema')=='H_PACKET_BUDGET_V1'
                and budget.get('branch')=='H' and budget.get('total_cap')==200000
                and budget.get('phase_limits')==self.cfg['phase_limits'],'Budget document and owner quotas disagree')
        for stage in self.cfg['stages']:
            for job in stage['jobs']:self.receipt_binding(job)
        self.budget_path=within(self.cfg['budget_path'],self.out)
        self.active=[];self.completed=[];self.current=None;self.stop_reason=None;self.gpu_reaped=[]
        self.started=time.time();self.monotonic_started=time.monotonic()
        self.initial_wall=self.started;self.stage_began=None;self.available_affinity=None;self.visual_lock=None
        self.qualified=False
        if self.cfg.get('prior_qualification_gate'):
            previous=self.cfg['prior_qualification_gate']
            require(self.reg['input_bindings'].get(previous['path'])==previous['sha256'] and sha(previous['path'])==previous['sha256'],'Prior qualification is not bound')
            receipt(previous['path'],previous['accepted_statuses'],self.receipt_binding(previous),previous.get('receipt_expect'))
            self.qualified=True
        require(self.qualified or any(s['id']=='qualification' for s in self.cfg['stages']),
                'A batch needs an actual qualification stage or a bound prior qualification')

    def wall(self):
        # Wall deadline cannot be postponed by a clock adjustment after launch.
        return max(time.time(),self.initial_wall+time.monotonic()-self.monotonic_started)

    def receipt_binding(self, specification):
        path=absolute(specification.get('receipt_registration',self.cfg['registration']))
        if path==self.regpath:return self.regsha
        require(self.reg['input_bindings'].get(str(path))==sha(path),'Scientific receipt registration is not bound')
        return sha(path)

    def snapshot(self,quiescent=False):
        return budget_snapshot(self.budget_path,self.cfg['budget_registration_sha256'],self.cfg['phase_limits'],quiescent)

    def request_stop(self,reason):
        if self.stop_reason is None:
            self.stop_reason=reason
            paths={self.out/'STOP'}|{Path(row['job']['out'])/'STOP' for row in self.active}
            for path in paths:
                path.parent.mkdir(parents=True,exist_ok=True)
                try:
                    with path.open('x',encoding='utf-8') as f:f.write(reason+'\n')
                except FileExistsError:pass
            save(self.owner_out/'stop_request.json',dict(time=time.time(),reason=reason,
                 status='STOP_REQUESTED_AT_SAFE_BOUNDARY',force_kill=False,signals_sent=False,
                 registration_sha256=self.regsha),exclusive=True)

    def boundary(self):
        if not self.qualified and self.wall()>=self.cfg['qualification_deadline_unix']:
            self.request_stop('96-hour64QAM qualification deadline reached')
        if self.wall()>=self.cfg['overall_deadline_unix']:self.request_stop('Registered H batch deadline reached')
        if self.current and time.monotonic()-self.stage_began>=self.current['max_seconds']:
            self.request_stop('Registered stage work deadline reached: '+self.current['id'])
        if (self.out/'STOP').exists() and self.stop_reason is None:self.request_stop('Explicit H STOP observed')
        require(self.stop_reason is None,self.stop_reason or 'STOP requested')

    def status(self,status):
        save(self.owner_out/'status.json',dict(status=status,time=time.time(),registration_sha256=self.regsha,
             elapsed_seconds=time.time()-self.started,current_stage=None if self.current is None else self.current['id'],
             active=[dict(job=r['job']['id'],identity=r['identity']) for r in self.active],
             completed=self.completed,qualification_passed=self.qualified,stop_reason=self.stop_reason,
             C_started=False,holdout_started=False,future_stages_started=False))

    def acquire_visual(self):
        import fcntl
        path=absolute(self.cfg['visual_lock_path']);path.parent.mkdir(parents=True,exist_ok=True)
        self.visual_lock=path.open('a+')
        try:
            fcntl.flock(self.visual_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            def query(timeout):
                result=subprocess.run([str(absolute(self.cfg['nvidia_smi'],resolve=False)),
                     '--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=timeout)
                return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            admission=wait_gpu_idle(query,self.gpu_reaped)
            save(self.owner_out/'stages'/self.current['id']/'gpu_admission.json',dict(**admission,
                 registration_sha256=self.regsha,prior_reaped=self.gpu_reaped),exclusive=True)
        except BaseException:
            self.visual_lock.close();self.visual_lock=None;raise

    def launch(self,job,slot):
        self.boundary()
        resource=self.current['resource'];threads=2 if resource=='cpu' else self.cfg['gpu_threads']
        affinity=set(self.cfg['cpu_affinities'][slot] if resource=='cpu' else self.cfg['gpu_affinity'])
        require(affinity<=self.available_affinity,'Registered affinity is unavailable')
        directory=absolute(job['out']);directory.mkdir(parents=True,exist_ok=True)
        require(not Path(job['completion']).exists() and not (directory/'failure.json').exists(),'Prior scientific attempt requires explicit recovery')
        execution=self.owner_out/'stages'/self.current['id']/'workers'/job['id'];execution.mkdir(parents=True,exist_ok=True)
        log=(execution/'worker.log').open('xb')
        env=dict(os.environ,**job.get('environment',{}))
        env.update(CUDA_VISIBLE_DEVICES='' if resource=='cpu' else str(self.cfg['gpu_device']),
             OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),
             NUMEXPR_NUM_THREADS=str(threads),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
             REGISTERED_H_STOP_FILE=str(self.out/'STOP'))
        try:
            os.sched_setaffinity(0,affinity)
            proc=subprocess.Popen(job['argv'],cwd=self.root,env=env,stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:
            log.close();raise
        else:
            row=dict(job=job,process=proc,log=log,execution=execution,started=time.time(),resource=resource,
                     identity=dict(pid=proc.pid,identity_capture='PENDING'))
            self.active.append(row)
        finally:os.sched_setaffinity(0,set(self.cfg['cpu_affinities'][0]))
        row['identity']=identity(proc.pid)
        require(row['identity']['argv']==job['argv'] and row['identity']['uid']==os.getuid(),'Launched identity differs')
        save(execution/'launch.json',dict(identity=row['identity'],argv=job['argv'],started=row['started'],
             threads=threads,affinity=sorted(os.sched_getaffinity(proc.pid)),resource=resource,
             registration_sha256=self.regsha),exclusive=True)

    def finish(self,row,validate=True):
        code=row['process'].wait();row['log'].close()
        value=dict(job_id=row['job']['id'],identity=row['identity'],exit_code=code,
                   closed_log_sha256=sha(row['execution']/'worker.log'),time=time.time(),
                   elapsed_seconds=time.time()-row['started'],registration_sha256=self.regsha)
        path=Path(row['job']['completion'])
        if path.exists():value.update(completion=str(path),completion_sha256=sha(path))
        failure=Path(row['job']['out'])/'failure.json'
        if failure.exists():value.update(failure=str(failure),failure_sha256=sha(failure))
        save(row['execution']/'exit_receipt.json',value,exclusive=True)
        if row.get('resource')=='gpu' and all(k in row['identity'] for k in ('pid','start_ticks','uid','argv')):
            self.gpu_reaped.append(dict(identity=row['identity'],wait_completed=True,exit_code=code,
                 exit_receipt=str(row['execution']/'exit_receipt.json'),
                 exit_receipt_sha256=sha(row['execution']/'exit_receipt.json')))
        self.active.remove(row)
        if validate:
            require(code==0 and path.exists() and not failure.exists(),'Worker failed or missing completion: '+row['job']['id'])
            receipt(path,row['job']['accepted_statuses'],self.receipt_binding(row['job']),row['job'].get('receipt_expect'),row['job']['out'])
        return value

    def drain(self):
        while self.active:
            for row in list(self.active):
                if row['process'].poll() is not None:self.finish(row,False)
            if self.active:self.status('STOP_PENDING_SAFE_DRAIN');time.sleep(2)
        if self.visual_lock is not None:self.visual_lock.close();self.visual_lock=None
        save(self.owner_out/'drain_receipt.json',dict(status='ALL_OWNED_CHILDREN_EXITED',time=time.time(),
             registration_sha256=self.regsha,force_kill=False,stop_reason=self.stop_reason),exclusive=True)

    def pause_h(self,reason):
        save(self.owner_out/'handoff_request.json',dict(status='PRIORITY_MAIN_CALIBRATION_REQUESTED',
             target='MAIN_SYSTEM_FULL_CALIBRATION',reason=reason,registration_sha256=self.regsha,
             requires_independent_bound_execution_registration=True,placeholder_command_launched=False,
             C_started=False,holdout_started=False,time=time.time()),exclusive=True)
        self.status('H_PAUSED_MAIN_CALIBRATION_HANDOFF_REQUIRED')

    def run(self):
        try:
            self.snapshot(quiescent=True)
            for stage in self.cfg['stages']:
                self.current=stage;self.stage_began=time.monotonic();self.boundary()
                verify(self.reg['source_bindings']);verify(self.reg['input_bindings'])
                self.snapshot(quiescent=True)
                if stage['resource']=='gpu':self.acquire_visual()
                exits=[]
                for slot,job in enumerate(stage['jobs']):self.launch(job,slot)
                while self.active:
                    self.boundary()
                    for row in list(self.active):
                        if row['process'].poll() is not None:exits.append(self.finish(row))
                        else:check_live(row['process'],row['identity'])
                    self.status('H_REGISTERED_STAGE_RUNNING')
                    if self.active:time.sleep(2)
                if self.visual_lock is not None:self.visual_lock.close();self.visual_lock=None
                budget=self.snapshot(quiescent=True)
                value=dict(status='REGISTERED_STAGE_COMPLETE',stage=stage['id'],registration_sha256=self.regsha,
                           jobs=exits,budget=budget,time=time.time())
                directory=self.owner_out/'stages'/stage['id']
                save(directory/'completion.json',value,exclusive=True)
                self.completed.append(dict(stage=stage['id'],completion=str(directory/'completion.json'),sha256=sha(directory/'completion.json')))
                if stage['id']=='qualification':self.qualified=True
                self.boundary()
            self.current=None
            verify(self.reg['source_bindings']);verify(self.reg['input_bindings'])
            result=dict(status='REGISTERED_H_STAGE_BATCH_COMPLETE',registration_sha256=self.regsha,
                allowed_stage_ids=self.reg['allowed_stage_ids'],completed=self.completed,budget=self.snapshot(True),
                qualification_passed=self.qualified,elapsed_seconds=time.time()-self.started,
                future_stages_started=False,H_full_delivery_claimed=False,C_started=False,holdout_started=False)
            save(self.owner_out/'completion.json',result,exclusive=True);self.status(result['status'])
        except BaseException:
            detail=traceback.format_exc();qualification_failure=not self.qualified
            self.request_stop(self.stop_reason or 'Registered H stage failed; preserve evidence')
            if not (self.owner_out/'failure.json').exists():
                save(self.owner_out/'failure.json',dict(status='FAILED_PRESERVE_AND_DIAGNOSE',traceback=detail,
                     registration_sha256=self.regsha,time=time.time(),current_stage=None if self.current is None else self.current['id'],
                     active=[dict(job=r['job']['id'],identity=r['identity']) for r in self.active]),exclusive=True)
            self.drain()
            if qualification_failure:self.pause_h(self.stop_reason)
            else:self.status('H_STAGE_FAILED_PRESERVE_AND_DIAGNOSE')
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();owner=Owner(args.config);owner.owner_out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (owner.owner_out/'execution.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not any((owner.owner_out/n).exists() for n in ('owner_identity.json','failure.json','completion.json'))
                and not (owner.out/'STOP').exists(),'Existing owner attempt/STOP needs an explicit recovery revision')
        owner.available_affinity=set(os.sched_getaffinity(0))
        require(set().union(*map(set,owner.cfg['cpu_affinities']),set(owner.cfg['gpu_affinity']))<=owner.available_affinity,
                'Registered CPU/GPU affinity unavailable')
        os.sched_setaffinity(0,set(owner.cfg['cpu_affinities'][0]));os.setpriority(os.PRIO_PROCESS,0,15)
        save(owner.owner_out/'owner_identity.json',dict(**identity(os.getpid()),registration_sha256=owner.regsha,
             config_sha256=sha(owner.config_path),time=time.time(),nice=15),exclusive=True)
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,lambda signum,frame:owner.request_stop('Owner received signal '+str(signum)))
        owner.run()


if __name__=='__main__':main()
