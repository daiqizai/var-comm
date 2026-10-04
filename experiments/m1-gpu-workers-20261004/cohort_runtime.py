"""Process-local admission for a sealed, explicitly owned M1 GPU cohort.

The original thermal Safety class and five-second HealthyPolling wrapper are
retained. Only the old rule rejecting every second GPU PID is replaced by exact
cohort identity checks. This module imports no CUDA or scientific implementation.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time


def require(value,message):
    if not value:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)


def verify_bindings(bindings):
    require(isinstance(bindings,dict) and bindings,'Worker source bindings are required')
    for path,digest in bindings.items():require(sha(path)==digest,'Bound worker source changed: '+str(path))


def process_identity(pid,proc_root=Path('/proc')):
    require(type(pid)is int and pid>0,'Invalid cohort PID')
    path=Path(proc_root)/str(pid)
    stat=(path/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    require(len(tail)>19 and tail[0]!='Z','Cohort PID is not live')
    lines=(path/'status').read_text().splitlines()
    uid=[int(x) for x in next(line for line in lines if line.startswith('Uid:')).split()[1:]]
    require(len(uid)==4 and len(set(uid))==1,'Cohort PID has inconsistent user identities')
    argv=[os.fsdecode(x) for x in (path/'cmdline').read_bytes().split(b'\0') if x]
    require(argv,'Cohort PID has no command')
    return dict(pid=pid,start_ticks=str(tail[19]),uid=uid[0],argv=argv)


def assert_process(record,reader=process_identity):
    wanted={k:record[k] for k in ('pid','start_ticks','uid','argv')}
    wanted['start_ticks']=str(wanted['start_ticks'])
    require(type(wanted['pid'])is int and type(wanted['uid'])is int
        and isinstance(wanted['argv'],list) and wanted['argv'],'Malformed cohort process record')
    require(reader(wanted['pid'])==wanted,'Cohort process identity changed: '+str(wanted['pid']))


def validate_manifest(manifest,config):
    require(manifest.get('schema')=='M1_GPU_COHORT_V1' and manifest.get('status')=='READY','Sealed ready cohort required')
    require(manifest.get('stage')==config['stage'] and config['stage'] in ('benchmark','production'),'Cohort stage differs')
    workers=manifest.get('workers');require(isinstance(workers,list) and 1<=len(workers)<=4,'Only one to four cohort workers are registered')
    ids=[w['worker_id'] for w in workers];pids=[w['pid'] for w in workers]
    require(len(set(ids))==len(ids) and len(set(pids))==len(pids) and manifest['owner']['pid'] not in pids,'Cohort worker identity duplicated')
    indices=[]
    for worker in workers:
        sources=worker['source_indices'];require(isinstance(sources,list) and sources and all(type(i)is int and 0<=i<1000 for i in sources),'Invalid source assignment')
        require(len(sources)==len(set(sources)),'Duplicate source within worker');indices.extend(sources)
        require(worker['uid']==manifest['owner']['uid'],'Cohort crosses user ownership')
    require(len(indices)==len(set(indices)),'Cohort source assignments overlap')
    mine=[w for w in workers if w['worker_id']==config['worker_id']]
    require(len(mine)==1 and mine[0]['source_indices']==config['source_indices'],'Worker source assignment differs from sealed cohort')
    return mine[0]


class CohortGuard:
    def __init__(self,config,manifest,manifest_sha256,reader=process_identity,current_pid=None):
        self.config=config;self.manifest=manifest;self.manifest_sha256=manifest_sha256;self.reader=reader
        self.mine=validate_manifest(manifest,config);self.current_pid=os.getpid() if current_pid is None else current_pid
        require(self.mine['pid']==self.current_pid,'Worker PID differs from sealed cohort')
        self.allowed={w['pid']:w for w in manifest['workers']};self.runtime=None

    def check_identities(self,foreign_processes):
        require(sha(self.config['cohort_manifest'])==self.manifest_sha256,'Sealed cohort manifest changed')
        assert_process(self.manifest['owner'],self.reader);assert_process(self.mine,self.reader)
        for item in foreign_processes:
            pid=item['pid'];require(pid in self.allowed,'GPU has an unregistered process: '+str(pid))
            assert_process(self.allowed[pid],self.reader)
        return None

    def require_available(self):
        require(self.runtime is not None,'Original runtime not attached to cohort guard')
        try:
            for attempt in range(3):
                try:
                    self.check_identities(self.original_foreign_gpu_processes());break
                except (FileNotFoundError,ProcessLookupError):
                    # NVML can report a just-exited peer. Requery only missing-PID
                    # races; an unknown PID or reused/mismatched identity fails now.
                    if attempt==2:raise
                    time.sleep(.05)
        except (RuntimeError,OSError,KeyError,StopIteration) as error:
            raise self.runtime.ResourceBusy(str(error)) from error

    def filtered_foreign_gpu_processes(self):
        self.require_available()
        # Every observed other PID has just passed exact cohort admission.
        return []

    def install(self,probe,runtime):
        """Change imported function references in this worker process only."""
        require(self.runtime is None,'Cohort admission already installed')
        self.runtime=runtime;original_safety=probe.Safety
        require(hasattr(original_safety,'check') and 'require_available' in original_safety.check.__globals__,
                'Original Safety interface differs')
        require(probe.require_available is runtime.require_available
            and original_safety.check.__globals__['require_available'] is runtime.require_available,
            'Probe and Safety do not share the original GPU admission function')
        self.original_foreign_gpu_processes=runtime.foreign_gpu_processes
        runtime.foreign_gpu_processes=self.filtered_foreign_gpu_processes
        require(probe.Safety is original_safety,'Original thermal Safety implementation changed')
        self.require_available()


def await_cohort(config,stopped=lambda:False,timeout=180.,clock=time.monotonic,sleep=time.sleep):
    began=clock();path=Path(config['cohort_manifest'])
    while not path.exists():
        require(not stopped() and not Path(config['stop_file']).exists(),'Worker stopped before cohort admission')
        require(clock()-began<timeout,'Cohort admission timed out before CUDA initialization');sleep(.1)
    manifest=read(path);digest=sha(path);guard=CohortGuard(config,manifest,digest)
    assert_process(manifest['owner']);assert_process(guard.mine)
    return guard


@contextmanager
def source_lock(directory,index):
    import fcntl
    require(type(index)is int and 0<=index<1000,'Invalid source lock index')
    path=Path(directory)/('%04d.lock'%index);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Another writer owns source '+str(index)) from error
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)
