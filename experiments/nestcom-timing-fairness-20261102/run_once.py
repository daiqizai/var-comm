"""One finite, serial timing queue; never invokes a historical launcher."""
from pathlib import Path
import os, sys, json, time, hashlib, subprocess, signal, traceback

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')

def read(p): return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def ident(pid):
    try:
        a=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=int(a[19]),state=a[0])
    except FileNotFoundError: return None
def same_process(identity):
    now=ident(identity['pid'])
    return now is not None and now['start_ticks']==identity['start_ticks']
def gpu_snapshot():
    def query(fields,kind='gpu'):
        return subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True).strip()
    return dict(gpus=query('index,uuid,pci.bus_id,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.sm'),
                apps=query('gpu_uuid,pid,used_gpu_memory','compute-apps'))
def cpu_snapshot():
    return [s for s in Path('/proc/stat').read_text().splitlines() if s.split()[0] in ['cpu38','cpu39','cpu40','cpu41']]
def foreign(snapshot,uuid,own_pid=None):
    return [row for row in snapshot['apps'].splitlines() if row.split(', ')[0]==uuid and (own_pid is None or int(row.split(', ')[1])!=own_pid)]
def terminate_owned(p,identity):
    if p.poll() is None:
        assert same_process(identity)
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            assert same_process(identity)
            os.killpg(p.pid,signal.SIGKILL);p.wait()

def main(request,expected):
    assert sha(request)==expected
    r=read(request);root=Path(r['study_root']);control=root/'queue_control'
    control.mkdir(exist_ok=False)
    save(control/'owner.json',dict(owner=ident(os.getpid()),registration=request,sha256=expected))
    os.sched_setaffinity(0,{44,46})
    gpu_used=cpu_used=0.0;results=[];start=time.time()
    try:
        for job in r['jobs']:
            name=job['name'];c=root/(name+'_control')
            c.mkdir(exist_ok=False)
            remain=min(r['deadline_unix']-time.time(),r['GPU_device_seconds']-gpu_used if job['gpu'] else r['CPU_process_wall_seconds']-cpu_used)
            if remain<60 or (root/'STOP').exists():
                status=dict(name=name,status='BLOCKED_BUDGET_OR_STOP_ZERO_LAUNCH',scientific_calls=0)
                save(c/'not_launched.json',status);results.append(status);continue
            snap=gpu_snapshot();save(c/'admission.json',dict(time=time.time(),**snap,cpu=cpu_snapshot()))
            matching=[s.split(', ') for s in snap['gpus'].splitlines() if s.split(', ')[1]==r['gpu_uuid']]
            assert len(matching)==1
            if job['gpu'] and (foreign(snap,r['gpu_uuid']) or int(matching[0][3])<60000 or int(matching[0][4])>20):
                status=dict(name=name,status='BLOCKED_GPU_NOT_IDLE_ZERO_LAUNCH',scientific_calls=0)
                save(c/'not_launched.json',status);results.append(status);continue
            assert sha(job['request'])==job['request_sha256']
            env=dict(os.environ,**r['environment'])
            if not job['gpu']:env['CUDA_VISIBLE_DEVICES']=''
            cmd=[job['python'],'-B',r['worker'],job['request'],job['request_sha256']]
            before=time.time();p=None;ch=None;reason=None
            with (c/'worker.log').open('xb') as log, (c/'telemetry.jsonl').open('x') as telemetry:
                p=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                ch=ident(p.pid);assert ch
                save(c/'child.json',dict(child=ch,started_unix=before,command=cmd))
                limit=min(r['worker_seconds'],remain-20)
                try:
                    while p.poll() is None:
                        now=time.time();snap=gpu_snapshot()
                        telemetry.write(json.dumps(dict(time=now,**snap,cpu=cpu_snapshot()))+'\n');telemetry.flush()
                        if job['gpu'] and foreign(snap,r['gpu_uuid'],p.pid):reason='FOREIGN_GPU_PROCESS';break
                        if now>=r['deadline_unix'] or now-before>=limit:reason='BOUNDED_DEADLINE';break
                        if (root/'STOP').exists():reason='USER_STOP_FILE';break
                        try:p.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                finally:
                    if p.poll() is None:terminate_owned(p,ch)
            rc=p.wait();elapsed=time.time()-before
            if job['gpu']:gpu_used+=elapsed
            else:cpu_used+=elapsed
            wait=dict(returncode=rc,child=ch,final_identity=ident(p.pid),started_unix=before,finished_unix=time.time(),
                      GPU_device_seconds=elapsed if job['gpu']else 0,CPU_process_wall_seconds=0 if job['gpu']else elapsed,
                      terminated_reason=reason,automatic_retry=False)
            save(c/'actual_wait.json',wait)
            complete=(root/name/'completion.json')
            status=dict(name=name,status='COMPLETE' if rc==0 and complete.exists() else 'FAILED_PRESERVED',actual_wait=wait)
            results.append(status);save(control/(name+'.json'),status)
            print(json.dumps(dict(stage=name,status=status['status'],seconds=elapsed)),flush=True)
        status='COMPLETE' if all(v['status']=='COMPLETE' for v in results) else 'INCOMPLETE_WITH_PRESERVED_DIAGNOSTICS'
        save(control/'completion.json',dict(status=status,stages=results,GPU_device_seconds=gpu_used,CPU_process_wall_seconds=cpu_used,elapsed=time.time()-start))
        return 0 if status=='COMPLETE' else 1
    except BaseException:
        save(control/'failure.json',dict(traceback=traceback.format_exc(),stages=results,GPU_device_seconds=gpu_used,CPU_process_wall_seconds=cpu_used,automatic_retry=False))
        raise

if __name__=='__main__':sys.exit(main(*sys.argv[1:]))
