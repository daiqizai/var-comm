"""Finite, one-owner continuation to step 150000, then independent codec testing.

Waits for the already-running role. Never clears stop markers or retries a role.
Each subsequent allocation has a fresh request and resource receipt. Test workers
cannot run while a training owner/rank is alive. No old P2 launcher is called.
"""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time,traceback
from runtime import L,OLD,R,PY,CHECKPOINT_ROOT,env,sha,write,setup
setup()

STOP=False
def stop(signum,frame):
    global STOP;STOP=True

def read(p):return json.loads(Path(p).read_text())
def live(v):
    if not v:return False
    p=Path('/proc')/str(v['pid'])/'stat'
    return p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==str(v['start_ticks'])

def boundary(protocol):
    assert (OLD/'STOP_P2').exists(),'old P2 pause marker missing'
    if STOP or (L/'STOP_MAIN').exists():raise InterruptedError('user/main stop retained; no continuation')
    if time.time()>protocol['deadline']:raise TimeoutError('finite chain deadline reached')
    for path,h in protocol['bindings'].items():assert sha(path)==h,path

def wait_previous(role,protocol):
    folder=L/'controllers'/role
    while True:
        boundary(protocol)
        if (folder/'actual_wait.json').exists():
            w=read(folder/'actual_wait.json')
            assert w['returncode']==0 and w['child_gone'],(role,w)
            if not live(read(folder/'owner.json')) and not live(read(folder/'child.json')):return w
        write(Path(protocol['root'])/'status.json',dict(phase='waiting_for_training',role=role,time=time.time()))
        time.sleep(15)

def wait_free(gpus,protocol):
    start=time.time()
    while True:
        boundary(protocol)
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        values={int(v[0]):v for v in [[s.strip() for s in line.split(',')] for line in rows.splitlines()]}
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        occupied={v.split(',')[0].strip() for v in apps.splitlines() if v.strip()}
        if all(int(values[g][2])<100 and int(values[g][3])<5 and values[g][1] not in occupied for g in gpus):return rows
        if time.time()-start>protocol['resource_wait_max_seconds']:raise TimeoutError('registered free-device wait exhausted')
        write(Path(protocol['root'])/'status.json',dict(phase='waiting_for_free_devices',gpus=gpus,time=time.time()))
        time.sleep(15)

def launch_owner(protocol,role,command,gpus,cpus,max_seconds,bindings,roles_root=None):
    root=Path(protocol['root']);source=Path(__file__).resolve().parent
    target=Path(roles_root) if roles_root else root/'roles'
    assert not (target/role).exists(),'exclusive role already exists'
    environment=env();environment['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
    request=dict(root=str(target),role=role,command=command,env=environment,cpus=cpus,gpus=gpus,max_seconds=max_seconds,bindings=bindings)
    path=root/'owner_requests'/(role+'.json');assert not path.exists();write(path,request)
    return subprocess.Popen([PY,str(source/'posttrain_owner_v1.py'),'--request',str(path)],stdin=subprocess.DEVNULL,
        stdout=(root/'owner_requests'/(role+'.stdout')).open('xb'),stderr=(root/'owner_requests'/(role+'.stderr')).open('xb'),env=environment,start_new_session=True)

def monitor(processes,protocol):
    try:
        while any(p.poll() is None for p in processes):
            boundary(protocol);time.sleep(10)
    except BaseException:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=45)
            except subprocess.TimeoutExpired:pass
        raise
    return [p.returncode for p in processes]

def continue_training(protocol):
    root=Path(protocol['root']);source=Path(__file__).resolve().parent;role=protocol['initial_training_role']
    for ordinal in range(protocol['max_new_training_segments']+1):
        wait=wait_previous(role,protocol);folder=L/role;done=read(folder/'segment_complete.json');cp=read(folder/'latest_checkpoint.json')
        assert cp['step']==done['step'] and sha(cp['path'])==cp['sha256']
        dest=CHECKPOINT_ROOT/'main_v1'/f"posttrain_boundary_step{cp['step']:06d}.pt"
        if not dest.exists():os.link(cp['path'],dest)
        assert sha(dest)==cp['sha256'];cp=dict(cp,path=str(dest))
        write(root/f'boundary_{ordinal:02d}.json',dict(predecessor=role,actual_wait=wait,checkpoint=cp,complete=done))
        if done['training_finished']:
            assert cp['step']==protocol['target_step']==150000
            write(root/'final_checkpoint.json',cp);return cp
        assert cp['step']<150000 and ordinal<protocol['max_new_training_segments'],'finite continuation allocation exhausted'
        old=read(folder/'request.json');assert old['global_batch']==32 and old['gpus']==list(range(8))
        inventory=wait_free(list(range(8)),protocol);newrole=f'training_world8_auto{ordinal+1:02d}';out=L/newrole;out.mkdir(exist_ok=False)
        request=dict(old,output=str(out),resume=cp,training_manifest=cp['corpus'],training_manifest_sha256=sha(cp['corpus']),
                     wall_seconds=39600,owner_max_seconds=43200,allocation_gpu_device_hours_max=96,
                     predecessor=role,continuation='fresh finite eight-GPU continuation of the unchanged 150000-update schedule',
                     prior_deadlines_archived={k:v for k,v in old.items() if 'deadline' in k})
        for k in list(request):
            if k.startswith('original_segment02_'):del request[k]
        rp=out/'request.json';write(rp,request)
        write(out/'admission.json',dict(checkpoint=cp,prior_actual_wait=wait,device_inventory=inventory,world=8,global_batch=32,old_P2_paused=True,training_scope_unchanged=True))
        command=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(source/'train_world8_v1.py'),'--request',str(rp)]
        p=launch_owner(protocol,newrole,command,list(range(8)),sorted(c for v in old['rank_cpus'].values() for c in v),43200,{**old['bindings'],str(rp):sha(rp)},L/'controllers')
        write(out/'launch.json',dict(owner_pid=p.pid,time=time.time(),resume_step=cp['step'],request_sha256=sha(rp)))
        write(root/'status.json',dict(phase='training_continuation',role=newrole,ordinal=ordinal+1,time=time.time()))
        rc=monitor([p],protocol);assert rc==[0],(newrole,rc)
        role=newrole

def evaluate_size(protocol,checkpoint,size):
    root=Path(protocol['root']);source=Path(__file__).resolve().parent;requests={}
    for shard in range(8):
        out=root/f'eval_{size}_{shard:02d}';out.mkdir(exist_ok=False)
        request=dict(output=str(out),size=size,shard=shard,checkpoint=checkpoint,manifest=protocol['manifest'],metric_assets=protocol['metric_assets'],
            baseline256=protocol['baseline256'],bindings=protocol['bindings'],
            rx_denied_roots=protocol['source_roots']+[str(root/f'eval_{s}_{k:02d}'/'tx') for s in [256,512] for k in range(8)])
        rp=root/'worker_requests'/f'{size}_{shard:02d}.json';write(rp,request);requests[shard]=rp
    # The first fixed shard completes TX, independent RX and scoring before the
    # other shards start. Failure is preserved; no replacement sources or retry.
    for shards in [[0],list(range(1,8))]:
        for stage in ['tx','rx','score']:
            boundary(protocol);wait_free(shards,protocol);jobs=[]
            for shard in shards:
                cpus=protocol['rank_cpus'][str(shard)];role=f'{size}_{shard:02d}_{stage}';rp=requests[shard]
                jobs.append(launch_owner(protocol,role,[PY,str(source/'posttrain_worker_v1.py'),'--request',str(rp),'--stage',stage],[shard],cpus,7200,{**protocol['bindings'],str(rp):sha(rp)}))
            write(root/'status.json',dict(phase='evaluation',size=size,stage=stage,shards=shards,time=time.time()))
            codes=monitor(jobs,protocol);assert all(c==0 for c in codes),(size,stage,codes)
    write(root/f'resolution_{size}_complete.json',dict(passed=True,sources=500,logical_positions=8000))

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();protocol=read(a.protocol);root=Path(protocol['root'])
    for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,stop)
    assert not (root/'started.json').exists();write(root/'started.json',dict(time=time.time(),pid=os.getpid(),protocol_sha256=sha(a.protocol)))
    try:
        checkpoint=continue_training(protocol)
        for size in [256,512]:
            try:evaluate_size(protocol,checkpoint,size)
            except (InterruptedError,TimeoutError):raise
            except Exception as e:
                write(root/f'resolution_{size}_failure.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        source=Path(__file__).resolve().parent
        p=launch_owner(protocol,'summary',[PY,str(source/'posttrain_summary_v1.py'),'--protocol',a.protocol],[],[20,22],3600,protocol['bindings'])
        assert monitor([p],protocol)==[0]
        write(root/'status.json',dict(phase='finished',time=time.time(),scientific_result=read(root/'delivery/complete.json')))
    except BaseException as e:
        write(root/'failure.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,training_finished=(root/'final_checkpoint.json').exists()))
        raise

if __name__=='__main__':main()
