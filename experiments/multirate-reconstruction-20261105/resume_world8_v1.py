"""User-authorized eight-GPU migration inside the current allocation window."""
from pathlib import Path
import json,subprocess,time,os
from runtime import L,OLD,R,PY,env,sha,write


def live(identity):
    p=Path('/proc')/str(identity['pid'])/'stat'
    return p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==str(identity['start_ticks'])


def main():
    source=Path(__file__).resolve().parent;transition=L/'migration_4to8_v1'
    proof=json.loads((transition/'preflight.json').read_text())
    cpu_wait=json.loads((transition/'preflight_cpu/actual_wait.json').read_text())
    assert proof['passed'] and cpu_wait['returncode']==0 and cpu_wait['child_gone']
    control=L/'controllers/training_segment02'
    wait=json.loads((control/'actual_wait.json').read_text());assert wait['returncode']==0 and wait['child_gone']
    assert not live(json.loads((control/'owner.json').read_text())) and not live(json.loads((control/'child.json').read_text()))
    reason='USER_AUTHORIZED_WORLD4_TO_WORLD8_20261105\n'
    assert (L/'STOP_MAIN').read_text()==reason and (OLD/'STOP_P2').exists()
    old=json.loads((L/'training_segment02/request.json').read_text())
    checkpoint=json.loads((transition/'checkpoint.json').read_text())
    assert checkpoint==proof['checkpoint'] and sha(checkpoint['path'])==checkpoint['sha256']
    assert sha(proof['corpus_path'])==proof['corpus_sha256']
    for name in ['runtime.py','model.py','codec.py','train_v2.py']:
        assert sha(source/name)==old['bindings'][str(source/name)],name
    for role in (L/'controllers').iterdir():
        if (role/'actual_wait.json').exists() or not (role/'registration.json').exists():continue
        reg=json.loads((role/'registration.json').read_text());assert not reg['gpu_devices'],role
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[[field.strip() for field in line.split(',')] for line in gpu.splitlines()]
    assert len(rows)==8 and {int(r[0]) for r in rows}==set(range(8))
    assert all(int(r[2])<100 and int(r[3])==0 for r in rows),rows
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    rank_cpus={str(r):[48+4*r,50+4*r] if r<4 else [138+4*(r-4),140+4*(r-4)] for r in range(8)}
    topology={}
    for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET,NODE'],text=True).splitlines():
        if line.startswith('#'):continue
        cpu,core,socket,node=map(int,line.split(','));topology[cpu]=(core,socket,node)
    cpus=sorted(c for values in rank_cpus.values() for c in values)
    assert len(set(topology[c][:2] for c in cpus))==16
    assert all(topology[c][2]==(0 if int(rank)<4 else 1) for rank,values in rank_cpus.items() for c in values)
    assert set(cpus)<=set(os.sched_getaffinity(0))
    owner_start=json.loads((control/'owner.json').read_text())['started']
    training_start=json.loads((L/'training_segment02/training_started.json').read_text())['time']
    owner_deadline=owner_start+old['owner_max_seconds'];soft_deadline=training_start+old['wall_seconds'];now=time.time()
    assert soft_deadline-now>1800 and owner_deadline-now>2400
    out=L/'training_world8_v1';out.mkdir(exist_ok=False)
    migration=dict(from_world=4,to_world=8,checkpoint_sha256=checkpoint['sha256'],
                   old_logical_rank_streams_retained=[0,1,2,3],new_rank_seeds=proof['new_rank_seeds'],
                   preflight_path=str(transition/'preflight.json'),same_global_samples=True,
                   bitwise_four_card_trajectory_claim=False)
    request=dict(old,output=str(out),resume=checkpoint,gpus=list(range(8)),rank_cpus=rank_cpus,
          training_manifest=proof['corpus_path'],training_manifest_sha256=proof['corpus_sha256'],
          world_migration=migration,wall_seconds=int(soft_deadline-now),owner_max_seconds=int(owner_deadline-now),
          original_segment02_owner_deadline=owner_deadline,original_segment02_training_deadline=soft_deadline,
          continuation='user-authorized 4-to-8 migration; current segment wall deadlines preserved',predecessor='training_segment02',
          allocation_gpu_device_hours_max=8*(owner_deadline-now)/3600,
          bindings={str(source/n):sha(source/n) for n in ['runtime.py','model.py','codec.py','train_world8_v1.py','world_migration_v1.py','resume_world8_v1.py']})
    rp=out/'request.json';write(rp,request)
    write(out/'admission.json',dict(passed=True,prior_actual_wait=wait,cpu_preflight_actual_wait=cpu_wait,
          checkpoint=checkpoint,gpu_inventory=gpu,rank_cpus=rank_cpus,world_migration=migration,
          global_batch=32,unchanged=['model','loss','optimizer','learning rate schedule','data cursor','corpus','precision'],
          wall_deadlines_preserved=True,remaining_max_gpu_device_hours=request['allocation_gpu_device_hours_max'],
          old_P2_remains_paused=True))
    owner=L/'training_world8_v1_owner.py';assert not owner.exists()
    command=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(source/'train_world8_v1.py'),'--request',str(rp)]
    body='import sys\nsys.path.insert(0,'+repr(str(source))+')\nfrom runtime import *\nsys.path.insert(0,str(R/"experiments/rdvq-prefix-20261104"))\nfrom owned_process import run_role\ne=env();e["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"\n'
    body+='run_role('+repr(str(L/'controllers'))+',"training_world8_v1",'+repr(command)+',e,'+repr(cpus)+','+repr(list(range(8)))+','+str(request['owner_max_seconds'])+','+repr({**request['bindings'],str(rp):sha(rp),str(transition/'preflight.json'):sha(transition/'preflight.json')})+')\n'
    compile(body,str(owner),'exec');owner.write_text(body)
    assert (L/'STOP_MAIN').read_text()==reason
    (L/'STOP_MAIN').unlink() # only the exact authorized migration marker
    process=subprocess.Popen([PY,str(owner)],stdin=subprocess.DEVNULL,stdout=(L/'training_world8_v1_owner.stdout').open('xb'),stderr=(L/'training_world8_v1_owner.stderr').open('xb'),start_new_session=True,env=env())
    write(out/'launch.json',dict(owner_pid=process.pid,time=time.time(),resume_step=checkpoint['step'],request_sha256=sha(rp)))
    print(json.dumps(dict(owner_pid=process.pid,resume_step=checkpoint['step'],world=8,owner_deadline=owner_deadline)))


if __name__=='__main__':main()
