"""One reviewed clean-device timing session after observed external contention.

Never resets the first deadline. Reuses the completed CPU-only BPG stages, but
does not mix GPU0 timings with the GPU6 comparison. No further device retry.
"""
from pathlib import Path
import sys,time,os,ast
import run_once as base

def main(request,expected):
    assert base.sha(request)==expected
    r=base.read(request);old=Path(r['previous_study_root']);root=Path(r['study_root'])
    control=root/'closure_gate';control.mkdir(exist_ok=False)
    base.save(control/'owner.json',dict(owner=base.ident(os.getpid()),registration=request,sha256=expected))
    for role in ['queue_owner_control','unlaunched_owner_control','granularity_owner_control']:
        while not(old/role/'actual_wait.json').exists():
            if time.time()>=r['deadline_unix']-60:raise RuntimeError('Original deadline; clean device not launched')
            time.sleep(5)
        w=base.read(old/role/'actual_wait.json');identity=base.read(old/role/'owner.json')['owner']
        for _ in range(30):
            if not base.same_process(identity):break
            time.sleep(.2)
        assert not base.same_process(w['child'])and not base.same_process(identity)
    spent_gpu=spent_cpu=0.;history=[]
    for c in old.glob('*_control'):
        if not c.name.startswith(('main_','diag_','grid_')):continue
        if not(c/'actual_wait.json').exists():continue
        w=base.read(c/'actual_wait.json');assert not base.same_process(w['child'])
        spent_gpu+=w['GPU_device_seconds'];spent_cpu+=w['CPU_process_wall_seconds'];history.append(dict(control=str(c),actual_wait=w))
    remaining=dict(r,GPU_device_seconds=r['GPU_device_seconds']-spent_gpu,CPU_process_wall_seconds=r['CPU_process_wall_seconds']-spent_cpu)
    base.save(control/'prior_resource_receipts.json',dict(GPU_device_seconds=spent_gpu,CPU_process_wall_seconds=spent_cpu,stages=history,deadline_unix=r['deadline_unix']))
    qp=root/'effective_queue_request.json';base.save(qp,remaining)
    text=Path(base.__file__).read_text()
    oldline="snap=gpu_snapshot();save(c/'admission.json',dict(time=time.time(),**snap,cpu=cpu_snapshot()))"
    newline="""snap=gpu_snapshot()
            admission_deadline=min(time.time()+300,r['deadline_unix']-60)
            idle_count=0
            while job['gpu'] and time.time()<admission_deadline and not(root/'STOP').exists():
                matching=[s.split(', ') for s in snap['gpus'].splitlines() if s.split(', ')[1]==r['gpu_uuid']]
                assert len(matching)==1
                idle=not foreign(snap,r['gpu_uuid']) and int(matching[0][3])>=60000 and int(matching[0][4])<=20
                idle_count=idle_count+1 if idle else 0
                if idle_count>=2:break
                time.sleep(2);snap=gpu_snapshot()
            remain=min(r['deadline_unix']-time.time(),r['GPU_device_seconds']-gpu_used if job['gpu'] else r['CPU_process_wall_seconds']-cpu_used)
            save(c/'admission.json',dict(time=time.time(),**snap,cpu=cpu_snapshot()))
            if remain<60 or (root/'STOP').exists():
                status=dict(name=name,status='BLOCKED_STOP_OR_BUDGET_ZERO_LAUNCH',scientific_calls=0)
                save(c/'not_launched.json',status);results.append(status);continue"""
    assert text.count(oldline)==1
    revised=text.replace(oldline,newline)
    assert ast.dump(ast.parse(revised.replace(newline,oldline)))==ast.dump(ast.parse(text))
    base.save(control/'controller_adaptations.json',dict(original=oldline,revised=newline))
    ns=dict(__name__='clean_device_bounded',__file__=base.__file__)
    exec(compile(revised,'<single-clean-device-no-further-retry>','exec'),ns)
    return ns['main'](str(qp),base.sha(qp))

if __name__=='__main__':sys.exit(main(*sys.argv[1:]))
