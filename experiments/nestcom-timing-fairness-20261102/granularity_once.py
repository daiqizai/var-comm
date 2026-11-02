"""Finish the three additional TX cells actually used by main Table I.

The token arm reuses main_g04_tx. This is a finite predeclared addition inside
the same absolute deadline and GPU/CPU budget, not a new optimization search.
"""
from pathlib import Path
import sys,time,os,ast
import run_once as base

def main(request,expected):
    assert base.sha(request)==expected
    r=base.read(request);root=Path(r['study_root']);control=root/'granularity_supervision'
    control.mkdir(exist_ok=False)
    base.save(control/'owner.json',dict(owner=base.ident(os.getpid()),registration=request,sha256=expected))
    histories=[]
    for owner,queue in [('queue_owner_control','queue_control'),('unlaunched_owner_control','unlaunched_queue_control')]:
        while not(root/owner/'actual_wait.json').exists():
            if time.time()>=r['deadline_unix']-60:raise RuntimeError('Original deadline reached; no granularity launch')
            time.sleep(5)
        w=base.read(root/owner/'actual_wait.json')
        for _ in range(30):
            if not base.same_process(base.read(root/owner/'owner.json')['owner']):break
            time.sleep(.2)
        assert not base.same_process(w['child'])and not base.same_process(base.read(root/owner/'owner.json')['owner'])
        histories.append(base.read(root/queue/'completion.json'))
    remaining=dict(r)
    for k in ['GPU_device_seconds','CPU_process_wall_seconds']:
        remaining[k]-=sum(h[k]for h in histories)
    base.save(control/'prior_resource_receipts.json',histories)
    qp=root/'granularity_queue_request.json';base.save(qp,remaining)
    text=Path(base.__file__).read_text()
    changes=[("control=root/'queue_control'","control=root/'granularity_queue_control'"),
      ("snap=gpu_snapshot();save(c/'admission.json',dict(time=time.time(),**snap,cpu=cpu_snapshot()))",
       """snap=gpu_snapshot()
            admission_deadline=min(time.time()+300,r['deadline_unix']-60)
            while job['gpu']:
                matching=[s.split(', ') for s in snap['gpus'].splitlines() if s.split(', ')[1]==r['gpu_uuid']]
                assert len(matching)==1
                if not foreign(snap,r['gpu_uuid']) and int(matching[0][3])>=60000 and int(matching[0][4])<=20:break
                if time.time()>=admission_deadline:break
                time.sleep(5);snap=gpu_snapshot()
            remain=min(r['deadline_unix']-time.time(),r['GPU_device_seconds']-gpu_used)
            save(c/'admission.json',dict(time=time.time(),**snap,cpu=cpu_snapshot()))""")]
    revised=text
    for old,new in changes:
        assert revised.count(old)==1;revised=revised.replace(old,new)
    undone=revised
    for old,new in reversed(changes):undone=undone.replace(new,old)
    assert ast.dump(ast.parse(undone))==ast.dump(ast.parse(text))
    base.save(control/'controller_adaptations.json',changes)
    ns=dict(__name__='granularity_bounded',__file__=base.__file__)
    exec(compile(revised,'<granularity-only-same-deadline>','exec'),ns)
    return ns['main'](str(qp),base.sha(qp))

if __name__=='__main__':sys.exit(main(*sys.argv[1:]))
