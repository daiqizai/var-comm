"""First launch only for preregistered stages refused before Popen.

This controller waits for the first queue's actual closure, never retries a
scientific failure, and keeps the original deadline and cumulative resources.
"""
from pathlib import Path
import sys, time, json, hashlib, ast
import run_once as base

def main(request, expected):
    assert base.sha(request)==expected
    r=base.read(request);root=Path(r['study_root']);control=root/'unlaunched_supervision'
    control.mkdir(exist_ok=False)
    base.save(control/'owner.json',dict(owner=base.ident(__import__('os').getpid()),request=request,sha256=expected))
    while not(root/'queue_owner_control/actual_wait.json').exists():
        if time.time()>=r['deadline_unix']-60:raise RuntimeError('Original absolute deadline; no launch')
        time.sleep(5)
    original=base.read(root/'queue_control/completion.json')
    ow=base.read(root/'queue_owner_control/actual_wait.json')
    for _ in range(30):
        if not base.same_process(base.read(root/'queue_owner_control/owner.json')['owner']):break
        time.sleep(.2)
    assert not base.same_process(ow['child']) and not base.same_process(base.read(root/'queue_owner_control/owner.json')['owner'])
    jobs=[];evidence=[]
    for job in r['jobs']:
        c=root/(job['name']+'_control')
        if not(c/'not_launched.json').exists():continue
        reason=base.read(c/'not_launched.json')
        if reason['status']!='BLOCKED_GPU_NOT_IDLE_ZERO_LAUNCH':continue
        assert not(c/'child.json').exists()and not(c/'actual_wait.json').exists()and not(root/job['name']).exists()
        assert base.sha(job['request'])==job['request_sha256']
        jobs.append(job);evidence.append(dict(stage=job['name'],original_refusal=reason,refusal_sha256=base.sha(c/'not_launched.json'),same_worker_registration_sha256=job['request_sha256'],prior_scientific_calls=0))
    new=dict(r,jobs=jobs,GPU_device_seconds=r['GPU_device_seconds']-original['GPU_device_seconds'],CPU_process_wall_seconds=r['CPU_process_wall_seconds']-original['CPU_process_wall_seconds'])
    base.save(control/'zero_launch_evidence.json',dict(stages=evidence,original_actual_wait=ow,original_resources=original,deadline_unchanged=True))
    qp=root/'unlaunched_queue_request.json';base.save(qp,new)
    text=Path(base.__file__).read_text()
    replacements=[
      ("control=root/'queue_control'", "control=root/'unlaunched_queue_control'"),
      ("c=root/(name+'_control')", "c=root/(name+'_first_launch_control')"),
      ("snap=gpu_snapshot();save(c/'admission.json',dict(time=time.time(),**snap,cpu=cpu_snapshot()))",
       """snap=gpu_snapshot()
            admission_deadline=min(time.time()+300,r['deadline_unix']-60)
            while job['gpu']:
                matching=[s.split(', ') for s in snap['gpus'].splitlines() if s.split(', ')[1]==r['gpu_uuid']]
                assert len(matching)==1
                if not foreign(snap,r['gpu_uuid']) and int(matching[0][3])>=60000 and int(matching[0][4])<=20:break
                if time.time()>=admission_deadline:break
                time.sleep(5);snap=gpu_snapshot()
            remain=min(r['deadline_unix']-time.time(),r['GPU_device_seconds']-gpu_used if job['gpu'] else r['CPU_process_wall_seconds']-cpu_used)
            save(c/'admission.json',dict(time=time.time(),**snap,cpu=cpu_snapshot()))"""),
    ]
    revised=text
    for old,newtext in replacements:
        assert revised.count(old)==1
        revised=revised.replace(old,newtext)
    undone=revised
    for old,newtext in reversed(replacements):undone=undone.replace(newtext,old)
    assert ast.dump(ast.parse(undone))==ast.dump(ast.parse(text))
    base.save(control/'controller_adaptations.json',replacements)
    ns=dict(__name__='bounded_first_launch',__file__=base.__file__)
    exec(compile(revised,'<first-launch-only-resource-wait>','exec'),ns)
    return ns['main'](str(qp),base.sha(qp))

if __name__=='__main__':sys.exit(main(*sys.argv[1:]))
