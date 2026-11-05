"""One recorded infrastructure transition, within the original allocation."""
from pathlib import Path
import json,subprocess,time,sys
from runtime import L,OLD,R,PY,env,sha,write
def main():
    S=Path(__file__).resolve().parent;prior=L/'training_v1';control=L/'controllers/training_v1'
    actual=json.loads((control/'actual_wait.json').read_text());assert actual['returncode']==0 and actual['child_gone']
    segment=json.loads((prior/'segment_complete.json').read_text());assert segment['paused']
    assert (OLD/'STOP_P2').exists() and (L/'STOP_MAIN').read_text()=='ENGINEERING_CHECKPOINT_METADATA_TRANSITION_V2\n'
    old=json.loads((prior/'request.json').read_text());checkpoint=json.loads((prior/'latest_checkpoint.json').read_text())
    assert checkpoint['step']==segment['step'] and sha(checkpoint['path'])==checkpoint['sha256']
    for n in ['runtime.py','model.py','codec.py']:assert sha(S/n)==old['bindings'][str(S/n)],n
    owner_start=json.loads((control/'owner.json').read_text())['started'];train_start=json.loads((prior/'training_started.json').read_text())['time']
    deadline=owner_start+43200;soft_deadline=train_start+old['wall_seconds'];now=time.time()
    assert soft_deadline-now>1200 and deadline-now>1800,'Original allocation has insufficient remaining time'
    for gpu in old['gpus']:
        used=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True);assert int(used)<100
    out=L/'training_v2';out.mkdir(exist_ok=False)
    request=dict(old,output=str(out),resume=checkpoint,wall_seconds=int(soft_deadline-now),owner_max_seconds=int(deadline-now),
       original_owner_deadline=deadline,original_training_deadline=soft_deadline,continuation='same initial allocation; no reset',
       bindings={str(S/n):sha(S/n) for n in ['runtime.py','model.py','codec.py','train_v2.py','resume_training_v2.py']})
    rp=out/'request.json';write(rp,request)
    write(out/'admission.json',dict(passed=True,prior_actual_wait=actual,checkpoint=checkpoint,original_deadlines_preserved=True,
        source_change='checkpoint now includes adopted corpus identity/cursor; final verified remainder can be admitted',
        model_loss_precision_global_batch_unchanged=True,old_P2_still_paused=True))
    owner=L/'training_owner_v2.py';assert not owner.exists()
    body='import sys\nsys.path.insert(0,'+repr(str(S))+')\nfrom runtime import *\nsys.path.insert(0,str(R/"experiments/rdvq-prefix-20261104"))\nfrom owned_process import run_role\ne=env();e["CUDA_VISIBLE_DEVICES"]="1,2,4,5"\n'
    command=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(S/'train_v2.py'),'--request',str(rp)]
    body+='run_role('+repr(str(L/'controllers'))+',"training_v2",'+repr(command)+',e,list(range(48,80)),[1,2,4,5],'+str(request['owner_max_seconds'])+','+repr({**request['bindings'],str(rp):sha(rp)})+')\n'
    owner.write_text(body)
    (L/'STOP_MAIN').unlink() # only our verified engineering-transition marker
    p=subprocess.Popen([PY,str(owner)],stdin=subprocess.DEVNULL,stdout=(L/'training_owner_v2.stdout').open('xb'),stderr=(L/'training_owner_v2.stderr').open('xb'),start_new_session=True,env=env())
    write(out/'launch.json',dict(owner_pid=p.pid,time=time.time(),resumed_step=checkpoint['step'],request_sha256=sha(rp)))
    print(json.dumps(dict(owner_pid=p.pid,resume_step=checkpoint['step'],owner_deadline=deadline,remaining_seconds=request['owner_max_seconds'])))
if __name__=='__main__':main()
