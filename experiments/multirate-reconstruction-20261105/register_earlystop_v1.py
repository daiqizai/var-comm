"""Preserve the stopped checkpoint and amend only the evaluation trigger/claims."""
from pathlib import Path
import json,os,time
from runtime import L,OLD,CHECKPOINT_ROOT,sha,write,setup
setup()
import torch
from collect_snapshot import collect

def main():
    source=Path(__file__).resolve().parent
    stopped=L/'user_early_stop_20261105_v1';root=L/'posttrain_earlystop_v1'
    assert not root.exists()
    s=collect();role='training_world8_v1';decision=json.loads((stopped/'decision.json').read_text())
    assert s['active_training_role']==role and s['old_P2_stop']
    for name in [role,'posttrain_chain_v1']:
        status=s['ledger'][name]
        assert not status['owner_live'] and not status['child_live']
        assert status['actual_wait']['child_gone']
        assert status['actual_wait']['returncode']==(0 if name==role else 1)
    assert s['ledger'][role]['actual_wait']['finished']<=decision['maximum_training_end_epoch']
    assert not list((L/'posttrain_v1/roles').glob('*'))
    cp=s['training']['latest_checkpoint.json'];done=s['training']['segment_complete.json']
    assert cp['step']==done['step']<100000 and not done['training_finished']
    assert sha(cp['path'])==cp['sha256']
    immutable=CHECKPOINT_ROOT/'main_v1'/f"early_stop_step{cp['step']:06d}.pt"
    assert not immutable.exists();os.link(cp['path'],immutable)
    assert sha(immutable)==cp['sha256']
    cp=dict(cp,path=str(immutable))
    state=torch.load(immutable,map_location='cpu',mmap=True,weights_only=False)
    assert state['step']==cp['step'] and state['cursor']==cp['cursor']
    assert state['world_size']==8 and len(state['rng_by_rank'])==8 and state['global_batch']==32
    assert state['schedule']==[{'resolution':256,'updates':100000},{'resolution':512,'updates':50000}]
    assert state['optimizer']['state'] and state['model']
    for rng in state['rng_by_rank']:assert {'python','numpy','torch','cuda'}<=set(rng)
    assert state['corpus_sha256']==sha(state['corpus_path'])
    proof=dict(checkpoint=cp,world_size=8,optimizer_state_entries=len(state['optimizer']['state']),
        rank_RNG_states=8,global_batch=32,corpus_sha256=state['corpus_sha256'],
        original_schedule=state['schedule'],training_actual_wait=s['ledger'][role]['actual_wait'],
        old_chain_actual_wait=s['ledger']['posttrain_chain_v1']['actual_wait'],
        old_chain_exit_reason='explicit user early-stop; no training/test child was launched',old_four_paused=True)
    del state
    write(stopped/'checkpoint_verification.json',proof)
    write(stopped/'complete.json',dict(passed=True,step=cp['step'],checkpoint=cp,training_stopped=True,old_chain_stopped=True,
        schedule_completed=False,updates_512=0,earlier_than_user_deadline=True,automatic_training_resume=False))
    original=L/'posttrain_v1/protocol.json';protocol=json.loads(original.read_text())
    for p,h in protocol['bindings'].items():assert sha(p)==h,p
    root.mkdir();(root/'worker_requests').mkdir();(root/'owner_requests').mkdir();(root/'roles').mkdir()
    protocol.update(root=str(root),initial_training_role=role,max_new_training_segments=0,
        deadline=min(protocol['deadline'],time.time()+48*3600),evaluation_checkpoint=cp,
        training_stop_sha256=sha(L/'STOP_MAIN'),supersedes_protocol=str(original),supersedes_protocol_sha256=sha(original),
        training_state=dict(stopped_step=cp['step'],schedule_target=150000,schedule_completed=False,
            updates_256=cp['step'],updates_512=0,stop_reason='user-authorized early stop after deteriorating training reconstruction losses'),
        resolution_interpretation={'256':'early-stopped trained-resolution codec comparison',
            '512':'untrained-resolution diagnostic; no scheduled 512 updates, mode embeddings not optimized at 512'},
        checkpoint_selection='last checkpoint after requested safe stop; no test-based selection',
        training_resume_authorized=False)
    protocol['bindings'].update({str(source/n):sha(source/n) for n in ['earlystop_eval_v1.py','earlystop_summary_v1.py','register_earlystop_v1.py']})
    write(root/'protocol.json',protocol)
    write(root/'registration_complete.json',dict(passed=True,protocol_sha256=sha(root/'protocol.json'),
        reused_sources=500,reused_test_protocol=True,new_model_calls=0,new_metric_calls=0,training_resumed=False,
        checkpoint=cp,training_state=protocol['training_state'],resolution_interpretation=protocol['resolution_interpretation']))
    print(json.dumps(dict(passed=True,step=cp['step'],checkpoint=cp,protocol_sha256=sha(root/'protocol.json'),new_model_calls=0)))

if __name__=='__main__':main()
