"""256-only scope and exact-step admission revision after zero-inference failures."""
from pathlib import Path
import json,time
from runtime import L,sha,write
from collect_snapshot import collect

def main():
    source=Path(__file__).resolve().parent;prior=L/'posttrain_earlystop_v1';root=L/'posttrain_256_v2'
    assert not root.exists()
    snap=collect();old=snap['ledger']['earlystop_evaluation_v1']
    assert not old['owner_live'] and not old['child_live'] and old['actual_wait']['returncode']==0
    assert not json.loads((prior/'delivery/complete.json').read_text())['passed']
    failures={}
    for size in [256,512]:
        stage=prior/'roles'/f'{size}_00_tx';wait=json.loads((stage/'actual_wait.json').read_text())
        assert wait['returncode']==1 and wait['child_gone']
        assert "assert state['step']==150000 and state['world_size']==8" in (stage/'stderr.log').read_text()
        assert not list((prior/f'eval_{size}_00').rglob('*.*'))
        failures[str(size)]=dict(actual_wait=wait,error_sha256=sha(stage/'stderr.log'),model_constructions=1,
            checkpoint_load_attempts=1,encoder_forward_calls=0,receiver_forward_calls=0,metric_calls=0)
    protocol=json.loads((prior/'protocol.json').read_text())
    assert sha(protocol['evaluation_checkpoint']['path'])==protocol['evaluation_checkpoint']['sha256']
    for p,h in protocol['bindings'].items():assert sha(p)==h,p
    assert protocol['evaluation_checkpoint']['step']==82858
    root.mkdir();(root/'worker_requests').mkdir();(root/'owner_requests').mkdir();(root/'roles').mkdir()
    protocol.update(root=str(root),evaluation_resolutions=[256],user_scope='256测试线: execute 256 only; do not start 512',
        supersedes_protocol=str(prior/'protocol.json'),supersedes_protocol_sha256=sha(prior/'protocol.json'),
        resolution_interpretation={'256':'early-stopped trained-resolution codec comparison','512':'NOT_RUN_USER_SCOPE'},
        max_evaluation_GPU_device_hours=48,max_logical_positions=8000,automatic_training_resume=False)
    protocol['bindings'].update({str(source/n):sha(source/n) for n in ['posttrain_worker_earlystop_v2.py','eval256_chain_v2.py','eval256_summary_v2.py','register_eval256_v2.py']})
    write(root/'protocol.json',protocol)
    write(root/'predecessor_failures.json',failures)
    write(root/'revision.json',dict(time=time.time(),cause='original worker still required completed step150000; early-stopped checkpoint was rejected before image encoding',
        fix='new version requires exact registered checkpoint SHA and step82858, with world_size8; no model or coding changes',
        worker_parent_sha256=sha(source/'posttrain_worker_v1.py'),worker_revision_sha256=sha(source/'posttrain_worker_earlystop_v2.py'),
        successful_scientific_calls_repeated=0,old_failure_bytes_preserved=True,training_resumed=False,scope=[256]))
    write(root/'registration_complete.json',dict(passed=True,protocol_sha256=sha(root/'protocol.json'),new_model_calls=0,scope=[256],
        exact_checkpoint_step=82858,old_512_attempt_before_scope_change='model load rejected; zero image/metric calls; no 512 continuation'))
    print(json.dumps(dict(passed=True,protocol_sha256=sha(root/'protocol.json'),scope=[256],training_resumed=False)))

if __name__=='__main__':main()
