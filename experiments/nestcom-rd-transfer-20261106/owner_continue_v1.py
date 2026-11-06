"""One explicitly authorized finite continuation, with no automatic retry."""
import json,time,os
from runtime import *
setup()
from owned_process import run_role

def main():
    r=json.loads((L/'continuation_v1.json').read_text());old_paused()
    assert time.time()<r['deadline']-300
    for role in ['qualification_tx_v2','qualification_rx_v2','training_v3']:
        a=json.loads((L/'roles'/role/'actual_wait.json').read_text())
        assert a['returncode']==0 and a['child_gone']
    e=env();e['CUDA_VISIBLE_DEVICES']=','.join(map(str,r['gpu_devices']))
    cmd=[PY,'-m','torch.distributed.run','--standalone','--nnodes=1',
         '--nproc_per_node='+str(r['world_size']),str(S/'train_continue_v1.py'),
         '--request',str(L/'continuation_v1.json')]
    bindings=dict(r['bindings']);bindings[str(L/'continuation_v1.json')]=sha(L/'continuation_v1.json')
    run_role(L/'roles','training_continue_v1',cmd,e,
             sum(r['rank_cpus'].values(),[]),r['gpu_devices'],
             r['deadline']-time.time(),bindings)

if __name__=='__main__':main()
