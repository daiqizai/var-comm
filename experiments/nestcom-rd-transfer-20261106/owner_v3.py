"""Finite role launcher. No retries or automatic training resumption."""
import argparse,json,os,time
from runtime import *
setup()
from owned_process import run_role
def main():
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['qualification_tx_v2','qualification_rx_v2','training_v3','publication_checks_v2']);a=ap.parse_args()
    r=json.loads((L/'protocol_v3.json').read_text());old_paused();e=env()
    if a.role.startswith('qualification'):
        e['CUDA_VISIBLE_DEVICES']='0';cmd=[PY,str(S/'qualify_v2.py'),'tx' if '_tx_' in a.role else 'rx'];gpus=[0];cpus=[0,2];seconds=900
    elif a.role=='training_v3':
        tx=json.loads((L/'qualification_v2/tx_pass.json').read_text());rx=json.loads((L/'qualification_v2/rx_pass.json').read_text());assert tx['passed'] and rx['passed'] and tx['expected']==rx['rows']
        for role in ['qualification_tx_v2','qualification_rx_v2']:
            actual=json.loads((L/'roles'/role/'actual_wait.json').read_text());assert actual['returncode']==0 and actual['child_gone']
        assert time.time()<r['deadline']-240
        e['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7';gpus=list(range(8));cpus=sum(r['rank_cpus'].values(),[])
        cmd=[PY,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',str(S/'train_v3.py'),'--request',str(L/'protocol_v3.json')];seconds=max(1,r['deadline']-time.time())
    else:
        e['CUDA_VISIBLE_DEVICES']='';gpus=[];cpus=[32,34];seconds=600
        cmd=[PY,str(S/'checks.py')]
    run_role(L/'roles',a.role,cmd,e,cpus,gpus,seconds,r['bindings'])
if __name__=='__main__':main()
