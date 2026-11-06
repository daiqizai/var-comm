"""Own one finite shard through TX -> independent RX -> five metrics."""
import argparse,json,time
from pathlib import Path
from runtime import *
setup()
from owned_process import run_role

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
    r=json.loads(Path(a.request).read_text());old_paused();stop=min(r['deadline'],time.time()+r['max_worker_seconds'])
    bindings=dict(r['bindings']);bindings[a.request]=sha(a.request)
    e=env();e['CUDA_VISIBLE_DEVICES']=str(r['gpu']);base=Path(r['output']);base.mkdir(exist_ok=False)
    for stage in ['tx','rx','score']:
        assert time.time()<stop-10
        run_role(base/'roles',stage,[PY,str(S/'formal_worker_v1.py'),'--request',a.request,'--stage',stage],
                 e,r['cpus'],[r['gpu']],stop-time.time(),bindings)
    write(base/'complete.json',dict(passed=True,time=time.time(),qualification=r['qualification']))

if __name__=='__main__':main()
