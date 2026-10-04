"""Run the qualified health-polling revision and stop on any parity failure."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import m1_common as u

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--original-output',required=True);a=p.parse_args()
    root=Path(a.root).resolve();out=Path(a.output).resolve();old=Path(a.original_output).resolve();here=Path(__file__).resolve().parent
    reg=u.read(out/'registration.json');u.verify(reg['source_bindings']);u.verify(reg['input_bindings'])
    try:
        for stage in ('qualify','benchmark'):
            subprocess.run([sys.executable,'-B',str(here/'m1_performance.py'),'--root',str(root),'--protocol',str(here/'protocol.json'),
                            '--output',str(out),'--stage',stage],check=True)
        before=u.read(old/'benchmark_registration.json');after=u.read(out/'benchmark_registration.json')
        for k in ('frozen_identity','numerical_runtime','original_population','action_grid','noise_seeds'):
            u.require(before[k]==after[k],'Performance revision changed scientific state: '+k)
        cases=[]
        for i in range(2):
            bp=old/'benchmark/source_checkpoints'/('%04d.json'%i);ap=out/'benchmark/source_checkpoints'/('%04d.json'%i)
            b=u.checkpoint(bp,u.identity(before),i);n=u.checkpoint(ap,u.identity(after),i)
            u.require(len(b['rows'])==len(n['rows'])==2070 and u.identity(b['rows'])==u.identity(n['rows']),
                      'Complete actual frame rows changed under health-poll optimization')
            cases.append(dict(source_index=i,rows=2070,row_sha256=u.identity(n['rows']),original_checkpoint_sha256=u.sha(bp),
                              optimized_checkpoint_sha256=u.sha(ap),original_seconds=b['seconds'],optimized_seconds=n['seconds']))
        u.seal(out/'performance_parity.json',dict(status='FULL_ROW_EXACT_PARITY_PASS',sources=2,rows=4140,cases=cases,
            old_registration_sha256=u.sha(old/'registration.json'),new_registration_sha256=u.sha(out/'registration.json'),
            healthy_query_seconds=5,hot_queries_unthrottled=True,scientific_changes=False))
        for stage in ('calibrate','development'):
            subprocess.run([sys.executable,'-B',str(here/'m1_performance.py'),'--root',str(root),'--protocol',str(here/'protocol.json'),
                            '--output',str(out),'--stage',stage],check=True)
        u.write(out/'owner_completion.json',dict(status='RECONSTRUCTIONS_COMPLETE',metrics='separate registered delivery owner'))
    except BaseException as e:
        u.write(out/'owner_failure.json',dict(error=repr(e),automatic_retry=False));raise

if __name__=='__main__':main()
