"""Sequential GPU ownership and bounded CPU parallelism for the frozen study.

This is a computation queue, not a source of scientific observations. Every
transition requires the corresponding immutable worker completion receipt.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from uep_common import read,write,seal,sha,require
import artifact_bridge as bridge

STOP=False
def stop(*_):
    global STOP
    STOP=True

class Owner:
    def __init__(self,config):
        self.config=config;self.root=Path(config['root']);self.out=Path(config['out']);self.code=Path(__file__).resolve().parent
        self.phy=self.out/'ldpc_environment/bin/python';self.native=Path(config['native_python'])
        self.shards=[self.out/'phy_shards'/str(i) for i in range(8)]
        self.profiles=self.out/'stage_a/candidate_profiles_N1024.json'
        self.children=[]

    def check(self):
        require(not STOP and not (self.out/'STOP').exists(),'Study stop requested; source/block checkpoints retained')
        for path,digest in self.config['source_bindings'].items():require(sha(path)==digest,'Registered study source changed: '+path)
        for d in self.shards:
            require(not (d/'failure.json').exists(),'PHY shard failure needs review: '+str(d))

    def status(self,status,**fields):
        write(self.out/'owner_status.json',dict(status=status,pid=os.getpid(),time=time.time(),**fields))

    def environment(self,gpu=False):
        e=os.environ.copy()
        if gpu:e.update(self.config['native_environment'])
        else:e.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        e.update(VAR_COMM_ROOT=str(self.root),PYTHONDONTWRITEBYTECODE='1')
        e['PYTHONPATH']=str(self.code)+':'+str(self.root/'src')+':'+e.get('PYTHONPATH','')
        return e

    def start(self,name,args,gpu=False):
        self.check()
        if gpu:
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            require(not active,'GPU has an existing compute owner: '+active)
        logs=self.out/'logs';logs.mkdir(exist_ok=True)
        with (logs/(name+'.log')).open('ab') as f:
            p=subprocess.Popen([str(self.native if gpu else self.phy),'-B',*[str(x) for x in args]],cwd=self.root,
                env=self.environment(gpu),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        self.children.append(p)
        write(self.out/'active_child.json',dict(name=name,pid=p.pid,args=[str(x) for x in args],gpu=gpu,time=time.time()))
        return p

    def wait(self,children):
        while any(p.poll() is None for p in children):
            self.check();time.sleep(15)
        require(all(p.returncode==0 for p in children),'Worker failed; inspect immutable stage logs and receipts')

    def run(self,name,args,gpu=False):
        self.status('RUNNING_'+name.upper());self.wait([self.start(name,args,gpu)])

    def wait_coarse(self):
        self.status('WAITING_FOR_CPU_COARSE')
        while not all((d/'coarse_completion.json').exists() for d in self.shards):
            self.check()
            for job in read(self.out/'coarse_launch.json')['jobs']:
                if (self.shards[job['shard']]/'coarse_completion.json').exists():continue
                state=self.shards[job['shard']]/'status.json'
                if state.exists() and read(state).get('status')=='PAUSED':
                    dispatcher=read(self.out/'coarse_dispatch_launch.json');proc=Path('/proc')/str(dispatcher['pid'])/'cmdline'
                    require(proc.exists() and 'coarse_dispatch.py' in proc.read_text(),'Queued CPU shards have no live dispatcher')
                    continue
                proc=Path('/proc')/str(job['pid'])/'stat'
                require(proc.exists() and proc.read_text().split()[21]==job['proc_start_ticks'],
                        'A coarse worker exited without its completion receipt')
            time.sleep(15)
        merged=self.out/'bler_coarse.json'
        if not merged.exists():self.merge('coarse',merged)
        return merged

    def merge(self,phase,destination,selection=None):
        args=[self.code/'merge_lookup.py','--profiles',self.profiles,'--shards',*self.shards,'--phase',phase,'--output',destination]
        if selection:args+=['--selection',selection]
        self.run('merge_'+destination.stem,args)

    def wait_old_gpu_delivery(self):
        upstream=Path(self.config['upstream_completion'])
        self.status('WAITING_FOR_M1_N2048_DELIVERY',upstream=str(upstream),CPU_lookup_continues=True)
        while not upstream.exists():
            self.check()
            for failure in self.config['upstream_failure_files']:
                require(not Path(failure).exists(),'Existing M1 task failed; GPU queue retained for review')
            time.sleep(15)
        value=read(upstream)
        require(value.get('status')==self.config['upstream_required_status'],'Existing M1 was not successfully delivered')
        # The receipt is written before its publisher exits; it does not own CUDA.

    def quality(self,states,output,limit,reuse=None,stage='run'):
        receipt=output/('benchmark.json' if stage=='benchmark' else 'completion.json')
        if receipt.exists():return
        args=[self.code/'quality_driver.py','--root',self.root,'--states',states,'--output',output,'--stage',stage,'--source-limit',limit]
        if reuse:args+=['--reuse-from',reuse]
        self.run('Q_'+output.name,args,True)
        require(receipt.exists(),'Quality worker returned without a completion receipt')

    def optimize(self,profiles,qbundle,bler,destination,stage):
        if not destination.exists():self.run('select_'+destination.stem,[self.code/'optimize.py','--profiles',profiles,'--quality',qbundle,'--bler',bler,'--output',destination,'--stage',stage])
        return read(destination)

    def refine(self,selection,iteration):
        jobs=[]
        for i,d in enumerate(self.shards):
            done=d/('refine_'+sha(selection)+'_completion.json')
            if done.exists():continue
            jobs.append(self.start('refine_%d_%d'%(iteration,i),[self.code/'phy_lookup.py','--root',self.root,
                '--profiles',self.profiles,'--out',d,'--qualification',self.out/'ldpc_qualification.json','--device','cpu',
                '--stage','refine','--refine',selection,'--shard-index',i,'--shard-count',8,'--benchmark-from',self.out/'phy_benchmark']))
        self.status('REFINING_BLER',iteration=iteration,workers=len(jobs));self.wait(jobs)
        path=self.out/('bler_refined_%02d.json'%iteration)
        if not path.exists():self.merge('refine',path,selection)
        return path

    def main(self):
        self.wait_old_gpu_delivery()
        states=self.out/'quality_states_registered.json'
        bridge.registered_states(self.profiles,states,{str(self.code/'protocol.json'):sha(self.code/'protocol.json')})
        pilot=self.out/'quality_pilot';self.quality(states,pilot,1000,stage='benchmark')
        qhours=read(pilot/'benchmark.json')['estimated_full1000_source_Q_hours']
        # Cost allowances are frozen engineering planning values, not observations.
        # Actual timings replace them in delivery. Coarse work usually completes
        # while the pre-existing M1 job owns the GPU.
        pending=0. if all((d/'coarse_completion.json').exists() for d in self.shards) else self.config['coarse_initial_upper_hours']
        cost=bridge.combined_cost_rule(qhours,pending,self.config['refinement_allowance_hours'],self.config['actual_allowance_hours'])
        cost.update(quality_pilot_sha256=sha(pilot/'benchmark.json'),planning_allowances_not_benchmark=True)
        seal(self.out/'stage_a_cost_decision.json',cost)
        self.status('COST_DECISION_FROZEN',**cost)
        limit=cost['default_quality_sources'];qout=self.out/('quality_screen300' if limit==300 else 'quality_final1000')
        self.quality(states,qout,limit)
        bundle=qout/'quality_bundle.json';bridge.quality_bundle(qout,bundle)
        bler=self.wait_coarse();profiles=self.profiles
        if limit==300:
            screen=self.out/'screen300_policies.json';self.optimize(profiles,bundle,bler,screen,'screen')
            profiles=self.out/'shortlist_profiles.json';bridge.freeze_shortlist(self.profiles,screen,profiles)
            states=self.out/'shortlist_states_registered.json';bridge.registered_states(profiles,states,{str(screen):sha(screen)})
            final=self.out/'quality_final1000';self.quality(states,final,1000,reuse=qout)
            qout=final;bundle=qout/'quality_bundle.json';bridge.quality_bundle(qout,bundle)
        selected=None
        for iteration in range(9):
            selected=self.out/('final_policy_revision_%02d.json'%iteration)
            result=self.optimize(profiles,bundle,bler,selected,'final')
            if bridge.ready_actual(result):break
            require(iteration<8,'Calibration refinement still unstable after eight rounds; review without opening development')
            bler=self.refine(selected,iteration)
        from evaluate import selected_schedule
        from profiles import freeze_codebook
        _,rows=selected_schedule(result);codebook=self.out/'final_codebook.json';seal(codebook,freeze_codebook(rows))
        seal(self.out/'final_freeze.json',dict(policies=str(selected),quality_bundle=str(bundle),quality_completion=str(qout/'completion.json'),
            bler=str(bler),codebook=str(codebook),training_updates=0,development_read=False,
            bindings={str(p):sha(p) for p in (selected,bundle,qout/'completion.json',bler,codebook)}))
        shared=['--policies',selected,'--quality-bundle',bundle,'--quality-completion',qout/'completion.json','--bler',bler,'--codebook',codebook]
        tx=self.out/'tx_tokens';events=self.out/'actual_events';scores=self.out/'actual_scores'
        if not (tx/'manifest.json').exists():self.run('tx_export',[self.code/'tx_export.py','--root',self.root,'--output',tx,*shared],True)
        if not (events/'completion.json').exists():self.run('actual_link',[self.code/'evaluate.py','--root',self.root,'--out',events,
            '--qualification',self.out/'ldpc_qualification.json','--device','cpu','--tx-manifest',tx/'manifest.json',*shared])
        if not (scores/'completion.json').exists():self.run('actual_scoring',[self.code/'score_actual.py','--root',self.root,'--events',events,
            '--output',scores,'--convnext-weights',self.out/'weights/convnext_tiny-983f1562.pth',*shared],True)
        # Delivery has its own immutable commands, tests, scope and publication
        # receipts. A missing delivery registration cannot be called completion.
        for item in self.config['delivery_commands']:
            receipt=self.out/item['completion_relative']
            if receipt.exists():continue
            args=[str(x).replace('{ROOT}',str(self.root)).replace('{OUT}',str(self.out)).replace('{CODE}',str(self.code))
                .replace('{POLICIES}',str(selected)).replace('{BLER}',str(bler)).replace('{QUALITY_BUNDLE}',str(bundle))
                .replace('{QUALITY_COMPLETION}',str(qout/'completion.json')).replace('{CODEBOOK}',str(codebook)) for x in item['args']]
            self.run(item['name'],args,item.get('gpu',False));require(receipt.exists(),'Missing delivery receipt: '+str(receipt))
        require(self.config['delivery_commands'],'No final delivery registered; scientific outputs require review')
        gate=read(self.out/'extension_gate.json')
        self.status('N1024_DELIVERED_N2048_EXTENSION_REQUIRED' if gate['N2048_extension_allowed'] else 'REGISTERED_STUDY_DELIVERED_AND_STOPPED',
                    N2048_extension_allowed=gate['N2048_extension_allowed'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();config=read(a.config)
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'study_owner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (out/'owner_failure.json').exists(),'Earlier controller failure needs explicit review')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    owner=Owner(config)
    try:owner.main()
    except Exception as error:
        if STOP or (out/'STOP').exists():
            write(out/'STOP',dict(reason='requested stop',time=time.time()))
            for job in read(out/'coarse_launch.json')['jobs']:
                proc=Path('/proc')/str(job['pid'])/'stat'
                if proc.exists() and proc.read_text().split()[21]==job['proc_start_ticks']:
                    os.kill(job['pid'],signal.SIGTERM)
        for child in owner.children:
            if child.poll() is None:child.terminate()
        owner.status('STOPPED_REQUIRES_REVIEW',error=repr(error));write(out/'owner_failure.json',dict(error=repr(error),time=time.time(),pid=os.getpid()));raise

if __name__=='__main__':main()
