"""Detached sidecar: qualify on CPU, wait for old work, score then publish."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from runner import HERE,ROOT,OUT,PARENT,read,write,sha,verify,source_bindings,parent_receipts

def execution_paths():
    return [ROOT/p for p in ('experiments/rx-posterior-step1-20260929','src',
        'experiments/var-latent-enhancement-20260917/src',
        'experiments/var-latent-enhancement-20260917/phase_b/src',
        'experiments/var-latent-enhancement-20260917/evaluation/src',
        'experiments/var-latent-enhancement-20260917/followup/src',
        'experiments/var-latent-enhancement-20260917/research/src',
        'experiments/var-latent-enhancement-20260917/mechanisms/src',
        'experiments/var-short-prefix-hybrid-20260923/src',
        'experiments/token_channel_efficiency_20260923/src')]

def gpu_users():
    output=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return [int(line.strip()) for line in output.splitlines() if line.strip() and line.strip()!='No running processes found']

def completed_scoring(bound,manifest_sha):
    receipt=read(OUT/'scoring_completion.json')
    required=dict(status='COMPLETE',synthetic=False,training_updates=0,policy_selection_updates=0,
        parity_passed=True,source_bindings=bound,modelmanifest_sha256=manifest_sha)
    if any(receipt.get(k)!=v for k,v in required.items()):raise RuntimeError('Scoring completion is stale or failed')
    if receipt.get('frames',0)<=0:raise RuntimeError('Empty completed scoring')
    for field in ('inputs','outputs','source_bindings'):verify(receipt[field])
    return receipt

def original_processes(parent=PARENT,processes=None):
    from publish import process_snapshot
    processes=process_snapshot() if processes is None else processes
    launch=read(parent/'supervisor_launch.json')
    owner=int(launch['pid']);ticks=str(launch['start_ticks'])
    needle='/experiments/scale-causal-partial-residual-20261002/'
    active=set()
    for pid,item in processes.items():
        if pid==owner and item.get('unreadable'):
            raise RuntimeError('Cannot verify current parent owner exit')
        if item.get('state')=='Z':continue
        if needle in item.get('command','') or (pid==owner and str(item.get('start_ticks'))==ticks):
            active.add(pid)
    return sorted(active)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bound=source_bindings();pid=os.getpid();start=time.time()
    def status(state,**extra):
        write(OUT/'supervisor_status.json',dict(status=state,pid=pid,time=time.time(),elapsed_seconds=time.time()-start,**extra))
    registration=OUT/'supervisor_registration.json'
    if registration.exists() and read(registration)['source_bindings']!=bound:
        raise RuntimeError('Supervisor registered source changed')
    write(registration,dict(source_bindings=bound,original_experiments=str(PARENT),
        scope=['N512','N1024','M1','M1_RATE','M2_ORACLE','M2_ACTUAL'],training_updates=0,selection_updates=0,
        source_commit_waits_for_original_completion=True,stop_after_verified_publication=True))
    def run(stage,script,args=(),python=None,cpu=False):
        verify(bound);log=OUT/f'{stage}_{time.time_ns()}.log'
        env=dict(os.environ,OMP_NUM_THREADS='2' if cpu else '6',OPENBLAS_NUM_THREADS='2',
            PYTHONPATH=os.pathsep.join(str(p) for p in execution_paths()))
        if cpu:env.update(CUDA_VISIBLE_DEVICES='',MKL_NUM_THREADS='2')
        with log.open('a') as f:
            p=subprocess.Popen([str(python or sys.executable),'-u',str(HERE/script),*args],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env)
            status('RUNNING',stage=stage,worker_pid=p.pid,log=str(log))
            code=p.wait()
        verify(bound)
        if code:raise RuntimeError(f'{stage} failed with exit {code}: {log}')
    try:
        while not (OUT/'assets_complete.json').exists():
            verify(bound)
            state=read(OUT/'assets_status.json') if (OUT/'assets_status.json').exists() else {}
            if state.get('status')=='FAILED':raise RuntimeError('Asset preparation failed: '+str(state))
            status('WAITING_FOR_METRIC_ASSETS',asset_status=state.get('status','PENDING'))
            time.sleep(30)
        assets=read(OUT/'assets_complete.json')
        if assets['status']!='ASSETS_READY' or sha(OUT/'modelmanifest.json')!=assets['manifest_sha256']:
            raise RuntimeError('Metric assets receipt differs')
        python=Path(assets['environment'])
        qualification=OUT/'models_qualification.json'
        if not qualification.exists():
            run('qualify_metrics','qualify_models.py',
                ['--manifest',str(OUT/'modelmanifest.json'),'--out-dir',str(OUT),'--device','cpu'],python,cpu=True)
        q=read(qualification)
        if q.get('status')!='REAL_MODEL_WEIGHTS_QUALIFICATION_PASS' or q.get('modelmanifest_sha256')!=assets['manifest_sha256']:
            raise RuntimeError('Metric qualification failed/stale')
        verify(q['source_bindings'])
        while True:
            verify(bound)
            old=read(PARENT/'supervisor_status.json')
            if old.get('status')=='FAILED':
                status('WAITING_FOR_ORIGINAL_REPAIR',parent_stage=old.get('stage'),parent_error=old.get('error'))
            elif old.get('status')=='COMPLETE':
                parent_receipts()
                active=original_processes()
                if not active:break
                status('WAITING_FOR_ORIGINAL_EXIT',active_pids=active)
            else:status('WAITING_FOR_ORIGINAL_EXPERIMENTS',parent_stage=old.get('stage'))
            time.sleep(30)
        source=OUT/'source_publication.json'
        if not source.exists() or read(source).get('status')!='PUSHED':
            run('publish_source','publish.py',['--phase','source'],python,cpu=True)
        publication=read(source)
        if publication.get('status')!='PUSHED' or publication.get('commit')!=publication.get('remote_commit'):
            raise RuntimeError('Source publication unverified')
        verify(publication['source_bindings'])
        if publication['source_bindings']!=bound:raise RuntimeError('Source publication inventory differs')
        if not (OUT/'scoring_completion.json').exists():
            while gpu_users():
                status('WAITING_FOR_FREE_GPU');verify(bound);time.sleep(30)
            run('replay_and_score','runner.py',python=python)
        completed_scoring(bound,assets['manifest_sha256'])
        result=ROOT/'results/unified_metrics_20261002'
        if not (result/'metrics_analysis_completion.json').exists():
            run('paired_analysis','analysis.py',['--results-dir',str(result)],python,cpu=True)
        analysis=read(result/'metrics_analysis_completion.json')
        if analysis.get('status')!='COMPLETE':raise RuntimeError('Analysis is incomplete')
        for key in ('inputs','outputs'):
            verify(analysis[key])
        if not (OUT/'completion.json').exists():run('publish_results','publish.py',['--phase','results'],python,cpu=True)
        final=read(OUT/'completion.json')
        if final.get('status')!='UNIFIED_METRICS_COMPLETE' or final.get('publication',{}).get('status')!='PUSHED':
            raise RuntimeError('Final metrics publication unverified')
        verify(bound);status('COMPLETE',stopped_after_authorized_scope=True)
    except Exception as error:
        status('FAILED',error=str(error));raise

if __name__=='__main__':main()
