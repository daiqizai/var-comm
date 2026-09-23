"""Durable handoff after verified predecessor retirement; preserve all completed stage receipts.

This milestone deliberately stops before unimplemented digital/evaluation stages.
The monitor must implement/qualify B1 evaluation and B2 before relaunching C.
The original controller lock is held while this successor owns the budget queue.
"""
import fcntl,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from latent_enhancement.runtime import digest,write_json,verify_snapshot,foreign_gpu_processes
from .common import ROOT,EXP,OUT,CONFIG,read,register,bindings

GATE=OUT/'scheduling_gate.json'

from .retired_gate import check_retired,acquire_predecessor_lock

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    predecessor_lock=acquire_predecessor_lock()
    gate=read(GATE);check_retired(gate)
    if not (OUT/'launch_identity_v3.json').exists():
        write_json(OUT/'launch_identity_v3.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pid':os.getpid(),'time':time.time(),'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'scope':'source A and new budget first20k; not whole supplement completion'})
    plan=[
      (['token_efficiency.source','--role','calibration','--preflight'],OUT/'source_preflight/calibration/completion.json'),
      (['token_efficiency.source','--role','calibration'],OUT/'source/calibration/completion.json'),
      (['token_efficiency.source','--role','development'],OUT/'source/development/completion.json'),
      (['token_efficiency.budget_train','--N','2048','--qualification-only'],OUT/'training/P2048_seed2026092304/qualification.json'),
      (['token_efficiency.budget_train','--N','3060','--qualification-only'],OUT/'training/P3060_seed2026092304/qualification.json'),
      (['token_efficiency.budget_train_microbatch','--N','2048'],OUT/'training/P2048_seed2026092304/completion.json'),
      (['token_efficiency.budget_train','--N','3060'],OUT/'training/P3060_seed2026092304/completion.json')]
    identity={'commands':[c for c,_ in plan],'bindings':bindings([__file__,EXP/'scripts/run_budgets_microbatch.sh',Path(__file__).with_name('retired_gate.py'),Path(__file__).with_name('coordinator.py'),CONFIG]),'retired_predecessor':{'receipt':gate['retirement_receipt'],'sha256':gate['retirement_receipt_sha256']},'cache_completion':gate['required_cache_completion']}
    register(OUT/'coordinator_identity_v3.json',identity)
    child=[None];stop=[False]
    def halt(*_):
        stop[0]=True
        if child[0] and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    def status(value,**extra):write_json(OUT/'status.json',{'status':value,'pid':os.getpid(),'time':time.time(),'retired_short_prefix_controller':gate['held_controller_pid'],'new_holdout':False,'content_selector':False,**extra})
    cache_done=Path(gate['required_cache_completion'])
    if not cache_done.exists():raise RuntimeError('retired predecessor requires completed cache')
    done=read(cache_done)
    if done['status']!='REAL_PHY_AND_LATENT_CACHE_COMPLETE' or done['sources']!=20000:raise RuntimeError('unexpected completed cache')
    if done['registration_sha256']!=digest(cache_done.parent/'registration.json'):raise RuntimeError('cache receipt binding')
    verify_snapshot(read(cache_done.parent/'registration.json')['bindings'])
    for name,sha in done['hashes'].items():
        if digest(cache_done.parent/name)!=sha:raise RuntimeError('cache hash mismatch')
    gate['status']='RETIRED_PREDECESSOR_A_B_PRIORITY_GATE';gate['cache_completion_sha256']=digest(cache_done);write_json(GATE,gate)
    receipts=OUT/'stages';receipts.mkdir(exist_ok=True)
    for index,(command,completion) in enumerate(plan):
        verify_snapshot(identity['bindings']);check_retired(gate);receipt=receipts/f'{index:02d}.json'
        if receipt.exists():
            saved=read(receipt)
            if saved['command']!=command or saved['completion_sha256']!=digest(completion):raise RuntimeError('completed stage identity')
            continue
        while foreign_gpu_processes():
            if stop[0]:status('PAUSED_WAITING',stage=index);return
            status('WAITING_FOR_AUTHORIZED_GPU0',stage=index);time.sleep(30)
        if stop[0]:status('PAUSED_WAITING',stage=index);return
        log=OUT/'logs'/f'{index:02d}_{command[0].rsplit(".",1)[-1]}_{int(time.time())}.log';log.parent.mkdir(exist_ok=True)
        with log.open('w') as handle:
            child[0]=subprocess.Popen([sys.executable,'-u','-m',*command],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            status('RUNNING',stage=index,command=command,child_pid=child[0].pid,log=str(log));rc=child[0].wait()
        if rc!=0:
            status('PAUSED_SAFE' if rc==75 else 'FAILED',stage=index,command=command,returncode=rc,log=str(log));return
        rec=read(completion)
        if rec.get('synthetic') is not False:raise RuntimeError('real execution receipt required')
        write_json(receipt,{'command':command,'completion':str(completion),'completion_sha256':digest(completion),'log':str(log),'returncode':rc})
    status('A_AND_LOW_BUDGET_20K_COMPLETE_B1_EVALUATION_PENDING',pending=['calibration-based extension decision','B1 selected development/online timing and QPSK grid','B2 actual16QAM grid','resume registered short-prefix C after A/B','paired statistics and resource figures'],C_gate='RETIRED_RELAUNCH_ONLY_AFTER_A_B_DELIVERY',new_quality_results_published=False)

if __name__=='__main__':
    try:main()
    except Exception:
        write_json(OUT/'status.json',{'status':'FAILED_COORDINATOR','pid':os.getpid(),'time':time.time(),'traceback':traceback.format_exc(),'action':'inspect evidence; never duplicate old held queue'});raise
