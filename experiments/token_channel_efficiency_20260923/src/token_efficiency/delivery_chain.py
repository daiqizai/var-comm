"""Serial real execution after budget training, with safe thermal resume and receipts.

No GPU work runs while the registered budget supervisor owns the queue. Unknown
failures stop. Completed stages are immutable snapshots; no historical files are
rewritten to bypass a failed check. The final C extension has a separate readiness
gate so unqualified code cannot silently run as a completed research stage.
"""
import fcntl,os,signal,subprocess,sys,time,traceback,shutil
from pathlib import Path
from latent_enhancement.runtime import digest,write_json,verify_snapshot,foreign_gpu_processes
from .common import ROOT,EXP,OUT,CONFIG,read,register,bindings
from .coordinator import proc_identity
from .retired_gate import check_retired,acquire_predecessor_lock
from .thermal_guard import sample,Window
from .budget_lifecycle import next_budget_action

CHAIN=OUT/'delivery_chain_v1'

class Runner:
    def __init__(self):
        self.stopping=False;signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
    def stop(self,*_):self.stopping=True
    def status(self,state,**extra):write_json(CHAIN/'status.json',{'status':state,'pid':os.getpid(),'start_ticks':proc_identity(os.getpid())['start_ticks'],'time':time.time(),**extra})
    def check_stop(self):
        if self.stopping:self.status('STOPPED_BY_REQUEST');raise SystemExit(75)
    def cool(self):
        window=Window()
        while True:
            self.check_stop();r=sample();_,ready=window.add(r)
            busy=foreign_gpu_processes();self.status('WAITING_FOR_COOL_FREE_GPU0',sample=r,cool_samples=window.cool_count,foreign_gpu_work=busy)
            if ready and not busy:return
            time.sleep(10)
    def run(self,key,module_args,completion,*,gpu=True,qualification=False):
        receipt=CHAIN/'stages'/f'{key}.json';completion=Path(completion)
        if receipt.exists():
            rec=read(receipt)
            if rec['command']!=module_args or digest(rec['snapshot'])!=rec['completion_sha256']:raise RuntimeError('completed delivery stage identity')
            return read(rec['snapshot'])
        attempt=0
        while True:
            self.check_stop()
            if gpu:self.cool()
            attempt+=1;run=CHAIN/'attempts'/f'{key}_{time.time_ns()}';run.mkdir(parents=True)
            command=[sys.executable,'-u','-m',*module_args]
            with (run/'console.log').open('w') as log:p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            expected=proc_identity(p.pid)
            if expected is None:raise RuntimeError('stage process disappeared before binding')
            write_json(run/'launch.json',{'command':command,'process':expected,'session_id':os.getsid(p.pid),'time':time.time()})
            window=Window();reason=None
            while p.poll() is None:
                if gpu:
                    try:r=sample();must_pause,_=window.add(r)
                    except Exception as exc:r={'sensor_error':repr(exc)};must_pause=True;reason='sensor_failure'
                else:r=None;must_pause=False
                self.status('RUNNING_STAGE',stage=key,command=module_args,process=expected,hardware=r,attempt=str(run))
                if self.stopping or must_pause:
                    reason=reason or ('requested_stop' if self.stopping else 'thermal')
                    actual=proc_identity(p.pid)
                    if actual and actual['state']!='Z':
                        if actual['start_ticks']!=expected['start_ticks'] or actual['cmdline']!=expected['cmdline']:raise RuntimeError('stage PID identity changed; signal refused')
                        os.kill(p.pid,signal.SIGTERM)
                    write_json(run/'stop_request.json',{'reason':reason,'hardware':r,'process':expected});break
                time.sleep(10 if gpu else 2)
            while p.poll() is None:self.status('WAITING_FOR_SAFE_STAGE_PAUSE',stage=key,process=expected,reason=reason);time.sleep(2)
            rc=p.returncode
            if self.stopping:self.status('STOPPED_BY_REQUEST',stage=key,returncode=rc);raise SystemExit(75)
            if reason=='sensor_failure':raise RuntimeError('thermal sensor failure; no automatic retry')
            if rc==75 or (qualification and reason=='thermal' and rc==-signal.SIGTERM):
                self.status('SAFE_STAGE_PAUSE_COOLDOWN',stage=key,attempt=str(run));continue
            if rc!=0:raise RuntimeError(f'{key} failed rc={rc}; see {run}/console.log')
            done=read(completion)
            if done.get('synthetic',False) is not False:raise RuntimeError('synthetic completion cannot advance delivery')
            snapshot=run/'completion_snapshot.json';shutil.copyfile(completion,snapshot)
            write_json(receipt,{'command':module_args,'completion':str(completion),'snapshot':str(snapshot),'completion_sha256':digest(snapshot),'attempt':str(run),'returncode':rc,'time':time.time()})
            return done

def main():
    CHAIN.mkdir(parents=True,exist_ok=True);lock=(CHAIN/'chain.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runner=Runner()
    local=Path(__file__).parent
    files=[CONFIG,EXP/'scripts/run_delivery_chain.sh',*sorted(local.glob('*.py'))]
    identity={'bindings':bindings(files),'scope':'finish calibration-based budget training; shared qualification; QPSK and16QAM calibration/development/timing; B1/B2 source statistics; original C first matrix then gated qualified C followups','new_holdout':False,'content_selector':False}
    register(CHAIN/'registration.json',identity)
    while True:
        runner.check_stop();verify_snapshot(identity['bindings']);g=read(OUT/'thermal_guard_v2/status.json');state=g['status']
        if state=='LOW_BUDGET_MILESTONE_COMPLETE_LATER_STAGES_PENDING':
            for N in (2048,3060):
                done=read(OUT/f'training/P{N}_seed2026092304/completion.json')
                if done['synthetic'] is not False or done['state']['step']!=20000:raise RuntimeError('initial budget milestone identity')
            p=proc_identity(g['pid'])
            if p and p['start_ticks']==g['start_ticks'] and p['state']!='Z':time.sleep(2);continue
            break
        if state in ('FAILED_GUARD','STOPPED_BY_REQUEST'):raise RuntimeError('budget guard stopped or failed; inspect before delivery')
        runner.status('WAITING_FOR_REGISTERED_BUDGET_TRAINING',budget_guard=g);time.sleep(30)
    release=CHAIN/'C_release.json'
    if not release.exists():check_retired(read(OUT/'scheduling_gate.json'))
    predecessor_lock=acquire_predecessor_lock()
    for N in (2048,3060):
        while True:
            runner.check_stop();action=next_budget_action(N)
            if action['kind']=='final':break
            until=action['until'];runner.run(f'budget_N{N}_until{until}',['token_efficiency.budget_extension','--N',str(N),'--plan',action['plan']],Path(action['plan']).parent/'completion.json')
    verify_snapshot(identity['bindings'])
    runner.run('shared_execution_qualification',['token_efficiency.qualify_execution'],OUT/'execution_qualification_v1/acceptance.json',qualification=True)
    runner.run('QPSK_calibration',['token_efficiency.digital_grid','--role','calibration','--mcs','QPSK'],OUT/'digital_grid_v1/QPSK/calibration/completion.json')
    runner.run('continuous_development',['token_efficiency.continuous_grid'],OUT/'continuous_grid_v1/completion.json')
    runner.run('QPSK_development',['token_efficiency.digital_grid','--role','development','--mcs','QPSK'],OUT/'digital_grid_v1/QPSK/development/completion.json')
    runner.run('16QAM_calibration',['token_efficiency.digital_grid','--role','calibration','--mcs','16QAM'],OUT/'digital_grid_v1/16QAM/calibration/completion.json')
    runner.run('16QAM_development',['token_efficiency.digital_grid','--role','development','--mcs','16QAM'],OUT/'digital_grid_v1/16QAM/development/completion.json')
    ab=ROOT/'results/token_channel_efficiency_20260923/B1_B2_v1/completion.json'
    runner.run('B1_B2_resource_report',['token_efficiency.resource_report'],ab,gpu=False)
    # Archive the exact old failure evidence before the unchanged C controller
    # legitimately writes a new live status. Do not alter the retirement receipt.
    if not release.exists():
        gate=read(OUT/'scheduling_gate.json');retired=check_retired(gate);archive=CHAIN/'pre_C_evidence';archive.mkdir(exist_ok=True);saved={}
        for index,(name,sha) in enumerate(retired['evidence_bindings'].items()):
            p=archive/f'{index:02d}_{Path(name).name}';shutil.copyfile(name,p)
            if digest(p)!=sha:raise RuntimeError('pre-C archive identity')
            saved[name]={'archive':str(p),'sha256':sha}
        write_json(release,{'status':'A_B_COMPLETE_ORIGINAL_C_RELEASED','AB_completion_sha256':digest(ab),'retirement_receipt_sha256':gate['retirement_receipt_sha256'],'archived_evidence':saved,'new_holdout':False,'content_selector':False})
    elif read(release)['AB_completion_sha256']!=digest(ab):raise RuntimeError('C release dependency changed')
    predecessor_lock.close()
    runner.run('C_first_matrix',['token_efficiency.C_first_matrix'],OUT/'C_followups/first_matrix_completion.json')
    # Readiness is supplied by separately CPU-qualified followup implementation.
    # This is a live wait, not a false completion or a request for user approval.
    ready=EXP/'C_followups_ready.json'
    while not ready.exists():runner.check_stop();runner.status('C_FIRST_MATRIX_COMPLETE_WAITING_FOR_QUALIFIED_FOLLOWUPS',missing=str(ready));time.sleep(30)
    readiness=read(ready);verify_snapshot(readiness['bindings'])
    runner.run('C_followups',['token_efficiency.C_followups'],OUT/'C_followups/completion.json')
    runner.status('REGISTERED_GRIDS_EXECUTED_REFERENCE_AUDIT_PUBLICATION_AND_REMOTE_ACCEPTANCE_PENDING',completion=str(OUT/'C_followups/completion.json'))

if __name__=='__main__':
    try:main()
    except Exception:
        write_json(CHAIN/'status.json',{'status':'FAILED_DELIVERY_CHAIN','pid':os.getpid(),'time':time.time(),'traceback':traceback.format_exc()});raise
