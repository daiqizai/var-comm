"""Checkpoint-boundary cooldown supervisor; never change GPU settings or numerics."""
import fcntl,os,signal,subprocess,sys,time
from pathlib import Path
from latent_enhancement.runtime import digest,write_json,verify_snapshot
from .common import ROOT,EXP,OUT,read,register,bindings
from .coordinator import proc_identity
from .retired_gate import check_retired

GUARD=OUT/'thermal_guard_v2'
QUERY='temperature.gpu,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown'

def parse_thermal(text):
    fields=[v.strip() for v in text.strip().split(',')]
    if len(fields)!=3:raise ValueError('complete thermal query required')
    temp=int(fields[0])
    if not 0<=temp<=120 or any(x.lower() not in ('active','not active') for x in fields[1:]):raise ValueError('unknown thermal state')
    return {'temperature':temp,'hardware_thermal_slowdown':fields[1].lower()=='active','software_thermal_slowdown':fields[2].lower()=='active'}

def sample():
    return {**parse_thermal(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu='+QUERY,'--format=csv,noheader,nounits'],text=True,timeout=20)),'time':time.time()}

def hot(row):
    return row['temperature']>=86 or row['hardware_thermal_slowdown'] or row['software_thermal_slowdown']

def cool(row):
    return row['temperature']<=75 and not hot(row)

class Window:
    def __init__(self):self.hot_count=0;self.cool_count=0
    def add(self,row):
        self.hot_count=self.hot_count+1 if hot(row) else 0
        self.cool_count=self.cool_count+1 if cool(row) else 0
        return self.hot_count>=3,self.cool_count>=6

def signal_verified(expected):
    actual=proc_identity(expected['pid'])
    if not actual or actual['state']=='Z':return False
    if actual['start_ticks']!=expected['start_ticks'] or actual['cmdline']!=expected['cmdline']:raise RuntimeError('PID identity changed; never signal recycled process')
    if ' -m token_efficiency.coordinator_v3 ' not in actual['cmdline']:raise RuntimeError('not the owned budget coordinator')
    os.kill(actual['pid'],signal.SIGTERM);return True

def active_owned_queue():
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            if path.stat().st_uid!=os.getuid() or (path/'cwd').resolve()!=ROOT:continue
            argv=(path/'cmdline').read_bytes().split(b'\0')
            if b'-m' in argv and argv[argv.index(b'-m')+1] in (b'token_efficiency.coordinator',b'token_efficiency.coordinator_v3',b'token_efficiency.budget_train',b'token_efficiency.budget_train_microbatch',b'token_efficiency.coordinator_v2'):
                p=proc_identity(int(path.name))
                if p and p['state']!='Z':found.append(p)
        except (OSError,IndexError):continue
    return found

def pause_receipt():
    state=read(OUT/'status.json')
    if state['status'] not in ('PAUSED_SAFE','PAUSED_WAITING'):raise RuntimeError('thermal stop did not reach a verified safe boundary')
    checkpoints={}
    for N in (2048,3060):
        p=OUT/f'training/P{N}_seed2026092304/latest.json'
        if p.exists():
            r=read(p)
            if digest(r['path'])!=r['sha256']:raise RuntimeError('paused checkpoint hash')
            checkpoints[f'P{N}']=r
    if state['status']=='PAUSED_SAFE':
        N=state['command'][state['command'].index('--N')+1]
        if checkpoints[f'P{N}']['reason']!='safe_pause':raise RuntimeError('missing safe checkpoint')
    if active_owned_queue():raise RuntimeError('queue child still alive after parent exit')
    return {'coordinator':state,'checkpoints':checkpoints}

def main():
    GUARD.mkdir(parents=True,exist_ok=True)
    lock=(GUARD/'guard.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity={'bindings':bindings([__file__,EXP/'scripts/run_budgets_microbatch_guard.sh',EXP/'scripts/run_budgets_microbatch.sh',Path(__file__).with_name('coordinator_v3.py')]),'sample_seconds':10,'pause_after_consecutive_hot':3,'cool_temperature_max':75,'resume_after_consecutive_cool':6,'changes_to_hardware':False,'changes_to_training_numerics':False}
    register(GUARD/'registration.json',identity)
    stopping=[False]
    def halt(*_):stopping[0]=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    def status(value,**extra):write_json(GUARD/'status.json',{'status':value,'pid':os.getpid(),'start_ticks':proc_identity(os.getpid())['start_ticks'],'time':time.time(),**extra})
    cycle=0
    while not stopping[0]:
        verify_snapshot(identity['bindings']);check_retired(read(OUT/'scheduling_gate.json'))
        if active_owned_queue():raise RuntimeError('existing queue; refuse duplicate supervisor launch')
        window=Window()
        while not stopping[0]:
            row=sample();_,ready=window.add(row);status('COOLING_BEFORE_RESUME',sample=row,cool_samples=window.cool_count)
            if ready:break
            time.sleep(10)
        if stopping[0]:break
        verify_snapshot(identity['bindings']);check_retired(read(OUT/'scheduling_gate.json'))
        if active_owned_queue():raise RuntimeError('queue appeared during cooldown')
        cycle+=1;run=GUARD/f'cycle_{time.time_ns()}';run.mkdir()
        with (run/'controller.log').open('w') as f:
            process=subprocess.Popen(['bash',str(EXP/'scripts/run_budgets_microbatch.sh')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        # The shell immediately execs Python; capture that final command identity.
        expected=None
        for _ in range(50):
            p=proc_identity(process.pid)
            if p and ' -m token_efficiency.coordinator_v3 ' in p['cmdline']:expected=p;break
            if process.poll() is not None:break
            time.sleep(.1)
        if expected is None:raise RuntimeError('could not bind successor process identity')
        write_json(run/'launch.json',{'controller':expected,'session':os.getsid(process.pid),'registration_sha256':digest(GUARD/'registration.json'),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
        window=Window();reason=None;history=[]
        while process.poll() is None:
            try:
                row=sample();history.append(row);history=history[-6:];must_pause,_=window.add(row)
                if stopping[0]:reason='requested_supervisor_stop'
                elif must_pause:reason='sustained_hardware_or_software_thermal_condition'
                status('MONITORING',controller=expected,sample=row,hot_samples=window.hot_count,cycle=str(run))
            except Exception as exc:
                reason='thermal_sensor_failure';write_json(run/'sensor_failure.json',{'error':repr(exc),'time':time.time()})
            if reason:
                signal_verified(expected);write_json(run/'stop_requested.json',{'reason':reason,'controller':expected,'samples':history,'time':time.time()});break
            time.sleep(10)
        # Wait for the trainer's own safe-checkpoint handler; never SIGKILL.
        while process.poll() is None:
            status('WAITING_FOR_SAFE_CHECKPOINT',controller=expected,reason=reason);time.sleep(2)
        state=read(OUT/'status.json')
        if state.get('pid')!=expected['pid']:raise RuntimeError('successor status identity')
        if state['status'] in ('PAUSED_SAFE','PAUSED_WAITING'):
            receipt=pause_receipt();write_json(run/'safe_pause.json',receipt)
            if reason=='thermal_sensor_failure':raise RuntimeError('sensor failed; manual inspection required')
            if stopping[0]:break
            # Includes the unchanged trainer's own thermal guard exit.
            status('SAFE_PAUSE_COOLDOWN',receipt=str(run/'safe_pause.json'))
            continue
        if state['status']=='A_AND_LOW_BUDGET_20K_COMPLETE_B1_EVALUATION_PENDING':
            status('LOW_BUDGET_MILESTONE_COMPLETE_LATER_STAGES_PENDING',coordinator=state);return
        raise RuntimeError('non-thermal queue failure; no automatic retry: '+str(state))
    status('STOPPED_BY_REQUEST')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_json(GUARD/'status.json',{'status':'FAILED_GUARD','pid':os.getpid(),'time':time.time(),'error':repr(exc)});raise
