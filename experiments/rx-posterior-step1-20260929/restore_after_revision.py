"""One-shot original-queue restoration after the revised RX hard gate stops."""
import os,sys,time,fcntl,subprocess,shutil,traceback
from pathlib import Path
import pause_lease as p
ROOT=p.ROOT;RX=p.OUT;OLD=p.OLD;HERE=RX/'revision_v2'/'queue_restoration'
def lock(path):
    h=Path(path).open('a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def bindings(reg):
    for path,h in reg['bindings'].items():assert p.sha(path)==h,('binding mismatch',path)
def archive(folder,dest):
    dest.mkdir(parents=True)
    out={}
    for leaf in ['status.json','worker_status.json','registration.json','worker_registration.json','launch_receipt.json','detached_console.log']:
        s=folder/leaf
        if s.exists():
            shutil.copyfile(s,dest/leaf);out[leaf]=p.sha(dest/leaf)
    return out
def launch(name,script,module,expected):
    folder=OLD/name
    leaf='status.json' if name=='delivery_chain_v1' else 'worker_status.json'
    regleaf='registration.json' if name=='delivery_chain_v1' else 'worker_registration.json'
    assert not p.same(p.identity(p.read(folder/'launch_receipt.json')['process']['pid']),p.read(folder/'launch_receipt.json')['process'])
    h=lock(folder/('chain.lock' if name=='delivery_chain_v1' else 'worker.lock'));h.close()
    p.write(HERE/'launch_intents'/f'{name}.json',dict(time=time.time(),script=script))
    env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
    with (folder/'detached_console.log').open('a') as log:
        log.write('\nRESTORE after user RX probe revised hard-gate stop; original source unchanged\n');log.flush()
        child=subprocess.Popen(['bash',str(ROOT/script)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    deadline=time.time()+90
    while True:
        actual=p.identity(child.pid);assert child.poll() is None,'restored process exited'
        status=p.read(folder/leaf)
        if (actual and ('-m '+module+' ') in actual['cmdline'] and status.get('pid')==child.pid
                and str(status.get('start_ticks'))==actual['start_ticks']
                and status['status'] in expected and time.time()-status['time']<40):break
        if time.time()>deadline:raise RuntimeError('no fresh valid restored identity: '+name)
        time.sleep(2)
    rec=dict(status='RESTORED_AFTER_RX_REVISED_HARD_GATE_STOP',process=actual,pid=actual['pid'],
             start_ticks=actual['start_ticks'],cmdline=actual['cmdline'],session_id=os.getsid(child.pid),
             registration_sha256=p.sha(folder/regleaf),previous_receipts=str(HERE/'previous'/name),time=time.time())
    p.write(folder/'launch_receipt.json',rec);p.write(HERE/(name+'_restored.json'),rec)
    return rec
def main():
    HERE.mkdir(parents=True,exist_ok=True);own=lock(HERE/'restore.lock')
    assert not (HERE/'registration.json').exists(),'inspect any prior restoration before retry'
    v=RX/'revision_v2';done=p.read(v/'calibration_completion.json')
    assert done['status']=='REVISED_HARD_GATE_FAILED_NO_DEVELOPMENT' and not done['passed'] and not done['development_accessed']
    assert not (v/'failure.json').exists() and not (v/'supervisor_failure.json').exists()
    sr=p.read(v/'launch_receipt.json')['process'];assert not p.same(p.identity(sr['pid']),sr)
    h=lock(v/'probe.lock');h.close()
    paused=p.read(RX/'pause_complete.json');assert p.sha(paused['latest']['path'])==paused['latest']['sha256']
    assert p.sha(OLD/'delivery_chain_v1/registration.json')==paused['original_registration_sha256']
    bindings(p.read(v/'registration.json'))
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    pause_time=p.read(RX/'pause_request.json')['time']
    specifications=[
        ('author_native_timing_v1','FAILED_AUTHOR_PARITY_TIMING_WORKER',"RuntimeError('upstream stopped/failed; preserve state for repair')"),
        ('n3060_native_timing_v1','FAILED_N3060_TIMING_WORKER',"RuntimeError('upstream failure or requested stop requires review')")]
    for name,failure,error in specifications:
        status=p.read(OLD/name/'worker_status.json');old=p.read(OLD/name/'launch_receipt.json')['process']
        assert status['status']==failure and status['error']==error and status['time']>=pause_time
        assert not p.same(p.identity(old['pid']),old)
        h=lock(OLD/name/'worker.lock');h.close()
    live=[]
    for name in ['reference_common_metrics_v1','n4084_common_replay_v1']:
        expected=p.read(OLD/name/'launch_receipt.json')['process'];assert p.same(p.identity(expected['pid']),expected)
        status=p.read(OLD/name/'worker_status.json');assert time.time()-status['time']<60 and status['status'].startswith('WAITING')
        live.append(dict(name=name,process=expected))
    names=['delivery_chain_v1','reference_common_metrics_v1','n4084_common_replay_v1','author_native_timing_v1','n3060_native_timing_v1']
    saved={}
    for name in names:
        folder=OLD/name;reg=folder/('registration.json' if name=='delivery_chain_v1' else 'worker_registration.json')
        bindings(p.read(reg));saved[name]=archive(folder,HERE/'previous'/name)
    p.write(HERE/'registration.json',dict(time=time.time(),source_sha256=p.sha(__file__),paused=paused,
            revised_gate_sha256=p.sha(v/'calibration_completion.json'),previous=saved,kept_waiters=live))
    lease=p.read(RX/'lease_status.json')['process'];assert p.same(p.identity(lease['pid']),lease)
    assert not (RX/'release_lease.json').exists()
    p.write(RX/'release_lease.json',dict(reason='Revised probe completed at hard-gate failure; restore original queue per execution sheet',time=time.time()))
    deadline=time.time()+35
    while p.same(p.identity(lease['pid']),lease):
        assert time.time()<deadline,'lease did not release';time.sleep(1)
    assert p.read(RX/'lease_released.json')['status']=='LOCKS_RELEASED_NO_IMPLICIT_RESTART'
    handles=[lock(path) for path in [OLD/'delivery_chain_v1/chain.lock',OLD/'C_followups/followups.lock',ROOT/'outputs/SHORT-PREFIX-20260923/controller.lock']]
    for h in handles:h.close()
    main=launch('delivery_chain_v1','experiments/token_channel_efficiency_20260923/scripts/run_delivery_chain.sh',
                'token_efficiency.delivery_chain',{'WAITING_FOR_COOL_FREE_GPU0','RUNNING_STAGE'})
    restored=[]
    for name,script,module,state in [
        ('author_native_timing_v1','tools/run_author_reference_timing.sh','tools.replay_author_reference_timing','WAITING_FOR_N4084_AND_ALL_PREDECESSORS'),
        ('n3060_native_timing_v1','tools/run_n3060_reference_timing.sh','tools.replay_n3060_reference_timing','WAITING_FOR_AUTHOR_AND_ALL_PREDECESSORS')]:
        assert p.same(p.identity(main['process']['pid']),main['process'])
        restored.append(launch(name,script,module,{state}))
    p.write(HERE/'completion.json',dict(status='ORIGINAL_MAIN_AND_TWO_PROTECTED_WAITERS_RESTORED_TWO_EXISTING_WAITERS_RETAINED',
            main=main,restored=restored,retained=live,time=time.time(),probe_development_accessed=False,
            safe_checkpoint=paused['latest'],original_model_and_sources_unchanged=True))
if __name__=='__main__':
    try:main()
    except Exception:
        p.write(HERE/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
