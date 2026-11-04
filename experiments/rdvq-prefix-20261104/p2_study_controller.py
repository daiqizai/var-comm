"""Finite first-launch queue, not a retry service. P3/P4 never enter this queue."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time,traceback
from owned_process import proc_identity

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)

def main():
    rp=Path(sys.argv[1]);r=json.loads(rp.read_text());L=Path(r['root']);O=Path(r['output']);O.mkdir(exist_ok=False);S=Path(__file__).resolve().parent
    write(O/'owner.json',dict(**proc_identity(os.getpid()),started=time.time()));write(O/'registration.json',r)
    assert r['reserved_gpu_hours']==190 and r['cap_gpu_hours']==256 and len(r['arms'])==10
    assert json.loads(Path(r['p1_gate']).read_text())['passed'] and json.loads(Path(r['dataset_gate']).read_text())['passed']
    w=json.loads(Path(r['p1_outer_wait']).read_text());assert w['returncode']==0 and w['child_gone']
    for p,h in r['bindings'].items():assert sha(p)==h,p
    pending=list(r['arms']);active={};finished=[];start=time.time();stopping=[False]
    def stop(sig,frame):
        stopping[0]=True
        if not (L/'STOP_P2').exists():(L/'STOP_P2').write_text(json.dumps(dict(signal=sig,requested=time.time())))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while pending or active:
            if time.time()>r['absolute_deadline'] or (L/'STOP_P2').exists():stopping[0]=True
            if time.time()>r['absolute_deadline'] and not (L/'STOP_P2').exists():(L/'STOP_P2').write_text('Registered study deadline; preserve checkpoints')
            for index,item in list(active.items()):
                proc=item['process'];rc=proc.poll()
                if rc is None:continue
                rc=proc.wait();row=dict(name=item['name'],gpu_index=index,returncode=rc,child_gone=not Path('/proc',str(proc.pid)).exists(),started=item['started'],finished=time.time(),request=item['request'])
                finished.append(row);write(O/'actual_arm_waits.json',finished);del active[index]
            if stopping[0] and not active:break
            if not stopping[0]:
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for gpu in r['gpus']:
                    if not pending:break
                    if gpu['index'] in active or gpu['uuid'] in apps:continue
                    eligible=[item for item in pending if item['gpu_index']==gpu['index']]
                    if not eligible:continue
                    item=eligible[0];pending.remove(item);req=json.loads(Path(item['request']).read_text());assert req['gpu']==gpu,'requests must be bound to their preregistered queue slot'
                    assert not Path(req['output_root']).exists(),'Do not repeat an existing arm'
                    p=subprocess.Popen([sys.executable,'-u',str(S/'p2_arm_chain.py'),item['request']],stdin=subprocess.DEVNULL,stdout=(O/(item['name']+'_stdout.log')).open('xb'),stderr=(O/(item['name']+'_stderr.log')).open('xb'))
                    active[gpu['index']]=dict(name=item['name'],process=p,started=time.time(),request=item['request'])
                    write(O/(item['name']+'_child.json'),dict(**proc_identity(p.pid),started=time.time(),request=item['request'],gpu=gpu))
            write(O/'progress.json',dict(active=[dict(name=v['name'],gpu_index=k,pid=v['process'].pid) for k,v in active.items()],finished=finished,pending=[v['name'] for v in pending],stopping=stopping[0],elapsed_seconds=time.time()-start))
            time.sleep(15)
        write(O/('paused.json' if stopping[0] else 'complete.json'),dict(passed=not stopping[0] and len(finished)==10 and all(v['returncode']==0 for v in finished),actual_waits=finished,not_started=[v['name'] for v in pending],scope='P2 only; science conclusions require matched analysis',no_automatic_retry=True))
    except BaseException:
        (O/'failure.txt').write_text(traceback.format_exc());stop(signal.SIGTERM,None)
        # Child arms notice STOP_P2; do not kill them before safe optimizer saves.
        for item in active.values():
            rc=item['process'].wait();finished.append(dict(name=item['name'],returncode=rc,child_gone=True,reason='controller failure stop'))
        write(O/'actual_arm_waits.json',finished);raise
if __name__=='__main__':main()
