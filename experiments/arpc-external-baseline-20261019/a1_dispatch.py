"""Outer once-only owner wait; run after prepare returned a pinned registration."""
import argparse,hashlib,json,os,pathlib,signal,subprocess,sys,time
from a1_owner import E,O,L,RUN,desc,verify,identity,gone,put
def main():
    a=argparse.ArgumentParser();a.add_argument('--expected-registration',required=True);args=a.parse_args()
    reg=desc(O/'a1_registration.json');assert reg['sha256']==args.expected_registration
    r=json.loads(pathlib.Path(reg['path']).read_text())
    for d in r['source_files']:verify(d)
    D=L/'a1_dispatch_v1';assert not D.exists() and not (RUN/'owner.launch.json').exists()
    D.mkdir();argv=[sys.executable,'-B',str(E/'a1_owner.py'),'run','--expected-registration',args.expected_registration]
    started=time.time();timed=terminated=killed=False
    with (D/'stdout.log').open('xb') as o,(D/'stderr.log').open('xb') as e:
        p=subprocess.Popen(argv,stdout=o,stderr=e,start_new_session=True);ident=identity(p.pid)
        put(D/'launch.json',{'process':ident,'parent_process':identity(os.getpid()),'argv':argv,'registration':reg,'started_unix':started,'timeout_seconds':7800})
        try:rc=p.wait(timeout=7800)
        except subprocess.TimeoutExpired:
            timed=terminated=True
            # Each owner child has its own group. Only verified same-generation
            # recorded children may be terminated, never another user's GPU job.
            for stage in ['tx','rx','audit']:
                q=RUN/f'{stage}.launch.json'
                if q.exists():
                    d=json.loads(q.read_text())['process']
                    if not gone(d):os.killpg(d['pid'],signal.SIGTERM)
            os.killpg(p.pid,signal.SIGTERM)
            try:rc=p.wait(timeout=30)
            except subprocess.TimeoutExpired:killed=True;os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
            for stage in ['tx','rx','audit']:
                q=RUN/f'{stage}.launch.json'
                if q.exists():
                    d=json.loads(q.read_text())['process']
                    if not gone(d):killed=True;os.killpg(d['pid'],signal.SIGKILL)
        put(D/'actual_wait.json',{'process':ident,'returncode':rc,'seconds':time.time()-started,'timed_out':timed,'terminated':terminated,'killed':killed,'metadata_error':None})
    if rc!=0 or timed:put(D/'failure.json',{'returncode':rc,'timed_out':timed,'science_retry_authorized':False});raise SystemExit(1)
    assert gone(ident)
if __name__=='__main__':main()
