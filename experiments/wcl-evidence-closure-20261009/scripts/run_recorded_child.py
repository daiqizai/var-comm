"""Wait for one finite authorized child and record its actual exit, once."""
import argparse, fcntl, hashlib, json, os
from pathlib import Path
import subprocess, sys, time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record-dir',required=True)
    parser.add_argument('--cwd',required=True);parser.add_argument('argv',nargs=argparse.REMAINDER)
    args=parser.parse_args();argv=args.argv[1:] if args.argv[:1]==['--'] else args.argv
    assert argv,'Child command required';out=Path(args.record_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/'owner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (out/'launch.json').exists(),'Existing attempt must be inspected, never silently restarted'
        with (out/'child.log').open('xb') as log:
            started=time.time();child=subprocess.Popen(argv,cwd=args.cwd,stdout=log,stderr=subprocess.STDOUT)
            (out/'launch.json').write_text(json.dumps(dict(argv=argv,cwd=args.cwd,owner_pid=os.getpid(),child_pid=child.pid,
                started_unix=started,owner_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
            code=child.wait()
        result=dict(exit_code=code,finished_unix=time.time(),elapsed_seconds=time.time()-started,actual_child_waited=True)
        (out/'exit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        return code

if __name__=='__main__':raise SystemExit(main())
