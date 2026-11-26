"""Advance only new formal and timing roles after validation is frozen; preserve failures."""
import json
from pathlib import Path
import subprocess
import sys
import time

def wait_job(path):
    until=time.monotonic()+3670.
    while not (path/"wait.json").exists():
        if time.monotonic()>until:
            raise TimeoutError(str(path))
        time.sleep(3)
    return json.loads((path/"wait.json").read_text())["actual_wait"]

def main():
    scratch,study,parent,py=map(Path,sys.argv[1:])
    def wait(stage):
        return wait_job(scratch/("job_"+stage))
    assert wait("development")==0 and (study/"development/frozen.json").is_file()
    assert wait("mechanism")==0 and (study/"mechanism/closed.json").is_file()
    status={}
    for name in ("formal","timing"):
        job=scratch/("job_"+name)
        if not job.exists():
            subprocess.run([sys.executable,"-B",str(study/"launch_worker.py"),name,
                            *map(str,(scratch,study,parent,py))],check=True)
        status[name]=wait(name)
    with (scratch/"continuation_exit.json").open("x") as f:
        json.dump(status,f)
    return 0 if all(v==0 for v in status.values()) else 1

if __name__=="__main__":
    sys.exit(main())
