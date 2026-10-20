"""One durable supervisor. A receipt alone never establishes completion."""
import argparse
import os
import signal
import subprocess
import sys
import time
from contract import *


def main(expected):
    r=binding(expected)
    d=T/'dispatch';d.mkdir()
    save(d/'supervisor.json',dict(process=process(os.getpid()),
        parent_process=process(os.getppid()),registration_sha256=expected))
    argv=[CPY,'-B',str(E/'owner.py'),'--registration',expected]
    start=time.monotonic();timed=False
    with (d/'stdout.log').open('xb') as stdout,(d/'stderr.log').open('xb') as stderr:
        p=subprocess.Popen(argv,cwd=E,stdout=stdout,stderr=stderr,start_new_session=True)
        ident=process(p.pid)
        save(d/'launch.json',dict(process=ident,parent_process=process(os.getpid()),argv=argv))
        maximum=5*r['stage_wall_seconds']['tx']+3*r['stage_wall_seconds']['phy']+5*r['stage_wall_seconds']['rx']+7200
        try:
            rc=p.wait(timeout=maximum)
        except subprocess.TimeoutExpired:
            timed=True
            # Stop only registered, same-generation owned children.
            for f in (T/'control').glob('*/launch.json'):
                child=read(f)['process'];now=process(child['pid'])
                if now is not None and now['start_ticks']==child['start_ticks']:
                    os.killpg(child['pid'],signal.SIGTERM)
            os.killpg(p.pid,signal.SIGTERM)
            try:rc=p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);rc=p.wait(timeout=30)
    save(d/'actual_wait.json',dict(returncode=rc,seconds=time.monotonic()-start,
        child=ident,timed_out=timed,child_gone=process(p.pid) is None))
    return rc


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True)
    sys.exit(main(ap.parse_args().registration))
