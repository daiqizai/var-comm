"""Once-only persistent outer wait for the resource-revised clean owner."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
ENTRY=Path(__file__).with_name('r0_clean_resource_v2.py')
REG=ROOT/'results/layered_mismatch_raw_20261013/r0_clean_resource_revision_v2/registration.json'
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013/clean_resource_dispatch_v2')

def put(p,v):
    with p.open('x',encoding='utf8') as f:
        json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def run(expected):
    spec=importlib.util.spec_from_file_location('_resource_revised_clean',ENTRY)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    r=m.registration(REG,expected)
    m.require(r['code_bindings'][str(Path(__file__))]==m.sha(__file__),'Dispatcher binding changed')
    m.retirement_gate(r,prelaunch=True)
    import fcntl
    TMP.mkdir(parents=True,exist_ok=True)
    with (TMP/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m.require(not (TMP/'launch.json').exists(),'Existing dispatch; no implicit repeat')
        argv=[str(m.PYTHON),'-B','-u',str(ENTRY),'run','--registration',str(REG),'--expected-registration',expected,'--reviewed']
        put(TMP/'launch.json',dict(process=m.process_identity(os.getpid()),entry=m.pin(ENTRY),registration=m.pin(REG),
            command=argv,created_unix=time.time(),automatic_retry=False,scientific_completion=False))
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1')
        child=None;start=time.monotonic();timeout=False
        try:
            with (TMP/'owner.stdout.log').open('xb') as out,(TMP/'owner.stderr.log').open('xb') as err:
                child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                identity=m.process_identity(child.pid)
                put(TMP/'owner_identity.json',dict(process=identity,pid=child.pid,created_unix=time.time()))
                try:code=child.wait(timeout=r['max_seconds']+120)
                except subprocess.TimeoutExpired:
                    timeout=True
                    m.require(os.getpgid(child.pid)==child.pid,'Owned process group identity changed')
                    os.killpg(child.pid,signal.SIGTERM)
                    try:code=child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            put(TMP/'actual_wait.json',dict(actual_wait=True,returncode=code,timed_out=timeout,
                elapsed_seconds=time.monotonic()-start,identity_at_launch=identity,automatic_retry=False))
            return code
        except BaseException:
            if child is not None and child.poll() is None:
                m.require(os.getpgid(child.pid)==child.pid,'Owned interruption group changed')
                os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);code=child.wait()
                put(TMP/'interrupted_actual_wait.json',dict(actual_wait=True,returncode=code,automatic_retry=False))
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expected-registration',required=True)
    a=p.parse_args();sys.exit(run(a.expected_registration))
