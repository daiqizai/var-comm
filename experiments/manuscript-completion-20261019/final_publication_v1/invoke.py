"""Local root invocation of an already reviewed/deployed publisher. No automatic phases."""
from pathlib import Path
import argparse
import json
import subprocess
import time

HOST='DevMachineModel-H100-4C-P1660-01'
R='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM'
CPY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
ENTRY=R+'/experiments/manuscript-completion-20261019/final_publication_v1/publication.py'
HERE=Path(__file__).resolve().parent


def args_for(role,d):
    import shlex
    argv=[CPY,'-B',ENTRY,role,'--descriptor',d['path'],'--sha256',d['sha256'],'--bytes',str(d['bytes'])]
    return ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30',
            HOST,'cd '+shlex.quote(R)+' && '+shlex.join(argv)]


def read_pin(path):
    d=json.loads(Path(path).read_text(encoding='utf8'))
    if set(d)!={'path','sha256','bytes'}:raise RuntimeError('Expected one final descriptor triple')
    return d


def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=('checks','commit'))
    p.add_argument('--descriptor-pin',required=True);a=p.parse_args();d=read_pin(a.descriptor_pin)
    out=HERE/('local_'+a.role+'_transport_v1');out.mkdir(exist_ok=False)
    argv=args_for(a.role,d)
    (out/'launch.json').write_text(json.dumps(dict(argv=argv,descriptor=d,started_unix=time.time()),indent=2))
    start=time.monotonic()
    with (out/'stdout.txt').open('xb')as so,(out/'stderr.txt').open('xb')as se:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=so,stderr=se)
        try: rc=child.wait(timeout=4200 if a.role=='checks' else 600)
        except subprocess.TimeoutExpired:
            # Do not infer remote failure or retry; the remote child may remain alive.
            (out/'uncertain_transport.json').write_text(json.dumps(dict(local_ssh_pid=child.pid,
                remote_result='UNKNOWN_REQUIRES_REMOTE_INSPECTION',automatic_retry=False)))
            raise RuntimeError('SSH wait timed out. Inspect remote publication receipts before any action.')
    (out/'actual_wait.json').write_text(json.dumps(dict(actual_wait=True,local_ssh_pid=child.pid,
        returncode=rc,seconds=time.monotonic()-start,scope='SSH transport only, not remote stage proof'),indent=2))
    if rc:raise RuntimeError('Remote publication phase stopped; preserve attempt and inspect receipts')
    print((out/'stdout.txt').read_text(encoding='utf8'))


if __name__=='__main__':main()
