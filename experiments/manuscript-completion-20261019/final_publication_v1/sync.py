"""Post-push exact light-file sync, Windows long paths, no arrays or full backup."""
from pathlib import Path,PurePosixPath
import argparse
import base64
import hashlib
import json
import os
import stat
import subprocess
import threading
import time

W=Path('C:/Users/11946/Documents/ChatGPT/comm')
HERE=Path(__file__).resolve().parent
HOST='DevMachineModel-H100-4C-P1660-01'
R='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM'
A='/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/manuscript_completion_20261019/final_publication_v1'
CPY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'


def digest(b):return hashlib.sha256(b).hexdigest()


def longpath(p):
    q=str(p.absolute())
    return '\\\\?\\'+q if os.name=='nt'and not q.startswith('\\\\?\\')else q


def safe_target(relative):
    q=PurePosixPath(relative)
    if q.is_absolute()or'..'in q.parts or q.as_posix()!=relative or ':'in relative or '\\'in relative:
        raise RuntimeError('Unsafe repository path')
    p=W.joinpath(*q.parts)
    for x in(p,*p.parents):
        try:s=os.lstat(longpath(x))
        except FileNotFoundError:continue
        if stat.S_ISLNK(s.st_mode)or getattr(s,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400):
            raise RuntimeError('Symlink/junction/reparse point in local sync path')
    if os.path.commonpath([str(W.absolute()),str(p.absolute())])!=str(W.absolute()):raise RuntimeError('Outside W')
    return p


def pin_bytes(b):return dict(bytes=len(b),sha256=digest(b))


def read_bytes(p):
    with open(longpath(p),'rb')as f:return f.read()


def write_new(p,b):
    os.makedirs(longpath(p.parent),exist_ok=True)
    with open(longpath(p),'xb')as f:f.write(b);f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--push-receipt',required=True)
    ap.add_argument('--expected-local-before',required=True)
    ap.add_argument('--attempt',required=True,help='New local receipt directory; never reused')
    a=ap.parse_args();receipt=json.loads(Path(a.push_receipt).read_text(encoding='utf8'))
    before=json.loads(Path(a.expected_local_before).read_text(encoding='utf8'))
    if receipt['normal_push']is not True or receipt['head']!=receipt['verified_remote']:
        raise RuntimeError('Normal push not verified')
    # before is an exact relative-path -> {bytes,sha256} allowlist, not a blanket overwrite switch.
    if not isinstance(before,dict):raise RuntimeError('Expected exact local-before mapping')
    for name,d in before.items():
        safe_target(name)
        if set(d)!={'bytes','sha256'}:raise RuntimeError('Invalid explicit local-before pin')
    control=Path(a.attempt).absolute()
    if control.parent!=HERE or not control.name.startswith('sync_attempt_'):raise RuntimeError('Use a new helper-local sync_attempt_*')
    control.mkdir(exist_ok=False)
    source=r'''
from pathlib import Path
import base64,hashlib,json,subprocess
R=Path(R_TEXT);A=Path(A_TEXT)
def h(b):return hashlib.sha256(b).hexdigest()
def git(*x):return subprocess.check_output(['git',*x],cwd=R)
assert git('rev-parse','HEAD').decode().strip()==EXPECTED['head']
assert not git('status','--porcelain','--untracked-files=no').strip()
actual=json.loads((A/'push_verified.json').read_text())
assert actual==EXPECTED
manifest=(A/'staged_files.json').read_bytes();m=EXPECTED['staged_files']
assert m['path']==str(A/'staged_files.json')and len(manifest)==m['bytes']and h(manifest)==m['sha256']
rows=json.loads(manifest)
print(json.dumps({'kind':'manifest','rows':rows,'receipt':actual}),flush=True)
for row in rows:
 assert type(row['bytes'])is int and 0<=row['bytes']<=10000000
 b=git('show',EXPECTED['head']+':'+row['path'])
 assert len(b)==row['bytes']and h(b)==row['sha256']
 print(json.dumps({'kind':'file','row':row,'base64':base64.b64encode(b).decode()}),flush=True)
assert git('rev-parse','HEAD').decode().strip()==EXPECTED['head']
print(json.dumps({'kind':'complete','count':len(rows),'head':EXPECTED['head']}),flush=True)
'''
    script='R_TEXT='+repr(R)+'\nA_TEXT='+repr(A)+'\nEXPECTED='+repr(receipt)+'\n'+source
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',
          '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3',HOST,CPY,'-B','-']
    (control/'launch.json').write_text(json.dumps(dict(argv=argv,head=receipt['head'],started_unix=time.time(),
        expected_local_before=before,scientific_calls=0,backup=False),indent=2))
    log=(control/'stderr.txt').open('xb')
    child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf8')
    expired=[]
    def stop_own_readonly_transport():
        expired.append(True)
        if child.poll()is None:child.terminate()
    timer=threading.Timer(3600,stop_own_readonly_transport);timer.daemon=True;timer.start()
    child.stdin.write(script);child.stdin.close();records=[];rows=None;done=False
    try:
        for line in child.stdout:
            item=json.loads(line)
            if item['kind']=='manifest':
                if rows is not None or item['receipt']!=receipt:raise RuntimeError('Duplicate/mismatched manifest')
                rows=item['rows']
                if len(rows)!=len({r['path']for r in rows}):raise RuntimeError('Duplicate manifest path')
                # Complete preflight before accepting the first replacement.
                for row in rows:
                    p=safe_target(row['path'])
                    if os.path.exists(longpath(p)):
                        current=pin_bytes(read_bytes(p));wanted={k:row[k]for k in('bytes','sha256')}
                        if current!=wanted and before.get(row['path'])!=current:
                            raise RuntimeError('Unknown local difference; no overwrite: '+row['path'])
                write_new(control/'remote_manifest.json',json.dumps(rows,indent=2).encode())
            elif item['kind']=='file':
                if rows is None:raise RuntimeError('No final exact manifest')
                row=item['row']
                if row!=rows[len(records)]:raise RuntimeError('Order/scope differs')
                b=base64.b64decode(item['base64'],validate=True)
                if pin_bytes(b)!={k:row[k]for k in('bytes','sha256')}:raise RuntimeError('Committed blob SHA differs')
                p=safe_target(row['path']);reused=False
                if os.path.exists(longpath(p)):
                    old=read_bytes(p);same=pin_bytes(old)==pin_bytes(b)
                    if same:reused=True
                    else:
                        if before.get(row['path'])!=pin_bytes(old):raise RuntimeError('Local file changed since preflight')
                        write_new(control/'replaced_local'/Path(*PurePosixPath(row['path']).parts),old)
                        tmp=p.with_name(p.name+'.publication-sync-tmp')
                        write_new(tmp,b);safe_target(row['path']);os.replace(longpath(tmp),longpath(p))
                else:write_new(p,b)
                if pin_bytes(read_bytes(p))!=pin_bytes(b):raise RuntimeError('Local verification failed')
                records.append(dict(row,reused_existing_same_sha=reused))
            elif item['kind']=='complete':
                if rows is None or item['count']!=len(records)or item['head']!=receipt['head']:
                    raise RuntimeError('Incomplete final sync')
                done=True
            else:raise RuntimeError('Unexpected sync stream')
        rc=child.wait(timeout=120)
        write_new(control/'actual_wait.json',json.dumps(dict(actual_wait=True,local_ssh_pid=child.pid,
            returncode=rc,scope='exact committed blob transport only'),indent=2).encode())
        if rc or not done or expired:raise RuntimeError('Sync transport incomplete; retain verified files; no automatic repeat')
        write_new(control/'local_verification.json',json.dumps(dict(status='PASS',head=receipt['head'],
            files=records,large_backup=False,scientific_calls=0),indent=2).encode())
        print(json.dumps(dict(status='PASS',files=len(records),receipt=str(control/'local_verification.json'))))
    except BaseException:
        write_new(control/'failure.json',json.dumps(dict(status='STOPPED',verified_files=records,
            message='Preserve this attempt. Inspect exact local/remote state before explicit continuation.'),indent=2).encode())
        raise
    finally:
        timer.cancel()
        if child.poll()is None:child.terminate();child.wait(timeout=30)
        log.close()


if __name__=='__main__':main()
