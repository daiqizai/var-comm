"""Publish the explicit reviewed UEP release; preserve unrelated working files."""
from pathlib import Path
import argparse
import subprocess
from uep_common import require,read,sha,seal

def call(root,args):
    subprocess.run(args,cwd=root,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--message',required=True);a=p.parse_args()
    manifest=read(a.manifest);paths=manifest['publish_paths']
    require(paths and len(set(paths))==len(paths),'Explicit unique publication paths required')
    root=a.root.resolve()
    for name in paths:
        f=(root/name).resolve();require(root in f.parents and f.is_file() and f.stat().st_size<10_000_000,'Unsafe/missing/oversized publication file')
        require(sha(f)==manifest['file_sha256'][name],'Publication source changed: '+name)
    require(not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).strip(),'Unexpected staged work; preserve it')
    call(root,['git','fetch','origin']);call(root,['git','merge-base','--is-ancestor','origin/main','HEAD'])
    call(root,['git','add','--',*paths])
    call(root,['python3','-B','tools/update_repository_manifest.py']);call(root,['git','add','--','release_manifest.json'])
    call(root,['python3','-B','tools/verify_repository.py']);call(root,['python3','-B','tools/run_cpu_checks.py'])
    call(root,['git','commit','-m',a.message]);call(root,['git','push','origin','HEAD:main'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=root,text=True).split()[0]
    require(commit==remote,'Remote publication SHA not verified')
    seal(a.out/'publication_completion.json',dict(status='PUSHED_AND_STOPPED',commit=commit,remote_commit=remote,
        checks='PASS',publication_manifest_sha256=sha(a.manifest),training_updates=0))

if __name__=='__main__':main()
