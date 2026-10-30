import sys,subprocess,time
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='9d55aacfcd740034ce85014ae65f20a33893b59c'
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=R,text=True).strip()
paths=[str(p.relative_to(R))for p in sorted(E.rglob('*'))if p.is_file()and'__pycache__'not in p.parts]
assert all((R/p).stat().st_size<10_000_000 and (R/p).suffix in ['.py','.txt','.json','.md']for p in paths)
q=L/'review_source_paths.nul';q.write_bytes(b''.join(p.encode()+b'\0'for p in paths));subprocess.run(['git','add','--pathspec-from-file='+str(q),'--pathspec-file-nul'],cwd=R,check=True)
actual=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=R).decode().strip('\0').split('\0');assert sorted(actual)==sorted(paths)
save(D/'source_review_stage.json',dict(status='STAGED_FOR_REVIEW_ONLY',HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),source_files=len(paths),paths=paths,commit=False,push=False,user_authorization='Work order prohibits automatic commit/push; staging reviewed changes is reversible and creates reviewable diff'))
os.sched_setaffinity(0,{26,27});out=L/'repository_verify_staged_source';out.mkdir(exist_ok=False);start=time.time();p=subprocess.run([sys.executable,'tools/verify_repository.py'],cwd=R,capture_output=True,text=True,timeout=240)
(out/'stdout.txt').write_text(p.stdout);(out/'stderr.txt').write_text(p.stderr);save(D/'repository_verify_staged_source.json',dict(returncode=p.returncode,seconds=time.time()-start,stdout=pin(out/'stdout.txt'),stderr=pin(out/'stderr.txt'),stage_receipt=pin(D/'source_review_stage.json'),no_commit_or_push=True))
print('VERIFY',p.returncode,p.stdout[-700:],p.stderr[-700:])