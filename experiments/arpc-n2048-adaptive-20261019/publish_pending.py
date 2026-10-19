"""Normal exact-path publication of the package-required pending delivery."""
from shared import *
import fcntl,re
A=L/'publication_pending_v1'
PARENT='980d74c4fb08f279c31e7a559afb0cf15cc92057'

def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True).strip()
def check(name,argv):
 c=A/name;c.mkdir();wait_child(c,argv,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),1800);return pin(c/'actual_wait.json')

def main():
 os.chdir(R);lock=(L/'execution.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert git('rev-parse','HEAD')==PARENT and git('branch','--show-current')=='main'
 assert not git('status','--porcelain','--untracked-files=no')and not git('diff','--cached','--name-only')
 assert read(O/'final_delivery_manifest.json')['status']=='PACKAGE_REQUIRED_PENDING_RESULT_DELIVERY_COMPLETE'
 assert read(O/'figures_calibration_v2/actual_visual_review.json')['status']=='PASS'
 counts=read(O/'B0_B1_delivery/actual_counts.json');assert counts['B2_objects']==counts['B3_objects']==counts['bootstrap']==0
 A.mkdir();put(A/'supervisor.json',dict(process=identity(os.getpid()),parent_process=identity(os.getppid()),source=pin(__file__)))
 c=L/'controls/close_evidence';w=read(c/'actual_wait.json');q=read(c/'launch.json');assert w['returncode']==0 and gone(w['process'])and gone(q['parent_process'])
 dest=O/'execution_receipts/close_evidence';dest.mkdir();copies=[]
 for p in sorted(c.iterdir()):
  assert p.is_file()and p.stat().st_size<1000000
  b=p.read_bytes();d=dest/p.name;d.open('xb').write(b);assert sha(p)==sha(d);copies.append(dict(source=pin(p),copy=pin(d)))
 put(O/'metadata_copy_exit.json',dict(status='PASS',copies=copies))
 note='''## 2026-10-19 N2048 adaptive cal100 delivery: candidate coverage pending

The separate low-SNR adaptive package has completed B0/B1 and all cal100 clean
source gaps. Both equal-protection fragments pay their68-symbol header and
13-bit length,16-bit CRC and96-bit reassembly record within the2048-use budget.
35,200 existing source objects were verified; no TX/tokenizer rerun.24 physical
qualification objects used48 subframes and96 decoder calls without repetition.
The original RX final exit1 remains preserved; a metadata-only closure audited
already saved decodes after an optional Python bytecode cache read was denied.

Cal100 now has3200 clean endpoints:1407 verified existing image/score states,
1793 new NeST native recoveries and1793 new DINO values, reusing100 original
source features. All384 complete policies use actual per-image length fallback.
At10dB five NeST policies remain plausible under historical packet-proxy bounds:
QPSK2/3,3/4,5/6 and16-QAM1/2,2/3. They cannot fit the two-policy condition cap.
Per the package uncertainty rule B2/B3 have NOT run; no new test score, paired
bootstrap or low-SNR wireless improvement is claimed. The other three conditions
are covered within two proposals. Using one plausible ARPC policy perSNR would
need nine calibration cells:27000logical/54000subframes/108000decoder attempts,
still beyond the original global and per-condition scope. This is a costed
option, not an authorized expansion. Old N1024/fixed-full N2048 remain unchanged.

See results/arpc_n2048_adaptive_20261019_v1/README.md, B0_B1_delivery/REPORT_zh.md,
all source-level diagnostic tables and the actually opened vector figures_v2
(directory figures_calibration_v2). Scientific continuation waits on scope;
no training, power extension, extra methods or heavy backup is launched.

'''
 paths=[];before=[]
 for root in (E,O):
  for p in sorted(root.rglob('*')):
   if p.is_file():
    assert not p.is_symlink()and p.stat().st_size<10000000
    assert p.suffix in ('.py','.md','.json','.csv','.png','.pdf','.svg','.txt')or p.name in ('stdout','stderr'),p
    paths.append(p.relative_to(R).as_posix())
 for name in ('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md'):
  p=R/name;raw=p.read_bytes();assert b'N2048 adaptive cal100 delivery: candidate coverage pending'not in raw
  before.append(pin(p));p.write_bytes(note.encode()+raw);paths.append(name)
 put(A/'before_docs.json',before)
 subprocess.run(['git','add','--sparse','--',*paths],cwd=R,check=True)
 check('manifest_update',[str(PY),'-B','tools/update_repository_manifest.py'])
 subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True);paths.append('release_manifest.json')
 assert set(git('diff','--cached','--name-only').splitlines())==set(paths);put(A/'exact_paths.json',paths)
 checks={};checks['diff_check']=check('diff_check',['git','diff','--cached','--check'])
 checks['verify_repository']=check('verify_repository',[str(PY),'-B','tools/verify_repository.py'])
 checks['run_cpu_checks']=check('run_cpu_checks',[str(PY),'-B','tools/run_cpu_checks.py'])
 put(A/'checks.json',checks);staged=[]
 for name in sorted(paths):
  b=subprocess.check_output(['git','show',':'+name],cwd=R);assert b==(R/name).read_bytes();staged.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 put(A/'staged_files.json',staged)
 record=dict(parent=PARENT,checks_sha256=sha(A/'checks.json'),staged_files_sha256=sha(A/'staged_files.json'))
 put(A/'checks_complete.json',dict(status='PASS',tree=git('write-tree'),**record))
 (A/'commit_message.txt').write_text('Deliver N2048 adaptive cal100 evidence and unresolved candidate scope\n\nComplete finite source/PHY qualification and exact cal100 recovery/score gaps. Preserve the original RX metadata failure and all candidate evidence. Stop before B2/B3 at the specified uncertainty cap; report the minimal additional budget without new training or wireless claims.\n')
 check('commit',['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-F',str(A/'commit_message.txt')])
 head=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==PARENT;put(A/'commit_verified.json',dict(head=head,normal_commit=True,**record))
 print(json.dumps(dict(head=head,files=len(paths),push_pending=True)),flush=True)

if __name__=='__main__':main()
