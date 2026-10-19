"""Normal exact-path publication of a CPU-only plan; no scientific launcher."""
from pathlib import Path
import hashlib,json,os,subprocess,time,sys
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/arpc-n2048-feasibility-20261019'
O=R/'results/arpc_n2048_feasibility_20261019_v1'
A=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/arpc_n2048_feasibility_20261019_v1/publication')
PY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
PARENT='fbc17ae300e46bfdab56c22f989752545a5c9b3c'
def git(*v):return subprocess.check_output(['git',*v],cwd=R,text=True).strip()
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def put(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def ident(pid):
    s=Path(f'/proc/{pid}/stat').read_text();return dict(pid=pid,start_ticks=int(s[s.rfind(')')+2:].split()[19]))
def check(name,argv,timeout=1800):
    start=time.monotonic()
    with (A/(name+'.stdout.txt')).open('xb') as out,(A/(name+'.stderr.txt')).open('xb') as err:
        p=subprocess.Popen(argv,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdout=out,stderr=err)
        identity=ident(p.pid);put(A/(name+'.launch.json'),dict(argv=argv,process=identity,parent_process=ident(os.getpid())))
        rc=p.wait(timeout=timeout)
    wait=A/(name+'.actual_wait.json');put(wait,dict(returncode=rc,process=identity,seconds=time.monotonic()-start,timed_out=False,killed=False,terminated=False,metadata_error=None))
    assert rc==0,(name,rc)
    return pin(wait)
def main():
    assert git('rev-parse','HEAD')==PARENT and git('branch','--show-current')=='main'
    assert not git('status','--porcelain','--untracked-files=no')
    A.mkdir(exist_ok=False)
    put(A/'supervisor.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),source=pin(Path(__file__)),science_execution=False))
    manifest=json.loads((O/'package_manifest.json').read_text());paths=[]
    for x in manifest['files']:
        p=R/x['path'];assert p.is_file() and not p.is_symlink();assert pin(p)['sha256']==x['sha256'] and p.stat().st_size==x['bytes'];paths.append(x['path'])
    paths.append((O/'package_manifest.json').relative_to(R).as_posix())
    note='''## 2026-10-19 ARPC N2048 feasibility plan delivered (not executed)

Preserved the completed N1024 comparison at fbc17ae. Joined the existing cal100
700 endpoint payload/quality records and exported reviewed vector curves. A
proposed 2 x (68-header + 956-body) layout pays both length/CRC fields and two
96-bit reassembly records: 9296 source bits at common 64-QAM 5/6. Both full
NeST-Com m10 and ARPC e7 fit 100/100 calibration sources. Historical low-SNR
reliability is a warning, not new two-fragment evidence. Existing complete
source objects are checked; new TX passes can be zero. Proposed fixed100
validation is 3600 subframes, <=3648 including qualification. No new channel,
model, metric, bootstrap or training calls occurred. N2048 execution requires
a later instruction; automation remains paused and backups deferred.
See results/arpc_n2048_feasibility_20261019_v1/PLAN_zh.md and proposed_scope.json.

'''
    before=[]
    for name in ['README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md']:
        p=R/name;before.append(pin(p));old=p.read_bytes();assert b'ARPC N2048 feasibility plan delivered' not in old;p.write_bytes(note.encode()+old);paths.append(name)
    put(A/'before_docs.json',before)
    subprocess.run(['git','add','--sparse','--',*paths],cwd=R,check=True)
    check('manifest_update',[PY,'-B','tools/update_repository_manifest.py'])
    subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True);paths.append('release_manifest.json')
    assert set(git('diff','--cached','--name-only').splitlines())==set(paths)
    checks={'diff_check':check('diff_check',['git','diff','--cached','--check']),
      'verify_repository':check('verify_repository',[PY,'-B','tools/verify_repository.py']),
      'run_cpu_checks':check('run_cpu_checks',[PY,'-B','tools/run_cpu_checks.py'])}
    put(A/'checks.json',checks)
    staged=[]
    for name in sorted(paths):
        data=subprocess.check_output(['git','show',':'+name],cwd=R);staged.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    put(A/'staged_files.json',staged)
    proof=dict(parent=PARENT,tree=git('write-tree'),checks_sha256=pin(A/'checks.json')['sha256'],staged_files_sha256=pin(A/'staged_files.json')['sha256'])
    put(A/'checks_complete.json',dict(proof,status='PASS'))
    (A/'commit_message.txt').write_text('Document ARPC payload-quality and bounded N2048 feasibility\n\nPreserve N1024 results; account for two paid subframes and reassembly. Report calibration fit, checked source reuse and proposed compute caps without running new science.\n')
    check('commit',['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-F',str(A/'commit_message.txt')])
    head=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==PARENT
    put(A/'commit_verified.json',dict(proof,head=head,normal_commit=True));print(json.dumps(dict(head=head,push_pending=True)))
if __name__=='__main__':main()
