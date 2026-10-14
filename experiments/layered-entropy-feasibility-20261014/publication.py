"""Normal publication of the closed common500 supplemental metric grid."""
from pathlib import Path
import argparse,csv,hashlib,json,os,signal,subprocess,time,sys
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/layered-entropy-feasibility-20261014'
O=R/'results/layered_entropy_feasibility_20261014_v1'
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/layered_entropy_feasibility_20261014_v1')
A=T/'publication_v1';PREP=O/'publication_preparation_v1'
PARENT='ccf9497e9661b75b9b229c4bf4e7d08b6a9c0b3e'
PY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def read(p):return json.loads(p.read_text())

def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def git(*a):return subprocess.check_output(['git',*a],cwd=R,text=True).strip()

def ident(pid):return dict(pid=pid,start_ticks=int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]))

def gone(v):
    p=Path('/proc')/str(v['pid'])/'stat';return not p.exists() or int(p.read_text().rsplit(')',1)[1].split()[19])!=v['start_ticks']

def require(v,msg):
    if not v:raise RuntimeError(msg)

def check(name,argv):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1');start=time.monotonic();timed_out=terminated=killed=False
    with (A/(name+'.stdout.txt')).open('xb') as out,(A/(name+'.stderr.txt')).open('xb') as err:
        child=subprocess.Popen(argv,cwd=R,env=env,stdout=out,stderr=err,start_new_session=True);process=ident(child.pid)
        save(A/(name+'.launch.json'),dict(argv=argv,parent_process=ident(os.getpid()),process=process,max_seconds=900))
        try:rc=child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            timed_out=terminated=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=15)
            except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=15)
    v=dict(actual_wait=True,exit_code=rc,process=process,elapsed_seconds=time.monotonic()-start,timed_out=timed_out,terminated=terminated,killed=killed,metadata_error=None,argv=argv,stdout=pin(A/(name+'.stdout.txt')),stderr=pin(A/(name+'.stderr.txt')))
    save(A/(name+'.actual_wait.json'),v);return v

def checks(a):
    closure();require(sha(PREP/'authorization.json')==a.descriptor_sha256,'Root descriptor differs');spec=read(PREP/'authorization.json')
    require(not git('diff','--cached','--name-only') and git('rev-parse','HEAD')==PARENT,'Existing index/parent differs')
    paths=[]
    for row in spec['files']+[pin(PREP/'authorization.json')]:
        p=Path(row['path']);require(pin(p)==row,'Exact reviewed file differs');paths.append(p.relative_to(R).as_posix())
    subprocess.run(['git','add','--sparse','--',*paths],cwd=R,check=True)
    update=check('manifest_update',[PY,'-B','tools/update_repository_manifest.py']);require(update['exit_code']==0,'Manifest update failed')
    subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True);paths.append('release_manifest.json')
    require(set(git('diff','--cached','--name-only').splitlines())==set(paths),'Exact staged set differs')
    results={};results['diff_check']=check('diff_check',['git','diff','--cached','--check'])
    if results['diff_check']['exit_code']!=0:
        save(A/'initial_checks.json',results)
        crlf=[n for n in paths if b'\r\n' in (R/n).read_bytes()]
        require(crlf and all(not any(line.rstrip(b'\r\n').endswith((b' ',b'\t')) for line in (R/n).read_bytes().splitlines(keepends=True) if line.rstrip(b'\r\n')) for n in crlf),'Only CRLF spelling can be waived')
        one=check('diff_exact_crlf',['git','-c','core.whitespace=cr-at-eol','diff','--cached','--check','--',*crlf])
        others=[n for n in paths if n not in crlf];two=check('diff_other_paths',['git','diff','--cached','--check','--',*others])
        require(one['exit_code']==0 and two['exit_code']==0,'Real whitespace error; preserve and diagnose')
        save(A/'exact_crlf_diagnosis.json',dict(paths=crlf,initial_default_preserved=True,only_one_off_exact_paths=True,global_config_unchanged=True))
        results['diff_check']=dict(one,other_paths=two)
    for name,argv in [('verify_repository',[PY,'-B','tools/verify_repository.py']),('run_cpu_checks',[PY,'-B','tools/run_cpu_checks.py'])]:
        results[name]=check(name,argv);require(results[name]['exit_code']==0 and not results[name]['timed_out'],name+' failed')
    save(A/'checks.json',results);staged=[]
    for n in paths:
        b=subprocess.check_output(['git','show',':'+n],cwd=R);staged.append(dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    save(A/'staged_files.json',staged)
    require(not git('diff','--name-only') and git('rev-parse','HEAD')==PARENT,'Tree changed during checks')
    save(A/'checks_complete.json',dict(status='CHECKS_PASS_NOT_COMMITTED',parent=PARENT,checks_sha256=sha(A/'checks.json'),staged_files_sha256=sha(A/'staged_files.json'),index_tree=git('write-tree')))
    print(json.dumps(dict(status='CHECKS_PASS',files=len(staged),checks= {k:v['exit_code'] for k,v in results.items()})))
def closure():
    sys.path.insert(0,str(E));from finalize import wait0
    for role in ('score_completion','candidate_plan','finalize'):wait0(role)
    c=read(O/'final/summary.json');require(c['status']=='COMPLETE_LE0_LE1_ONLY_WITH_DISCLOSED_HISTORICAL_FAILURES' and c['TX_probability_traversals']==50 and c['independent_source_roundtrips']==24 and c['new_clean_states']==450 and c['clean_quality_rows']==1200 and c['wireless_image_frames']==0 and c['packet_trials']==0,'Exact finite LE0/LE1 closure required')
    require(c['old_raw_NO_GO_preserved']['status']=='NO_GO_STOP' and c['old_raw_NO_GO_preserved']['r2_authorized'] is False,'Preserved old raw NO_GO required')
    require(c['failed_initial_metric_argument_attempts']==1 and c['successful_scientific_calls_repeated']==0,'Disclose history without replay')
    for row in read(O/'final/output_manifest.json')['files']:require(pin(Path(row['path']))==row,'Final CPU output differs')
    for row in read(O/'registration.json')['source_bindings']:require(sha(Path(row['path']))==row['sha256'],'Original frozen source changed')
    require(read(O/'LE0/actual_visual_review.json')['actual_opened_both_PNGs'] is True,'Real visual review required')

def prepare():
    closure();require(git('rev-parse','HEAD')==PARENT and git('branch','--show-current')=='main' and not git('status','--porcelain','--untracked-files=no'),'Verified clean main parent required')
    require(not A.exists() and not PREP.exists(),'Existing publication attempt must be diagnosed');A.mkdir();PREP.mkdir()
    evidence=O/'evidence_v1';evidence.mkdir();sources=[]
    controls=('asset_audit_control_v1','asset_audit_resume_control_v1','tests_control_v1','analyze_control_v1','science_initial_control_v1','pilot_driver_control_v1','resource_stop_diagnosis_v1','resource_resume_control_v1','resource_resume_qualification_control_v1','resource_resume_initial_control_v1','score_api_qualification_control_v1','score_completion_initial_control_v1','score_completion_control_v1','candidate_plan_initial_control_v1','candidate_plan_control_v1','finalize_control_v1')
    for n in controls:sources += [p for p in (T/n).iterdir() if p.is_file() and p.suffix in ('.json','.txt')]
    for n in ('science','resource_resume_v1','score_completion_v1'):
        sources += [p for p in (T/n).iterdir() if p.is_file() and p.suffix=='.json' and p.stat().st_size<10000000]
    for n in ('science/streams','science/received_proofs'):sources += list((T/n).glob('*.json'))
    copies=[]
    for source in sorted(set(sources)):
        require(not source.is_symlink() and source.stat().st_size<10000000,'Only closed ordinary small evidence');before=pin(source);target=evidence/source.relative_to(T);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(source.read_bytes())
        require(pin(source)==before and sha(target)==before['sha256'],'Closed evidence changed');copies.append(dict(source=before,copy=pin(target)))
    save(evidence/'copy_manifest.json',dict(files=copies,originals_unchanged=True,new_scientific_calls=0))
    prefix=b"""## 2026-10-14 LE0/LE1 layered entropy source feasibility delivered

Only LE0 and LE1 were authorized and completed. See
results/layered_entropy_feasibility_20261014_v1/REPORT_zh.md.
The deterministic pilot50 from original raw cal200 completed50 TX traversals,
24 independent terminated two-stream RX proofs, and1200 clean source/endpoints
(750 exact reuse,450 missing-state renders and DINO-L scores). Wireless image
frames, packet trials, tokenization and training updates are all zero. The actual
extra segmentation information cost is median30bit, range28--32bit, including
the second13bit length field and16bit CRC. All terminal lengths are exact;
first38 streams retain an explicit missing auxiliary finish-increment field.
Source/model/CDF identities match. GPU3 resource stop and metric API failure
remain disclosed with actual exits; successful science was not repeated.
Actual Sionna group constructors and the capacity domain are independent design
evidence; original ep_phy.Backend.plan is unchanged. Prospective LE2 has at most
21504 packet trials if separately authorized. LE2/LE3/LE4 are NOT_RUN and no
automatic next stage is scheduled. All evaluated old raw two-group candidates
are dominated in the DINO and PSNR true7/10 planes; original R1 NO_GO is preserved.
Clean-prefix quality and arithmetic packability do not assert wireless gains.

"""
    docs=[]
    for n in ('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md'):
        p=R/n;before=pin(p);old=p.read_bytes();docs.append(dict(path=str(p),before=before));p.write_bytes(prefix+old)
    files=sorted(set([p for base in (E,O) for p in base.rglob('*') if p.is_file()]+[R/n for n in ('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md')]))
    require(all(p.stat().st_size<10000000 and not p.is_symlink() and p.suffix in ('.py','.md','.json','.csv','.txt','.pdf','.svg','.png') for p in files),'Only exact reviewed source/light files; no arrays or weights')
    descriptor=dict(schema='CLOSED_LE01_ENTROPY_FINITE_NORMAL_PUBLICATION_V1',parent=PARENT,status='PASS',reviewer='/root',files=[pin(p) for p in files],before_documents=docs,copy_manifest=pin(evidence/'copy_manifest.json'),actual_completion=pin(O/'final/summary.json'),stages=['LE0','LE1'],wireless_frames=0,packet_trials=0,later_stages_not_authorized=True,historical_failures_preserved=True)
    save(PREP/'authorization.json',descriptor);print(json.dumps(dict(descriptor=pin(PREP/'authorization.json'),files=len(files))))

def commit():
    p=read(A/'checks_complete.json');require(p['checks_sha256']==sha(A/'checks.json') and p['staged_files_sha256']==sha(A/'staged_files.json') and p['index_tree']==git('write-tree'),'Checked exact scope differs')
    require(git('rev-parse','HEAD')==PARENT and not git('diff','--name-only') and not (A/'commit.json').exists(),'Commit already attempted or tree differs')
    name,email=subprocess.check_output(['git','show','-s','--format=%an%n%ae',PARENT],cwd=R,text=True).splitlines()
    save(A/'one_off_identity.json',dict(identity_source='previous verified main commit',parent=PARENT,command_scope_only=True,global_config_unchanged=True))
    result=check('normal_commit',['git','-c','user.name='+name,'-c','user.email='+email,'commit','-m','Deliver finite LE0 LE1 layered entropy source feasibility']);require(result['exit_code']==0,'Normal commit failed')
    require(git('rev-parse','HEAD^')==PARENT and not git('status','--porcelain','--untracked-files=no'),'Normal commit closure differs')
    v=dict(parent=PARENT,head=git('rev-parse','HEAD'),normal_commit=True,checks_sha256=sha(A/'checks.json'),staged_files_sha256=sha(A/'staged_files.json'));save(A/'commit.json',v);print(json.dumps(v))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=('prepare','checks','commit'));p.add_argument('--descriptor-sha256');a=p.parse_args()
    if a.role=='prepare':prepare()
    elif a.role=='checks':checks(a)
    else:commit()
