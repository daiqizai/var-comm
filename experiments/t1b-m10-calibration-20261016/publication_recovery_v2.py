"""Publish only actually closed T1-B light results, never execute science."""
from tb_common import *
import subprocess,signal
A=T/'publication_v1';PREP=O/'publication_preparation_v1';PARENT=HEAD
PY=str(RT/'envs/frozen_visual_py310/bin/python')
def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True).strip()
def staged_names():return subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=R).decode().rstrip('\0').split('\0')
def closure():
 from deliver_recovery_v2 import REG,complete_wait
 from recovery_binding import verify_recovery
 verify_recovery('4f6b2f22b1514c0c21508ef1cf1162d1deda0cb1673daccd5eb67d3b0e2ed057')
 from metric_device_recovery_v2 import verify_metric_recovery
 verify_metric_recovery('567b8e8b39459ec8548db7ca7f2cd0fe3075a11c7d6c03d062a5e7731400036f')
 registration(REG);complete_wait(T/'dispatch_recovery_v2/actual_wait.json');complete_wait(T/'delivery_control/actual_wait.json')
 for p in(T/'dispatch_recovery_v2/supervisor.json',T/'control_recovery_v2/supervisor.json'):
  assert gone(read(p)['process'])
 complete_wait(T/'control/phy/actual_wait.json')
 complete_wait(T/'control_recovery_v1/receiver_recovery_v1/actual_wait.json')
 for stage in('metrics_recovery_v2','selection'):complete_wait(T/'control_recovery_v2'/stage/'actual_wait.json')
 for role in('dispatch_recovery_v2','control_recovery_v2','phy','receiver_recovery_v1','metrics_recovery_v2','selection','delivery'):assert not(T/role/'failure.json').exists()
 d=read(T/'delivery/completion.json');checked(d['manifest']);manifest=read(O/'delivery/manifest.json')
 for f in manifest['files']:checked(f)
 audit=read(O/'delivery/closure_audit.json');assert audit['status']=='T1B_DELIVERED_CALIBRATION_ONLY' and audit['actual_new_physical']==22139 and audit['actual_packet_calls']<=44278 and audit['T1C']is False and audit['training']is False
 return audit
def check(name,argv,limit=900):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1');start=time.monotonic();timed=terminated=killed=False
 with(A/(name+'.stdout.txt')).open('xb')as so,(A/(name+'.stderr.txt')).open('xb')as se:
  child=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True);ident=pid_identity(child.pid);assert ident
  save(A/(name+'.launch.json'),dict(argv=argv,process=ident,parent_process=pid_identity(os.getpid()),max_seconds=limit))
  try:rc=child.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   timed=terminated=True;os.killpg(child.pid,signal.SIGTERM)
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=20)
 v=dict(actual_wait=True,exit_code=rc,elapsed_seconds=time.monotonic()-start,process=ident,argv=argv,timed_out=timed,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(A/(name+'.stdout.txt')),stderr=pin(A/(name+'.stderr.txt')))
 save(A/(name+'.actual_wait.json'),v);return v
def prepare():
 audit=closure();assert git('rev-parse','HEAD')==PARENT and git('branch','--show-current')=='main'and not git('status','--porcelain','--untracked-files=no')and not A.exists()and not PREP.exists();A.mkdir();PREP.mkdir()
 evidence=O/'execution_evidence';evidence.mkdir();copies=[]
 scopes=[T/'dispatch_recovery_v2',T/'control_recovery_v2',T/'metrics',B/'t1b_metric_device_qualification_v2']+[T/'control_recovery_v2'/s for s in('metrics_recovery_v2','selection')]+[T/'dispatch_recovery_v1',T/'control_recovery_v1',T/'receiver',B/'t1b_runtime_import_boundary_qualification_v1',B/'t1b_runtime_import_boundary_qualification_v1/synthetic',B/'t1b_runtime_import_boundary_qualification_v1/import_control']+[T/'control_recovery_v1'/s for s in('receiver_recovery_v1','metrics')]+[T/'registration_control',T/'dispatch',T/'delivery_control',T/'control']+[T/'control'/s for s in('phy','receiver')]+[B/'t1b_m10_execution_qualification_v1',B/'t1b_m10_preparation_control_v1/test_plan',B/'t1b_m10_preparation_control_v1/prepare']
 for ix,root in enumerate(scopes):
  for p in sorted(root.iterdir()):
   if not p.is_file()or p.suffix not in('.json','.txt'):continue
   assert not p.is_symlink() and p.stat().st_size<1000000
   before=pin(p);q=evidence/str(ix)/p.name;q.parent.mkdir(parents=True,exist_ok=True)
   with q.open('xb')as f:f.write(p.read_bytes())
   assert pin(p)==before and sha(q)==before['sha256'];copies.append(dict(source=before,copy=pin(q)))
 for stage in('phy','receiver_recovery_v1','metrics_recovery_v2','selection'):
  q=evidence/(stage+'_completion.json');p=T/stage/'completion.json'
  with q.open('xb')as f:f.write(p.read_bytes())
  copies.append(dict(source=pin(p),copy=pin(q)))
 save(evidence/'copy_manifest.json',dict(records=copies,originals_preserved=True,runtime_arrays_and_ledgers_not_copied=True))
 import csv
 with(O/'delivery/policy_comparison.csv').open(newline='',encoding='utf8')as f:comparisons=list(csv.DictReader(f))
 changed=[dict(snr_db=x['snr_db'],family=x['family'],winner=x['winner'],DINO_delta=x['delta'])for x in comparisons if x['changed']=='True']
 save(O/'final_closure.json',dict(status='T1B_DELIVERED_STOP',reviewer='/root',basis_HEAD=PARENT,closure=pin(O/'delivery/closure_audit.json'),policy_changes=changed,new_source_TX=0,new_tokenizer=0,T1C=False,LE3=False,training=False,backup_deferred=True,automatic_followup=False))
 prefix=f'''## 2026-10-16 T1-B symmetric m10 calibration delivered; stop before T1-C

Both whole and partial families now include maximum target m10 with their
separately frozen fallback chains. All old candidates/winners are retained.
Source lengths collapse20 exact calibration aliases; four changed strategies
are completely evaluated on originalcal1000 at4/10/19dB and4101/4102/4103.
The36000 logical grid required22139 new physical events and{audit['actual_packet_calls']}
packet decodes, within44278. Exact existing events/images/scores were reused;
no source TX/tokenizer/training/test-set calls were added. One original RX
initialization failed before Torch import; its failed model-load reservation
is preserved. A separate runtime-library read boundary recovery inherited
closed PHY without rerunning it and completed RX. A later UUID spelling
assertion stopped metric initialization with zero model constructions/scores.
Exact UUID normalization retained the same physical GPU and completed
metrics and selection without repeating PHY or RX.

Frozen old/new winners and all four metrics, all new logical frame scores,
source means, paid-profile evidence and actual exits are delivered in
results/t1b_m10_calibration_20261016_v1/delivery/REPORT_zh.md.
There are{len(changed)} changed family/SNR winners; calibration maximum differences
are not independent test gains or novelty evidence. Old raw R1 NO_GO,
LE0/LE1/LE2, fixed16 and shared timing stay unchanged.
T1-C/common500/new100 source-boundary work and LE3 were NOT_RUN.
No automatic successor. Scientific backups remain deferred by the user.

'''.encode();before=[]
 for name in('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md'):
  p=R/name;before.append(pin(p));p.write_bytes(prefix+p.read_bytes())
 files=sorted({p for root in(E,O)for p in root.rglob('*')if p.is_file()}|{R/n for n in('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md')})
 assert all(not p.is_symlink() and p.stat().st_size<16000000 and p.suffix in('.py','.md','.json','.csv','.txt')for p in files)
 save(O/'output_manifest.json',dict(files=[pin(p)for p in files if p.is_relative_to(O)],excludes_self_and_later_publication=True));files.append(O/'output_manifest.json')
 save(PREP/'authorization.json',dict(schema='T1B_CLOSED_NORMAL_PUBLICATION_V1',status='PASS',reviewer='/root',parent=PARENT,files=[pin(p)for p in files],before_docs=before,registration=pin(O/'registration.json'),backup_deferred=True,automatic_followup=False))
 print(canonical(pin(PREP/'authorization.json')))
def checks():
 closure();assert git('rev-parse','HEAD')==PARENT and not git('diff','--cached','--name-only');spec=read(PREP/'authorization.json');paths=[]
 for row in spec['files']+[pin(PREP/'authorization.json')]:paths.append(checked(row).relative_to(R).as_posix())
 subprocess.run(['git','add','--sparse','--',*paths],cwd=R,check=True)
 v=check('manifest_update',[PY,'-B','tools/update_repository_manifest.py']);assert v['exit_code']==0
 subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True);paths.append('release_manifest.json');assert set(staged_names())==set(paths)
 results={}
 for name,argv in(('diff_check',['git','diff','--cached','--check']),('verify_repository',[PY,'-B','tools/verify_repository.py']),('run_cpu_checks',[PY,'-B','tools/run_cpu_checks.py'])):
  v=check(name,argv);results[name]=v
  if v['exit_code']!=0:save(A/'initial_checks.json',results);raise AssertionError(name+' failed; preserve and diagnose')
 finish_checks(results)
def finish_checks(results):
 assert all(v['actual_wait']and type(v['exit_code'])is int and v['exit_code']==0 for v in results.values());save(A/'checks.json',results);staged=[]
 for name in staged_names():
  b=subprocess.check_output(['git','show',':'+name],cwd=R);staged.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 save(A/'staged_files.json',staged);assert git('rev-parse','HEAD')==PARENT and not git('diff','--name-only')
 save(A/'checks_complete.json',dict(status='CHECKS_PASS_NOT_COMMITTED',parent=PARENT,checks_sha256=sha(A/'checks.json'),staged_files_sha256=sha(A/'staged_files.json'),index_tree=git('write-tree')))
 print('CHECKS_PASS',len(staged))
def commit():
 c=read(A/'checks_complete.json');assert git('write-tree')==c['index_tree']and git('rev-parse','HEAD')==PARENT and c['checks_sha256']==sha(A/'checks.json')and c['staged_files_sha256']==sha(A/'staged_files.json')and not git('diff','--name-only')and not(A/'commit.json').exists()
 name,email=subprocess.check_output(['git','show','-s','--format=%an%n%ae',PARENT],cwd=R,text=True).splitlines();body=A/'commit_body.txt'
 with body.open('x',encoding='utf8',newline='\n')as f:f.write('Complete symmetric m10 T1-B calibration and freeze candidates\n\nRetain legacy candidates and winners, add both m10 fallback families, verify old event/pixel/metric reuse and fill only missing calibration events. Deliver all36000 logical frames and source-level means across4/10/19dB with unchanged original scoring mathematics. Preserve actual failures and paid profile boundaries.\n\nValidation: actual finite science exits, closure audit, original repository verifier and264 CPU checks. No source TX, tokenizer, training, T1-C/test-set work, LE3 or automatic successor. Backups deferred.\n')
 v=check('commit',['git','-c','user.name='+name,'-c','user.email='+email,'commit','-F',str(body)]);assert v['exit_code']==0
 head=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==PARENT
 save(A/'commit.json',dict(parent=PARENT,head=head,normal_commit=True,checks_sha256=c['checks_sha256'],staged_files_sha256=c['staged_files_sha256']));print(canonical(dict(head=head,parent=PARENT)))
if __name__=='__main__':{'prepare':prepare,'checks':checks,'commit':commit}[sys.argv[1]]()
