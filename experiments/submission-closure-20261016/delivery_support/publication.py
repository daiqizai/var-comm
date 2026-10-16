"""Publish the closed first submission batch; never execute science."""
from common import *
import sys,subprocess,signal
A=T/'publication_v1';PREP=O/'publication_preparation_v1';PARENT='bdb3e3c12d2ea2c18f4371e11cd912d5e8529100';PY=str(RT/'envs/frozen_visual_py310/bin/python')
def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True).strip()
def staged_names():return subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=R).decode().rstrip('\0').split('\0')
def closed_control(name,expected=0):
 d=T/name;w=read(d/'actual_wait.json');assert w['actual_wait'] and type(w['returncode'])is int and w['returncode']==expected and not w['timed_out']
 assert not any(w.get(k,False) for k in ('terminated','killed')) and w.get('metadata_error')is None
 for k in ('process','parent_process'):assert gone(w[k]),(name,k,w[k])
 for k in ('stdout','stderr'):checked(w[k])
 return pin(d/'actual_wait.json')
def closure():
 assert sha(O/'source_registration.json')=='13c9057b42cdc9d36a8934b2479e8a9385073d5205402852a355e32803c60845'
 assert sha(O/'metric_registration.json')=='16b31b8d62bdc91170ca7b5e5c3057b70db40cba86cfab7dd9c1a0f91f980c04'
 for n in ('source_registration.json','metric_registration.json'):
  for row in read(O/n)['code']:checked(row)
 for n in ('source_audit_control_v1','fixed16_score_control_v1','test_m10_control_v1','figures_control_v2'):closed_control(n)
 closed_control('figures_control_v1',1)
 for n in ('m10_source','fixed16_score'):assert not(T/n/'failure.json').exists()
 c=read(O/'m10/completion.json');assert (c['new_TX'],c['new_RX'],c['new_prior_scales'])==(1000,8,10080)and c['counts']['unresolved']==0
 c=read(O/'fixed16/completion.json');assert (c['new_raw_metric_scalars'],c['new_psnr_scalars'])==(48,64)and c['counts']['unresolved']==0
 for name in ('m10','fixed16'):
  for row in read(O/name/'closure_audit.json')['outputs']:checked(row)
 for row in read(O/'m10/candidate_delta_manifest.json').get('outputs',[]):checked(row)
 v=read(O/'actual_visual_review.json');assert v['status']=='PASS'and v['all_nine_pngs_actually_opened']and len(v['PNG'])==9;checked(v['manifest'])
 for fig in read(O/'figures_v2/manifest.json')['figures']:
  for row in fig['files']:checked(row)
 for row in v['PNG']:checked(row)
 assert read(O/'scope_actual.json')['automatic_followup']is False
def check(name,argv,limit=900):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1');start=time.monotonic();timed=terminated=killed=False
 with(A/(name+'.stdout.txt')).open('xb')as so,(A/(name+'.stderr.txt')).open('xb')as se:
  child=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True);ident=pid_identity(child.pid);assert ident;save(A/(name+'.launch.json'),dict(argv=argv,process=ident,parent_process=pid_identity(os.getpid()),max_seconds=limit))
  try:rc=child.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   timed=terminated=True;os.killpg(child.pid,signal.SIGTERM)
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=20)
 v=dict(actual_wait=True,exit_code=rc,elapsed_seconds=time.monotonic()-start,process=ident,argv=argv,timed_out=timed,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(A/(name+'.stdout.txt')),stderr=pin(A/(name+'.stderr.txt')));save(A/(name+'.actual_wait.json'),v);return v
def prepare():
 closure();assert git('rev-parse','HEAD')==PARENT and git('branch','--show-current')=='main'and not git('status','--porcelain','--untracked-files=no')and not A.exists()and not PREP.exists();A.mkdir();PREP.mkdir()
 evidence=O/'evidence_v1';evidence.mkdir(exist_ok=True);copies=[]
 for n in ('source_audit_initial_v1','source_audit_control_v1','fixed16_score_initial_v1','fixed16_score_control_v1','test_m10_control_v1','figures_control_v1','figures_control_v2'):
  for p in sorted((T/n).iterdir()):
   assert p.is_file()and not p.is_symlink()and p.suffix in ('.json','.txt')and p.stat().st_size<10000000
   before=pin(p);q=evidence/n/p.name;q.parent.mkdir(parents=True,exist_ok=True)
   with q.open('xb')as f:f.write(p.read_bytes())
   assert pin(p)==before and sha(q)==before['sha256'];copies.append(dict(source=before,copy=pin(q)))
 save(evidence/'control_copy_manifest.json',dict(records=copies,originals_preserved=True,scientific_arrays_and_ledgers_remain_in_registered_runtime=True))
 save(O/'final_closure.json',dict(status='FIRST_BATCH_DELIVERED_STOP',reviewer='/root',basis_HEAD=PARENT,new_TX=1000,new_independent_RX=8,new_prior_scales=10080,new_raw_metric_scalars=48,new_psnr_scalars=64,reused_metric_scalars=144,new_channel_packets=0,new_wireless_images=0,new_renders=0,new_training=0,new_diffusion=0,new_bootstrap=0,new_timing=0,automatic_followup=False,all_science_parent_child_gone=True,source_closure=pin(O/'m10/closure_audit.json'),metric_closure=pin(O/'fixed16/closure_audit.json'),figure_review=pin(O/'actual_visual_review.json'),first_figure_metadata_failure_preserved=True,backup_deferred=True))
 prefix='''## 2026-10-16 Submission first batch delivered; m10 correction required

T0/T1-A/T2/T3 only. Originalcal1000 completed1000 source TX and8 independent
source RX; all24000 existing endpoints match byte-exact. Fullm10 lengths
min/median/P95/max=3823/6071/6568.05/6855bit. Capacity4273 fits4/1000;
capacity4751 fits18/1000. Both whole and partial families need symmetricm10
candidate correction. Old candidates/results are retained; no new calibration
or test link run was started. Changed-prefix subset396logical frames is not
the full calibration budget; see the explicit prospective upper bound.

Native raw fixed16 was recovered from its historical assets and scored at the
original13dB condition:48 new DISTS/DreamSim/MS-SSIM plus64 new PSNR scalars,
144 old scalars reused. Raw has lower perceptual distances and lower PSNR/MS-SSIM
means; these16 development sources do not establish population superiority.
Source roles/intersections, actual LDPC layouts, model/training facts, existing
common500 metrics and shared-H800 timing are delivered with9 reviewed figure
sets. The common500 layout is explicitly provisional pendingm10 correction.

No new channel packets, wireless images, renders, diffusion, training, bootstrap
or timing. Raw R1 NO_GO, LE0/LE1/LE2 and completed timing remain unchanged.
T1-B/T1-C/LE3/LE4 and power expansion are NOT_RUN. First batch stops here.
See results/submission_closure_20261016_v1/REPORT_zh.md.
Scientific backups remain deferred at the user's request.

'''.encode();before=[]
 for name in ('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md'):
  p=R/name;before.append(pin(p));p.write_bytes(prefix+p.read_bytes())
 with(E/'README.md').open('xb')as f:f.write(prefix+b'\nThe user package remains a plan. Actual execution is bounded by scope_actual.json and the two immutable registrations.\n')
 files=sorted({p for root in(E,O)for p in root.rglob('*')if p.is_file()}|{R/name for name in('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md')})
 assert all(not p.is_symlink()and p.stat().st_size<10000000 and p.suffix in('.py','.md','.json','.csv','.txt','.png','.pdf','.svg')for p in files)
 save(O/'output_manifest.json',dict(files=[pin(p)for p in files if p.is_relative_to(O)],excludes_self_and_later_publication_receipts=True))
 files.append(O/'output_manifest.json')
 save(PREP/'authorization.json',dict(schema='SUBMISSION_CLOSED_FIRST_BATCH_NORMAL_PUBLICATION_V1',status='PASS',reviewer='/root',parent=PARENT,files=[pin(p)for p in files],before_docs=before,source_registration=pin(O/'source_registration.json'),metric_registration=pin(O/'metric_registration.json'),backup_deferred=True,automatic_followup=False));print(canonical(pin(PREP/'authorization.json')))
def checks():
 closure();assert git('rev-parse','HEAD')==PARENT and not git('diff','--cached','--name-only');spec=read(PREP/'authorization.json');paths=[]
 for row in spec['files']+[pin(PREP/'authorization.json')]:paths.append(checked(row).relative_to(R).as_posix())
 subprocess.run(['git','add','--sparse','--',*paths],cwd=R,check=True);v=check('manifest_update',[PY,'-B','tools/update_repository_manifest.py']);assert v['exit_code']==0
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
 save(A/'staged_files.json',staged);assert git('rev-parse','HEAD')==PARENT and not git('diff','--name-only');save(A/'checks_complete.json',dict(status='CHECKS_PASS_NOT_COMMITTED',parent=PARENT,checks_sha256=sha(A/'checks.json'),staged_files_sha256=sha(A/'staged_files.json'),index_tree=git('write-tree')));print('CHECKS_PASS',len(staged))
def commit():
 c=read(A/'checks_complete.json');assert git('write-tree')==c['index_tree']and git('rev-parse','HEAD')==PARENT and c['checks_sha256']==sha(A/'checks.json')and c['staged_files_sha256']==sha(A/'staged_files.json')and not git('diff','--name-only')and not(A/'commit.json').exists()
 name,email=subprocess.check_output(['git','show','-s','--format=%an%n%ae',PARENT],cwd=R,text=True).splitlines();body=A/'commit_body.txt'
 with body.open('x',encoding='utf8',newline='\n')as f:f.write('Close submission batch: m10 source boundary and native fixed16 evidence\n\nComplete1000 source TX and8 independent RX with exact regression of24000 old endpoints. Fullm10 fits18 calibration sources at4751bit; require symmetric whole/partial candidate correction without running followup channels. Complete112 new native fixed16 scores and reuse144 existing values. Deliver all-source descriptive comparisons, source roles, LDPC/model facts, existing metrics/timing and reviewed figures. Preserve initial plotting metadata failure and all original science.\n\nValidation: actual closed scientific receipts, original repository verifier and264 CPU checks. No new channel, image generation, training, bootstrap or timing. Backups deferred; stop before T1-B/T1-C or LE3.\n')
 v=check('commit',['git','-c','user.name='+name,'-c','user.email='+email,'commit','-F',str(body)]);assert v['exit_code']==0;head=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==PARENT
 save(A/'commit.json',dict(parent=PARENT,head=head,normal_commit=True,checks_sha256=c['checks_sha256'],staged_files_sha256=c['staged_files_sha256']));print(canonical(dict(head=head,parent=PARENT)))
if __name__=='__main__':{'prepare':prepare,'checks':checks,'commit':commit}[sys.argv[1]]()
