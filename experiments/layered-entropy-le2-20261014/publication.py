"""Normal publication of closed LE2, including every immutable failure."""
from common import *
import sys,subprocess,signal,shutil
A=T/'publication_v1';PREP=O/'publication_preparation_v1';PARENT='bbd75f967a612ce968e5f0ea0aa73b88c774ca4c';PY=str(RT/'envs/frozen_visual_py310/bin/python')
def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True).strip()
def staged_names():return subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=R).decode().rstrip('\0').split('\0')
def closure():
 w=read(T/'cpu_delivery_metadata_revision_control_v2/actual_wait.json');assert w['actual_wait'] and type(w['returncode'])is int and w['returncode']==0 and w['metadata_error']is None and not any(w[k] for k in ('timed_out','terminated','killed'))
 # The root separately verifies PIDs on the execution host (leo). Publication
 # runs on H100; cross-host /proc checks would be invalid.
 c=read(O/'execution_host_final_closure.json');assert c['host_verified']=='leo-tactile' and c['all_processes_gone'] and c['actual_cpu_delivery_wait_sha256']==sha(T/'cpu_delivery_metadata_revision_control_v2/actual_wait.json')
 s=read(O/'delivery_v2/summary.json');assert s['packet_trials']==18688 and s['initial_cells']==92 and s['topup_cells']==18 and s['wireless_image_frames']==0 and s['LE3_executed'] is False and s['measured_positive_group_point_estimates']==0 and s['status']=='NO_PROMISING_MEASURED_CANDIDATE'
 for r in read(O/'delivery_v2/output_manifest.json')['files']:checked(r)
 for r in read(O/'registration.json')['bindings']:checked(r)
 for r in read(O1/'registration.json')['source_bindings']:checked(r)
 assert read(O1/'final/summary.json')['old_raw_NO_GO_preserved']['status']=='NO_GO_STOP'
 assert (O/'delivery_v2/metadata_revision.json').is_file()
def check(name,argv,limit=900):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1');start=time.monotonic();timed=terminated=killed=False
 with (A/(name+'.stdout.txt')).open('xb') as out,(A/(name+'.stderr.txt')).open('xb') as err:
  child=subprocess.Popen(argv,cwd=R,env=env,stdout=out,stderr=err,start_new_session=True);identity=process(child.pid);assert identity;save(A/(name+'.launch.json'),dict(argv=argv,process=identity,parent_process=process(os.getpid()),max_seconds=limit))
  try:rc=child.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   timed=terminated=True;os.killpg(child.pid,signal.SIGTERM)
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=20)
 v=dict(actual_wait=True,exit_code=rc,elapsed_seconds=time.monotonic()-start,process=identity,argv=argv,timed_out=timed,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(A/(name+'.stdout.txt')),stderr=pin(A/(name+'.stderr.txt')));save(A/(name+'.actual_wait.json'),v);return v
def prepare():
 closure();assert git('rev-parse','HEAD')==PARENT and git('branch','--show-current')=='main' and not git('status','--porcelain','--untracked-files=no') and not git('diff','--cached','--name-only') and not A.exists() and not PREP.exists();A.mkdir();PREP.mkdir();evidence=O/'evidence_v1';evidence.mkdir()
 dirs=('gray_initial_control_v1','gray_control_v1','gray_revision_initial_control_v1','gray_pixel_identity_revision_control_v1','tests_control_v1','packets_initial_control_v1','packets_control_v1','cpu_delivery_initial_control_v1','cpu_delivery_control_v1','cpu_delivery_metadata_revision_initial_control_v2','cpu_delivery_metadata_revision_control_v2','gray','gray_pixel_identity_revision_v1')
 copies=[]
 paths=[p for name in dirs for p in (T/name).iterdir() if p.is_file() and p.suffix in ('.json','.txt')]+[p for p in (T/'packets').iterdir() if p.is_file() and p.suffix=='.json']+[p for p in T.glob('cpu_delivery*.json')]
 for p in sorted(set(paths)):
  before=pin(p);assert before['bytes']<10000000 and not p.is_symlink();target=evidence/p.relative_to(T);target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(p.read_bytes())
  assert pin(p)==before and sha(target)==before['sha256'];copies.append(dict(source=before,copy=pin(target)))
 save(evidence/'copy_manifest.json',dict(records=copies,science_arrays_and_ledgers_not_copied=True,originals_preserved=True))
 note='''# LE2 delivered; stop before LE3\n\nCurrent delivered files are results/layered_entropy_le2_20261014_v1/delivery_v2/.\nLE0/LE1 and their immutable source/registration are preserved. The new LE2 scope\ncorrected29bit overhead in each proportional group allocation, replaced capacity\nfrequency nomination with complete source-fallback strategies, and measured\nall15 single956-symbol MCS plus nominated grouped layouts at both7 and10dB.\nAll18688 packet trials completed,92 initial cells and18 paired-SNR topups,\nbelow21504; zero wireless image frames, training, VAR TX, clean renders or LE3.\nThe best measured two-group proxy J=.6919812165440816 is below single robust\nJ=.7021032469582041: delta=-.010122030414122518. No measured positive candidate.\n409 grouped target/fallback aliases are covered;103001 combinations are unmeasured.\nThis is a correct-prefix packet-probability proxy, not real wireless image quality\nor a global negative proof. Cellwise Wilson sensitivity is not simultaneous or\npaired image confidence. Nine deduplicated exploratory/diagnostic LE3 candidates\nand a prospective paid12bit directory are delivered, with zero gain recommendations\nand no execution authorization. U and EEP or nominal/backoff anchors may coincide.\nDeduplication equivalence is on Pilot50; future-source equivalence is not asserted.\n\nFailures remain immutable: first gray pixel metadata check exit1 (no scoring);\nsource byte-SHA convention repair scored33 missing fixed-gray pairs and reused17.\nOne necessary extra DINO construction is disclosed. Packet owner exit1 only at\nCSV export after all science; first CPU delivery exit1 at duplicate physical-n\nand trial-n fields; separate CPU metadata revision exit0 generated final reports\nwithout any new packet/model/metric call and reused its completed CSV byte-exact.\nSee evidence_v1 and delivery_v2/metadata_revision.json. Never rerun old entries.\nBackups remain deferred at the user's request; this is lightweight publication.\n'''
 with (E/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write(note)
 with (O/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write(note)
 prefix='''## 2026-10-15 LE2 finite packet proxy delivered; stop before LE3\n\nLE0/LE1 and old raw R1 NO_GO are preserved. The new LE2 revision corrects\nthe29bit length+CRC fields in each proportional allocation and nominates complete\nstrategies from every Pilot50 source's actual fallback prefix quality. It covers\nsingle956 MCS and grouped reliability at7/10dB:18688 new packets,92 initial cells\nand18 topups within21504; no new wireless image/VAR/clean/training calls.\nBest grouped proxy J=.6919812165440816 versus single robust .7021032469582041,\ndelta=-.010122030414122518: NO_PROMISING_MEASURED_CANDIDATE. Unmeasured layouts\nremain unknown and no global NO_GO or wireless image gain is claimed.\nNine deduplicated diagnostic/exploratory LE3 candidates are delivered, no\nrecommended positive proxy candidate; LE3 and power expansion are NOT_RUN.\nGray hash-convention and CSV metadata failures with their real exits remain\nexplicit; CPU-only final delivery exited0 without repeating any packet/score.\nSee results/layered_entropy_le2_20261014_v1/delivery_v2/REPORT_zh.md.\nScientific backups remain deferred at the user's explicit request.\n\n'''.encode()
 before=[]
 for name in ('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md'):
  q=R/name;before.append(pin(q));q.write_bytes(prefix+q.read_bytes())
 files=sorted({p for root in (E,O) for p in root.rglob('*') if p.is_file()}|{R/name for name in ('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md')});assert all(p.stat().st_size<10000000 and not p.is_symlink() and p.suffix in ('.py','.md','.json','.csv','.txt') for p in files)
 save(PREP/'authorization.json',dict(schema='LE2_CLOSED_FINITE_NORMAL_PUBLICATION_V1',status='PASS',reviewer='/root',parent=PARENT,files=[pin(p) for p in files],before_docs=before,packet_trials=18688,zero_LE3=True,original_scientific_failure_preserved=True,CPU_metadata_revision_actualwait=pin(T/'cpu_delivery_metadata_revision_control_v2/actual_wait.json'),backup_deferred=True));print(canonical(pin(PREP/'authorization.json')))
def checks():
 closure();assert git('rev-parse','HEAD')==PARENT and not git('diff','--cached','--name-only');spec=read(PREP/'authorization.json');paths=[]
 for row in spec['files']+[pin(PREP/'authorization.json')]:q=checked(row);paths.append(q.relative_to(R).as_posix())
 subprocess.run(['git','add','--sparse','--',*paths],cwd=R,check=True);v=check('manifest_update',[PY,'-B','tools/update_repository_manifest.py']);assert v['exit_code']==0;subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True);paths.append('release_manifest.json');assert set(staged_names())==set(paths)
 results={}
 for name,argv in (('diff_check',['git','diff','--cached','--check']),('verify_repository',[PY,'-B','tools/verify_repository.py']),('run_cpu_checks',[PY,'-B','tools/run_cpu_checks.py'])):
  v=check(name,argv);results[name]=v
  if v['exit_code']!=0:save(A/'initial_checks.json',results);raise AssertionError(name+' failed; original receipts preserved')
 save(A/'checks.json',results);staged=[]
 for name in paths:
  b=subprocess.check_output(['git','show',':'+name],cwd=R);staged.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 save(A/'staged_files.json',staged);assert git('rev-parse','HEAD')==PARENT and not git('diff','--name-only');save(A/'checks_complete.json',dict(status='CHECKS_PASS_NOT_COMMITTED',parent=PARENT,checks_sha256=sha(A/'checks.json'),staged_files_sha256=sha(A/'staged_files.json'),index_tree=git('write-tree')));print('CHECKS_PASS',len(staged))
def commit():
 c=read(A/'checks_complete.json');assert git('write-tree')==c['index_tree'] and git('rev-parse','HEAD')==PARENT and c['checks_sha256']==sha(A/'checks.json') and c['staged_files_sha256']==sha(A/'staged_files.json') and not git('diff','--name-only') and not (A/'commit.json').exists()
 name,email=subprocess.check_output(['git','show','-s','--format=%an%n%ae',PARENT],cwd=R,text=True).splitlines();body=A/'commit_body.txt'
 with body.open('x',encoding='utf8',newline='\n') as f:f.write('Complete finite LE2 packet screening and stop before LE3\n\nCorrect the two29bit allocation fields and nominate complete source-fallback strategies. Cover single956 and grouped reliability at7/10dB with18688 actual packets within21504. Best measured group proxy trails robust single by.010122; deliver9 distinct diagnostic candidates without executing LE3. Preserve LE01, raw NO_GO and all metadata failures; independent CPU delivery reuses closed science.\n\nValidation: original repository verifier and264 synthetic CPU checks, new20 parser/budget tests, metadata-only regressions, and full92-cell actual-hard-RX closure. No new training or wireless images. Backups remain deferred.\n')
 v=check('commit',['git','-c','user.name='+name,'-c','user.email='+email,'commit','-F',str(body)]);assert v['exit_code']==0;head=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==PARENT
 save(A/'commit.json',dict(parent=PARENT,head=head,normal_commit=True,checks_sha256=c['checks_sha256'],staged_files_sha256=c['staged_files_sha256']));print(canonical(dict(head=head,parent=PARENT)))
if __name__=='__main__':{'prepare':prepare,'checks':checks,'commit':commit}[sys.argv[1]]()
