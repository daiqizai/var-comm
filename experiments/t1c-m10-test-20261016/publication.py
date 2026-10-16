"""Normal main publication of closed T1-C code and light results only."""
from tc_common import *
from tc_protocol import closed_wait
import subprocess,signal,shutil
REG='32d148496df2e35912a35d878207a96fbe6d5d20345350276d748365ca3e1ef8'
A=T/'publication_v1';PREP=O/'publication_preparation_v1';PY=str(RT/'envs/frozen_visual_py310/bin/python')
def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True).strip()
def staged():return [p for p in subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=R).decode().split('\0') if p]
def closure():
 registration('link',REG)
 for s in ('source','phy','receiver','metrics','delivery'):
  closed_wait(read(T/(s+'_control/actual_wait.json')),gone);assert not(T/s/'failure.json').exists() and not(T/(s+'_control/failure.json')).exists()
 d=read(O/'delivery/completion.json');assert d['status']=='COMPLETE_T1C_19DB_DELIVER_AND_STOP' and d['registration_sha256']==REG
 for p in checkread(d['manifest'])['files']:checked(p)
 return read(O/'delivery/closure_audit.json')
def check(name,argv,limit=900):
 start=time.monotonic();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1');timed=terminated=killed=False
 with(A/(name+'.stdout.txt')).open('xb') as so,(A/(name+'.stderr.txt')).open('xb') as se:
  child=subprocess.Popen(argv,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True);ident=pid_identity(child.pid)
  save(A/(name+'.launch.json'),dict(argv=argv,process=ident,parent_process=pid_identity(os.getpid()),max_seconds=limit))
  try:rc=child.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   timed=terminated=True;os.killpg(child.pid,signal.SIGTERM)
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=20)
 v=dict(actual_wait=True,exit_code=rc,process=ident,argv=argv,elapsed_seconds=time.monotonic()-start,timed_out=timed,terminated=terminated,killed=killed,metadata_error=None,stdout=pin(A/(name+'.stdout.txt')),stderr=pin(A/(name+'.stderr.txt')));save(A/(name+'.actual_wait.json'),v);return v
def prepare():
 audit=closure();assert git('rev-parse','HEAD')==HEAD and git('branch','--show-current')=='main' and not git('status','--porcelain','--untracked-files=no')
 A.mkdir();PREP.mkdir();ev=O/'execution_evidence';ev.mkdir();copies=[]
 def copy(p,q):
  assert not p.is_symlink() and p.stat().st_size<10000000;d=pin(p);q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb') as f:f.write(p.read_bytes())
  assert pin(p)==d and sha(q)==d['sha256'];copies.append(dict(source=d,copy=pin(q)))
 for stage in ('source','tests','phy','receiver','metrics','delivery'):
  for role in (stage+'_initial_control',stage+'_control'):
   for p in sorted((T/role).iterdir()):
    if p.is_file() and p.suffix in ('.json','.txt'):copy(p,ev/role/p.name)
  if stage!='tests':copy(T/stage/'completion.json',ev/stage/'completion.json')
 for role in ('source_preparation','link_preparation','link_preparation_v2','link_preparation_v3','link_preflight_cross_family_reuse_v1'):
  for p in sorted((T/role).iterdir()):
   if p.is_file() and p.name in ('completion.json','launch.json','failure.json','observation.json'):copy(p,ev/role/p.name)
 copy(T/'receiver/actual_native_identity.json',ev/'receiver/actual_native_identity.json')
 for name in ('four_metric_identity.json','supplemental_identity.json'):copy(T/'metrics'/name,ev/'metrics'/name)
 copy(T/'whole_evidence_restored/restoration_manifest.json',ev/'historical_whole_metadata_restoration.json')
 save(ev/'copy_manifest.json',dict(records=copies,originals_preserved=True,no_arrays_or_weights=True))
 save(O/'final_closure.json',dict(status='T1C_DELIVERED_STOP',reviewer='root',basis_HEAD=HEAD,closure=pin(O/'delivery/closure_audit.json'),m10_fit={'common500':6,'new100':1},logical_frames=3600,new_physical=1125,new_packets=audit['scientific_stages']['phy']['new_packets']['total'],frozen_policies_unchanged=True,training=0,selection=0,other_SNRs=0,LE3=False,automatic_successor=False,backup_deferred=True))
 prefix=f'''## 2026-10-16 T1-C corrected m10 test evaluation delivered; stop

Frozen entropy whole and partial strategies completed19dB on common500 and
independent new100, each with three original noise realizations. The600 actual
m10 source traversals retain all24 old endpoints byte-exact. Fullm10 fits6/500
and1/100 sources; the rest follow the frozen family-specific fallback chains.
The3600 logical denominator uses2454 exact reused frames (including378
cross-family equivalent events) and1125 new physical events, with
{audit['scientific_stages']['phy']['new_packets']['total']} actual packets within2250. All new PHY/RX/metric/statistics stages actually
exit0. Full-denominator source means and direct paired intervals use10000
source resamples, seed2026100701, separately for500 and100 sources.

Historical common500 whole-code streams from the older4090 environment differ
from the current frozen H800 implementation. Old-to-new whole-code deltas
therefore mix this version effect withm10; no solem10 attribution is claimed.
Old results remain intact. common500 retains allseven metrics; new100 retains
its originalfour. No4/10dB, other method, training, selection, LE3, HiFi or
shared-timing rerun. Existing timing remains labeled as the old version.
Scientific backups remain deferred. See
results/t1c_m10_test_20261016_v1/delivery/REPORT_zh.md.

'''.encode('utf8');before=[]
 for name in ('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md'):
  p=R/name;before.append(pin(p));p.write_bytes(prefix+p.read_bytes())
 (E/'README.md').write_text('T1-C only: frozen corrected entropy whole and partial, original common500 and independent new100,19dB and originalthree noise realizations. No training or selection. Read source_registration.json and link_registration.json in results/t1c_m10_test_20261016_v1. Completed scientific stages must never be repeated. See delivery/REPORT_zh.md for all results and version limitations. CPU qualification is synthetic, not scientific evidence. Scientific backups are deferred.\n',encoding='utf8')
 files=sorted({p for root in(E,O) for p in root.rglob('*') if p.is_file()}|{R/n for n in('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md')})
 assert all(not p.is_symlink() and p.stat().st_size<10000000 and p.suffix in('.py','.json','.csv','.md','.txt') for p in files)
 save(PREP/'authorization.json',dict(schema='T1C_CLOSED_NORMAL_PUBLICATION_V1',status='PASS',reviewer='root',parent=HEAD,files=[pin(p) for p in files],before_docs=before,registration=pin(O/'link_registration.json'),backup_deferred=True,automatic_followup=False));print('PUBLICATION_PREPARED',len(files))
def checks():
 closure();assert git('rev-parse','HEAD')==HEAD and not staged();spec=read(PREP/'authorization.json');paths=[checked(x).relative_to(R).as_posix() for x in spec['files']+[pin(PREP/'authorization.json')]]
 subprocess.run(['git','add','--sparse','--',*paths],cwd=R,check=True)
 assert check('manifest_update',[PY,'-B','tools/update_repository_manifest.py'])['exit_code']==0
 subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True);paths.append('release_manifest.json');assert set(staged())==set(paths)
 results={}
 for name,argv in [('diff_check',['git','diff','--cached','--check']),('verify_repository',[PY,'-B','tools/verify_repository.py']),('run_cpu_checks',[PY,'-B','tools/run_cpu_checks.py'])]:
  v=check(name,argv);results[name]=v
  if v['exit_code']!=0:save(A/'initial_checks.json',results);raise AssertionError(name+' failed; diagnose preserved evidence')
 finish_checks(results)
def finish_checks(results):
 assert all(v['actual_wait'] and type(v['exit_code']) is int and v['exit_code']==0 for v in results.values());save(A/'checks.json',results)
 rows=[]
 for name in staged():
  b=subprocess.check_output(['git','show',':'+name],cwd=R);rows.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 save(A/'staged_files.json',rows);assert not git('diff','--name-only')
 save(A/'checks_complete.json',dict(status='CHECKS_PASS_NOT_COMMITTED',parent=HEAD,checks_sha256=sha(A/'checks.json'),staged_files_sha256=sha(A/'staged_files.json'),index_tree=git('write-tree')));print('CHECKS_PASS',len(rows))
def commit():
 c=read(A/'checks_complete.json');assert git('rev-parse','HEAD')==HEAD and git('write-tree')==c['index_tree'] and not git('diff','--name-only')
 assert c['checks_sha256']==sha(A/'checks.json') and c['staged_files_sha256']==sha(A/'staged_files.json')
 name,email=subprocess.check_output(['git','show','-s','--format=%an%n%ae',HEAD],cwd=R,text=True).splitlines();body=A/'commit_body.txt'
 with body.open('x',encoding='utf8') as f:f.write('Complete frozen m10 T1-C evaluation at19dB\n\nDeliver separate common500 and new100 full-denominator results for corrected entropy whole and partial strategies. Verify600 source lengths, reuse2454 exact logical events and fill1125 changed physical events. Preserve the older common500 whole-stream version difference and all historical outputs.\n\nValidation: closed actual stages, source-paired10000 bootstrap,15 synthetic tests, original repository verifier and264 CPU checks. No other SNR, training, reselection, timing, LE3 or backup run. Stop after delivery.\n')
 assert check('commit',['git','-c','user.name='+name,'-c','user.email='+email,'commit','-F',str(body)])['exit_code']==0
 h=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==HEAD;save(A/'commit.json',dict(parent=HEAD,head=h,normal_commit=True,checks_sha256=c['checks_sha256'],staged_files_sha256=c['staged_files_sha256']));print(h)
if __name__=='__main__':{'prepare':prepare,'checks':checks,'commit':commit}[sys.argv[1]]()
