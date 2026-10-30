"""Create a reviewable staged diff and lightweight archive; NEVER commit or push."""
from opt_common import *
import subprocess,shutil,zipfile,ast

HEAD='9d55aacfcd740034ce85014ae65f20a33893b59c'

def files():
 excluded=set(read(D/'delivery/large_file_index.json')['excluded_from_review_package']);result=[]
 for root in [E,D]:
  for p in sorted(root.rglob('*')):
   if not p.is_file()or '__pycache__'in p.parts:continue
   if root==D and str(p.relative_to(D))in excluded:continue
   assert not p.is_symlink()and p.stat().st_size<=10_000_000,p
   assert p.suffix not in {'.pt','.pth','.npz','.npy','.sqlite','.zip','.log','.pyc'},p
   result.append(p)
 return result

def add_exact(paths):
 q=L/'review_package'/'stage_paths.nul';q.parent.mkdir(exist_ok=True)
 q.write_bytes(b''.join(str(p.relative_to(R)).encode()+b'\0'for p in paths))
 subprocess.run(['git','add','--pathspec-from-file='+str(q),'--pathspec-file-nul'],cwd=R,check=True)

def main():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==HEAD
 out=D/'delivery/validation';assert read(out/'data_checks.json')['status']=='PASS'
 assert read(out/'artifact_review.json')['status']=='VISUALLY_REVIEWED_ALL_8_PNG_AND_8_PDF_PAGES'
 assert read(D/'resource_delivery/resource_ledger.json')['all_owner_and_child_identities_gone']
 # Preserve original failed checks and the real passing environment check.
 tests=read(D/'CPU_checks_visual_environment.json');assert tests['returncode']==0
 receipts=out/'repository_receipts';receipts.mkdir(exist_ok=False)
 for label,v in [('cpu_checks',tests),('source_verify',read(D/'repository_verify_staged_source.json'))]:
  for stream in ['stdout','stderr']:shutil.copyfile(checked(v[stream]),receipts/(label+'.'+stream+'.txt'))
 for item in read(D/'repository_checks.json')['results']:
  for stream in ['stdout','stderr']:shutil.copyfile(checked(item[stream]),receipts/('initial_'+item['name']+'.'+stream+'.txt'))
 compiled=[]
 for p in sorted(E.rglob('*.py')):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));compiled.append(str(p.relative_to(R)))
 save(out/'source_syntax_check.json',dict(status='PASS',files=len(compiled),method='AST parse only, no execution of launchers or scientific stages',paths=compiled))
 states=read(D/'delivery/stage_completion.json')['stages'];lines=['本轮验收清单','状态为实际执行结果；blocked 不算完成。','']
 for name,v in states.items():lines.append(name+': '+v['status']+' — '+v.get('reason',v.get('evidence',v.get('role',''))).__str__())
 lines+=['','统计数据核验：done','全部8张图PNG及PDF逐页实际检查：done','CPU工程检查：done，264项通过；不等于图像质量证据','Git：按用户要求只暂存待审查，不 commit/push。','C第二种子：前置质量保护线未通过，不启动。','NEW200：原始合格数据缺失，独立新留出验证未完成。']
 with(D/'delivery/completion_checklist.txt').open('x',encoding='utf8')as f:f.write('\n'.join(lines)+'\n')
 initial=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=R).decode().strip('\0').split('\0')
 assert all(p.startswith('experiments/nestcom-opt-20261030/')for p in initial if p),('Unexpected staged change; preserve and stop',initial)
 paths=files();add_exact(paths)
 actual=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=R).decode().strip('\0').split('\0');assert set(actual)=={str(p.relative_to(R))for p in paths}
 assert not subprocess.check_output(['git','diff','--name-only'],cwd=R,text=True).strip(),'Unexpected tracked unstaged changes'
 os.sched_setaffinity(0,{26,27});start=time.time();p=subprocess.run([sys.executable,'tools/verify_repository.py'],cwd=R,capture_output=True,text=True,timeout=300)
 for stream,value in [('stdout',p.stdout),('stderr',p.stderr)]:
  with(receipts/('final_verify.'+stream+'.txt')).open('x')as f:f.write(value)
 save(out/'repository_final.json',dict(returncode=p.returncode,seconds=time.time()-start,HEAD=HEAD,stdout=pin(receipts/'final_verify.stdout.txt'),stderr=pin(receipts/'final_verify.stderr.txt'),CPU_checks=pin(D/'CPU_checks_visual_environment.json'),original_failures_preserved=pin(D/'repository_checks.json'),commit=False,push=False))
 assert p.returncode==0,(p.stdout,p.stderr)
 save(out/'git_delivery_state.json',dict(status='STAGED_UNCOMMITTED_REVIEW_PACKAGE',HEAD=HEAD,branch=subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip(),scope=['experiments/nestcom-opt-20261030','results/nestcom_opt_20261030_v1'],commit=False,push=False,original_unrelated_untracked_preserved=True,reason='Latest user work order explicitly reserves Git upload to the user'))
 paths=files();add_exact(paths)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==HEAD
 assert not subprocess.check_output(['git','diff','--name-only'],cwd=R,text=True).strip()
 target=L/'review_package/NeST_Optimization_20261030_v1.zip';assert not target.exists();manifest=[]
 for p in paths:manifest.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)))
 with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as z:
  for p in paths:z.write(p,str(p.relative_to(R)))
  z.writestr('FILE_MANIFEST.json',json.dumps(dict(HEAD=HEAD,files=manifest,excluded_oversized_originals=read(D/'delivery/large_file_index.json')['excluded_from_review_package'],all_original_rows_or_bytes_in_parts=True,commit=False,push=False),ensure_ascii=False,indent=2)+'\n')
 with zipfile.ZipFile(target)as z:assert z.testzip()is None and len(z.namelist())==len(paths)+1
 result=dict(status='READY_FOR_USER_REVIEW',archive=pin(target),file_count=len(paths),all_workers_exited=True,NEW200='BLOCKED',Git='STAGED_UNCOMMITTED_NO_PUSH',HEAD=HEAD)
 save(L/'review_package/completion.json',result);print(json.dumps(result))

if __name__=='__main__':main()
