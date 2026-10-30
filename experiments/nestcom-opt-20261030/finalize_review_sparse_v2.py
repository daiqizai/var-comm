"""Finish packaging after exact sparse-index admission failure; no science rerun."""
from finalize_review import files,HEAD
from opt_common import *
import ast,subprocess,zipfile

def add_exact(paths):
 q=L/'review_package/stage_paths_sparse_v2.nul'
 q.write_bytes(b''.join(str(p.relative_to(R)).encode()+b'\0'for p in paths))
 subprocess.run(['git','add','--sparse','--pathspec-from-file='+str(q),'--pathspec-file-nul'],cwd=R,check=True,capture_output=True,text=True)

def main():
 out=D/'delivery/validation';receipts=out/'repository_receipts'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==HEAD
 assert not(out/'repository_final.json').exists()and not(L/'review_package/completion.json').exists()
 assert read(out/'data_checks.json')['status']=='PASS'
 assert read(out/'artifact_review.json')['status']=='VISUALLY_REVIEWED_ALL_8_PNG_AND_8_PDF_PAGES'
 assert read(out/'final_reporting_checks.json')['status']=='PASS'
 assert read(out/'packaging_sparse_failure/transport.json')['returncode']==1
 assert 'sparse-checkout definition'in(out/'packaging_sparse_failure/stderr.txt').read_text()
 assert(D/'delivery/completion_checklist.txt').exists()and receipts.is_dir()
 assert read(D/'CPU_checks_visual_environment.json')['returncode']==0
 assert read(D/'resource_delivery/resource_ledger.json')['all_owner_and_child_identities_gone']
 compiled=[]
 for p in sorted(E.rglob('*.py')):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));compiled.append(str(p.relative_to(R)))
 save(out/'source_syntax_check_sparse_v2.json',dict(status='PASS',files=len(compiled),method='AST only, includes final reporting and packaging amendments',paths=compiled))
 initial=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=R).decode().strip('\0').split('\0')
 assert all(p.startswith(('experiments/nestcom-opt-20261030/','results/nestcom_opt_20261030_v1/'))for p in initial if p)
 paths=files();add_exact(paths)
 actual=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=R).decode().strip('\0').split('\0')
 assert set(actual)=={str(p.relative_to(R))for p in paths}
 assert not subprocess.check_output(['git','diff','--name-only'],cwd=R,text=True).strip()
 os.sched_setaffinity(0,{26,27});start=time.time();p=subprocess.run([sys.executable,'tools/verify_repository.py'],cwd=R,capture_output=True,text=True,timeout=300)
 for stream,value in [('stdout',p.stdout),('stderr',p.stderr)]:
  with(receipts/('final_verify.'+stream+'.txt')).open('x')as f:f.write(value)
 save(out/'repository_final.json',dict(returncode=p.returncode,seconds=time.time()-start,HEAD=HEAD,stdout=pin(receipts/'final_verify.stdout.txt'),stderr=pin(receipts/'final_verify.stderr.txt'),CPU_checks=pin(D/'CPU_checks_visual_environment.json'),original_failures_preserved=pin(D/'repository_checks.json'),packaging_sparse_admission_failure=pin(out/'packaging_sparse_failure/transport.json'),commit=False,push=False))
 assert p.returncode==0,(p.stdout,p.stderr)
 save(out/'git_delivery_state.json',dict(status='STAGED_UNCOMMITTED_REVIEW_PACKAGE',HEAD=HEAD,branch=subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip(),scope=['experiments/nestcom-opt-20261030','results/nestcom_opt_20261030_v1'],commit=False,push=False,original_unrelated_untracked_preserved=True,sparse_checkout_definition_unchanged=True,reason='Latest user work order explicitly reserves Git upload to the user'))
 paths=files();add_exact(paths)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==HEAD
 assert not subprocess.check_output(['git','diff','--name-only'],cwd=R,text=True).strip()
 target=L/'review_package/NeST_Optimization_20261030_v1.zip';assert not target.exists()
 manifest=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p))for p in paths]
 with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as z:
  for p in paths:z.write(p,str(p.relative_to(R)))
  z.writestr('FILE_MANIFEST.json',json.dumps(dict(HEAD=HEAD,files=manifest,excluded_oversized_originals=read(D/'delivery/large_file_index.json')['excluded_from_review_package'],all_original_rows_or_bytes_in_parts=True,commit=False,push=False),ensure_ascii=False,indent=2)+'\n')
 with zipfile.ZipFile(target)as z:assert z.testzip()is None and len(z.namelist())==len(paths)+1
 result=dict(status='READY_FOR_USER_REVIEW',archive=pin(target),file_count=len(paths),all_workers_exited=True,NEW200='BLOCKED',Git='STAGED_UNCOMMITTED_NO_PUSH',HEAD=HEAD)
 save(L/'review_package/completion.json',result);print(json.dumps(result))

if __name__=='__main__':main()
