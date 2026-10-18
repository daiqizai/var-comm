"""Preserve three oversized closed JSON records on PFS; normal lightweight release."""
import hashlib,importlib.util,json,os,pathlib,subprocess,sys,time,traceback
E=pathlib.Path(__file__).parent.parent;sys.path.insert(0,str(E))
import a1_owner as r
R,L,O=r.R,r.L,r.O;A=L/'final_publication_v1'
PY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
SIZES={'results/arpc_external_baseline_20261019_v1/a2b_link_v1/logical_frames.json':10075177,
       'results/arpc_external_baseline_20261019_v1/a2b_metrics_v1/pair_plan.json':11639848,
       'results/arpc_external_baseline_20261019_v1/a2b_rx_v1/logical_receivers.json':17464942}

def worker():
    source=E/'final_publication_v1/publish.py';assert r.desc(source)['sha256']=='aec83acc71aee6016078ba84a11b90b1dc2519511c03901c5b0b1ae9c5cfdae8'
    spec=importlib.util.spec_from_file_location('original_final_publication',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    auth=json.loads((A/'authorization.json').read_text());parent=auth['parent'];assert m.git('rev-parse','HEAD')==parent and not m.git('diff','--name-only')
    d=L/'final_publication_format_control_v1';x=json.loads((d/'child.actual_wait.json').read_text());assert x['returncode']==1 and r.gone(x['process']) and r.gone(json.loads((d/'supervisor.json').read_text())['process'])
    w=json.loads((A/'verify_repository.actual_wait.json').read_text());assert w['returncode']==1 and r.gone(w['process'])
    assert not(A/'run_cpu_checks.launch.json').exists() and not(A/'checks_complete.json').exists() and not(A/'commit_verified.json').exists()
    paths=json.loads((A/'reviewed_final_paths.json').read_text());assert set(m.git('diff','--cached','--name-only').splitlines())==set(paths)
    records=[]
    for name,size in SIZES.items():
        p=R/name;pin=r.desc(p);assert pin['bytes']==size
        assert subprocess.run(['git','cat-file','-e','HEAD:'+name],cwd=R,capture_output=True).returncode!=0
        data=subprocess.check_output(['git','show',':'+name],cwd=R);assert hashlib.sha256(data).hexdigest()==pin['sha256'];records.append(pin)
    assert {name for name in m.git('ls-files').splitlines() if (R/name).stat().st_size>10_000_000}==set(SIZES)
    # Remove only new index entries. The actual scientific files are untouched.
    subprocess.run(['git','rm','--cached','--sparse','--',*SIZES],cwd=R,check=True)
    for pin in records:r.verify(pin)
    review=O/'final_publication_review_v1/large_closed_metadata.json'
    r.put(review,{'status':'EXACT_LARGE_CLOSED_RECORDS_RETAINED_ON_PFS','files':records,'reason':'Original repository verifier enforces 10 MB per tracked file. Only these new index entries are removed; full files and scientific SHA/closures remain unchanged on PFS. No filter changes and no restoration-backup claim.',
       'failed_repository_check':r.desc(A/'verify_repository.actual_wait.json'),'failed_stderr':r.desc(A/'verify_repository.stderr'),'source':r.desc(__file__),'observed_unix':time.time()})
    additions=[pathlib.Path(__file__).relative_to(R).as_posix(),review.relative_to(R).as_posix()]
    subprocess.run(['git','add','--sparse','--',*additions],cwd=R,check=True)
    paths=[n for n in paths if n not in SIZES]+additions
    m.check('manifest_update_size_evidence',[PY,'-B','tools/update_repository_manifest.py'])
    subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True)
    assert set(m.git('diff','--cached','--name-only').splitlines())==set(paths)
    immutable=json.loads((O/'final_publication_review_v1/format_review.json').read_text())['exceptions']
    exact=[]
    for row in immutable:
        r.verify(row['file']);name=pathlib.Path(row['file']['path']).relative_to(R).as_posix();exact.append(name)
        assert hashlib.sha256(subprocess.check_output(['git','show',':'+name],cwd=R)).hexdigest()==row['file']['sha256']
    prior=A/'diff_check_exact_immutable8.actual_wait.json';x=json.loads(prior.read_text());assert type(x['returncode']) is int and x['returncode']==0 and r.gone(x['process'])
    checks={'diff_check':r.desc(prior),
      'default_diff_other_paths':m.check('diff_check_size_revision_default',['git','diff','--cached','--check','--',*sorted(set(paths)-set(exact))]),
      'verify_repository':m.check('verify_repository_size_revision',[PY,'-B','tools/verify_repository.py']),
      'run_cpu_checks':m.check('run_cpu_checks',[PY,'-B','tools/run_cpu_checks.py'])}
    r.put(A/'checks.json',checks);r.put(A/'reviewed_release_paths.json',paths)
    files=[]
    for name in sorted(paths):
        data=subprocess.check_output(['git','show',':'+name],cwd=R);files.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    r.put(A/'staged_files.json',files)
    proof={'parent':parent,'tree':m.git('write-tree'),'checks_sha256':r.desc(A/'checks.json')['sha256'],'staged_files_sha256':r.desc(A/'staged_files.json')['sha256']}
    r.put(A/'checks_complete.json',dict(proof,status='PASS'))
    message=A/'commit_message.txt';assert not message.exists();message.write_text('Complete bounded ARPC calibration and fixed100 matched evaluation\n\nPreserve independent reception, all seven source-balanced metrics and directly paired intervals. Record frozen paid policies, failure evidence and reviewed vector figures without extending science. Large closed scientific metadata remains on PFS.\n')
    m.check('commit',['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-F',str(message)])
    head=m.git('rev-parse','HEAD');assert m.git('rev-parse','HEAD^')==parent
    r.put(A/'commit_verified.json',dict(proof,head=head,normal_commit=True));print(json.dumps({'head':head,'push_pending':True}))

def main():
    if len(sys.argv)>1:assert sys.argv[1:] == ['_worker'];worker();return
    root=L/'final_publication_size_control_v1';assert not root.exists();root.mkdir()
    r.put(root/'supervisor.json',{'process':r.identity(os.getpid()),'parent_process':r.identity(os.getppid()),'argv':sys.argv,'source':r.desc(__file__),'started_unix':time.time()});r.RUN=root
    try:
        r.run_stage('child',[PY,'-B',__file__,'_worker'],dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),2400)
        r.put(root/'completion.json',{'status':'PASS','wait':r.desc(root/'child.actual_wait.json')})
    except BaseException:r.put(root/'failure.json',{'error':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':main()
