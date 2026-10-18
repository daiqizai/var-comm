"""Exact immutable SVG/log whitespace review; retain default checks elsewhere."""
import collections,hashlib,importlib.util,json,pathlib,re,subprocess,sys,time
E=pathlib.Path(__file__).parent.parent;sys.path.insert(0,str(E))
import a1_owner as r
R,L,O=r.R,r.L,r.O;A=L/'final_publication_v1'
path=E/'final_publication_v1/publish.py'
assert r.desc(path)['sha256']=='aec83acc71aee6016078ba84a11b90b1dc2519511c03901c5b0b1ae9c5cfdae8'
spec=importlib.util.spec_from_file_location('original_final_publication',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PREFIX='results/arpc_external_baseline_20261019_v1/'
EXPECTED={PREFIX+'final_execution_receipts_v1/'+p:1 for p in ['a2b_rx_v1/rx.stdout.txt','a2b_source_v1/tx.stdout.txt','a3_rx_v1/rx.stdout.txt','a3_source_v1/tx.stdout.txt']}
EXPECTED.update({PREFIX+f'final_figures_v{v}/{n}.svg':count for v in [1,2] for n,count in [('matched100_paired_differences',234),('matched100_seven_metrics',878)]})

def main():
    auth=json.loads((A/'authorization.json').read_text());parent=auth['parent']
    assert m.git('branch','--show-current')=='main' and m.git('rev-parse','HEAD')==parent
    assert not m.git('diff','--name-only')
    for base in ['final_publication_execution_v1','final_publication_copy_resume_control_v1']:
        d=L/base;x=json.loads((d/'child.actual_wait.json').read_text());assert type(x['returncode']) is int and x['returncode']==1 and r.gone(x['process']) and r.gone(json.loads((d/'supervisor.json').read_text())['process'])
    w=json.loads((A/'diff_check.actual_wait.json').read_text());assert w['returncode']==2 and r.gone(w['process'])
    paths=json.loads((A/'reviewed_paths.json').read_text())+['release_manifest.json']
    assert set(m.git('diff','--cached','--name-only').splitlines())==set(paths)
    for pin in json.loads((A/'reviewed_input_files.json').read_text()):r.verify(pin)
    counts=collections.Counter()
    for line in (A/'diff_check.stdout').read_text().splitlines():
        match=re.fullmatch(r'([^+].*?):(\d+): (trailing whitespace\.|new blank line at EOF\.)',line)
        if not match:continue
        name,number,kind=match.groups();assert name in EXPECTED and kind=='trailing whitespace.'
        row=(R/name).read_bytes().splitlines(keepends=True)[int(number)-1]
        assert row.endswith(b' \n')
        if name.endswith('.txt'):assert row==b'[constructor]  ==== customized_flash_attn=False (using_flash=0/32), fused_mlp=False (fused_mlp=0/32) ==== \n'
        else:assert row.lstrip().startswith((b'<path ',b'M ',b'L ',b'Q ',b'C ',b'z '))
        counts[name]+=1
    assert dict(counts)==EXPECTED and sum(counts.values())==2228
    review=O/'final_publication_review_v1';assert not review.exists();review.mkdir()
    r.put(review/'format_review.json',{'status':'EXACT_IMMUTABLE_FORMAT_ONLY','reviewer':'Codex root actual diff and original bytes inspection','observed_unix':time.time(),
       'original_default_exit':r.desc(A/'diff_check.actual_wait.json'),'original_stdout':r.desc(A/'diff_check.stdout'),
       'exceptions':[{'file':r.desc(R/p),'lines':n,'one_call_option':'core.whitespace=-blank-at-eol'} for p,n in EXPECTED.items()],
       'scope':'Four original constructor log lines plus four immutable Matplotlib SVG path whitespace; preserve bytes. All other paths use default whitespace checks.',
       'source':r.desc(__file__),'original_copy_failure':r.desc(L/'final_publication_execution_v1/child.actual_wait.json'),
       'copy_resume':r.desc(A/'copy_resume_review.json')})
    additions=[p.relative_to(R).as_posix() for p in sorted(pathlib.Path(__file__).parent.iterdir()) if p.is_file()]+[(review/'format_review.json').relative_to(R).as_posix()]
    assert all(p not in paths for p in additions)
    subprocess.run(['git','add','--sparse','--',*additions],cwd=R,check=True);paths+=additions
    m.check('manifest_update_format_evidence',[m.PY,'-B','tools/update_repository_manifest.py'])
    subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True)
    assert set(m.git('diff','--cached','--name-only').splitlines())==set(paths)
    r.put(A/'reviewed_final_paths.json',paths)
    checks={
      'diff_check':m.check('diff_check_exact_immutable8',['git','-c','core.whitespace=-blank-at-eol','diff','--cached','--check','--',*sorted(EXPECTED)]),
      'default_diff_other_paths':m.check('diff_check_remaining_default',['git','diff','--cached','--check','--',*sorted(set(paths)-set(EXPECTED))]),
      'verify_repository':m.check('verify_repository',[m.PY,'-B','tools/verify_repository.py']),
      'run_cpu_checks':m.check('run_cpu_checks',[m.PY,'-B','tools/run_cpu_checks.py'])}
    r.put(A/'checks.json',checks)
    files=[]
    for name in sorted(paths):
        data=subprocess.check_output(['git','show',':'+name],cwd=R)
        files.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    r.put(A/'staged_files.json',files)
    proof={'parent':parent,'tree':m.git('write-tree'),'checks_sha256':r.desc(A/'checks.json')['sha256'],'staged_files_sha256':r.desc(A/'staged_files.json')['sha256']}
    r.put(A/'checks_complete.json',dict(proof,status='PASS'))
    message=A/'commit_message.txt';assert not message.exists();message.write_text('Complete bounded ARPC calibration and fixed100 matched evaluation\n\nPreserve independent reception, all seven source-balanced metrics and directly paired intervals. Record frozen paid policies, original failure evidence and reviewed vector figures without extending science.\n')
    m.check('commit',['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-F',str(message)])
    head=m.git('rev-parse','HEAD');assert m.git('rev-parse','HEAD^')==parent
    r.put(A/'commit_verified.json',dict(proof,head=head,normal_commit=True));print(json.dumps({'head':head,'push_pending':True}))

if __name__=='__main__':main()
