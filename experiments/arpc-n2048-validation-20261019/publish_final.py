"""Normal final publication of the finite N2048 validation and honest failures."""
from shared import *
import fcntl,re

A=L/'publication_final_v1'
PARENT='5f9baf7c7258168e8aef56e5ec4d7ee0406c3ffa'

def git(*args):
    return subprocess.check_output(['git',*args],cwd=R,text=True).strip()

def check(name,argv):
    c=A/name;c.mkdir()
    wait_child(c,argv,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),1800)
    return pin(c/'actual_wait.json')

def main():
    os.chdir(R)
    lock=(L/'execution.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert git('rev-parse','HEAD')==PARENT and git('branch','--show-current')=='main'
    assert not git('status','--porcelain','--untracked-files=no')
    assert not git('diff','--cached','--name-only')
    assert read(O/'final_delivery_manifest.json')['status']=='SCIENTIFIC_AND_VISUAL_DELIVERY_COMPLETE'
    assert read(O/'figures_v2/actual_visual_review.json')['status']=='PASS'
    assert read(O/'delivery/actual_counts.json')['total_decoder_attempts']==7300
    A.mkdir()
    put(A/'supervisor.json',dict(process=identity(os.getpid()),parent_process=identity(os.getppid()),source=pin(__file__)))
    # The metadata copier's own exit is preserved only after that process ends.
    c=L/'controls/resume_closed_evidence'
    assert read(c/'actual_wait.json')['returncode']==0 and gone(read(c/'actual_wait.json')['process']) and gone(read(c/'launch.json')['parent_process'])
    dest=O/'execution_receipts/resume_closed_evidence';dest.mkdir()
    copies=[]
    for p in sorted(c.iterdir()):
        assert p.is_file() and p.stat().st_size<1000000
        d=dest/p.name;d.open('xb').write(p.read_bytes());assert sha(d)==sha(p)
        copies.append(dict(source=pin(p),copy=pin(d)))
    put(O/'final_readiness/metadata_copy_exit.json',dict(status='PASS',exact_copies=copies))
    note='''## 2026-10-19 N2048 two-method full-representation validation completed

The authorized fixed100 validation is complete under the shared two-subframe
2048-use budget, paying both headers, lengths, CRCs and96-bit reassembly records.
Complete NeST-Com m10 and ARPC e7 objects both fit; fixed64-QAM rate5/6 delivers
0/300 objects per method at4/10dB and300/300 at19dB, with no wrongly accepted
objects. Rejected objects remain gray in all denominators. This fixed full-source
layout is capacity-feasible but does not solve low-SNR reliability.

At19dB direct paired intervals favor NeST-Com for PSNR, LPIPS, DreamSim and
MS-SSIM, and ARPC for DISTS; DINO-L and source-prediction agreement intervals
cross zero. All seven metrics and pointwise intervals are reported. The single
10000-replicate bootstrap samples sources after averaging three noise draws.
Qualification and formal work used3648 transmitted subframes and7300 decoder
attempts, including exactly four explicitly authorized repeated decodes after
the preserved import failure.200 formal native reconstructions and300 distinct
seven-metric pairs completed. No training, tokenizer or TX probability rerun.

Original N1024 results and all failure lineage remain intact. Reviewed final
PDF/editableSVG/600dpiPNG are figures_v2; the first title-clipped export is retained.
See results/arpc_n2048_validation_20261019_v1/README.md and delivery/REPORT_zh.md.
No additional scientific stage is launched; restoration backups remain deferred.

'''
    before=[];paths=[]
    for root in [E,O]:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                assert not p.is_symlink() and p.stat().st_size<=10000000
                paths.append(p.relative_to(R).as_posix())
    for name in ['README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md']:
        p=R/name;raw=p.read_bytes();assert b'N2048 two-method full-representation validation completed' not in raw
        before.append(pin(p));p.write_bytes(note.encode()+raw);paths.append(name)
    put(A/'before_docs.json',before)
    subprocess.run(['git','add','--sparse','--',*paths],cwd=R,check=True)
    check('manifest_update',[str(PY),'-B','tools/update_repository_manifest.py'])
    subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True);paths.append('release_manifest.json')
    assert set(git('diff','--cached','--name-only').splitlines())==set(paths)
    put(A/'exact_paths.json',paths)
    checks={}
    try:
        checks['diff_check']=check('diff_check',['git','diff','--cached','--check'])
    except AssertionError:
        w=read(A/'diff_check/actual_wait.json');assert w['returncode']==2
        output=(A/'diff_check/stdout').read_text()
        warnings=re.findall(r'^([^\r\n:]+):\d+: (trailing whitespace\.|new blank line at EOF\.)$',output,re.M)
        assert warnings and len(warnings)==len(re.findall(r'^.+?:\d+: ',output,re.M))
        exact_crlf=(E/'inputs/asset_audit.json').relative_to(R).as_posix()
        allowed_svg={str((O/v/'n2048_full_comparison.svg').relative_to(R)) for v in ['figures_v1','figures_v2']}
        copied={str(Path(q['copy']['path']).relative_to(R)) for q in read(O/'closed_receipts_manifest.json')['copies']}
        flagged={p for p,why in warnings}
        assert flagged <= {exact_crlf}|allowed_svg|copied,flagged-({exact_crlf}|allowed_svg|copied)
        configs={}
        for name in sorted(flagged):
            raw=(R/name).read_bytes();own=[why for p,why in warnings if p==name]
            if name==exact_crlf:
                assert b'\r\n'in raw and not any(x.endswith((b' ',b'\t'))for x in raw.splitlines())
                configs[name]='cr-at-eol'
            else:
                assert name in allowed_svg or name in copied
                configs[name]='-blank-at-eol'+(',-blank-at-eof' if 'new blank line at EOF.'in own else '')
        put(A/'exact_whitespace_diagnosis.json',dict(default_wait=pin(A/'diff_check/actual_wait.json'),warnings=warnings,per_command_exact_exceptions=configs,bytes_unchanged=[pin(R/n)for n in sorted(flagged)],persistent_configuration_changed=False))
        checks['diff_check']=check('diff_remaining',['git','diff','--cached','--check','--',*[x for x in paths if x not in flagged]])
        for number,(name,config) in enumerate(configs.items()):
            checks['diff_immutable_'+str(number)]=check('diff_immutable_'+str(number),['git','-c','core.whitespace='+config,'diff','--cached','--check','--',name])
    checks['verify_repository']=check('verify_repository',[str(PY),'-B','tools/verify_repository.py'])
    checks['run_cpu_checks']=check('run_cpu_checks',[str(PY),'-B','tools/run_cpu_checks.py'])
    put(A/'checks.json',checks)
    staged=[]
    for name in sorted(paths):
        b=subprocess.check_output(['git','show',':'+name],cwd=R)
        assert b==(R/name).read_bytes()
        staged.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    put(A/'staged_files.json',staged)
    record=dict(parent=PARENT,checks_sha256=sha(A/'checks.json'),staged_files_sha256=sha(A/'staged_files.json'))
    put(A/'checks_complete.json',dict(status='PASS',tree=git('write-tree'),**record))
    (A/'commit_message.txt').write_text('Complete fixed100 N2048 full-representation comparison\n\nDeliver symmetric two-subframe NeST-Com/ARPC results, seven metrics and direct source-paired intervals. Preserve all low-SNR failures and four explicitly approved repeated decoder calls; no training or TX probability reruns.\n')
    check('commit',['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-F',str(A/'commit_message.txt')])
    head=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==PARENT
    put(A/'commit_verified.json',dict(head=head,normal_commit=True,**record))
    print(json.dumps(dict(head=head,files=len(paths),push_pending=True)),flush=True)

if __name__=='__main__':main()
