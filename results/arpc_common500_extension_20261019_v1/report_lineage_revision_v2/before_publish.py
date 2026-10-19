"""Normal reviewed light-result publication after actual scientific/visual closure."""
import fcntl,subprocess,time
from common import *
import a1_owner as owner
A=L/'publication_v1'
PARENT='02ee120324e3f2fa16b2b05b602346db34256586'

def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True).strip()

def check(name,argv):
    d=A/name;d.mkdir();owner.RUN=d
    owner.run_stage('child',argv,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),1800)
    return desc(d/'child.actual_wait.json')

def main():
    os.chdir(R)
    lock=(PL/'arpc_science.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert git('rev-parse','HEAD')==PARENT and git('branch','--show-current')=='main'
    assert not git('status','--porcelain','--untracked-files=no') and not git('diff','--cached','--name-only')
    final=read(O/'final_report_v1/completion.json');assert final['status']=='SAVED_RESULTS_REPORTED_WITH_DISCLOSED_ENGINEERING_LIMITATION'and final['source_count']==500 and not final['uncompleted_scientific_items']
    for d in final['inputs']+final['outputs']:verify(d)
    reportcontrol=L/'report_control_v1';check_wait(reportcontrol/'child.actual_wait.json');assert gone(read(reportcontrol/'supervisor.json')['process'])
    assert not A.exists();A.mkdir()
    put(A/'supervisor.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'source':desc(__file__),'started_unix':time.time()})
    actual=read(O/'delivery_v1/actual_compute_and_reuse.json')
    note=f'''## 2026-10-20 ARPC N1024 common500 extension delivered

The fixed ARPC external100 test is extended to the original common500, retaining
original source indices and frozen 4/10/19 dB policies. All4500 logical events
are represented (1500/SNR):900 verified existing plus3600 additional events.
New400 captions/tokenizations/probability traversals produced2800 endpoint streams;
no TX clean rendering, recalibration, training or other baseline inference.
New link work used1200 encoded waveforms and3600 header+3600 body decodes.
Actual RX produced{actual['new_rx_image_renders']} new images; seven-metric scoring
covered{actual['new_metric_pairs']} unique new pairs with all failures retained.
All105 five-method mean/CI cells and21 direct source-paired NeST-minus-ARPC
intervals use the existing10000x500 index bank, with zero new random draws.
The original four-method main means are unchanged within1e-10; NeST19dB uses
the current T1-C m10-corrected PARTIAL version. Historical noise realizations
differ across baselines: pairing is by source, not common noise across all five.

The original PHY RX exited1 after saving all3600 outputs because its final
denied-open assertion failed. Exact denied paths were lost from its memory-only
list. The original enforcing hook and failure remain unchanged; a separately
recorded saved-output audit completed without replaying a decode/model call.
This attempt-telemetry limitation is disclosed, never labeled originalexit0.
Native RX and scoring have actual exit0 and post-exit audits. The admitted
PRO6000 host/runtime and resource selection, including Python build difference,
are recorded; no cross-hardware timing claim is made.

See results/arpc_common500_extension_20261019_v1/final_report_v1/REPORT_zh.md,
delivery_v1/ for all seven metrics and original100-to500 comparisons, and
figures_v1/ for the actually reviewed five-method2x2 vector Fig.2, plotCSV,
English caption and outline changes. The full manuscript is not rewritten.
N2048 remains the completed100-source scope; prior ablations/timing are retained.
Backups remain deferred. The normal commit/push is evidenced by publication_v1;
this note does not pre-fill an unobserved remote SHA.

---

'''
    paths=[]
    allowed={'.py','.md','.json','.csv','.txt','.pdf','.svg','.png','.stdout','.stderr'}
    for root in (E,O):
        for p in sorted(root.rglob('*')):
            if p.is_file():
                assert not p.is_symlink()and p.stat().st_size<=10000000 and p.suffix in allowed,p
                paths.append(p.relative_to(R).as_posix())
    before=[]
    for name in ('README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md'):
        p=R/name;raw=p.read_bytes();assert b'ARPC N1024 common500 extension delivered'not in raw
        before.append(desc(p));p.write_bytes(note.encode()+raw);paths.append(name)
    put(A/'before_docs.json',before)
    subprocess.run(['git','add','--sparse','--',*paths],cwd=R,check=True)
    check('manifest_update',[str(PY),'-B','tools/update_repository_manifest.py'])
    subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True);paths.append('release_manifest.json')
    assert set(git('diff','--cached','--name-only').splitlines())==set(paths)
    put(A/'exact_paths.json',paths)
    checks={'diff_check':check('diff_check',['git','diff','--cached','--check'])}
    checks['verify_repository']=check('verify_repository',[str(PY),'-B','tools/verify_repository.py'])
    checks['run_cpu_checks']=check('run_cpu_checks',[str(PY),'-B','tools/run_cpu_checks.py'])
    put(A/'checks.json',checks)
    staged=[]
    for name in sorted(paths):
        b=subprocess.check_output(['git','show',':'+name],cwd=R);assert b==(R/name).read_bytes()
        staged.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    put(A/'staged_files.json',staged)
    record={'parent':PARENT,'checks_sha256':desc(A/'checks.json')['sha256'],'staged_files_sha256':desc(A/'staged_files.json')['sha256']}
    put(A/'checks_complete.json',dict(status='PASS',tree=git('write-tree'),**record))
    (A/'commit_message.txt').write_text('Extend ARPC N1024 evaluation to common500\n\nPreserve frozen policies and original100, add remaining400, reuse existing baselines and the common500 bootstrap bank. Deliver seven metrics, direct source-paired intervals and five-method main figure. Disclose preserved PHY exit1 and missing denied-path attempt telemetry; no successful scientific replay.\n')
    check('commit',['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-F',str(A/'commit_message.txt')])
    head=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==PARENT
    put(A/'commit_verified.json',dict(head=head,normal_commit=True,**record))
    print(json.dumps({'head':head,'files':len(paths),'push_pending':True}))

if __name__=='__main__':main()
