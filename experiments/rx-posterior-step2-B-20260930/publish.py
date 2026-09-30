"""Publish the completed authorized B run after repository and result checks."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/RX-POSTERIOR-STEP2-B-20260930-R1'
RESULT=ROOT/'results/rx_posterior_step2_B_20260930_R1'
RECEIPT=OUT/'publication.json'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');os.replace(temporary,path)
def command(args,log):
    subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    log.flush()


def main():
    if RECEIPT.exists() and read(RECEIPT).get('status')=='PUSHED':return
    training=read(OUT/'training/completion.json');evaluation=read(OUT/'evaluation/completion.json')
    config=read(RESULT/'config.json')
    if training['development_read'] or not evaluation['frozen_weights_unchanged']:
        raise RuntimeError('scientific completion boundary failed')
    for path,expected in config['source_bindings'].items():
        if sha(path)!=expected:raise RuntimeError('executed source changed '+path)
    provenance=RESULT/'provenance';provenance.mkdir(exist_ok=True)
    original={str(path.relative_to(RESULT)):sha(path) for path in RESULT.rglob('*')
        if path.is_file() and 'provenance' not in path.relative_to(RESULT).parts and 'table_shards' not in path.relative_to(RESULT).parts}
    write(provenance/'original_result_sha256.json',original)
    for folder,name in [(OUT/'training','training'),(OUT/'evaluation','evaluation')]:
        for path in folder.glob('*.json'):
            if path.name in ('status.json','latest.json'):continue
            target=provenance/name/path.name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=path.read_bytes():raise RuntimeError('provenance differs '+str(target))
            shutil.copyfile(path,target)
    for path in (OUT/'training/calibration').glob('*.csv'):
        target=RESULT/'training_calibration'/path.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
        shutil.copyfile(path.with_suffix('.json'),target.with_suffix('.json'))
    from tables import export
    export(RESULT)
    selected=training['selected']['P_low'];state=training['state'];decisions=read(RESULT/'decisions.json')
    records=decisions.get('records',decisions.get('decisions',decisions)) if isinstance(decisions,dict) else decisions
    if isinstance(records,dict):records=list(records.values())
    labels=', '.join(f"{r['snr_db']}dB {r['label']}" for r in records if r['snr_db'] in (-5,-2,1))
    note=(f"## Step2 B completed 2026-09-30\n\n"
          f"The authorized single P_low arm continued selected27500 with the original populated AdamW, order and RNG. "
          f"It completed {state['step']} added updates and selected added step {selected['step']} only by full1000 calibration utility. "
          f"The frozen evaluation then recalibrated receiver policies on200 calibration sources and evaluated100 development sources at -5/-2/1/4/13dB. "
          f"G2: {labels}. The13dB result is an out-of-training-range side-effect diagnostic.\n\n"
          "[Report and full tables](results/rx_posterior_step2_B_20260930_R1/README.md). "
          "The development set is reused; this is not a new holdout or a training-seed replication.\n\n")
    for filename in ('RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md'):
        path=ROOT/filename;text=path.read_text()
        if '## Step2 B completed 2026-09-30' not in text:path.write_text(note+text)
    readme=RESULT/'README.md'
    readme.write_text(f"# Step2 B: selected P_low receiver study\n\n"
        f"Completed added updates: {state['step']}. Selected added step: {selected['step']} (global {selected['total_updates']}).\n\n"
        "- [Scientific report](report.md)\n- [Frozen configuration](config.json)\n- [Calibration-only training evidence](provenance/training/completion.json)\n"
        "- [Receiver calibration policy](selected_policy.json)\n- [G2 decisions](decisions.json)\n\n"
        "Training checkpoints, models and tensor caches remain outside Git. Full paired tables are stored as exact byte-preserving shards below10MB. Restore them with:\n\n"
        "```bash\npython3 experiments/rx-posterior-step2-B-20260930/tables.py restore\n```\n\n"
        "P_low receiver methods share one P_low observation. Comparisons with the old P4084 pair source/noise seed/N/E and retain each model's distinct waveform and observation hashes.\n")
    own=[str(HERE.relative_to(ROOT)),str(RESULT.relative_to(ROOT)),
         'reports/rx_posterior_step2_B_20260930_R1.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md','.gitignore']
    logpath=OUT/'publication_checks.log'
    with logpath.open('a') as log:
        command(['git','add','--',*own],log)
        # Refuse to commit any staged file outside this authorized study.
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
        if any(not any(p==prefix or p.startswith(prefix+'/') for prefix in own) for p in staged):
            raise RuntimeError('unrelated staged changes require a separate review')
        command([sys.executable,'tools/update_repository_manifest.py'],log)
        command(['git','add','release_manifest.json'],log)
        command([sys.executable,'tools/verify_repository.py'],log)
        command([sys.executable,'tools/run_cpu_checks.py'],log)
        command([sys.executable,str(HERE/'test_protocol.py')],log)
        command([sys.executable,str(HERE/'tables.py'),'verify'],log)
        command(['git','fetch','origin'],log)
        command(['git','merge-base','--is-ancestor','origin/main','HEAD'],log)
        # Check staged source bytes against actual execution hashes after all checks.
        for path,expected in config['source_bindings'].items():
            relative=str(Path(path).relative_to(ROOT))
            data=subprocess.check_output(['git','show',':'+relative],cwd=ROOT)
            if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('staged source binding differs '+relative)
        message=OUT/'commit_message.txt';message.write_text('Publish selected P_low Step2 B calibration and receiver results\n')
        if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).strip():
            command(['git','commit','-F',str(message)],log)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write(RECEIPT,dict(status='COMMITTED',commit=commit,checks='PASS',log=str(logpath),time=time.time()))
        command(['git','push','origin','main'],log)
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
        if remote!=commit:raise RuntimeError('remote commit SHA differs')
    write(RECEIPT,dict(status='PUSHED',commit=commit,remote_sha256=remote,checks='PASS',log=str(logpath),time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        write(OUT/f'publication_failure_{time.time_ns()}.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()))
        raise
