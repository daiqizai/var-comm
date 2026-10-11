"""One-shot detached successor. Single lock, finite stages, fail closed, no retry."""
from pathlib import Path
import argparse,csv,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam')
ROOT=BASE/'code/liulu/code/VAR_COMM';RT=BASE/'code/liulu/var_comm_runtime_20261010'
RESULT=ROOT/'results/baseline_strength_20261011';OUT=RESULT/'automatic_followthrough_v1'
TMP=BASE/'liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011';LOG=TMP/'automatic_followthrough_v1'
HERE=Path(__file__).resolve().parent;PY=RT/'envs/frozen_visual_py310/bin/python'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    with q.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(q,p)
def pin(p):return dict(path=str(p),sha256=sha(p))
def verify(p):assert sha(p['path'])==p['sha256'],p['path']

def training_gate():
    selections={};bindings={}
    for branch,name,job in [('P','P_continue','P_train_gpu1_v2'),('Swin','Swin_1to19','Swin_train_v1')]:
        d=RESULT/name
        assert not (d/'failure.json').exists(),'Training failure'
        done=read(d/'completion.json');selected=read(d/'selected_checkpoint.json');history=read(d/'calibration_history.json')
        wait=read(TMP/'jobs'/job/'exit.json')
        assert done['status']=='TRAINING_AND_CALIBRATION_COMPLETE' and wait['exit_code']==0 and wait['actual_child_waited']
        assert selected['step']==min(history,key=lambda x:(x['objective'],x['step']))['step']
        assert not selected['holdout_read'] and selected['registration_sha256']==sha(d/'registration.json')
        assert selected['completed_step']<=({'P':100000,'Swin':200000}[branch])
        verify(selected['checkpoint']);assert sha(selected['csv'])==selected['sha256']
        for f in ['completion.json','selected_checkpoint.json','registration.json','calibration_history.json','stopping_checks.json']:bindings[str(d/f)]=sha(d/f)
        bindings[str(TMP/'jobs'/job/'exit.json')]=sha(TMP/'jobs'/job/'exit.json')
        selections[branch]=selected
    return selections,bindings

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'registration.json').exists(),'Already registered'
    selected,training_bindings=training_gate()
    checkpoints={};milestones=[];final=[]
    for branch,name,steps in [('P','P_continue',[40000,60000,80000]),('Swin','Swin_1to19',[80000,120000,160000])]:
        history=read(RESULT/name/'calibration_history.json');steps=sorted(set(steps+[selected[branch]['step'],selected[branch]['completed_step']]))
        for step in steps:
            cp=next(x['checkpoint'] for x in history if x['step']==step);verify(cp);key=branch+str(step)
            checkpoints[key]=dict(cp,step=step)
            milestones.append(dict(name=key,branch=branch,checkpoint=key,seeds=[4101,4102,4103] if branch=='P' else [4101]))
        key=branch+str(selected[branch]['step']);final.append(dict(name=key,branch=branch,checkpoint=key))
    # Existing100-source diagnostic: reuse exact matching cells only.
    interim={}
    for name,old in [('P40000','P40k'),('P80000','P80k'),('Swin80000','Swin80k')]:
        interim[name]=pin(RESULT/'interim_cal100_v1'/f'{old}_per_frame.csv')
    assert read(RESULT/'interim_cal100_v1/validation.json')['status'].startswith('PASS')
    pfinal=final[0]['name']
    bottleneck=[dict(name='F_direct_Dc_free_latent_reference',branch='P',checkpoint=pfinal,mode='direct_F',snrs=[19],seeds=[0])]
    for key in ['P40000',pfinal]:
        bottleneck.extend([dict(name=key+'_zero_noise',branch='P',checkpoint=key,mode='zero_noise',snrs=[19],seeds=[0]),dict(name=key+'_19dB',branch='P',checkpoint=key,reuse_method=key,snrs=[19],seeds=[4101,4102,4103])])
    common=TMP/'common500_original_sources';receipt=read(TMP/'common500_restore_verified.json')
    assert sha(common/'manifest.json')==receipt['manifest_sha256']
    manifest=read(common/'manifest.json');assert manifest['source_count']==500
    records=[]
    for i,row in enumerate(manifest['records']):
        assert row['source_index']==i and row['source_id']==manifest['source_ids'][i]
        p=common/'assets'/f'{i:04d}.npz';f=common/'F_latents'/f'{i:04d}.npz'
        assert sha(p)==row['archive_sha256'] and sha(f)==row['F_archive']['sha256']
        records.append(dict(source_id=row['source_id'],source_index=i,pixels=pin(p),latent=pin(f),preprocessing_id=row['preprocessing_id']))
    populations={'common500':dict(source_count=500,manifest=pin(common/'manifest.json'),records=records)}
    closure=RT/'qualification/h800_ep_new100_source_v1_attempt1/registered/run/completion.json';closed=read(closure)
    assert closed['actual_wait']['success'] and closed['worker_exit_codes']==[0]
    verify(closed['worker_completion']);source=read(closed['worker_completion']['path']);assert source['source_count']==100
    populations['new100']=dict(source_count=100,closure=pin(closure),records=[])
    for i,row in enumerate(source['sources']):
        assert row['source_index']==i;verify(row['source_assets'])
        populations['new100']['records'].append(dict(source_id=row['source_id'],source_index=i,pixels=row['source_assets'],preprocessing_id=row['preprocessing_id']))
    config=read(RT/'qualification/h800_ep_new100_metrics_v3_attempt1/registered/request.json')['metric_config']
    config={k:v for k,v in config.items() if k in ['sources','manifest','registration','weight_receipt','dino_receipt','whole_policy']}
    files=list(HERE.glob('*.py'))+list((ROOT/'experiments/baseline-strength-20261011').glob('*.py'))
    files+=[ROOT/'experiments/entropy-partial-mismatch-20261010/scripts/h800_ep_pilot_metrics_v1.py']
    files+=list((ROOT/'experiments/external-comparison-20261004').glob('swin_*.py'))
    reg=dict(schema='BASELINE_AUTOMATIC_FOLLOWTHROUGH_V1',created_unix=time.time(),gpu=0,allocator_cap_bytes=16<<30,
             selections=selected,training_bindings=training_bindings,checkpoints=checkpoints,milestone_methods=milestones,
             final_methods=final,bottleneck_methods=bottleneck,populations=populations,interim_reuse=interim,
             metric_config=config,code_bindings={str(p):sha(p) for p in files},stage_order=['milestones','bottleneck','common500','new100'],
             stage_frames={'milestones':96000,'bottleneck':900,'common500':18000,'new100':1800},
             stage_timeout_seconds=12*3600,resource_wait_seconds=6*3600,
             common500_noise_seeds=[2001,2002,2003],new100_noise_seeds=[9301,9302,9303],
             new_final_method_frames=19800,new_training_updates=0,bootstrap_calls=0,automatic_retries=False,
             statistical_intervals='PENDING_SEPARATE_ORIGINAL_BOOTSTRAP_ADMISSION',timing='PENDING_EXCLUSIVE_WINDOW',
             population_note='Subsequent frozen-baseline same-source supplement; neither population is newly blind for this study.',
             bottleneck_note='Full F is a free-latent reference, not an equal-budget upper bound. Zero-noise paths retain real N1024 bottleneck and receiver SNR19.',
             old_Swin19dB_OOD_preserved=True,new_Swin_optimizer_reset_disclosed=True)
    write(OUT/'registration.json',reg)
    print(json.dumps({'registered':str(OUT/'registration.json'),'selected_steps':{k:v['step'] for k,v in selected.items()},'frames':reg['stage_frames']}))

def run():
    lock=(OUT/'chain.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists(),'Already attempted; no automatic retry'
    reg=read(OUT/'registration.json');LOG.mkdir(parents=True,exist_ok=True)
    for p,digest in reg['code_bindings'].items():assert sha(p)==digest,p
    selections,bindings=training_gate();assert bindings==reg['training_bindings'] and selections==reg['selections']
    write(OUT/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),registration_sha256=sha(OUT/'registration.json')))
    os.nice(10);os.sched_setaffinity(0,{25,29,31,35,39,43})
    env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',BASELINE_GPU_ID='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6')
    for key,name in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('TORCH_HOME','torch'),('HF_HOME','huggingface'),('MPLCONFIGDIR','matplotlib')]:
        p=TMP/name;p.mkdir(exist_ok=True);env[key]=str(p)
    children=[]
    for stage in reg['stage_order']:
        wait_started=time.monotonic()
        while True:
            assert not (OUT/'STOP').exists(),'User stop requested'
            assert time.monotonic()-wait_started<reg['resource_wait_seconds'],'Resource wait expired; no work launched'
            free,temp=map(int,subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            if free>=20480 and temp<80:break
            write(OUT/'status.json',dict(status='WAITING_FOR_RESOURCES',stage=stage,pid=os.getpid(),free_MiB=free,temperature=temp));time.sleep(30)
        for p,digest in reg['code_bindings'].items():assert sha(p)==digest,p
        with (LOG/f'{stage}.log').open('xb') as log:
            child=subprocess.Popen([str(PY),str(HERE/'evaluate_stage.py'),stage],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            write(OUT/'status.json',dict(status='RUNNING',stage=stage,owner_pid=os.getpid(),worker_pid=child.pid,updated_unix=time.time()))
            started=time.monotonic();reason=None
            while child.poll() is None:
                if (OUT/'STOP').exists() or time.monotonic()-started>reg['stage_timeout_seconds']+300:
                    reason='STOP_OR_STAGE_TIMEOUT';child.terminate();break
                time.sleep(5)
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();code=child.wait()
        waited=dict(stage=stage,pid=child.pid,exit_code=code,actual_child_waited=True,stop_reason=reason,seconds=time.monotonic()-started)
        write(OUT/f'{stage}_actual_wait.json',waited);children.append(waited)
        assert code==0 and reason is None,'Stage failed; successors not launched'
        done=read(OUT/stage/'completion.json');assert done['frames']==reg['stage_frames'][stage] and done['status']=='COMPLETE_FROZEN_FRAME_SCORING'
        for p,digest in done['outputs'].items():assert sha(p)==digest,p
    done=dict(status='ALL_DIAGNOSTIC_AND_FROZEN_FRAME_SCORING_COMPLETE',actual_waits=children,final_method_frames=19800,statistics='PENDING',timing='PENDING',training_updates=0)
    write(OUT/'completion.json',done);write(OUT/'status.json',done)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run']);a=parser.parse_args()
    try:prepare() if a.command=='prepare' else run()
    except BaseException:
        if a.command=='run':write(OUT/'failure.json',dict(status='STOPPED_NO_RETRY',traceback=traceback.format_exc(),time=time.time()))
        raise
