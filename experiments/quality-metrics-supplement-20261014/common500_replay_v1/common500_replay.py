"""Authorized missing-pixel replay: frozen P100k/Swin200k only, 9000 frames."""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,sqlite3,subprocess,sys,time,traceback
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/quality-metrics-supplement-20261014/common500_replay_v1'
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1')
O=R/'results/quality_metrics_supplement_20261014_v1/common500_replay_v1';OLD=R/'results/baseline_strength_20261011/automatic_followthrough_v1'
REG=O/'registration.json';RUN=T/'common500_replay_v1';METHODS=('P100000','Swin200000');SNRS=(4,10,19);SEEDS=(2001,2002,2003)
EXPECTED_REG='20562b16cdc6c8807c30fdde523f50c80f34c3918dbd8145b1647d47e5d867ee'
def require(v,msg):
    if not v:raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def pin(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def checked(p):
    q=Path(p['path']);require(q.is_absolute() and not q.is_symlink() and sha(q)==p['sha256'],'Input binding changed: '+str(q));return q
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def atomic(p,v):
    temp=Path(p).with_suffix('.next')
    with temp.open('x') as f:json.dump(v,f,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)
def ash(x):
    import numpy as np
    x=np.ascontiguousarray(x);return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def roster(rows):
    groups={}
    for row in rows:
        key=(row['method'],int(row['source_index']),int(row['snr_db']),int(row['noise_seed']))
        require(key not in groups and key[0] in METHODS and key[2] in SNRS and key[3] in SEEDS and int(row['N'])==1024,'Unexpected/repeated frame')
        groups[key]=row
    require(set(groups)=={(m,i,s,n) for m in METHODS for i in range(500) for s in SNRS for n in SEEDS},'Exactly9000 frozen frames required')
    return groups
def register():
    require(not O.exists() and not RUN.exists(),'No duplicate registration/attempt')
    require(sha(OLD/'registration.json')==EXPECTED_REG,'Original formal registration differs')
    old=read(OLD/'registration.json');done=read(OLD/'common500/completion.json');wait=read(OLD/'common500_actual_wait.json')
    require(done['frames']==18000 and done['source_count']==500 and wait['exit_code']==0 and wait['actual_child_waited'] is True,'Original actual common500 closure')
    frames=[]
    for method in METHODS:
        p=OLD/'common500'/f'{method}.csv';require(sha(p)==done['outputs'][str(p)],'Old row bytes differ')
        with p.open() as f:
            frames += [r for r in csv.DictReader(f) if int(r['snr_db']) in SNRS]
    keyed=roster(frames);sources=old['populations']['common500']['records'];require(len(sources)==500,'Fixed common500 required')
    import numpy as np
    for i,src in enumerate(sources):
        require(src['source_index']==i,'Original source order required')
        with np.load(checked(src['pixels']),allow_pickle=False) as z:x=z['pixels'].copy()
        require(x.dtype==np.uint8 and x.shape==(3,256,256) and hashlib.sha256(x.tobytes()).hexdigest()==src['preprocessing_id'],'Original pixels differ')
        with np.load(checked(src['latent']),allow_pickle=False) as z:F=z['F'];require(F.dtype==np.float32 and F.shape==(32,16,16),'Original F differs')
        target=ash(x.astype(np.float32)/np.float32(255))
        for method in METHODS:
            for snr in SNRS:
                for seed in SEEDS:
                    row=keyed[method,i,snr,seed];require(row['source_id']==src['source_id'] and row['reference_sha256']==target,'Old target identity differs')
    for p,h in old['code_bindings'].items():require(sha(p)==h,'Original source changed: '+p)
    for method in METHODS:checked(old['checkpoints'][method])
    metric_identity=read(OLD/'common500/metric_identity.json');numeric=metric_identity['metadata']['numeric_flags']
    require(numeric==dict(matmul_tf32=False,cudnn_tf32=False,precision='highest',cudnn_benchmark=False,cudnn_deterministic=True,deterministic=True,threads=6,interop_threads=2),'Original numeric flags differ')
    O.mkdir();value=dict(schema='AUTHORIZED_COMMON500_MISSING_PIXEL_REPLAY_V1',user_authorization='User said 允许 after explicit frozen9000-frame channel replay and per-frame SHA comparison proposal',
        methods=list(METHODS),SNRS=list(SNRS),noise_seeds=list(SEEDS),N=1024,sources=sources,frames=frames,source_count=500,logical_frames=9000,
        old_registration=pin(OLD/'registration.json'),old_completion=pin(OLD/'common500/completion.json'),old_actual_wait=pin(OLD/'common500_actual_wait.json'),
        old_csv={m:pin(OLD/'common500'/f'{m}.csv') for m in METHODS},checkpoints={m:old['checkpoints'][m] for m in METHODS},code_bindings=old['code_bindings'],
        numeric_flags=numeric,source=pin(Path(__file__).resolve()),private_dependency_overlay=pin(T/'dependency_overlay_v1/manifest.json'),
        resources=dict(physical_gpu=0,allocator_GiB=16,startup_free_MiB=8192,runtime_free_MiB=2048,CPU_affinity=[25,29,31,35,39,43],nice=10,max_seconds=7200),
        P_execution='Original source-batch1 TX/RX; per-source decoder batch3 of original three noisy latents',Swin_execution='Original source-batch1 encode/paid-header/channel/RX/gray-failure path',
        new_training=0,reselection=False,old_metrics_recomputed=0,extra_snrs_or_samples=0,unknown_failure_auto_retry=False,mismatch='Save and report every mismatch, never relabel as original')
    save(REG,value);print(json.dumps(pin(REG)),flush=True)
class Ledger:
    caps={'load_P':1,'load_Swin':1,'P_transmit':1500,'P_receive':4500,'P_decoder_batch3':1500,'Swin_physical_frame':4500}
    def __init__(self,p):self.db=sqlite3.connect(p);self.db.execute('CREATE TABLE calls(id INTEGER PRIMARY KEY,kind TEXT,identity TEXT,state TEXT)');self.db.commit()
    def call(self,kind,identity,fn):
        n=self.db.execute('SELECT count(*) FROM calls WHERE kind=?',(kind,)).fetchone()[0];require(n<self.caps[kind],'Replay call cap: '+kind)
        cur=self.db.execute('INSERT INTO calls(kind,identity,state) VALUES(?,?,?)',(kind,identity,'RESERVED'));self.db.commit()
        try:v=fn()
        except BaseException:self.db.execute('UPDATE calls SET state=? WHERE id=?',('FAILED',cur.lastrowid));self.db.commit();raise
        self.db.execute('UPDATE calls SET state=? WHERE id=?',('COMPLETE',cur.lastrowid));self.db.commit();return v
    def summary(self):
        rows=self.db.execute('SELECT kind,state,count(*) FROM calls GROUP BY kind,state').fetchall();return dict(rows=rows,unresolved=sum(n for _,s,n in rows if s!='COMPLETE'))
def run(expected):
    require(sha(REG)==expected,'Registration SHA differs');r=read(REG);checked(r['source']);checked(r['old_registration']);checked(r['private_dependency_overlay'])
    require(os.environ['CUDA_VISIBLE_DEVICES']=='0' and os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','Registered device and numerical environment required')
    RUN.mkdir(exist_ok=False);cache=RUN/'runtime_cache'
    for n in ('tmp','inductor','triton','torch'):(cache/n).mkdir(parents=True)
    import tempfile
    os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),TORCH_HOME=str(cache/'torch'));tempfile.tempdir=str(cache/'tmp')
    import fcntl
    lock=(RUN/'owner.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old_lock=(OLD/'chain.lock').open('rb');fcntl.flock(old_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.nice(10);os.sched_setaffinity(0,set(r['resources']['CPU_affinity']));start=time.monotonic();ledger=Ledger(RUN/'calls.sqlite');rows=[];mismatches=[]
    def guard(startup=False):
        require(not (T/'common500_replay_STOP').exists() and time.monotonic()-start<7200,'STOP or finite replay window')
        free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());require(free>=(8192 if startup else 2048),'Private resource admission/floor')
    try:
        guard(True);sys.path[:0]=[str(T/'dependency_overlay_v1/sitepackages'),str(R/'experiments/baseline-strength-20261011')]
        import numpy as np,torch
        import baseline_runtime as b,run_continue as training
        runtime=b.configure();require(runtime['threads']==6 and runtime['interop_threads']==2 and runtime['FP32'] and not runtime['TF32'],'Actual original numerics required')
        save(RUN/'numerical_runtime.json',runtime)
        # Construction only; none of the old training/probe/metric entrypoints run.
        for name in ('train','qualification','optimizer','p_update','calibrate'):
            def forbid(*a,_name=name,**kw):raise RuntimeError('Forbidden old training/probe entry: '+_name)
            setattr(training,name,forbid)
        expected_rows=roster(r['frames']);state=None
        def emit(method,i,snr,seed,image,link):
            old=expected_rows[method,i,snr,seed];require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and image.min()>=0 and image.max()<=1,'Original native RGB contract')
            actual=ash(image);same=actual==old['reconstruction_sha256'];header_same=str(bool(link['header_accepted']))==old['header_accepted']
            row=dict(method=method,source_index=i,source_id=r['sources'][i]['source_id'],N=1024,snr_db=snr,noise_seed=seed,old_reconstruction_sha256=old['reconstruction_sha256'],actual_reconstruction_sha256=actual,pixel_sha_match=same,header_match=header_same,**link)
            target=RUN/'images'/method/f'{i:04d}_{snr}_{seed}.npz';target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:np.savez_compressed(f,reconstruction=image);f.flush();os.fsync(f.fileno())
            row['archive']=pin(target);rows.append(row)
            if not same or not header_same:mismatches.append(row)
        with torch.inference_mode():
            for method in METHODS:
                branch='P' if method=='P100000' else 'Swin';guard();state=ledger.call('load_'+branch,method,lambda:training.load(branch))
                if branch=='P':del state['lp'];torch.cuda.empty_cache()
                else:state['sp']._PHY.ROOT=RUN/'native_build'
                for p,h in state['pins'].items():require(sha(p)==h,'Loaded model source differs')
                save(RUN/f'{method}_loaded_sources.json',state['pins'])
                cp=checked(r['checkpoints'][method]);payload=torch.load(cp,map_location='cpu',weights_only=True);state['model'].load_state_dict(payload['model'],strict=True);del payload
                state['model'].eval().requires_grad_(False)
                for snr in SNRS:
                    for i,source in enumerate(r['sources']):
                        guard();identity=f'{method}|{i}|{snr}'
                        if branch=='P':
                            with np.load(checked(source['latent']),allow_pickle=False) as z:F=z['F'].copy()
                            wave=ledger.call('P_transmit',identity,lambda:state['model'].transmit(torch.from_numpy(F[None]).cuda())[0].cpu().numpy())
                            require(wave.shape==(1024,2) and wave.dtype==np.float32,'Original P waveform shape/dtype');energy=float(np.square(wave.astype(np.float64)).sum());require(abs(energy-2048)<.02,'Original Es2 energy')
                            zs=[]
                            for seed in SEEDS:
                                noise=state['ns']['seeded_noise']('VAR-CONTINUOUS-1024|'+source['source_id'],seed,(1024,2)).astype(np.float32)
                                y=wave+noise*np.float32(10.**(-float(snr)/20))
                                zs.append(ledger.call('P_receive',identity+'|'+str(seed),lambda y=y:state['model'].receive(torch.as_tensor(y[None],device='cuda:0'),torch.tensor([snr],dtype=torch.float32,device='cuda:0'))[0]))
                            images=ledger.call('P_decoder_batch3',identity,lambda:state['dc'](torch.stack(zs)).cpu().numpy())
                            for j,seed in enumerate(SEEDS):emit(method,i,snr,seed,images[j],dict(header_accepted=True,header_crc_accepted=True,header_fields_legal=True,actual_energy=energy,fallback='none'))
                        else:
                            with np.load(checked(source['pixels']),allow_pickle=False) as z:pixels=z['pixels'].copy()
                            x=torch.from_numpy(pixels[None]).to(device='cuda:0',dtype=torch.float32)/255
                            for seed in SEEDS:
                                noise=np.stack([state['sp'].standard_noise(source['source_id'],seed,1024,snr)])
                                images,links=ledger.call('Swin_physical_frame',identity+'|'+str(seed),lambda noise=noise:state['sr'].physical_batch(state['model'],x,1024,snr,noise))
                                emit(method,i,snr,seed,images[0].cpu().numpy(),links[0])
                        if (i+1)%25==0:
                            atomic(RUN/'progress.json',dict(method=method,snr_db=snr,sources_completed=i+1,logical_frames=len(rows),mismatches=len(mismatches),seconds=time.monotonic()-start));print('SAVED',method,snr,i+1,len(rows),'mismatches',len(mismatches),flush=True)
                del state;state=None;torch.cuda.empty_cache()
        require(len(rows)==9000 and ledger.summary()['unresolved']==0,'Incomplete replay')
        for p,h in r['code_bindings'].items():require(sha(p)==h,'Original source changed during replay')
        for p in r['old_csv'].values():checked(p)
        for p in r['checkpoints'].values():checked(p)
        save(RUN/'frame_index.json',rows);save(RUN/'mismatches.json',mismatches);counts=ledger.summary();ledger.db.close()
        save(RUN/'completion.json',dict(status='COMPLETE_EXACT_ORIGINAL_PIXEL_REPLAY' if not mismatches else 'COMPLETE_WITH_PIXEL_MISMATCHES_NOT_ORIGINAL',logical_frames=9000,methods=list(METHODS),mismatch_count=len(mismatches),old_hash_matched=len(rows)-len(mismatches),counts=counts,frame_index=pin(RUN/'frame_index.json'),mismatches=pin(RUN/'mismatches.json'),ledger=pin(RUN/'calls.sqlite'),registration=pin(REG),seconds=time.monotonic()-start,new_training=0,old_metric_calls=0,new_model_selection=False,score_admitted=not mismatches))
    except BaseException:
        save(RUN/'failure.json',dict(status='FAILED_PRESERVE_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc(),logical_frames=len(rows),mismatches=len(mismatches),counts=ledger.summary()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=('register','run'));p.add_argument('--expected-registration');a=p.parse_args();register() if a.role=='register' else run(a.expected_registration)
