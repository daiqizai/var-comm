"""New metrics on the fixed, already saved final-link development16 only."""
from pathlib import Path
import argparse, collections, csv, hashlib, importlib.util, json, math, os, sqlite3, subprocess, sys, time, traceback

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
TASK=B/'quality_metrics_supplement_20261014_v1'
OUT=R/'results/quality_metrics_supplement_20261014_v1/hifi_fixed16'
OLD=R/'results/hifi_final_fixed16_20261012'
LOADER=R/'experiments/unified-metrics-20261002/metric_models.py'
LOADER_SHA='dbc6f6c78c92ffd8a6e914d79008bbee8fc8b485a776884de45c6bdce1703066'
METHODS=('HiFiSwin200000','Swin200000','P100000')
METRICS=('dists','dreamsim','ms_ssim')
INDICES=(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
REG=OUT/'dependency_recovery_registration.json'
RUN=TASK/'hifi_score_dependency_recovery_v1'

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def checked(p):
    q=Path(p['path']);require(q.is_absolute() and '..' not in q.parts and not q.is_symlink() and q.is_file(),'Canonical ordinary input required')
    require(sha(q)==p['sha256'] and ('bytes' not in p or q.stat().st_size==p['bytes']),'Bound input bytes changed: '+str(q));return q
def ash(a):
    import numpy as np
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def csvsave(p,rows):
    with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def validate_grid(rows,sources):
    require(len(rows)==48 and len(sources)==16,'All48 saved method/source outputs and16 references required')
    require(tuple(s['source_index'] for s in sources)==INDICES and len({s['source_id'] for s in sources})==16,'Original fixed development16 order required')
    ids={s['source_index']:s for s in sources};keyed={}
    for row in rows:
        key=row['method'],row['source_index'];require(key not in keyed and key[0] in METHODS and key[1] in ids,'Duplicate/unknown output')
        require(row['source_id']==ids[key[1]]['source_id'] and row['source_pixels_sha256']==ids[key[1]]['preprocessing_id'],'Reference identity differs')
        require(row['N']==1024 and row['snr_db']==13 and row['noise_seed']==2001,'Fixed workpoint required; no19dB substitution')
        keyed[key]=row
    require(set(keyed)=={(m,i) for m in METHODS for i in INDICES},'Incomplete output roster')
    for i in INDICES:
        h,s=keyed['HiFiSwin200000',i],keyed['Swin200000',i]
        require(h['paired_Swin_archive']==s['archive'] and h['observation_sha256']==s['observation_sha256'] and h['checkpoint_sha256']==s['checkpoint_sha256'],'Actual same-observation final Swin/HiFi binding differs')
    return keyed
def load_pixels(source,row):
    import numpy as np
    with np.load(checked(source['archive']),allow_pickle=False) as z:x=z['pixels'].copy()
    require(x.dtype==np.uint8 and x.shape==(3,256,256) and hashlib.sha256(x.tobytes()).hexdigest()==source['preprocessing_id'],'Source native pixels differ')
    x=x.astype(np.float32)/np.float32(255)
    with np.load(checked(row['archive']),allow_pickle=False) as z:y=z['reconstruction'].copy()
    require(y.dtype==np.float32 and y.shape==(3,256,256) and np.isfinite(y).all() and y.min()>=0 and y.max()<=1,'Native reconstruction RGB contract differs')
    require(ash(y)==row['reconstruction_sha256'],'Saved original reconstruction hash differs')
    return x,y
def register():
    require(not REG.exists() and not RUN.exists(),'Existing registration/attempt must not be repeated')
    old=read(OLD/'registration.json');rows=read(OLD/'receipts.json');done=read(OLD/'completion.json')
    require(done['status']=='COMPLETE_FIXED16_FINAL_HIFI' and done['rows']==48 and done['HiFi_recoveries']==16 and done['metric_calls']==0,'Actual original16 HiFi closure required')
    require(read(OLD/'actual_wait.json')['returncode']==0 and sha(OLD/'receipts.json')==done['receipts_sha256'],'Actual original exit/receipts mismatch')
    keyed=validate_grid(rows,old['sources']);input_pixels=[]
    for source in old['sources']:
        for method in METHODS:
            row=keyed[method,source['source_index']];x,y=load_pixels(source,row)
            input_pixels.append(dict(method=method,source_index=source['source_index'],source_id=source['source_id'],
                reference=pin(checked(source['archive'])),reconstruction=pin(checked(row['archive'])),reference_rgb_sha256=ash(x),reconstruction_rgb_sha256=ash(y)))
    require(sha(LOADER)==LOADER_SHA,'Original official-loader source differs')
    assets=read(TASK/'metric_assets_posix_manifest.json');require(set(assets['models'])==set(METRICS) and assets['all_file_SHA_PASS'] is True,'Complete immutable metric assets required')
    protocol=dict(dists=dict(direction='lower',RGB='float32 original256x256, [0,1], no resize/crop/quantization',normalization='official internal ImageNet mean/std',batch_average=False),
        dreamsim=dict(direction='lower',branch='ensemble',release='v0.2.0-checkpoints',use_patch_model=False,normalize_embeds=True,
            preprocessing='official square224 bicubic antialiased tensor geometry; original RGB floats retained, no PIL requantization or crop',
            normalization='unchanged official per-backbone internal normalization'),
        ms_ssim=dict(direction='higher',RGB='float32 original256x256, [0,1], no resize/crop/quantization',data_range=1.0,size_average=False,win_size=11,win_sigma=1.5,
            weights=[0.0448,0.2856,0.3001,0.2363,0.1333],K=[0.01,0.03]))
    value=dict(schema='HIFI_FIXED16_SUPPLEMENTAL_THREE_METRICS_V1',methods=list(METHODS),metrics=list(METRICS),source_indices=list(INDICES),
        source_count=16,logical_pairs=48,logical_scalar_scores=144,HiFi_pairs=16,N=1024,snr_db=13,noise_count=1,noise_seed=2001,
        population='original_fixed16_development_display_set',cohort_never_merged_with_common500=True,bootstrap_calls=0,inferential_CI=False,
        model_or_policy_selection=False,new_training=0,new_channel=0,new_receiver=0,no_main_figure_change=True,
        originals={n:pin(OLD/n) for n in ['registration.json','receipts.json','completion.json','actual_wait.json']},
        sources=old['sources'],original_rows=rows,input_pixels=input_pixels,metric_assets=pin(TASK/'metric_assets_posix_manifest.json'),
        loader=pin(LOADER),protocol=protocol,code=pin(Path(__file__).resolve()),resource=dict(physical_gpu=3,batch=1,FP32=True,TF32=False,
            deterministic=True,math_threads=6,interop_threads=2,CPU_affinity=list(range(6)),allocator_GiB=12,startup_free_MiB=8192,runtime_free_MiB=2048),max_seconds=7200)
    require(OUT.is_dir(), 'Original failed registration directory required');value['path_recovery_evidence']=pin(TASK/'path_recovery_preparation_v1/review.json');value['dependency_recovery_evidence']=pin(TASK/'dependency_recovery_preparation_v1/review.json');value['dependency_overlay']=pin(TASK/'dependency_overlay_v1/manifest.json');value['original_failed_registration']=pin(OUT/'registration.json');require(read(OUT/'registration.json')['input_pixels']==input_pixels, 'Recovery must retain exact original registered pixels');save(REG,value);print(json.dumps(pin(REG)))
class Ledger:
    def __init__(self,p):
        self.db=sqlite3.connect(p);self.db.execute('CREATE TABLE calls(id INTEGER PRIMARY KEY,kind TEXT,identity TEXT,state TEXT)');self.db.commit()
    def call(self,kind,identity,fn):
        cur=self.db.execute('INSERT INTO calls(kind,identity,state) VALUES(?,?,?)',(kind,identity,'RESERVED'));self.db.commit()
        try:value=fn()
        except BaseException:self.db.execute('UPDATE calls SET state=? WHERE id=?',('FAILED',cur.lastrowid));self.db.commit();raise
        self.db.execute('UPDATE calls SET state=? WHERE id=?',('COMPLETE',cur.lastrowid));self.db.commit();return value
    def summary(self):
        rows=self.db.execute('SELECT kind,state,count(*) FROM calls GROUP BY kind,state').fetchall();return dict(rows=rows,unresolved=sum(n for _,s,n in rows if s!='COMPLETE'))
def run(expected):
    require(sha(REG)==expected,'Expected registration SHA differs');r=read(REG);checked(r['code']);checked(r['loader']);assets=read(checked(r['metric_assets']))
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='3','Only registered GPU3 required')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8' and os.environ.get('NVIDIA_TF32_OVERRIDE')=='0' and os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='0','Frozen deterministic FP32 environment required')
    RUN.mkdir(parents=True,exist_ok=False);cache=RUN/'runtime_cache'
    for sub in ('tmp','inductor','triton','torch'): (cache/sub).mkdir(parents=True)
    os.environ.update(TMPDIR=str(cache/'tmp'),TEMP=str(cache/'tmp'),TMP=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),TORCH_HOME=str(cache/'torch'))
    import tempfile;tempfile.tempdir=str(cache/'tmp')
    import fcntl
    lock=(RUN/'owner.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=Ledger(RUN/'calls.sqlite');started=time.monotonic()
    def guard():
        require(time.monotonic()-started<r['max_seconds'] and not (TASK/'STOP').exists(),'Finite window or user STOP')
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        require(free>=2048,'Registered runtime free-memory floor')
    try:
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());require(free>=8192,'Registered startup memory admission')
        os.sched_setaffinity(0,set(range(6)))
        import numpy as np,torch,torch.nn.functional as F
        torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
        torch.manual_seed(2026101401)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
        spec=importlib.util.spec_from_file_location('_fixed_official_metric_loader',LOADER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        models={}
        for name,loader in [('dists',m.load_dists),('dreamsim',m.load_dreamsim),('ms_ssim',m.load_ms_ssim)]:
            guard();models[name]=ledger.call('load_'+name,name,lambda loader=loader,name=name:loader(assets['models'][name],torch.device('cuda:0')))
        save(RUN/'metric_runtime_identity.json',dict(registration=pin(REG),models=assets['models'],protocol=r['protocol'],versions=m.package_versions(),
            device=torch.cuda.get_device_name(0),CUDA=torch.version.cuda,FP32=True,TF32=False,batch=1,threads=torch.get_num_threads(),interop=torch.get_num_interop_threads()))
        keyed=validate_grid(r['original_rows'],r['sources']);frozen={(p['method'],p['source_index']):p for p in r['input_pixels']};rows=[];cache_pairs={};features={}
        with torch.inference_mode():
            for source in r['sources']:
                for method in METHODS:
                    guard();old=keyed[method,source['source_index']];x,y=load_pixels(source,old);identity=frozen[method,source['source_index']]
                    require(ash(x)==identity['reference_rgb_sha256'] and ash(y)==identity['reconstruction_rgb_sha256'],'Registered pair pixels changed')
                    pair=hashlib.sha256((expected+ash(x)+ash(y)).encode()).hexdigest();xt=torch.from_numpy(x[None]).cuda();yt=torch.from_numpy(y[None]).cuda()
                    if pair not in cache_pairs:
                        fx=features.get(ash(x))
                        if fx is None:
                            fx=ledger.call('dreamsim_reference',ash(x),lambda:models['dreamsim'].embed(m.preprocess_dreamsim(xt)).float().detach());features[ash(x)]=fx
                        dv=ledger.call('dists_pair',pair,lambda:models['dists'](xt,yt,require_grad=False,batch_average=False).reshape(-1)[0].item())
                        fy=ledger.call('dreamsim_reconstruction',pair,lambda:models['dreamsim'].embed(m.preprocess_dreamsim(yt)).float())
                        ds=(1-F.cosine_similarity(fx,fy,dim=-1))[0].item()
                        ms=ledger.call('ms_ssim_pair',pair,lambda:models['ms_ssim'](xt,yt,data_range=1.0,size_average=False,win_size=11,win_sigma=1.5,weights=r['protocol']['ms_ssim']['weights'],K=(0.01,0.03))[0].item())
                        values=dict(dists=dv,dreamsim=ds,ms_ssim=ms);require(all(math.isfinite(v) for v in values.values()),'Nonfinite metric; preserve failure, never drop row');cache_pairs[pair]=values;origin='ACTUAL_NEW_PAIR'
                    else:values=cache_pairs[pair];origin='EXACT_NATIVE_PIXEL_PAIR_REUSE'
                    rows.append(dict(method=method,source_index=source['source_index'],source_id=source['source_id'],N=1024,snr_db=13,noise_seed=2001,
                        population=r['population'],**values,pair_key=pair,score_origin=origin,reference_rgb_sha256=ash(x),reconstruction_rgb_sha256=ash(y),
                        original_archive=old['archive']['path'],original_archive_sha256=old['archive']['sha256'],original_header_accepted=old.get('header_accepted',old.get('sampler_receipt',{}).get('header_accepted','NOT_APPLICABLE'))))
                    save(RUN/f'row_{len(rows):03d}.json',rows[-1]);print('SCORED',len(rows),method,source['source_index'],flush=True)
        require(len(rows)==48 and ledger.summary()['unresolved']==0,'Incomplete metric grid/ledger')
        for value in r['originals'].values():checked(value)
        for value in r['input_pixels']:
            checked(value['reference']);checked(value['reconstruction'])
        checked(r['code']);checked(r['loader']);checked(r['metric_assets'])
        csvsave(RUN/'frame_scores.csv',rows);summary=ledger.summary();ledger.db.close()
        save(RUN/'completion.json',dict(status='COMPLETE_SAVED_DEVELOPMENT16_THREE_METRICS',logical_pairs=48,HiFi_pairs=16,unique_pairs=len(cache_pairs),scalar_scores=144,
            registration=pin(REG),scores=pin(RUN/'frame_scores.csv'),ledger=pin(RUN/'calls.sqlite'),counts=summary,seconds=time.monotonic()-started,
            new_training=0,new_channel=0,new_receiver=0,bootstrap=0,model_or_policy_selection=False))
    except BaseException:
        save(RUN/'failure.json',dict(status='FAILED_PRESERVE_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc(),counts=ledger.summary()));raise
def finalize(expected,wait_path,wait_sha):
    import numpy as np
    require(sha(REG)==expected and sha(wait_path)==wait_sha,'Actual closure pins required');wait=read(wait_path)
    require(wait['actual_wait'] is True and type(wait['returncode']) is int and wait['returncode']==0 and all(wait[k] is False for k in ('timed_out','terminated','killed')) and wait['metadata_error'] is None,'Real clean metric process exit0 required')
    r=read(REG);checked(r['code']);checked(r['loader']);checked(r['metric_assets'])
    expected_argv=['/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python','-B',r['code']['path'],'run','--expected-registration',expected]
    require(wait['registration_sha256']==expected and wait['argv']==expected_argv,'Actual metric child argv/registration differs')
    checked(wait['stdout']);checked(wait['stderr'])
    for ident in (wait['process'],wait['parent_process']):
        p=Path('/proc')/str(ident['pid'])/'stat'
        require(not p.exists() or int(p.read_text().rsplit(')',1)[1].split()[19])!=ident['start_ticks'],'Original parent/child generation remains')
    require(not (RUN/'failure.json').exists(),'Failed metric attempt must be diagnosed')
    done=read(RUN/'completion.json');require(done['logical_pairs']==48 and done['counts']['unresolved']==0,'Full actual score closure')
    require(done['registration']==pin(REG) and done['scalar_scores']==144 and done['HiFi_pairs']==16,'Scientific scope differs')
    p=checked(done['scores'])
    with p.open() as f:rows=list(csv.DictReader(f))
    require(len(rows)==48 and len({(r['method'],int(r['source_index'])) for r in rows})==48,'All native rows retained')
    summary=[];paired=[]
    for method in METHODS:
        for metric in METRICS:
            a=np.array([float(next(r[metric] for r in rows if r['method']==method and int(r['source_index'])==i)) for i in INDICES],dtype=np.float64)
            summary.append(dict(method=method,metric=metric,mean=float(a.mean()),std_ddof1=float(a.std(ddof=1)),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),source_count=16,noise_count=1,N=1024,snr_db=13,inferential_CI='NOT_COMPUTED_FIXED_DISPLAY_SET'))
    for reference in METHODS[1:]:
        for metric in METRICS:
            a=np.array([float(next(r[metric] for r in rows if r['method']==METHODS[0] and int(r['source_index'])==i))-float(next(r[metric] for r in rows if r['method']==reference and int(r['source_index'])==i)) for i in INDICES],dtype=np.float64)
            paired.append(dict(method=METHODS[0],reference=reference,metric=metric,mean_difference=float(a.mean()),median_difference=float(np.median(a)),std_difference_ddof1=float(a.std(ddof=1)),source_count=16,delta='HiFi minus reference',CI='NOT_COMPUTED_FIXED_DISPLAY_SET'))
    csvsave(OUT/'frame_scores.csv',rows);csvsave(OUT/'summary.csv',summary);csvsave(OUT/'paired_descriptive.csv',paired)
    table=['| Method | DISTS ↓ | DreamSim ↓ | MS-SSIM ↑ |','|---|---:|---:|---:|']
    for method in METHODS:table.append('| '+method+' | '+' | '.join(f"{next(r['mean'] for r in summary if r['method']==method and r['metric']==metric):.6f}" for metric in METRICS)+' |')
    text='\n'.join(table)
    report='# Fixed16 final-link supplementary quality\n\n'+text+'\n\nAll16 HiFi outputs and their saved final Swin200k and P100k counterparts were scored at N1024/13dB, channel seed2001 and frozen HiFi posterior seed23. Native float32 arrays were used, with original file and pixel hashes verified. This is a single-noise, originally selected development display set; its descriptive means and directly source-paired differences are not common500 results or population confidence intervals. No image generation, receiver/channel replay, training, checkpoint/policy selection, bootstrap or main-figure change occurred. Every outcome remains. DISTS and DreamSim are lower-is-better; MS-SSIM is higher-is-better. Full-precision frame, distribution and signed paired tables are supplied. DreamSim uses the fixed official ensemble and square224 bicubic antialiased tensor preprocessing, preserving native floats without PIL re-quantization; official per-backbone normalization is unchanged. DISTS/MS-SSIM consume original256 RGB with frozen settings in registration.json. Metrics share learned backbones and are not independent evidence.\n\nThe common500 six-method addition is WAITING_FOR_ORIGINAL_P100k_Swin200k_RECONSTRUCTIONS by explicit user choice; it is not scored or merged here. Old quality scores and main figures are unchanged.\n'
    (OUT/'REPORT.md').write_text(report)
    (OUT/'REPORT_zh.md').write_text('# 最终 HiFi 固定16张补充评分\n\n'+text+'\n\n全部16张 HiFi、同源同工作点的 Swin200k 和 P100k 均使用原始重建数组评分。条件为 N1024、13 dB、信道种子2001、HiFi 后验种子23，每源一个既有噪声实例。该集合是原先选定的 development 展示集，均值与逐源配对差值仅作描述，不作为 common500 或总体置信区间。没有重训、重跑信道、重建图像、重新选模型、bootstrap 或修改主图。三指标均完整报告，DISTS/DreamSim 越低越好，MS-SSIM 越高越好。DreamSim 固定官方 ensemble 分支，原生浮点 RGB 通过作者方形224、bicubic、antialias几何预处理，保留原浮点精度而不转PIL再量化；内部归一化不变。DISTS/MS-SSIM 固定原生256 RGB设置。详细原始分数、均值/标准差/中位数/范围及直接逐源配对差值见CSV。\n\n依用户选择，common500 六方法补充等待 P100k/Swin200k 原始像素，不在此执行或混入16张结果。\n')
    save(OUT/'completion.json',dict(status='DELIVERED_COMPLETE_FIXED16_SUPPLEMENTAL_METRICS',metric_completion=pin(RUN/'completion.json'),actual_wait=pin(Path(wait_path)),
        process_generations_gone=True,all16HiFi=True,logical_pairs=48,scalar_scores=144,new_channel=0,new_receiver=0,new_training=0,bootstrap=0,
        common500_status='WAITING_FOR_ORIGINAL_P100k_Swin200k_RECONSTRUCTIONS',outputs={p.name:pin(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(pin(OUT/'completion.json')))
if __name__=='__main__':
    overlay=TASK/'dependency_overlay_v1/manifest.json';require(sha(overlay)=='c3372713ca0237ade062a3a286b9ce7a531858d607f247a97a3d793e8c9bee4d','Frozen private dependencies differ');[checked(p) for p in read(overlay)['files']];sys.path.insert(0,str(TASK/'dependency_overlay_v1/sitepackages'))
    a=argparse.ArgumentParser();a.add_argument('role',choices=('register','run','finalize'));a.add_argument('--expected-registration');a.add_argument('--actual-wait');a.add_argument('--actual-wait-sha256');v=a.parse_args()
    if v.role=='register':register()
    elif v.role=='run':run(v.expected_registration)
    else:finalize(v.expected_registration,v.actual_wait,v.actual_wait_sha256)
