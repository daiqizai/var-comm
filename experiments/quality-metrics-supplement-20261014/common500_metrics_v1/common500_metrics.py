"""Frozen supplemental metrics on 27000 verified original common500 RGB pairs."""
from pathlib import Path
import argparse,collections,csv,hashlib,importlib.util,json,math,os,sqlite3,subprocess,sys,time,traceback
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
T=B/'quality_metrics_supplement_20261014_v1'
E=R/'experiments/quality-metrics-supplement-20261014/common500_metrics_v1'
O=R/'results/quality_metrics_supplement_20261014_v1/common500'
REG=O/'registration.json';RUN=T/'common500_metrics_v1'
DATA=T/'common500_metrics_registration_data_v1'
REPLAY=T/'common500_replay_v1';REPLAY_REG=R/'results/quality_metrics_supplement_20261014_v1/common500_replay_v1/registration.json'
BRIDGE=T/'digital_input_bridge_v2';EMAIN=R/'results/entropy_main_common500_20261013'
LOADER=R/'experiments/unified-metrics-20261002/metric_models.py'
LOADER_SHA='dbc6f6c78c92ffd8a6e914d79008bbee8fc8b485a776884de45c6bdce1703066'
METHODS=('EC_VAR_PARTIAL','RAW_PARTIAL','EC_VAR_WHOLE','P100000','Swin200000','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024')
METRICS=('dists','dreamsim','ms_ssim');SNRS=(4,10,19)
SEEDS={m:((2001,2002,2003) if m in METHODS[3:] else (6201,6202,6203)) for m in METHODS}
def require(v,msg):
    if not v:raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def pin(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def checked(v):
    p=Path(v['path']);require(p.is_absolute() and '..' not in p.parts and not p.is_symlink() and p.is_file(),'Canonical ordinary input required: '+str(p))
    before=p.stat();require(('bytes' not in v or before.st_size==v['bytes']) and sha(p)==v['sha256'],'Input SHA differs: '+str(p));after=p.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Input changed during read');return p
def ash(a):
    import numpy as np
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def pixels(a):
    import numpy as np
    require(a.dtype==np.float32 and a.shape==(3,256,256) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1,'Native float RGB contract differs');return np.ascontiguousarray(a)
def boolean(v):
    require(v in (True,False,'True','False','true','false'),'Explicit boolean required');return v in (True,'True','true')
def key(r):return r['method'],int(r['source_index']),int(r['snr_db']),int(r['noise_seed'])
def validate_grid(rows,sources):
    require(len(rows)==27000 and len(sources)==500,'Complete27000/500 scope required')
    require([s['source_index'] for s in sources]==list(range(500)) and len({s['source_id'] for s in sources})==500,'Original common500 source order required')
    keyed={}
    for r in rows:
        k=key(r);require(k not in keyed and k[0] in METHODS and k[1] in range(500) and k[2] in SNRS and k[3] in SEEDS[k[0]] and r['N']==1024,'Unexpected/duplicate logical frame')
        require(r['source_id']==sources[k[1]]['source_id'] and r['reference_rgb_sha256']==sources[k[1]]['reference_rgb_sha256'],'Source pixel identity differs');keyed[k]=r
    require(set(keyed)=={(m,i,s,n) for m in METHODS for i in range(500) for s in SNRS for n in SEEDS[m]},'No omitted failure frames or points allowed');return keyed
def pair_key(r,registration):return hashlib.sha256((registration+r['reference_rgb_sha256']+r['reconstruction_sha256']).encode()).hexdigest()
def closed(wait_path):
    w=read(wait_path);require(w['actual_wait'] is True and type(w['returncode']) is int and w['returncode']==0 and all(w[k] is False for k in ('timed_out','terminated','killed')) and w['metadata_error'] is None,'Actual clean child exit0 required')
    for ident in (w['process'],w['parent_process']):
        p=Path('/proc')/str(ident['pid'])/'stat';require(not p.exists() or int(p.read_text().rsplit(')',1)[1].split()[19])!=ident['start_ticks'],'Bound process generation remains live')
    checked(w['stdout']);checked(w['stderr']);return w
def load_source(src):
    import numpy as np
    with np.load(checked(src['archive']),allow_pickle=False) as z:u=z['pixels'].copy()
    require(u.dtype==np.uint8 and u.shape==(3,256,256) and hashlib.sha256(u.tobytes()).hexdigest()==src['preprocessing_id'],'Original native reference differs')
    x=pixels(u.astype(np.float32)/np.float32(255));require(ash(x)==src['reference_rgb_sha256'],'Canonical target differs');return x
def load_image(row,archives):
    import numpy as np
    p=row['archive']['path']
    if p not in archives:
        with np.load(checked(row['archive']),allow_pickle=False) as z:archives[p]={k:z[k].copy() for k in z.files}
    a=archives[p][row['image_key']]
    if row['image_slot'] is not None:a=a[row['image_slot']]
    y=pixels(a);require(ash(y)==row['reconstruction_sha256'],'Original reconstruction pixels differ');return y
def register():
    require(not O.exists() and not RUN.exists() and not DATA.exists(),'No repeated registration/score attempt')
    rw=closed(T/'common500_replay_dispatch_v1/actual_wait.json');rd=read(REPLAY/'completion.json');rr=read(REPLAY_REG)
    require(rd['status']=='COMPLETE_EXACT_ORIGINAL_PIXEL_REPLAY' and rd['mismatch_count']==0 and rd['old_hash_matched']==9000 and rd['counts']['unresolved']==0 and rd['score_admitted'] is True,'Original9000 replay closure required')
    require(rd['registration']==pin(REPLAY_REG) and rw['registration_sha256']==sha(REPLAY_REG),'Replay registration identity differs')
    wanted=[('P_decoder_batch3','COMPLETE',1500),('P_receive','COMPLETE',4500),('P_transmit','COMPLETE',1500),('Swin_physical_frame','COMPLETE',4500),('load_P','COMPLETE',1),('load_Swin','COMPLETE',1)]
    require([tuple(x) for x in rd['counts']['rows']]==wanted and rw['source']==rr['source'],'Exact real replay call counts/source required')
    require(not (REPLAY/'failure.json').exists() and not (T/'common500_replay_dispatch_v1/failure.json').exists(),'Failed replay must not enter scoring')
    for p,h in rr['code_bindings'].items():require(sha(p)==h,'Original replay bound source changed')
    for p in rr['checkpoints'].values():checked(p)
    loaded={}
    for m in ('P100000','Swin200000'):
        p=REPLAY/f'{m}_loaded_sources.json';loaded[m]=pin(p)
        for q,h in read(p).items():require(sha(q)==h,'Actual loaded source identity changed')
    for p in rr['old_csv'].values():checked(p)
    checked(rr['old_registration']);checked(rr['old_completion']);checked(rr['old_actual_wait']);checked(rd['ledger']);checked(rd['mismatches'])
    require(read(checked(rd['mismatches']))==[],'Every mismatch must be disclosed and excluded from original-output claim')
    ec=read(EMAIN/'scoring_completion.json');ew=read(EMAIN/'actual_wait_scoring.json')
    require(ec['status']=='COMPLETE_EMAIN4500_SCORING' and ec['logical_frames']==4500 and ew['actual_child_waited'] is True and type(ew['returncode']) is int and ew['returncode']==0,'Original entropy-partial4500 closure required')
    with checked(ec['frame_rows']).open() as f:em=list(csv.DictReader(f))
    bridge=read(BRIDGE/'manifest.json');require(bridge['logical_frames']==13500 and bridge['all_original_pixel_SHA_PASS'] is True and bridge['all_500_reference_SHA_PASS'] is True and all(bridge[k]==0 for k in ('new_channel','new_reconstruction','new_training','new_metric_scores')),'Verified byte-only bridge required')
    br=read(checked(bridge['frame_index']));bs=read(checked(bridge['sources']));require(len(bs)==500,'Bridge source roster required')
    sources=[]
    for i,old in enumerate(rr['sources']):
        require(old['source_index']==i and bs[i]['source_id']==old['source_id'],'Shared original source order differs')
        sources.append(dict(source_index=i,source_id=old['source_id'],archive=old['pixels'],preprocessing_id=old['preprocessing_id'],reference_rgb_sha256=bs[i]['reference_rgb_sha256']))
    frames=[]
    for r in br:
        r=dict(r);r['image_slot']=int(r['image_slot']);r['fixed_gray']=boolean(r['fixed_gray']);r['header_accepted']=boolean(r['header_accepted']);frames.append(r)
    for r in read(checked(rd['frame_index'])):
        require(r['pixel_sha_match'] is True and r['header_match'] is True and r['actual_reconstruction_sha256']==r['old_reconstruction_sha256'],'Nonoriginal replay pixels cannot be relabelled')
        i=r['source_index'];frames.append(dict(method=r['method'],source_index=i,source_id=r['source_id'],N=1024,snr_db=r['snr_db'],noise_seed=r['noise_seed'],archive=r['archive'],image_key='reconstruction',image_slot=None,
            reconstruction_sha256=r['actual_reconstruction_sha256'],reference_rgb_sha256=sources[i]['reference_rgb_sha256'],fixed_gray=r['fallback']!='none',header_accepted=r['header_accepted'],failure_status=r['fallback'],original_evidence=dict(replay_row=r,old_csv=rr['old_csv'][r['method']])))
    for r in em:
        require(r['method']=='EC_VAR_PARTIAL','Only original entropy-partial method required');i=int(r['source_index']);frames.append(dict(method=r['method'],source_index=i,source_id=r['source_id'],N=int(r['N']),snr_db=int(r['snr_db']),noise_seed=int(r['noise_seed']),
            archive=pin(Path(r['image_path'])),image_key='image',image_slot=None,reconstruction_sha256=r['image_sha256'],reference_rgb_sha256=sources[i]['reference_rgb_sha256'],fixed_gray=boolean(r['fixed_gray']),header_accepted=boolean(r['header_ok']),failure_status=r['physical_status'],original_evidence=dict(original_row=r,old_csv=ec['frame_rows'])))
    keyed=validate_grid(frames,sources);unique=set();input_counts=[]
    for source in sources:
        x=load_source(source);archives={}
        for m in METHODS:
            for s in SNRS:
                for n in SEEDS[m]:
                    r=keyed[m,source['source_index'],s,n];y=load_image(r,archives);unique.add((ash(x),ash(y)))
        input_counts.append(dict(source_index=source['source_index'],logical_pairs=54,unique_image_pairs=len({r['reconstruction_sha256'] for r in frames if r['source_index']==source['source_index']})))
        if (source['source_index']+1)%25==0:print('VERIFIED_INPUTS',source['source_index']+1,flush=True)
    require(sha(LOADER)==LOADER_SHA,'Official-loader binding differs')
    hifi=read(R/'results/quality_metrics_supplement_20261014_v1/hifi_fixed16/dependency_recovery_registration.json')
    assets=checked(hifi['metric_assets']);overlay=checked(hifi['dependency_overlay']);require(set(read(assets)['models'])==set(METRICS),'All three frozen official metrics required')
    DATA.mkdir();save(DATA/'frame_index.json',frames)
    scope=dict(schema='COMMON500_SIX_METHOD_THREE_SUPPLEMENTAL_METRICS_V1',population='original_common500',source_count=500,N=1024,SNRS=list(SNRS),methods=list(METHODS),metrics=list(METRICS),noise_seeds=SEEDS,
        logical_pairs=27000,logical_scalar_scores=81000,unique_pairs=len(unique),source_noise_average='First average original three noise frames per source/method/SNR/metric, including every original failure',
        statistics=dict(source_count=500,bootstrap_repetitions=10000,seed=2026100701,CI95='NumPy linear quantiles .025/.975 of directly resampled source-paired mean differences',
            paired_comparisons=[dict(method=METHODS[a],reference=METHODS[b]) for a in range(6) for b in range(a+1,6)],multiplicity='Exploratory pointwise intervals; no simultaneous-significance or global winner claim',metric_sign='Raw signed method minus reference; lower DISTS/DreamSim and higher MS-SSIM are favorable'),
        new_channel=0,new_reconstruction=0,new_training=0,new_checkpoint_or_policy_selection=False,no_main_figure_change=True,all_three_metrics_reported=True,
        prior_replay=dict(authorized_missing_frames=9000,actual_channel_replay_disclosed=True,all_9000_exact_old_pixel_SHA_PASS=True,actual_wait=pin(T/'common500_replay_dispatch_v1/actual_wait.json'),completion=pin(REPLAY/'completion.json'),registration=pin(REPLAY_REG),loaded_sources=loaded),
        original_closure=dict(emain_completion=pin(EMAIN/'scoring_completion.json'),emain_actual_wait=pin(EMAIN/'actual_wait_scoring.json'),emain_registration=pin(EMAIN/'registration_v4.json'),bridge_manifest=pin(BRIDGE/'manifest.json'),bridge_restoration=pin(BRIDGE/'restoration_completion.json'),raw_frozen_points=pin(B/'postprocess_original500_v1/points.json'),original_source_statistics=pin(B/'postprocess_original500_v1/raw64_common500_source_statistics.py')),
        sources=sources,frames=pin(DATA/'frame_index.json'),input_counts=input_counts,metric_assets=pin(assets),dependency_overlay=pin(overlay),loader=pin(LOADER),protocol=hifi['protocol'],
        scientific_code=pin(Path(__file__).resolve()),resources=dict(physical_gpu=3,FP32=True,TF32=False,precision='highest',batch=1,deterministic=True,cudnn_deterministic=True,cudnn_benchmark=False,math_threads=6,interop_threads=2,CPU_affinity=list(range(6)),allocator_GiB=12,startup_free_MiB=8192,runtime_free_MiB=2048,seed=2026101401,max_seconds=21600),
        failure='Preserve unknown failures; no automatic retry, dropping frames, re-rendering or branch switch')
    O.mkdir();save(REG,scope);print(json.dumps(pin(REG)),flush=True)
class Ledger:
    def __init__(self,p,caps):
        self.caps=caps;self.db=sqlite3.connect(p);self.db.execute('CREATE TABLE calls(id INTEGER PRIMARY KEY,kind TEXT,identity TEXT,state TEXT)');self.db.execute('CREATE INDEX call_kind_state ON calls(kind,state)');self.db.execute('CREATE TABLE frames(logical_key TEXT PRIMARY KEY,row_json TEXT NOT NULL)');self.db.commit()
    def call(self,kind,identity,fn):
        n=self.db.execute('SELECT count(*) FROM calls WHERE kind=?',(kind,)).fetchone()[0];require(n<self.caps[kind],'Frozen metric call cap exceeded: '+kind)
        c=self.db.execute('INSERT INTO calls(kind,identity,state) VALUES(?,?,?)',(kind,identity,'RESERVED'));self.db.commit()
        try:v=fn()
        except BaseException:self.db.execute('UPDATE calls SET state=? WHERE id=?',('FAILED',c.lastrowid));self.db.commit();raise
        self.db.execute('UPDATE calls SET state=? WHERE id=?',('COMPLETE',c.lastrowid));self.db.commit();return v
    def frame(self,r):self.db.execute('INSERT INTO frames VALUES(?,?)',(json.dumps(key(r)),json.dumps(r,allow_nan=False)));self.db.commit()
    def summary(self):
        rows=self.db.execute('SELECT kind,state,count(*) FROM calls GROUP BY kind,state').fetchall();return dict(rows=rows,unresolved=sum(n for _,s,n in rows if s!='COMPLETE'),logical_frames=self.db.execute('SELECT count(*) FROM frames').fetchone()[0])
def csvsave(p,rows):
    with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
def run(expected):
    require(sha(REG)==expected,'Frozen registration SHA required');r=read(REG);checked(r['scientific_code']);checked(r['loader']);assets=read(checked(r['metric_assets']));overlay=read(checked(r['dependency_overlay']))
    require(len(overlay['files'])==55,'Complete frozen private dependency overlay required')
    for v in overlay['files']:checked(v)
    sys.path.insert(0,str(T/'dependency_overlay_v1/sitepackages'))
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='3' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8' and os.environ.get('NVIDIA_TF32_OVERRIDE')=='0' and os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='0','Registered FP32 environment required')
    RUN.mkdir(exist_ok=False);cache=RUN/'runtime_cache'
    for n in ('tmp','inductor','triton','torch'):(cache/n).mkdir(parents=True)
    import tempfile,fcntl
    os.environ.update(TMPDIR=str(cache/'tmp'),TMP=str(cache/'tmp'),TEMP=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),TORCH_HOME=str(cache/'torch'));tempfile.tempdir=str(cache/'tmp')
    lock=(RUN/'owner.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for n in ('common500_replay_dispatch_v1','common500_metrics_dispatch_v1'):
        if n=='common500_replay_dispatch_v1':closed(T/n/'actual_wait.json')
    caps={k:1 for k in ('load_dists','load_dreamsim','load_ms_ssim')};caps.update(dreamsim_reference=500,dists_pair=r['unique_pairs'],dreamsim_reconstruction=r['unique_pairs'],ms_ssim_pair=r['unique_pairs'])
    ledger=Ledger(RUN/'calls.sqlite',caps);start=time.monotonic();next_gpu_check=0;history=[]
    def guard(force=False):
        nonlocal next_gpu_check
        now=time.monotonic();require(now-start<r['resources']['max_seconds'] and not (T/'STOP').exists(),'Finite scoring window or user STOP')
        if force or now>=next_gpu_check:
            free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());require(free>=2048,'Runtime absolute free-memory floor');history.append(dict(elapsed_seconds=now-start,free_MiB=free));next_gpu_check=now+10
    try:
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());require(free>=8192,'Registered startup free-memory admission')
        os.sched_setaffinity(0,set(range(6)))
        import numpy as np,torch,torch.nn.functional as F
        torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.manual_seed(2026101401);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
        spec=importlib.util.spec_from_file_location('_common_frozen_official_loader',LOADER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);models={}
        for name,loader in [('dists',m.load_dists),('dreamsim',m.load_dreamsim),('ms_ssim',m.load_ms_ssim)]:
            guard(True);models[name]=ledger.call('load_'+name,name,lambda loader=loader,name=name:loader(assets['models'][name],torch.device('cuda:0')))
        save(RUN/'metric_runtime_identity.json',dict(registration=pin(REG),model_metadata=assets['models'],protocol=r['protocol'],versions=m.package_versions(),
            physical_gpu=3,device=torch.cuda.get_device_name(0),CUDA=torch.version.cuda,FP32=True,TF32=False,precision=torch.get_float32_matmul_precision(),batch=1,deterministic=torch.are_deterministic_algorithms_enabled(),cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),CPU_affinity=sorted(os.sched_getaffinity(0))))
        keyed=validate_grid(read(checked(r['frames'])),r['sources']);cache_pairs={};reference_features={};rows=[]
        with torch.inference_mode():
            for source in r['sources']:
                guard();x=load_source(source);xt=torch.from_numpy(x[None]).cuda();archives={};xh=ash(x)
                if xh not in reference_features:reference_features[xh]=ledger.call('dreamsim_reference',xh,lambda:models['dreamsim'].embed(m.preprocess_dreamsim(xt)).float().detach().cpu())
                fx=reference_features[xh].cuda()
                for method in METHODS:
                    for snr in SNRS:
                        for seed in SEEDS[method]:
                            guard();old=keyed[method,source['source_index'],snr,seed];y=load_image(old,archives);pair=pair_key(old,expected)
                            if pair not in cache_pairs:
                                yt=torch.from_numpy(y[None]).cuda()
                                dv=ledger.call('dists_pair',pair,lambda:models['dists'](xt,yt,require_grad=False,batch_average=False).reshape(-1)[0].item())
                                fy=ledger.call('dreamsim_reconstruction',pair,lambda:models['dreamsim'].embed(m.preprocess_dreamsim(yt)).float())
                                ds=(1-F.cosine_similarity(fx,fy,dim=-1))[0].item()
                                ms=ledger.call('ms_ssim_pair',pair,lambda:models['ms_ssim'](xt,yt,data_range=1.0,size_average=False,win_size=11,win_sigma=1.5,weights=r['protocol']['ms_ssim']['weights'],K=(0.01,0.03))[0].item())
                                values=dict(dists=dv,dreamsim=ds,ms_ssim=ms);require(all(math.isfinite(v) for v in values.values()),'Nonfinite supplementary score, preserve every failure');cache_pairs[pair]=values;origin='ACTUAL_NEW_NATIVE_PIXEL_PAIR'
                            else:values=cache_pairs[pair];origin='EXACT_VERIFIED_NATIVE_PIXEL_PAIR_REUSE'
                            row=dict(method=method,source_index=source['source_index'],source_id=source['source_id'],N=1024,snr_db=snr,noise_seed=seed,**values,reference_rgb_sha256=xh,reconstruction_rgb_sha256=old['reconstruction_sha256'],pair_key=pair,score_origin=origin,fixed_gray=old['fixed_gray'],header_accepted=old['header_accepted'],failure_status=old['failure_status'],original_archive=old['archive']['path'],original_archive_sha256=old['archive']['sha256'])
                            ledger.frame(row);rows.append(row)
                if (source['source_index']+1)%10==0:
                    p=RUN/'progress.json';temp=p.with_suffix('.next');save(temp,dict(sources_completed=source['source_index']+1,logical_pairs=len(rows),scalar_scores=len(rows)*3,unique_pairs=len(cache_pairs),seconds=time.monotonic()-start));os.replace(temp,p);print('SCORED',source['source_index']+1,len(rows),len(cache_pairs),flush=True)
        require(len(rows)==27000 and len(cache_pairs)==r['unique_pairs'] and ledger.summary()['unresolved']==0,'Incomplete scoring/ledger or pair reuse scope differs')
        # Close the actual input bytes after all scores; no model/channel call.
        all_archives={v['archive']['path']:v['archive'] for v in keyed.values()}
        all_archives.update({s['archive']['path']:s['archive'] for s in r['sources']})
        for archive in all_archives.values():guard();checked(archive)
        for section in (r['prior_replay'],r['original_closure']):
            for v in section.values():
                if isinstance(v,dict) and {'path','bytes','sha256'}<=set(v):checked(v)
        checked(r['scientific_code']);checked(r['loader']);checked(r['metric_assets']);checked(r['dependency_overlay'])
        csvsave(RUN/'frame_scores.csv',rows);save(RUN/'resource_history.json',history);summary=ledger.summary();ledger.db.close()
        save(RUN/'completion.json',dict(status='COMPLETE_COMMON500_SIX_METHOD_THREE_SUPPLEMENTARY_METRICS',logical_pairs=27000,scalar_scores=81000,unique_pairs=len(cache_pairs),counts=summary,registration=pin(REG),scores=pin(RUN/'frame_scores.csv'),ledger=pin(RUN/'calls.sqlite'),resource_history=pin(RUN/'resource_history.json'),seconds=time.monotonic()-start,new_channel=0,new_reconstruction=0,new_training=0,new_selection=False,statistics_completed=False))
    except BaseException:
        save(RUN/'failure.json',dict(status='FAILED_PRESERVE_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc(),counts=ledger.summary()));raise
def source_values(rows,method,snr,metric):
    import numpy as np
    values=collections.defaultdict(dict)
    for r in rows:
        if r['method']==method and int(r['snr_db'])==snr:
            i,n=int(r['source_index']),int(r['noise_seed']);require(n not in values[i],'Duplicate noise frame');values[i][n]=float(r[metric])
    require(set(values)==set(range(500)) and all(set(v)==set(SEEDS[method]) for v in values.values()),'Every source must retain all original three noises')
    return np.array([np.mean([values[i][n] for n in SEEDS[method]],dtype=np.float64) for i in range(500)],dtype=np.float64)
def statistics(rows):
    import numpy as np
    # The registered original draw shape/seed and directly paired source means.
    rng=np.random.default_rng(2026100701);draws=rng.integers(0,500,size=(10000,500));arrays={};summary=[];paired=[];source_rows=[]
    for method in METHODS:
        for snr in SNRS:
            for metric in METRICS:
                a=source_values(rows,method,snr,metric);arrays[method,snr,metric]=a;boot=a[draws].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975],method='linear')
                summary.append(dict(method=method,snr_db=snr,metric=metric,mean=float(a.mean()),CI95_low=float(lo),CI95_high=float(hi),source_count=500,noise_count=3,logical_frames=1500,fixed_gray_frames=sum(boolean(r['fixed_gray']) for r in rows if r['method']==method and int(r['snr_db'])==snr),direction='higher' if metric=='ms_ssim' else 'lower',bootstrap_repetitions=10000,bootstrap_seed=2026100701))
                source_rows.extend(dict(method=method,snr_db=snr,metric=metric,source_index=i,source_mean=float(a[i]),noise_count=3) for i in range(500))
    for a in range(6):
        for b in range(a+1,6):
            for snr in SNRS:
                for metric in METRICS:
                    delta=arrays[METHODS[a],snr,metric]-arrays[METHODS[b],snr,metric];boot=delta[draws].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975],method='linear')
                    paired.append(dict(method=METHODS[a],reference=METHODS[b],snr_db=snr,metric=metric,mean_difference=float(delta.mean()),CI95_low=float(lo),CI95_high=float(hi),source_count=500,noise_count=3,delta='method minus reference',direction='higher difference favorable' if metric=='ms_ssim' else 'lower difference favorable',interval='direct source-paired pointwise95%',bootstrap_repetitions=10000,bootstrap_seed=2026100701))
    return summary,paired,source_rows,ash(draws)
def finalize(expected,wait_path,wait_sha):
    require(sha(REG)==expected and sha(wait_path)==wait_sha,'Actual wait/registration pin required');w=closed(wait_path);r=read(REG)
    require(w['registration_sha256']==expected and w['argv'][-3:]==['run','--expected-registration',expected] and w['source']==r['scientific_code'],'Exact scientific argv/source binding required')
    require(not (RUN/'failure.json').exists(),'Unknown failure cannot be finalized');done=read(RUN/'completion.json');require(done['registration']==pin(REG) and done['logical_pairs']==27000 and done['scalar_scores']==81000 and done['counts']['unresolved']==0,'Complete scientific closure required')
    with checked(done['scores']).open() as f:rows=list(csv.DictReader(f))
    validate_grid([{**row,'N':int(row['N'])} for row in rows],r['sources']);summary,paired,source_rows,drawsha=statistics(rows)
    frame_files=[]
    for method in METHODS:
        name=f'{method}_frame_scores.csv';csvsave(O/name,[row for row in rows if row['method']==method]);frame_files.append(name)
    csvsave(O/'source_noise_means.csv',source_rows);csvsave(O/'summary.csv',summary);csvsave(O/'paired_source_intervals.csv',paired)
    table=['| SNR (dB) | Method | DISTS ↓ (95% CI) | DreamSim ↓ (95% CI) | MS-SSIM ↑ (95% CI) |','|---:|---|---:|---:|---:|']
    for snr in SNRS:
        for method in METHODS:
            cells=[]
            for metric in METRICS:
                a=next(q for q in summary if (q['method'],q['snr_db'],q['metric'])==(method,snr,metric));cells.append(f"{a['mean']:.6f} [{a['CI95_low']:.6f}, {a['CI95_high']:.6f}]")
            table.append(f'| {snr} | {method} | '+' | '.join(cells)+' |')
    explanation=('All original common500 sources, N1024, SNR4/10/19 and each method\'s original three noise seeds are retained. There are27000 logical image pairs and81000 supplemental scalar scores. Failed/gray frames retain their original denominator. Scores reuse only exactly verified native reference/reconstruction pairs. DISTS/DreamSim are lower-is-better; MS-SSIM is higher-is-better. Official frozen implementations, weights and RGB preprocessing are the same as the delivered fixed16 supplement. The original P100k/Swin200k evaluator had not saved pixels; the user separately authorized frozen9000-frame deterministic replay, and all9000 recovered outputs exactly match the old per-frame reconstruction SHA and header status. The other four methods use sealed original native arrays. No training, new checkpoint/policy selection, new samples/SNRs, recomputation of old scores or change to the main figure occurred.\n\nEach source first averages its original three noise scores. The registered10000 source resamples, seed2026100701, produce pointwise95% intervals. The separate paired table resamples signed per-source method differences directly, not differences of marginal intervals. All15 method pairs and all three metrics/workpoints are reported; these exploratory intervals do not constitute simultaneous significance or a universal winner. HiFi development16 remains separate at13dB. Supplemental scores cannot feed back into main policy/checkpoint selection.\n')
    (O/'REPORT.md').write_text('# Common500 supplemental quality metrics\n\n'+'\n'.join(table)+'\n\n'+explanation)
    zh=('原 common500、N1024、4／10／19 dB，每源三次原噪声；六方法共27000个逻辑图像对、81000个标量分数。全部灰图和失败帧保留分母，仅精确相同的原图—重建像素对复用评分。P100k／Swin200k 的9000帧是在用户单独授权后按原检查点、信道噪声和数学设置恢复，全部与旧重建SHA及头部状态一致。其余四方法读取封存原始数组。没有训练、重新选检查点／主配置、扩展样本或SNR、修改旧分数及原主图。\n\n先在每源内平均三次噪声，再按来源级直接配对重采样10000次，seed2026100701。完整精度CSV报告全部三指标及15个方法对的逐点95%配对区间；它们不等于同时显著性或普遍优胜结论，不能反馈到配置选择。HiFi 的13dB development16表保持独立。\n')
    (O/'REPORT_zh.md').write_text('# common500 三指标补充评价\n\n'+'\n'.join(table)+'\n\n'+zh)
    files={n:pin(O/n) for n in frame_files+['source_noise_means.csv','summary.csv','paired_source_intervals.csv','REPORT.md','REPORT_zh.md']}
    save(O/'completion.json',dict(status='COMPLETE_COMMON500_SUPPLEMENTAL_METRICS_AND_PAIRED_SOURCE_INTERVALS',logical_pairs=27000,scalar_scores=81000,unique_pairs=done['unique_pairs'],source_count=500,summary_rows=54,paired_rows=135,registration=pin(REG),scientific_completion=pin(RUN/'completion.json'),actual_wait=pin(wait_path),draws_sha256=drawsha,files=files,old_scores_changed=False,main_figure_changed=False,selection_feedback=False))
    print(json.dumps(pin(O/'completion.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=('register','run','finalize'));p.add_argument('--expected-registration');p.add_argument('--actual-wait');p.add_argument('--actual-wait-sha256');a=p.parse_args()
    if a.role=='register':register()
    elif a.role=='run':run(a.expected_registration)
    else:finalize(a.expected_registration,a.actual_wait,a.actual_wait_sha256)
