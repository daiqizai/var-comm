"""48 new raw scores +64 missing PSNR; preserve144 old scalars exactly."""
from common import *
import sys,csv,tempfile,subprocess,traceback,importlib.util,math
METHODS=('RAW_PARTIAL','P100000','Swin200000','HiFiSwin200000')
METRICS=('psnr_db','dists','dreamsim','ms_ssim')
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def run(expected):
 reg=O/'metric_registration.json';assert sha(reg)==expected;r=read(reg)
 for p in r['code']+[r['old_registration'],r['old_scores'],r['metric_assets'],r['loader'],r['dependency_overlay'],r['raw_manifest']]:checked(p)
 overlay=read(r['dependency_overlay']['path']);[checked(p)for p in overlay['files']];sys.path.insert(0,str(Path(r['dependency_overlay']['path']).parent/'sitepackages'))
 out=T/'fixed16_score';out.mkdir(exist_ok=False)
 for sub in ['runtime_cache/tmp','runtime_cache/inductor','runtime_cache/triton','runtime_cache/torch','rows']:(out/sub).mkdir(parents=True)
 os.environ.update(TMPDIR=str(out/'runtime_cache/tmp'),TEMP=str(out/'runtime_cache/tmp'),TMP=str(out/'runtime_cache/tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'runtime_cache/inductor'),TRITON_CACHE_DIR=str(out/'runtime_cache/triton'),TORCH_HOME=str(out/'runtime_cache/torch'));tempfile.tempdir=os.environ['TMPDIR']
 os.sched_setaffinity(0,set(range(2,8)));ledger=Ledger(out/'calls.sqlite',r['caps']);start=time.monotonic()
 save(out/'launch.json',dict(process=pid_identity(os.getpid()),parent_process=pid_identity(os.getppid()),argv=sys.argv,registration=pin(reg)))
 def boundary():
  assert time.monotonic()-start<7200 and not(T/'STOP').exists()
  free=int(subprocess.check_output(['nvidia-smi','-i',str(r['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=2048
 try:
  import numpy as np,torch,torch.nn.functional as F
  torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(2026101401)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
  torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
  m=module(r['loader']['path'],'_original_metric_loader');assets=read(r['metric_assets']['path']);models={}
  for name in ['dists','dreamsim','ms_ssim']:
   boundary();models[name]=ledger.call('model_load',name,lambda name=name:getattr(m,'load_'+name)(assets['models'][name],torch.device('cuda:0')))
  save(out/'runtime_identity.json',dict(protocol=r['protocol'],models=assets['models'],versions=m.package_versions(),device=torch.cuda.get_device_name(0),CUDA=torch.version.cuda,FP32=True,TF32=False,batch=1,threads=6,interop=2))
  with Path(r['old_scores']['path']).open()as f:old={(x['method'],int(x['source_index'])):x for x in csv.DictReader(f)}
  result=[];features={}
  with torch.inference_mode():
   for row in r['pairs']:
    boundary();key=f"{row['method']}:{row['source_index']}"
    with np.load(checked(row['reference']),allow_pickle=False)as z:x=z['pixels'].copy()
    assert x.dtype==np.uint8 and x.shape==(3,256,256)and hashlib.sha256(x.tobytes()).hexdigest()==row['preprocessing_id'];x=x.astype(np.float32)/np.float32(255)
    with np.load(checked(row['archive']),allow_pickle=False)as z:y=z['reconstruction'].copy()
    assert y.dtype==np.float32 and y.shape==(3,256,256)and np.isfinite(y).all()and 0<=y.min()<=y.max()<=1 and asha(y)==row['reconstruction_sha256']
    psnr=ledger.call('psnr',key,lambda:float(-10*np.log10(np.mean((x.astype(np.float64)-y.astype(np.float64))**2))))
    assert math.isfinite(psnr)
    if row['method']=='RAW_PARTIAL':
     xt=torch.from_numpy(x[None]).cuda();yt=torch.from_numpy(y[None]).cuda();sid=row['source_id']
     fx=ledger.call('dreamsim_reference',sid,lambda:models['dreamsim'].embed(m.preprocess_dreamsim(xt)).float().detach())
     dv=ledger.call('dists',key,lambda:models['dists'](xt,yt,require_grad=False,batch_average=False).reshape(-1)[0].item())
     def dream():
      fy=models['dreamsim'].embed(m.preprocess_dreamsim(yt)).float();return(1-F.cosine_similarity(fx,fy,dim=-1))[0].item()
     ds=ledger.call('dreamsim',key,dream)
     ms=ledger.call('ms_ssim',key,lambda:models['ms_ssim'](xt,yt,data_range=1.0,size_average=False,win_size=11,win_sigma=1.5,weights=r['protocol']['ms_ssim']['weights'],K=(0.01,0.03))[0].item())
     vals=dict(dists=dv,dreamsim=ds,ms_ssim=ms);origin='NEW_RAW_NATIVE_FLOAT_SCORES'
    else:
     prev=old[row['method'],row['source_index']];assert prev['reference_rgb_sha256']==asha(x)and prev['reconstruction_rgb_sha256']==asha(y)
     vals={k:prev[k]for k in ['dists','dreamsim','ms_ssim']};origin='EXACT_OLD_SCORE_TEXT_REUSE'
    assert all(math.isfinite(float(v))for v in vals.values())
    rec=dict(method=row['method'],source_index=row['source_index'],source_id=row['source_id'],N=1024,snr_db=13,noise_seed=row['noise_seed'],posterior_seed=23 if row['method']=='HiFiSwin200000'else'',psnr_db=psnr,**vals,three_metric_origin=origin,psnr_origin='NEW_NATIVE_FLOAT64_MSE',reference_rgb_sha256=asha(x),reconstruction_rgb_sha256=asha(y),archive_sha256=row['archive']['sha256'])
    result.append(rec);save(out/'rows'/f'{len(result):03d}.json',rec);print('SCORED',len(result),key,flush=True)
  assert len(result)==64 and ledger.summary()['unresolved']==0
  csvsave(out/'frame_scores.csv',result);summary=[];paired=[]
  for method in METHODS:
   for met in METRICS:
    a=np.array([float(x[met])for x in result if x['method']==method]);summary.append(dict(method=method,metric=met,mean=float(a.mean()),median=float(np.median(a)),std_ddof1=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max()),sources=16,N=1024,snr_db=13,CI='NOT_COMPUTED_DESCRIPTIVE_FIXED16'))
  for method in METHODS[1:]:
   for met in METRICS:
    a=np.array([float(next(x[met]for x in result if x['method']=='RAW_PARTIAL'and x['source_index']==i))-float(next(x[met]for x in result if x['method']==method and x['source_index']==i))for i in r['source_indices']]);paired.append(dict(method='RAW_PARTIAL',reference=method,metric=met,mean_difference=float(a.mean()),median_difference=float(np.median(a)),std_ddof1=float(a.std(ddof=1)),sources=16,delta='RAW_PARTIAL minus reference',CI='NOT_COMPUTED_DESCRIPTIVE_FIXED16'))
  csvsave(out/'summary.csv',summary);csvsave(out/'paired_descriptive.csv',paired)
  save(out/'completion.json',dict(status='COMPLETE_FIXED16_RAW_ADDITION',registration=pin(reg),new_raw_metric_scalars=48,new_psnr_scalars=64,reused_scalars=144,total_logical_scalars=256,new_image_reconstructions=0,new_channel=0,new_bootstrap=0,counts=ledger.summary(),seconds=time.monotonic()-start,outputs=[pin(out/n)for n in ['frame_scores.csv','summary.csv','paired_descriptive.csv']]))
 except BaseException:
  save(out/'failure.json',dict(status='BLOCKED_PRESERVE_NO_RETRY',traceback=traceback.format_exc(),counts=ledger.summary()));raise
if __name__=='__main__':run(sys.argv[1])
