"""Frozen original PSNR/LPIPS-Alex/DINO-L on closed cal100 clean outputs only."""
import argparse,csv,hashlib,json,math,os,pathlib,sys,time,traceback
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/arpc-external-baseline-20261019';OLD=R/'experiments/entropy-partial-mismatch-20261010/scripts'
L=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/arpc_external_20261019_v1')
RUN=L/'a2_metrics_v1';O=R/'results/arpc_external_baseline_20261019_v1/a2_metrics_v1'
sys.path.insert(0,str(E));from a1_owner import desc,verify,put
CAPS={'model_load':2,'reference_dino':100,'image_scores':800,'lpips_pair':800,'lpips_backbone_forward':1600,'reconstruction_dino':800}
def require_image(image):
    import numpy as np
    assert image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and (image>=0).all() and (image<=1).all()
    return image
class Ledger:
    def __init__(self):
        self.counts={};self.directory=RUN/'calls';self.directory.mkdir()
    def call(self,kind,key,fn):
        import torch
        assert torch.cuda.mem_get_info()[0]>=2*1024**3
        n=self.counts.get(kind,0)+1;assert n<=CAPS[kind];self.counts[kind]=n
        p=self.directory/f'{kind}_{n:04d}';started=time.time();put(p.with_suffix('.reserved.json'),{'kind':kind,'key':key,'started_unix':started,'pid':os.getpid()})
        try:
            result=fn();put(p.with_suffix('.complete.json'),{'seconds':time.time()-started});return result
        except BaseException:
            put(p.with_suffix('.failed.json'),{'seconds':time.time()-started,'error':traceback.format_exc()});raise
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True);ap.add_argument('--sha256',required=True);a=ap.parse_args()
    assert desc(a.registration)['sha256']==a.sha256;reg=json.loads(pathlib.Path(a.registration).read_text())
    for d in reg['inputs']+reg['source_files']:verify(d)
    assert not (RUN/'calls').exists();cache=RUN/'runtime_cache';cache.mkdir();(cache/'tmp').mkdir()
    os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),
      CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import tempfile;tempfile.tempdir=str(cache/'tmp')
    import ctypes
    sys.path.insert(0,str(E/'cuda_recovery_v1'));from runtime import check_mapping
    mapping=check_mapping(ctypes.CDLL('libcuda.so.1'),reg['resource']['cuda_admission'],os.environ.get('CUDA_VISIBLE_DEVICES'))
    put(RUN/'actual_cuda_mapping.json',mapping)
    sys.path.insert(0,str(OLD));import h800_ep_pilot_metrics_v1 as old
    config=reg['metric_config'];assets=old.bound_assets(config)
    import numpy as np,torch,lpips
    torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    assert old.flags_function(config['sources']['step0_reference_metrics.py']['path'])(torch)==old.FLAGS
    assert torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8'
    torch.cuda.set_device(0);gpu=torch.cuda.get_device_properties(0)
    assert torch.cuda.mem_get_info()[0]>=8*1024**3;torch.cuda.set_per_process_memory_fraction(6*1024**3/gpu.total_memory)
    ledger=Ledger();psnr,psnr_proof=old.psnr_projection(config['sources']['score_bpg_cached_holdout_r2.py']['path'])
    loader,pair,lpips_proof=old.lpips_projection(config['sources']['quality.py']['path'],dict(Path=pathlib.Path,lpips=lpips,torch=torch))
    assert old.core.sha(pathlib.Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==old.WEIGHT_SHA['lpips']
    metric=old.load_module(config['sources']['metric_models.py'],'_ARPC_original_metrics')
    with metric.no_network():
        perceptual=ledger.call('model_load','LPIPS_Alex',lambda:loader({'alexnet_checkpoint':assets['weights']['alexnet']['path']},'cuda:0'))
        evaluator=ledger.call('model_load','DINO_L',lambda:metric.MetricEvaluator(assets['dinov2_manifest'],device='cuda:0',requested=('dinov2_vitl14',)))
    evaluator.metadata['metric_selection_usage']='ARPC fixed cal100 screening only; DINO-L selects, PSNR/LPIPS diagnose; no evaluation sources.'
    original_forward=perceptual.net.forward
    perceptual.net.forward=lambda *args,**kwargs:ledger.call('lpips_backbone_forward','pair_backbone',lambda:original_forward(*args,**kwargs))
    metadata={'schema':'ARPC_A2_THREE_FROZEN_METRICS_V1','assets':assets,'config':config,'numeric_flags':old.FLAGS,
      'DINO_L_metadata':evaluator.metadata,'LPIPS_math_proof':lpips_proof,'PSNR_math_proof':psnr_proof,
      'image_input':'Native float_output, float32 CHW [0,1], no uint8 requantization; source uint8/255 float32.',
      'failure_baseline':'constant float32 0.5 RGB, same existing digital failure image; one pair/source.',
      'float_dtype':'float32','AMP':False,'batch_size':1,'device':mapping,'host':os.uname().nodename,
      'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'ConvNeXt_constructed':False,
      'DINO_score_label_argument':'fixed0 unused by requested DINO-only metric; no labels used for selection.'}
    metric_identity=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest();put(O/'metric_identity.json',metadata)
    states=json.loads(pathlib.Path(reg['clean_states']['path']).read_text())['states'];assert len(states)==700
    groups={i:[x for x in states if x['rank']==i] for i in range(100)};rows=[];cached={}
    for i,items in groups.items():
        assert len(items)==7 and sorted(x['endpoint'] for x in items)==list(range(1,8));src=items[0]['source'];verify(src)
        with np.load(src['path'],allow_pickle=False) as z:pixels=z['pixels'].copy()
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==items[0]['source_pixels_sha256']
        target=require_image(pixels.astype(np.float32)/255);x=torch.as_tensor(target[None],device='cuda:0')
        prepared=ledger.call('reference_dino',str(i),lambda:evaluator.prepare_reference(x))
        np.savez(RUN/f'{i:02d}_reference.npz',features=prepared['dinov2_vitl14'].numpy())
        todo=[(state['endpoint'],state) for state in sorted(items,key=lambda x:x['endpoint'])]+[(0,None)]
        for endpoint,state in todo:
            if state is None:image=np.full((3,256,256),.5,dtype=np.float32);image_pin=None
            else:
                image_pin=state['image'];verify(image_pin)
                with np.load(image_pin['path'],allow_pickle=False) as z:image=z['float_output'].copy()
            require_image(image);key=old.pair_cache_key(target,image,metric_identity)
            reused=key in cached
            if not reused:
                def compute():
                    y=torch.as_tensor(image[None],device='cuda:0')
                    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                        lp=ledger.call('lpips_pair',key,lambda:pair(perceptual,y,x))
                        di=ledger.call('reconstruction_dino',key,lambda:evaluator.score(x,y,[0],label_conditioned=False,prepared=prepared))
                    values={'psnr_db':float(psnr(target,image)),'lpips_alex':float(lp[0].item()),'dinov2_vitl14_cosine':float(di[0]['dinov2_vitl14_cosine'])}
                    assert all(math.isfinite(v) for v in values.values());return values
                cached[key]=ledger.call('image_scores',key,compute)
                put(RUN/(key+'.json'),{'pair_key':key,'source':src,'image':image_pin,'image_float_sha256':old.array_sha(image),'metric_identity':metric_identity,'scores':cached[key]})
            rows.append({'rank':i,'source_id':items[0]['source_id'],'source_index':items[0]['source_index'],'endpoint':endpoint,
              'kind':'fixed_failure_baseline' if endpoint==0 else 'clean_prefix','pair_key':key,'metric_identity':metric_identity,'exact_pair_reuse':reused,**cached[key]})
    assert len(rows)==800 and ledger.counts['reference_dino']==100 and ledger.counts['model_load']==2
    assert ledger.counts['lpips_backbone_forward']==2*ledger.counts['image_scores'] and ledger.counts['image_scores']==ledger.counts['lpips_pair']==ledger.counts['reconstruction_dino']
    with (O/'clean_quality.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    put(O/'completion.json',{'status':'COMPLETE_CAL100_THREE_METRICS','registration_sha256':a.sha256,'metric_identity':metric_identity,
      'counts':ledger.counts,'clean_logical_pairs':700,'fixed_failure_pairs':100,'unique_pairs':len(cached),'unresolved':0,
      'new_channel_frames':0,'VAR_calls':0,'bootstrap_calls':0,'model_training':0,'ended_unix':time.time(),
      'outputs':[desc(O/'clean_quality.csv'),desc(O/'metric_identity.json')],
      'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved()})
if __name__=='__main__':main()
