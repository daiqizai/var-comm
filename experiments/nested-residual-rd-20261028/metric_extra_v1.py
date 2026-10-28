"""Only DreamSim or CLIP-I; reuse identity-matched native features and pairs."""
from residual_runtime import *
from fit_common import cr,digest
import importlib.util,traceback,ctypes,tempfile
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main(path,expected):
    assert sha(path)==expected;r=read(path);[checked(v)for v in r['code']];plan=cr(r['plan'])
    out=pathlib.Path(r['output']);out.mkdir(exist_ok=False);(out/'features').mkdir();(out/'pairs').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
    os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');os.sched_setaffinity(0,set(r['cpu_affinity']))
    import torch,torch.nn.functional as F
    torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
    assert torch.__version__==r['torch'];lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(dev),0)==0;bus=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(bus,32,dev)==0 and bus.value.decode().lower()==r['pci'].lower()
    torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
    ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic();cfg=r['config'];met=module(checked(cfg['clip']['loader']),'_B_FROZEN_EXTRA');dream=r['mode']=='dreamsim';metric='dreamsim'if dream else'clip_image_cosine'
    features={};scores={};cache={};reuse={}
    try:
        for desc in r['prior_dirs']:
            p=pathlib.Path(desc);identv=read(p/'identity.json');assert identv['torch']==r['torch']and identv['pci'].lower()==r['pci'].lower()and identv['loader']==cfg['clip']['loader']and identv['config']==r['frozen_metric_registration']and identv['FP32']and not identv['TF32']and identv['batch']==1
            assert identv['mode']==('three'if dream else'clip');done=read(p/'completion.json');assert done['status']=='COMPLETE_FORMAL_METRICS';cache.update(cr(done['features']));reuse.update(cr(done['scores']))
        if dream:
            sup=cfg['supplement'];overlay=cr(sup['dependency_overlay']);[checked(v)for v in overlay['files']];sys.path.insert(0,cfg['config']['supplemental_overlay_root']);ma=cr(sup['metric_assets']);torch.manual_seed(2026101401)
            with met.no_network():network=ledger.call('model_load','once',lambda:met.load_dreamsim(ma['models']['dreamsim'],torch.device('cuda:0')))
            preprocess=met.preprocess_dreamsim
        else:
            with met.no_network():network=ledger.call('model_load','once',lambda:met.load_clip(cfg['clip']['clip_assets'],torch.device('cuda:0')))
            preprocess=lambda x:met.preprocess_clip(met.validate_images(x))
        save(out/'identity.json',dict(metric=metric,torch=torch.__version__,pci=bus.value.decode(),FP32=True,TF32=False,AMP=False,batch=1,threads=6,interop=2,loader=cfg['clip']['loader'],config=r['frozen_metric_registration']))
        def feature(a,h):
            if h in cache:return torch.from_numpy(np.load(checked(cache[h]),allow_pickle=False)).cuda()
            if h not in features:
                def embed():
                    x=torch.from_numpy(a[None]).cuda();v=(network.embed(preprocess(x))if dream else network.encode_image(preprocess(x))).float().cpu().numpy();p=out/'features'/(h+'.npy')
                    with p.open('xb')as f:np.save(f,v,allow_pickle=False)
                    features[h]=pin(p);return v
                v=ledger.call('embedding',h,embed)
            else:v=np.load(checked(features[h]),allow_pickle=False)
            return torch.from_numpy(v).cuda()
        sources={s['source_id']:s for s in plan['sources']};ref={}
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            for j in plan['pairs']:
                assert time.monotonic()-begin<r['max_seconds']and not(L/'STOP').exists();key=j['pair_key'];s=sources[j['source_id']]
                if key in reuse:
                    z=reuse[key];assert z['pixel_sha256']==j['pixel_sha256']and z['reference_sha256']==s['reference_sha256'];scores[key]=dict(z,origin='EXACT_PAIR_REUSE');continue
                if s['source_id']not in ref:
                    with np.load(checked(s['archive']),allow_pickle=False)as z:a=z['pixels'].astype(np.float32)/np.float32(255)
                    assert asha(a)==s['reference_sha256'];ref[s['source_id']]=feature(a,s['reference_sha256'])
                with np.load(checked(j['image']),allow_pickle=False)as z:y=z['image'].copy()
                assert asha(y)==j['pixel_sha256']and y.dtype==np.float32 and y.shape==(3,256,256)and np.isfinite(y).all()and 0<=y.min()<=y.max()<=1
                fy=feature(y,j['pixel_sha256']);fx=ref[s['source_id']]
                def score():
                    cosine=F.cosine_similarity(fx,fy,dim=-1);v=float(((1-cosine)if dream else cosine)[0].item());assert np.isfinite(v);record=dict(pair_key=key,source_id=j['source_id'],source_index=j['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=j['pixel_sha256'],origin='NEW_EXTRA_PAIR',**{metric:v});save(out/'pairs'/(key+'.json'),record);return record
                scores[key]=ledger.call('pair',key,score)
        save(out/'scores.json',scores);save(out/'features.json',features);save(out/'completion.json',dict(status='COMPLETE_EXTRA_METRIC',metric=metric,scores=pin(out/'scores.json'),features=pin(out/'features.json'),count=len(scores),ledger=ledger.summary(),seconds=time.monotonic()-begin,new_reconstruction=0,new_PHY=0))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
