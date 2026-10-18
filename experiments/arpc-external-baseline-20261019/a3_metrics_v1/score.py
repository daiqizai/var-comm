"""Seven frozen metrics on actual independent evaluation pixels, no policy selection."""
import argparse,csv,gc,hashlib,importlib.util,json,math,os,pathlib,sys,time,traceback
import control as c
from data import METRICS
CAPS=dict(model_constructions=3,reference_preparations=100,image_scores=800,
 dinov2_vitl14_reference=100,dinov2_vitl14_reconstruction=800,convnext_reference=100,
 convnext_reconstruction=800,lpips_pair=800,lpips_alexnet_backbone_forward=1600,
 load_dists=1,load_dreamsim=1,load_ms_ssim=1,dreamsim_reference=100,dists_pair=800,
 dreamsim_reconstruction=800,ms_ssim_pair=800)
class Ledger:
    def __init__(self,guard):
        self.directory=c.RUN/'calls';self.directory.mkdir();self.counts={};self.guard=guard
    def call(self,kind,fn,**metadata):
        self.guard();n=self.counts.get(kind,0)+1;assert n<=CAPS[kind];self.counts[kind]=n
        p=self.directory/f'{kind}_{n:04d}';start=time.time();c.put(p.with_suffix('.reserved.json'),{'kind':kind,'ordinal':n,'metadata':metadata,'started_unix':start})
        try:
            value=fn();c.put(p.with_suffix('.complete.json'),{'seconds':time.time()-start});return value
        except BaseException:c.put(p.with_suffix('.failed.json'),{'error':traceback.format_exc()});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-registration',required=True);a=p.parse_args()
    assert c.desc(c.REG)['sha256']==a.expected_registration;reg=json.loads(c.REG.read_text())
    for d in reg['source_files']+reg['inputs']:c.verify(d)
    cache=c.RUN/'runtime_cache';cache.mkdir();(cache/'tmp').mkdir()
    os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import tempfile;tempfile.tempdir=str(cache/'tmp')
    import ctypes,numpy as np,torch
    sys.path.insert(0,str(c.E/'cuda_recovery_v1'));from runtime import check_mapping
    mapping=check_mapping(ctypes.CDLL('libcuda.so.1'),reg['resource']['cuda_admission'],os.environ.get('CUDA_VISIBLE_DEVICES'));c.put(c.RUN/'actual_cuda_mapping.json',mapping)
    assert torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8'
    torch.cuda.set_device(0);assert torch.cuda.mem_get_info()[0]>=8*1024**3
    torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
    def guard():
        assert time.time()<reg['absolute_deadline_unix'] and not (c.RUN/'STOP').exists()
        assert torch.cuda.mem_get_info()[0]>=2*1024**3
    ledger=Ledger(guard);plan=json.loads(pathlib.Path(reg['pair_plan']['path']).read_text())['sources']
    assert len(plan)==100;npairs=sum(len(s['pairs']) for s in plan);assert npairs<=800
    def reference(s):
        c.verify(s['archive'])
        with np.load(s['archive']['path'],allow_pickle=False) as z:x=z['pixels'].copy()
        assert x.dtype==np.uint8 and x.shape==(3,256,256)
        return x.astype(np.float32)/np.float32(255)
    def image(pair):
        if pair['image'] is None:y=np.full((3,256,256),.5,dtype=np.float32)
        else:
            c.verify(pair['image'])
            with np.load(pair['image']['path'],allow_pickle=False) as z:y=z['float_output'].copy()
        assert y.dtype==np.float32 and y.shape==(3,256,256) and np.isfinite(y).all() and (y>=0).all() and (y<=1).all()
        assert hashlib.sha256(y.tobytes()).hexdigest()==pair['image_float_sha256'];return y
    original=pathlib.Path(reg['original_four_metrics']['path']);sys.path.insert(0,str(original.parent))
    from metric_adapter import load
    metric=load(original);ev=metric.FourMetrics(reg['metric_config'],ledger,guard)
    ev.metadata.update(population='external_common100_v1',selection_objective=None,policy_selection=False,ARPC_source_native_float=True,metric_adapter=c.desc(pathlib.Path(__file__).parent/'metric_adapter.py'))
    ev.identity=metric.core.digest(ev.metadata)
    four_id=c.OUT/'four_metric_identity.json';c.put(four_id,{'identity':ev.identity,'metadata':ev.metadata})
    pairsdir=c.RUN/'pairs';pairsdir.mkdir();allpairs={};source_keys={}
    for s in plan:
        x=reference(s);prepared=ev.prepare(x);source_keys[s['source_id']]=[]
        for pair in s['pairs']:
            y=image(pair);key=metric.pair_cache_key(x,y,ev.identity)
            assert key not in allpairs
            values=ev.score(x,y,0,prepared)
            record={'pair_key':key,'source':s['archive'],'image':pair['image'],'image_float_sha256':pair['image_float_sha256'],'metric_identity':ev.identity,'scores':values}
            c.put(pairsdir/(key+'_four.json'),record);allpairs[key]=values;source_keys[s['source_id']].append(key)
        if (s['rank']+1)%10==0:print(json.dumps({'stage':'FOUR_METRICS','sources':s['rank']+1,'pairs':len(allpairs)}),flush=True)
    assert len(allpairs)==npairs and ledger.counts['image_scores']==npairs and ledger.counts['reference_preparations']==100
    del ev,prepared;gc.collect();torch.cuda.empty_cache()
    supplemental=json.loads(pathlib.Path(reg['supplemental_registration']['path']).read_text())
    for k in ['loader','metric_assets','dependency_overlay']:c.verify(supplemental[k])
    assets=json.loads(pathlib.Path(supplemental['metric_assets']['path']).read_text());overlay=json.loads(pathlib.Path(supplemental['dependency_overlay']['path']).read_text())
    for pin in overlay['files']:c.verify(pin)
    sys.path.insert(0,reg['supplemental_overlay_root'])
    spec=importlib.util.spec_from_file_location('_arpc_frozen_official_loader',supplemental['loader']['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    torch.manual_seed(2026101401);models={}
    with m.no_network():
        for name,loader in [('dists',m.load_dists),('dreamsim',m.load_dreamsim),('ms_ssim',m.load_ms_ssim)]:
            models[name]=ledger.call('load_'+name,lambda name=name,loader=loader:loader(assets['models'][name],torch.device('cuda:0')))
    suppid=c.OUT/'supplemental_identity.json';c.put(suppid,{'original_registration':reg['supplemental_registration'],'model_metadata':assets['models'],'protocol':supplemental['protocol'],'versions':m.package_versions(),'FP32':True,'TF32':False,'population':'external_common100_v1','no_selection':True,'mapping':mapping,'seed':2026101401})
    logical=[];pairrows=[]
    with torch.inference_mode():
        for s in plan:
            x=reference(s);xt=torch.from_numpy(x[None]).cuda()
            feat=ledger.call('dreamsim_reference',lambda:models['dreamsim'].embed(m.preprocess_dreamsim(xt)).float().detach().cpu())
            for pair,key in zip(s['pairs'],source_keys[s['source_id']]):
                y=image(pair);yt=torch.from_numpy(y[None]).cuda()
                dv=ledger.call('dists_pair',lambda:models['dists'](xt,yt,require_grad=False,batch_average=False).reshape(-1)[0].item())
                fy=ledger.call('dreamsim_reconstruction',lambda:models['dreamsim'].embed(m.preprocess_dreamsim(yt)).float())
                ds=(1-torch.nn.functional.cosine_similarity(feat.cuda(),fy,dim=-1))[0].item()
                ms=ledger.call('ms_ssim_pair',lambda:models['ms_ssim'](xt,yt,data_range=1.0,size_average=False,win_size=11,win_sigma=1.5,weights=supplemental['protocol']['ms_ssim']['weights'],K=(0.01,0.03))[0].item())
                values=dict(dists=dv,dreamsim=ds,ms_ssim=ms);assert all(math.isfinite(v) for v in values.values())
                c.put(pairsdir/(key+'_supplement.json'),{'pair_key':key,'scores':values,'identity':c.desc(suppid)})
                values={**allpairs[key],**values};assert set(values)==set(METRICS)
                pairrows.append({'source_id':s['source_id'],'source_index':s['source_index'],'pair_key':key,'image_float_sha256':pair['image_float_sha256'],**values})
                logical.extend([{**frame,'pair_key':key,**values} for frame in pair['frames']])
            if (s['rank']+1)%10==0:print(json.dumps({'stage':'SUPPLEMENTAL_METRICS','sources':s['rank']+1,'pairs':len(pairrows)}),flush=True)
    assert len(logical)==900 and len(pairrows)==npairs
    for kind,n in ledger.counts.items():
        assert len(list((c.RUN/'calls').glob(kind+'_*.complete.json')))==n
    assert not list((c.RUN/'calls').glob('*.failed.json'))
    assert ledger.counts['lpips_alexnet_backbone_forward']==2*npairs
    for name,rows in [('logical_scores.csv',logical),('actual_pair_scores.csv',pairrows)]:
        with (c.OUT/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    c.put(c.OUT/'score_completion.json',{'status':'EVALUATION_SEVEN_METRICS_COMPLETE','registration_sha256':a.expected_registration,'logical_frames':900,'unique_pairs':npairs,'source_count':100,'counts':ledger.counts,'unresolved':0,'bootstrap':0,'policy_selection':False,'truth_used_by_RX':False,'outputs':[c.desc(c.OUT/n) for n in ['logical_scores.csv','actual_pair_scores.csv','four_metric_identity.json','supplemental_identity.json']]})
if __name__=='__main__':main()
