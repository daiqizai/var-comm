"""DINO-only scoring of closed calibration outputs; no policy or formal test."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from contracts import image,check_identity,check_rx_closed,validate_grid,METRIC,RUNTIME_KEYS
import argparse,ctypes,traceback,math
import numpy as np


def main(role,path,expected):
    assert sha(path)==expected;r=read(path)
    for d in r['code']:checked(d)
    checked(r['scope']);checked(r['revision'])
    assert r['revision']['sha256']=='fb80b863dad31fa37cac9cc2e8d8ee4a7d5248071b3a73fdcc74c704517fc254'
    assert type(r['max_seconds'])is int and 0<r['max_seconds']<=43200
    assert r['calibration_only']is True and r['source_reference_calls']==0
    for d in r['RX_closures']:
        w=cr(d['actual_wait']);audit=cr(d['audit']);observation=cr(d['closed_process_observation'])
        assert observation['actual_wait']==d['actual_wait']and observation['audit']==d['audit']
        check_rx_closed(w,audit,observation)
    plan=cr(r['pair_plan']);sources=plan['sources'];assert len(sources)==1000
    assert {s['source_index']for s in sources}==set(range(1000))
    assert sum(len(p['frames'])for s in sources for p in s['pairs'])==18000
    count=sum(p.get('reuse')is None for s in sources for p in s['pairs'])
    assert r['caps']==dict(model_load=1,reconstruction_dino=count)
    out=L/role;out.mkdir(exist_ok=False);runtime_cache(out);os.sched_setaffinity(0,set(range(6)))
    beg=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps'])
    save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
    try:
        sys.path.insert(0,str(R/'experiments/arpc-external-baseline-20261019/cuda_recovery_v1'))
        from runtime import check_mapping
        mapping=check_mapping(ctypes.CDLL('libcuda.so.1'),r['cuda_admission'],os.environ.get('CUDA_VISIBLE_DEVICES'))
        sys.path.insert(0,str(R/'experiments/entropy-partial-mismatch-20261010/scripts'))
        import h800_ep_pilot_metrics_v1 as old
        config=r['metric_config'];assets=old.bound_assets(config)
        import torch
        torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
        assert old.flags_function(config['sources']['step0_reference_metrics.py']['path'])(torch)==old.FLAGS
        assert torch.__version__=='2.11.0+cu128'and torch.version.cuda=='12.8'
        torch.cuda.set_device(0);gpu=torch.cuda.get_device_properties(0)
        assert torch.cuda.mem_get_info()[0]>=8*1024**3
        torch.cuda.set_per_process_memory_fraction(6*1024**3/gpu.total_memory)
        metric=old.load_module(config['sources']['metric_models.py'],'_C_CALIBRATION_FROZEN_DINO')
        with metric.no_network():
            evaluator=ledger.call('model_load','DINO_L',lambda:metric.MetricEvaluator(assets['dinov2_manifest'],device='cuda:0',requested=('dinov2_vitl14',)))
        evaluator.metadata['metric_selection_usage']='C predicted-class fixed-frame cal1000 DINO-only calibration; no formal sources or model training.'
        runtime=dict(numeric_flags=old.FLAGS,float_dtype='float32',AMP=False,batch_size=1,
            device=mapping,host=os.uname().nodename,torch=torch.__version__,numpy=np.__version__,python=sys.version)
        donors={}
        for d in r['donor_metric_metadata']:
            m=cr(d);check_identity(m,evaluator.metadata,runtime)
            ident_key=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest();donors[ident_key]=d
        save(out/'metric_identity.json',dict(DINO_L_metadata=evaluator.metadata,**runtime,
            source_reference_reuse='Exact source pixels, finite float32(1,1024) features and unchanged numeric metric identity',
            no_other_metric_models=True,donors=r['donor_metric_metadata']))
        metric_id=evaluator.identity();scores=[];frames=[];cached={};(out/'pairs').mkdir()
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            for s in sources:
                assert time.monotonic()-beg<r['max_seconds']
                with np.load(checked(s['archive']),allow_pickle=False)as z:pixels=z['pixels']
                assert pixels.dtype==np.uint8 and pixels.shape==(3,256,256)
                target=image(pixels.astype(np.float32)/255);assert asha(target)==s['source_float_sha256']
                x=torch.as_tensor(target[None],device='cuda:0')
                with np.load(checked(s['reference']),allow_pickle=False)as z:features=z['features'].copy()
                assert features.dtype==np.float32 and features.shape==(1,1024)and np.isfinite(features).all()
                assert asha(features)==s['reference_feature_sha256']
                prepared=dict(digest=metric.image_digest(x),batch=1,evaluator_identity=evaluator.identity(),dinov2_vitl14=torch.as_tensor(features))
                for p in s['pairs']:
                    assert time.monotonic()-beg<r['max_seconds']and torch.cuda.mem_get_info()[0]>=2*1024**3
                    if p['image']is None:y=np.full((3,256,256),.5,np.float32)
                    else:
                        with np.load(checked(p['image']),allow_pickle=False)as z:y=z['image'].copy()
                    image(y);assert asha(y)==p['pixel_sha256']
                    key=old.pair_cache_key(target,y,metric_id)
                    if key in cached:
                        value=cached[key];origin='EXACT_CURRENT_PAIR_REUSE'
                    elif p.get('reuse')is not None:
                        reuse=cr(p['reuse']);assert reuse['metric_identity']in donors
                        assert reuse['source']==s['archive']and reuse['image_float_sha256']==asha(y)
                        assert reuse['pair_key']==old.pair_cache_key(target,y,reuse['metric_identity'])
                        value=float(reuse['scores'][METRIC]);origin='EXACT_CLOSED_PAIR_REUSE'
                    else:
                        def compute():
                            result=evaluator.score(x,torch.as_tensor(y[None],device='cuda:0'),[0],label_conditioned=False,prepared=prepared)
                            assert len(result)==1 and set(result[0])=={METRIC,'label_conditioned'}
                            return float(result[0][METRIC])
                        value=ledger.call('reconstruction_dino',key,compute);origin='NEW_ACTUAL_DINO_PAIR'
                    assert math.isfinite(value)
                    row=dict(source_id=s['source_id'],source_index=s['source_index'],pair_key=key,origin=origin,
                        source=s['archive'],source_float_sha256=asha(target),image=p['image'],pixel_sha256=asha(y),**{METRIC:value})
                    if key not in cached:
                        save(out/'pairs'/(key+'.json'),row);cached[key]=value
                    scores.append(row)
                    for f in p['frames']:
                        assert f['source_id']==s['source_id']and f['source_index']==s['source_index']
                        frames.append(dict(f,pair_key=key,**{METRIC:value}))
                if(s['source_index']+1)%50==0:print('C_DINO_SCORED_SOURCE',s['source_index']+1,len(scores),flush=True)
        validate_grid(frames,sources)
        assert ledger.summary()['unresolved']==0
        save(out/'actual_pair_scores.json',scores);csvsave(out/'logical_scores.csv',frames)
        actual_new=sum(v['count']for v in ledger.summary()['counts']if v['kind']=='reconstruction_dino'and v['status']=='COMPLETE')
        assert actual_new<=count
        save(out/'completion.json',dict(status='CAL1000_DINO_CLOSED_POLICY_NOT_SELECTED',registration_sha256=expected,
            logical_frames=18000,source_count=1000,source_pair_rows=len(scores),unique_pairs=len(cached),
            new_DINO_pairs=actual_new,reused_source_pair_rows=len(scores)-actual_new,source_feature_reuses=1000,
            source_reference_calls=0,model_training=0,PHY_calls=0,bootstrap_calls=0,
            ledger=ledger.summary(),metric_identity=pin(out/'metric_identity.json'),
            pair_scores=pin(out/'actual_pair_scores.json'),logical_scores=pin(out/'logical_scores.csv'),elapsed_seconds=time.monotonic()-beg))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True)
    p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
