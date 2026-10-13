"""Four unchanged metrics, source averages left to the registered CPU analysis."""
import argparse
import csv
import time
import traceback
import numpy as np
from common import *

def run(path,parent):
    r=registration(path);out=ART/'score';out.mkdir();guard=guard_for(r,out,parent);device,g=resource_device(r,out);last=0.;ledger=None
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>5:
            snap,_=device.resource_snapshot(g.helper());last=time.monotonic();status(out/'resource.json',snap)
    caps=dict(model_constructions=3,reference_preparations=500,image_scores=4500,dinov2_vitl14_reference=500,convnext_reference=500,
        dinov2_vitl14_reconstruction=4500,convnext_reconstruction=4500,lpips_pair=4500,lpips_alexnet_backbone_forward=9000)
    try:
        require(read(ART/'visual/actual_wait.json')['returncode']==0,'Visual actual exit0 required')
        done=read(ART/'visual/completion.json');pins=read(checked(done['frames']));require(len(pins)==4500,'Full actual visual grid')
        ledger=Ledger(out/'calls.sqlite',caps,guard);write(out/'launch.json',dict(pid=os.getpid(),registration=descriptor(path)))
        import torch
        torch.cuda.set_per_process_memory_fraction((16<<30)/torch.cuda.get_device_properties(0).total_memory)
        import h800_ep_pilot_metrics_v1 as metric
        config=dict(r['metric_config']);config['runtime_identity']=dict(population='original_common500_frozen_policy_extension',physical_gpu=0,
            torch=torch.__version__,cuda=torch.version.cuda,FP32=True,TF32=False,policy_selection=False)
        boundary();evaluator=metric.FourMetrics(config,ledger,boundary)
        evaluator.evaluator.metadata['metric_selection_usage']='Frozen common500 extension evaluation; no selection'
        evaluator.metadata.update(population='original_common500_frozen_policy_extension',selection_objective=None,policy_selection=False)
        evaluator.identity=digest(evaluator.metadata);write(out/'metric_identity.json',evaluator.metadata)
        refs={};cache={};rows=[];started=time.monotonic()
        for index,pin in enumerate(pins):
            boundary();row=read(checked(pin));i=row['source_index'];record=r['sources'][i]
            require(row['frame_index']==index and row['source_id']==record['source_id'],'Scoring source order')
            if i not in refs:
                with np.load(checked(record['archive']),allow_pickle=False) as z:pixels=z['pixels'].copy()
                require(hashlib.sha256(pixels.tobytes()).hexdigest()==record['preprocessing_id'],'Frozen source pixel identity')
                target=pixels.astype(np.float32)/255;refs={i:(target,evaluator.prepare(target))}
            target,prepared=refs[i];recon=row['reconstruction'];key=digest(dict(source=array_sha(target),image=recon['image_sha256'],metric=evaluator.identity))
            if key not in cache:
                with np.load(checked(recon['image_archive']),allow_pickle=False) as z:image=z['image'].copy()
                require(array_sha(image)==recon['image_sha256'],'Actual float32 reconstruction differs')
                values=evaluator.score(target,image,record['evaluation_class_index'],prepared);cache[key]=values;origin='ACTUAL_NEW_PIXEL_PAIR'
            else:values=cache[key];origin='EXACT_PIXEL_PAIR_REUSE'
            public={k:row[k] for k in ('frame_index','method','source_index','source_id','snr_db','noise_seed','N','actual_m','actual_K','actual_T','arithmetic_bits','capacity_bits','received_m','received_K','physical_status','source_status','header_ok','crc_accepted','parser_accepted','fixed_gray')}
            public.update(values,source_prediction=prepared['convnext_prediction'],metric_identity=evaluator.identity,image_path=recon['image_archive']['path'],
                image_sha256=recon['image_sha256'],metric_pair_key=key,score_reuse=origin,policy_id=row['policy']['candidate_id'],model_identity=digest(r['native_identity']))
            write(out/'rows'/f'{index:04d}.json',public);rows.append(public)
            if (index+1)%90==0:status(OUT/'score_status.json',dict(completed_frames=index+1,total_frames=4500,actual_unique_pairs=len(cache),elapsed_seconds=time.monotonic()-started,counts=ledger.summary()))
        require(len(rows)==4500 and len({(x['source_id'],x['snr_db'],x['noise_seed']) for x in rows})==4500,'Exact final grid')
        csvpath=OUT/'frame_rows.csv'
        with csvpath.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        counts=ledger.summary();require(counts['unresolved']==0 and counts['completed']['reference_preparations']==500,'Complete finite metrics')
        write(out/'completion.json',dict(status='COMPLETE_SCORE4500',logical_frames=4500,metric_cells=18000,actual_unique_pixel_pairs=len(cache),
            frame_rows=descriptor(csvpath),metric_identity=descriptor(out/'metric_identity.json'),counts=counts,bootstrap_calls=0));ledger.close()
    except BaseException as e:
        write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(e),traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--parent',type=int,required=True);a=p.parse_args();run(a.registration,a.parent)
