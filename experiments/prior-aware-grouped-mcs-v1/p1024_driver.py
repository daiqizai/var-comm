"""Final development-only P1024 baseline: old 4/7/13 dB plus new 10 dB.

Uses the original selected 40k model and exact historical decoder batch shape.
Old float caches are preferred; missing images are decoded from the original
sealed receive latents and must match the previously scored RGB hashes.
No training, calibration, new source selection or policy selection is exposed.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import source_quality as q
import quality_driver as driver
import validation_metrics as validation
import p1024_10db as pcore

SNRS=(4,7,10,13)
SEEDS=(2001,2002,2003)
STATUS='P1024_BASELINE_EVALUATION_COMPLETE'
METRICS=('psnr_db','lpips_alex','dino_cosine','dinov2_vitl14_cosine','clip_image_cosine',
    'dists','dreamsim','ms_ssim','dino_specificity','resnet50_top1_label',
    'resnet50_top1_source_prediction','semantic_error','confidently_wrong',
    'convnext_top1_label','convnext_source_prediction_agreement','E','F_sq_error_zero_erasure_proxy')


class Paused(Exception):pass


def verify(bindings):
    for path,digest in bindings.items():q.require(q.sha(path)==digest,'Bound P baseline input changed: '+str(path))


def build_native(root,runtime,performance_runtime):
    """The external Native has the original replay engine; M1 Native does not."""
    root,runtime,performance_runtime=map(lambda x:Path(x).resolve(),(root,runtime,performance_runtime))
    sys.path.insert(0,str(runtime))
    common=driver.module('own_controls_common',runtime/'own_controls_common.py')
    implementation=driver.module('_uep_p1024_original_native',runtime/'own_controls_native.py')
    native=implementation.Native(root,driver.stop)
    performance=driver.module('_uep_p1024_health_polling',performance_runtime/'m1_performance.py')
    probe=native.assets.old.b;q.require(probe.SAFETY is not None,'Original safety monitor absent')
    probe.SAFETY=performance.HealthyPolling(probe.SAFETY);native.health_polling=probe.SAFETY
    native.driver_bindings={str(p):q.sha(p) for p in (Path(__file__).resolve(),Path(driver.__file__).resolve(),
        Path(q.__file__).resolve(),Path(validation.__file__).resolve(),Path(pcore.__file__).resolve(),
        runtime/'own_controls_common.py',runtime/'own_controls_native.py',runtime/'own_controls_phy.py',
        runtime/'external_eval_common.py',performance_runtime/'m1_performance.py')}
    native.driver_bindings.update(native.common.source_bindings());verify(native.driver_bindings)
    native.baseline_common=common
    return native


def cache_receipts(root):
    root=Path(root);base=root/'outputs/EXTERNAL-COMPARISON-20261004'
    paths=[(base/'own_controls/export-n1024_completion.json','OWN_CONTROL_STAGE_COMPLETE'),
        (base/'fixed80k_revision/five_column_comparison/replay_all16/manifest.json','EXACT_COMPLETED_RGB_EXPORT_PASS'),
        (base/'fixed80k_revision/m1_headline_16qam13/replay_all16/manifest.json','EXACT_COMPLETED_RGB_EXPORT_PASS')]
    outputs={};bindings={}
    for path,status in paths:
        if not path.exists():continue
        value=q.read(path)
        q.require(value['status']==status and value.get('synthetic') is False,'Existing cache receipt is not complete and real')
        if status=='OWN_CONTROL_STAGE_COMPLETE':q.require(value['stage']=='export-n1024','Wrong control export receipt')
        bindings[str(path)]=q.sha(path)
        for p,digest in value['outputs'].items():
            q.require(p not in outputs or outputs[p]==digest,'Conflicting existing float cache receipts')
            outputs[p]=digest
    return outputs,bindings


def expected_source_rows(engine,index):
    rows=[r for r in engine.by_source['N1024'][index] if r['method']=='P1024'
          and int(r['snr_db']) in (4,7,13) and int(r['noise_seed']) in SEEDS]
    by={(int(r['snr_db']),int(r['noise_seed'])):r for r in rows}
    q.require(len(rows)==len(by)==9 and set(by)=={(s,n) for s in (4,7,13) for n in SEEDS},'Original P population is incomplete')
    return by


def old_metadata(engine,index,original,measured,image,mode,bindings):
    snr,seed=int(original['snr_db']),int(original['noise_seed']);record=engine.records[index]
    digest=q.rgb_sha(image);target_hash=q.rgb_sha(engine.target(index))
    q.require(measured['image_sha256']==digest and measured['reference_sha256']==target_hash,
              'Existing/redecoded P pixels differ from the completed original float reconstruction')
    q.require(measured['source_id']==record['image_id'] and int(measured['N'])==1024
              and int(measured['snr_db'])==snr and int(measured['noise_seed'])==seed,'Original metric frame identity differs')
    row=dict(source_index=index,source_id=record['image_id'],preprocessing_id=record['preprocessing_id'],
        true_class_index=int(record['class_index']),N=1024,snr_db=snr,noise_seed=seed,method='P1024',
        phy_family='continuous',condition_label=False,selected_step=40000,decoder_id='Dc',
        original_replay_row_id=measured['replay_row_id'],
        image_sha256=digest,reference_sha256=target_hash,E=float(original['E']),
        waveform_sha256=original['waveform_sha256'],observation_sha256=original['observation_sha256'],
        latent_sq_err_final=float(original['latent_sq_err_final']),
        reconstruction_origin=mode,exact_completed_rgb_match=True,reconstruction_input_bindings=bindings,
        noise_namespace='original token_efficiency.execution.apply_channel / plugin.CELL')
    return row


def restore_source(native,api,index,cache_outputs,original_inventory,guard):
    e,p,t=api.engine,api.adapter,native.torch;target=e.target(index);record=e.records[index]
    original_path=native.root/f'outputs/UNIFIED-METRICS-20261002/source_checkpoints/{index:03d}.json'
    completed=native.baseline_common.original_checkpoint(original_path,original_inventory,index,record)
    measured={r['replay_row_id']:r for r in completed['rows']};wanted=expected_source_rows(e,index)
    inputs={str(original_path):q.sha(original_path)};cached={}
    for path in pcore.known_cache_paths(native.root,index):
        admitted=cache_outputs.get(str(path))
        if admitted is None:continue
        value=pcore.read_existing_cache(path,checkpoint_sha256=admitted,source_index=index,reference_sha256=q.rgb_sha(target))
        q.require(value['status']=='CACHE_AVAILABLE','Previously completed float cache disappeared')
        cp=q.read(path);proof=cp['float_reconstructions'];inputs[str(path)]=admitted;inputs[proof['path']]=proof['sha256']
        for row in value['rows']:
            key=(int(row['snr_db']),int(row['noise_seed']));image=value['images'][row['image_slot']]
            q.require(key not in cached or np.array_equal(cached[key][0],image),'Existing P cache copies disagree')
            cached[key]=(image,{str(path):admitted,proof['path']:proof['sha256']})
    newrows,newimages,newtarget=api.infer_source(index,guard=guard)
    q.require(np.array_equal(target,newtarget),'P new-point source reference differs')
    by={(10,s):(dict(row,reconstruction_origin='new_10dB_original_chain'),image) for s,row,image in zip(SEEDS,newrows,newimages)}
    decoded=0
    for snr in (4,7,13):
        missing=[seed for seed in SEEDS if (snr,seed) not in cached]
        rebuilt={}
        if missing:
            guard();indices=[p.obs_index[index,snr,seed] for seed in SEEDS]
            Z=p.obs['Z'][indices];ii=p.obs['source_indices'][indices]
            q.require(ii.tolist()==[index]*3,'Original sealed P latent source mapping differs')
            with t.no_grad():
                scalars,images=p.plugin.images_metrics(Z,ii,e.records,p.models,
                    e.data['reference'].to(e.loaded['device']),t.tensor(e.mismatch),save_images=True)
            q.require(len(images)==3,'Original P decoder batch omitted a noise repeat')
            decoded+=3
            for seed,image in zip(SEEDS,images):
                image=pcore.rgb(image.numpy());original=wanted[snr,seed]
                historical=measured[native.replay.row_id('N1024',original)]
                q.require(q.rgb_sha(image)==historical['image_sha256'],'P-only three-noise decoder is not exact old RGB')
                rebuilt[seed]=image
        for seed in SEEDS:
            original=wanted[snr,seed];historical=measured[native.replay.row_id('N1024',original)]
            if (snr,seed) in cached:
                image,proofs=cached[snr,seed];mode='existing_lossless_float_cache'
            else:
                image=rebuilt[seed];proofs={str(original_path):inputs[str(original_path)]};mode='original_sealed_latent_redecoded_exact_rgb'
            row=old_metadata(e,index,original,historical,image,mode,proofs)
            q.require(row['waveform_sha256']==newrows[0]['waveform_sha256'],'10dB changed original P transmission')
            by[snr,seed]=(row,image)
    rows=[];images=[]
    for snr in SNRS:
        for seed in SEEDS:
            row,image=by[snr,seed]
            row.update(image_slot=len(images),row_id=q.identity(dict(method='P1024',source_id=record['image_id'],snr_db=snr,noise_seed=seed)),
                label_conditioned=False,decoder_id='Dc',synthetic=False,training_updates=0,selection_updates=0,
                selected_checkpoint_sha256=api.identity['selected_checkpoint_sha256'],
                decoder_sha256=api.identity['decoder_sha256'],energy_kind='strict_per_frame_2N_continuous',
                F_sq_error_zero_erasure_proxy=row['latent_sq_err_final'])
            rows.append(row);images.append(image)
    counts=dict(existing_cached_frames=sum(r['reconstruction_origin']=='existing_lossless_float_cache' for r in rows),
        exact_old_redecoded_frames=sum(r['reconstruction_origin']=='original_sealed_latent_redecoded_exact_rgb' for r in rows),
        old_decoder_images_executed=decoded,new_10dB_frames=3)
    return rows,images,target,inputs,counts


def validate_source(saved,binding,index,record):
    q.require(saved['binding']==binding and saved['source_index']==index and saved['source_id']==record['image_id']
        and saved['payload_sha256']==q.identity({k:v for k,v in saved.items() if k!='payload_sha256'}),'P checkpoint seal differs')
    rows=saved['rows'];q.require([(r['snr_db'],r['noise_seed'],r['method']) for r in rows]
        ==[(s,n,'P1024') for s in SNRS for n in SEEDS],'P source frame grid differs')
    proof=saved['float_reconstructions'];q.require(q.sha(proof['path'])==proof['sha256'],'P float archive changed')
    with np.load(proof['path'],allow_pickle=False) as archive:
        q.require(set(archive.files)=={'images','source_rgb','row_ids','image_slots'},'P archive schema differs')
        images=archive['images'];reference=q.rgb_sha(archive['source_rgb'])
        q.require(images.shape==(12,3,256,256) and archive['row_ids'].tolist()==[r['row_id'] for r in rows]
            and archive['image_slots'].tolist()==list(range(12)),'P archive row mapping differs')
        for slot,row in enumerate(rows):
            q.require(row['source_index']==index and row['source_id']==record['image_id']
                and row['preprocessing_id']==record['preprocessing_id'] and row['N']==1024
                and row['selected_step']==40000 and row['image_slot']==slot
                and row['image_sha256']==q.rgb_sha(images[slot]) and row['reference_sha256']==reference,
                'P source/image identity differs')
            q.require(all(row.get(m) is not None and math.isfinite(float(row[m])) for m in METRICS),'Missing final P metric')
    verify(saved['input_bindings'])


def summarize(rows):
    q.require(len(rows)==1200,'P final population must have 1200 rows');result=[]
    for snr in SNRS:
        by={(r['source_index'],r['noise_seed']):r for r in rows if r['snr_db']==snr}
        q.require(len(by)==300 and set(by)=={(i,n) for i in range(100) for n in SEEDS},'Incomplete P source/noise population')
        for metric in METRICS:
            values=[np.mean([float(by[i,n][metric]) for n in SEEDS]) for i in range(100)]
            result.append(dict(N=1024,snr_db=snr,method='P1024',metric=metric,sources=100,frames=300,
                bootstrap_unit='source_mean_over_three_noise_repeats',bootstrap_replicates=10000,bootstrap_seed=20261002,
                **validation.interval(values)))
    return result


def run(native,api,admission,output,scorer,classifier):
    from score_actual import extra_metrics,write_csv
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    data=native.data('development');records=data['records'];e=api.engine
    q.require(native.loaded['identity']==admission['visual_identity'],'P visual identity differs from frozen UEP visual interface')
    q.require([r['image_id'] for r in records]==[r['image_id'] for r in e.records]
        and set(r['image_id'] for r in records).isdisjoint(admission['policies']['source_ids']),'P development source order or split differs')
    inventory=native.baseline_common.old_source_inventory(native.root)
    cache_outputs,cache_bindings=cache_receipts(native.root)
    bindings=dict(admission['input_bindings'],**inventory['bindings'],**cache_bindings,**e.manifest()['input_sha256'])
    sources=[dict(source_index=i,source_id=r['image_id'],preprocessing_id=r['preprocessing_id']) for i,r in enumerate(records)]
    source_bindings=dict(native.driver_bindings)
    import score_actual
    source_bindings[str(Path(score_actual.__file__).resolve())]=q.sha(score_actual.__file__)
    registration=dict(status='P1024_BASELINE_EVALUATION_REGISTERED',population='development',sources=100,source_identity=sources,
        N=1024,snrs=list(SNRS),noise_seeds=list(SEEDS),method='P1024',expected_rows=1200,
        selected_policies_sha256=admission['selected_policies_sha256'],input_bindings=bindings,source_bindings=source_bindings,
        p1024_identity=api.identity,visual_identity=native.loaded['identity'],metric_identity=scorer.identity,
        independent_classifier=classifier.identity,training_updates=0,selection_updates=0,calibration_executed=False,
        old_points=[4,7,13],new_point=10,missing_old_cache_rule='decode original sealed latent, require exact historical RGB hash',
        numerical_runtime=native.flags,pairing='same original source and nominal seed; original P and UEP noise namespaces differ')
    q.seal(output/'registration.json',registration);binding=q.identity(registration)
    outputs={};allrows=[];started=time.monotonic();totals={}
    classifier_sha=native.assets.old.b.state_sha256(classifier.model)
    def guard():
        if driver.STOP or (output/'STOP').exists():raise Paused('Source checkpoint boundary requested')
        native.common.check()
    for index,record0 in enumerate(records):
        if driver.boundary(native) or (output/'STOP').exists():raise Paused('Source checkpoint boundary requested')
        destination=output/'source_checkpoints'/f'{index:04d}.json';record=dict(record0,source_index=index)
        if destination.exists():saved=q.read(destination)
        else:
            began=time.monotonic();rows,images,target,inputs,counts=restore_source(native,api,index,cache_outputs,inventory,guard)
            scorer.prepare_source(record,target,index);values=scorer(record,images)
            extra_metrics(scorer,native,data,index,images,values,classifier,record)
            q.require(len(rows)==len(values)==12,'P scorer omitted a frame')
            for row,metrics in zip(rows,values):
                q.require(not(set(row)&set(metrics)),'Unified metric callback modifies P context')
                row.update(q.jsonable(metrics))
            archive=output/'reconstructions'/f'{index:04d}.npz';archive.parent.mkdir(parents=True,exist_ok=True)
            temp=archive.with_name(archive.name+'.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,images=np.stack(images),source_rgb=target,
                row_ids=np.asarray([r['row_id'] for r in rows]),image_slots=np.arange(12,dtype=np.int64))
            os.replace(temp,archive)
            saved=dict(binding=binding,source_index=index,source_id=record['image_id'],rows=rows,input_bindings=inputs,
                float_reconstructions=dict(path=str(archive),sha256=q.sha(archive)),counts=counts,
                source_total_seconds=time.monotonic()-began,synthetic=False)
            saved['payload_sha256']=q.identity(saved);q.seal(destination,saved)
        validate_source(saved,binding,index,record)
        for k,v in saved['counts'].items():totals[k]=totals.get(k,0)+v
        allrows.extend(saved['rows']);outputs[str(destination)]=q.sha(destination)
        outputs[saved['float_reconstructions']['path']]=saved['float_reconstructions']['sha256']
        q.write(output/'status.json',dict(status='RUNNING',sources=index+1,rows=len(allrows),counts=totals,elapsed_seconds=time.monotonic()-started))
    summary=summarize(allrows);classification,transitions=validation.summarize(allrows)
    for name,rows in [('metrics_per_frame.csv.gz',allrows),('metrics_summary.csv',summary),
        ('convnext_summary.csv',classification),('convnext_transitions.csv',transitions)]:
        path=output/name;write_csv(path,rows);outputs[str(path)]=q.sha(path)
    api.finish();q.require(native.assets.old.b.state_sha256(classifier.model)==classifier_sha,'ConvNeXt weights changed')
    verify(bindings);verify(source_bindings)
    complete=dict(status=STATUS,population='development',sources=100,rows=1200,snrs=list(SNRS),noise_seeds=list(SEEDS),
        selected_step=40000,registration_sha256=q.sha(output/'registration.json'),outputs=outputs,input_bindings=bindings,
        counts=totals,synthetic=False,training_updates=0,selection_updates=0,calibration_executed=False,
        independent_classifier_used_for_selection=False,elapsed_seconds=time.monotonic()-started)
    q.seal(output/'completion.json',complete);q.write(output/'status.json',dict(status='COMPLETE',sources=100,rows=1200))
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','policies','quality-bundle','quality-completion','bler','codebook','convnext-weights'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-runtime',type=Path);parser.add_argument('--performance-runtime',type=Path)
    args=parser.parse_args();args.root=args.root.resolve();args.output=args.output.resolve()
    from evaluate import admit_frozen
    admission=admit_frozen(args.policies,args.quality_bundle,args.quality_completion,args.bler,args.codebook)
    q.require(not (args.output/'failure.json').exists(),'Prior P failure requires review before retry')
    with driver.single_writer(args.output):
        signal.signal(signal.SIGTERM,driver.stop);signal.signal(signal.SIGINT,driver.stop)
        try:
            native=build_native(args.root,args.native_runtime or args.root/'outputs/EXTERNAL-COMPARISON-20261004/runtime',
                args.performance_runtime or args.root/'experiments/m1-n2048-full-grid-20261004')
            api=pcore.P1024Frozen10dB(native,stage='development',policy_path=args.policies,policy_sha256=q.sha(args.policies))
            scorer=driver.MetricScorer(native,args.root,args.output)
            classifier=validation.ConvNeXtValidation(args.convnext_weights,validation.WEIGHTS_SHA256,
                device=native.loaded['device'],stage='development',policy_path=args.policies,policy_sha256=q.sha(args.policies))
            complete=run(native,api,admission,args.output,scorer,classifier)
            print(complete['status'],complete['rows'],flush=True)
        except Paused as error:
            q.write(args.output/'status.json',dict(status='PAUSED',reason=str(error),pid=os.getpid()));return 75
        except Exception as error:
            q.write(args.output/'failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error)));raise
    return 0


if __name__=='__main__':raise SystemExit(main())
