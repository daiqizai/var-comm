"""Render sealed real UEP receive events and score all frozen development rows.

Only the actual accepted prefix and partial payload reach ReceivedRenderer.
TX payloads, class labels, source pixels and error truth stay outside recovery.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import csv
import gzip
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import source_quality as q
import quality_driver as driver
from actual_phy import accepted_state
import validation_metrics as validation

SNRS=(4,7,10,13)
SEEDS=(2001,2002,2003)
METRICS=('psnr_db','lpips_alex','dino_cosine','dinov2_vitl14_cosine','clip_image_cosine',
         'dists','dreamsim','ms_ssim','dino_specificity','resnet50_top1_label',
         'resnet50_top1_source_prediction','semantic_error','confidently_wrong',
         'convnext_top1_label','convnext_source_prediction_agreement','E',
         'F_sq_error_zero_erasure_proxy','header_ok','accepted_groups')


def verify(bindings):
    for path,digest in bindings.items():q.require(q.sha(path)==digest,'Bound real-event input changed: '+str(path))


def actual_state(event,codebook):
    """Reconstruct state from actual header-selected format and accepted bits."""
    q.require(event['N']==1024,'This deployment scorer is registered at N1024 only')
    expected=dict(kind='gray',prefix=[],partial_values=[],m=0,K=0,accepted_groups=0)
    if event['header_ok']:
        profiles={int(p['profile_id']):p for p in codebook['entries']}
        q.require(int(event['profile_id']) in profiles,'Actual header profile is not in frozen public codebook')
        profile=profiles[int(event['profile_id'])]
        q.require(profile['N']==event['N'] and len(event['groups'])==profile['G'],'Actual decoded group layout differs')
        payloads=[];accepts=[]
        for index,(received,group) in enumerate(zip(event['groups'],profile['groups'])):
            q.require(received['index']==index and received['source_bits']==group['source_bits']
                      and received['phy_key']==group['phy_key'] and type(received['crc_accept']) is bool,'Actual group format differs')
            ok=received['crc_accept'];value=received['accepted_payload']
            if ok:
                value=np.asarray(value)
                q.require(value.shape==(group['source_bits'],) and np.issubdtype(value.dtype,np.integer)
                          and np.isin(value,(0,1)).all(),'Invalid accepted group bits')
                value=value.astype(np.uint8)
            else:
                q.require(value is None,'Rejected hard bits must not be exposed as accepted information')
                value=np.empty(0,dtype=np.uint8)
            payloads.append(value);accepts.append(ok)
        expected=accepted_state(profile,payloads,accepts)
    else:
        q.require(not event['groups'],'Header rejection must not provide decoded body groups')
    q.require(event['state']==expected,'Saved state differs from actual accepted groups')
    if 'state_sha256' in event:q.require(event['state_sha256']==q.identity(expected),'Accepted state checksum differs')
    return expected


def event_completion(events,admission):
    events=Path(events).resolve();reg=q.read(events/'registration.json');done=q.read(events/'completion.json')
    q.require(reg['status']=='UEP_ACTUAL_EVENTS_REGISTERED' and done['status']=='UEP_ACTUAL_EVENTS_COMPLETE'
              and done['sources']==100 and len(reg['source_ids'])==len(set(reg['source_ids']))==100
              and done['registration_sha256']==q.sha(events/'registration.json')
              and done['synthetic'] is False and done['actual_bit_chain_executed'] is True,
              'All 100 actual development receive checkpoints must be sealed before scoring')
    q.require(reg['selected_policies_sha256']==admission['selected_policies_sha256']
              and reg['codebook_sha256']==q.identity(admission['codebook']['entries']), 'Actual events use different frozen policies/codebook')
    q.require(reg['received_format']=='actual_crc_accepted_prefix_v1','Actual receiver format differs')
    q.require(reg['labels_by_snr']==q.jsonable(admission['labels_by_snr']), 'Actual method labels differ from frozen policies')
    q.require(set(reg['source_ids']).isdisjoint(admission['policies']['source_ids']),'Calibration/development overlap')
    verify(reg['input_bindings']);verify(done.get('input_bindings',{}))
    for i in range(100):
        p=events/'source_checkpoints'/('%04d.json'%i)
        q.require(done['outputs'].get(str(p))==q.sha(p),'Unsealed/missing actual receive source')
    return reg,done


def frame_rows(checkpoint,source_index,source_id,codebook,labels_by_snr=None):
    q.require(checkpoint['source_index']==source_index and checkpoint['source_id']==source_id and checkpoint['complete'] is True,
              'Actual receive source identity differs')
    if labels_by_snr is not None:
        from evaluate import schedule_for_source
        expected=schedule_for_source(dict(source_index=source_index,source_id=source_id),labels_by_snr)
        actual=[{k:frame[k] for k in ('physical_frame_id','spec','methods')} for frame in checkpoint['frames']]
        q.require(actual==expected,'Actual frames differ from the complete frozen method/source/noise schedule')
    rows=[];seen=set()
    for frame in checkpoint['frames']:
        q.require(frame['actual_bit_chain_executed'] is True and frame['synthetic'] is False,'Real physical decoding is required')
        spec=frame['spec'];event=frame['receiver_event']
        q.require(spec['source_index']==source_index and spec['source_id']==source_id and spec['N']==1024
                  and spec['snr_db'] in SNRS and spec['noise_seed'] in SEEDS,'Unexpected actual development context')
        q.require(event['N']==spec['N'] and event['frame_counter']==spec['frame_counter']
                  and event['codebook_sha256']==q.identity(codebook['entries']),'Receive context differs')
        q.require(frame['physical_frame_id'] not in seen,'Duplicate physical frame');seen.add(frame['physical_frame_id'])
        state=actual_state(event,codebook)
        for method in frame['methods']:
            rows.append((frame,method,state))
    keys=[(r[0]['spec']['snr_db'],r[0]['spec']['noise_seed'],r[1]['family']) for r in rows]
    q.require(len(keys)==len(set(keys)),'Duplicate actual method/source/noise row')
    return rows


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'wt',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def method_context(frame,method):
    q.require(frame['actual_bit_chain_executed'] is True and frame['synthetic'] is False,
              'Scored actual context requires verified real physical execution')
    return dict(method=method['family'],family=method['family'],stable_id=method['stable_id'],
                actual_bit_chain_executed=True,synthetic=False)


def analyze(rows):
    groups=defaultdict(dict)
    for row in rows:
        key=(int(row['source_index']),int(row['noise_seed']));group=(int(row['snr_db']),row['method'])
        q.require(key not in groups[group],'Duplicate paired output');groups[group][key]=row
    expected={(i,s) for i in range(100) for s in SEEDS};draws=np.random.default_rng(20261002).integers(0,100,(10000,100))
    summary=[];means={}
    def interval(values):
        lo,hi=np.quantile(values[draws].mean(1),[.025,.975])
        return dict(mean=float(values.mean()),ci_low=float(lo),ci_high=float(hi),sources=100,frames=300,
                    bootstrap_unit='source_mean_over_three_noise_repeats',bootstrap_replicates=10000,bootstrap_seed=20261002)
    for (snr,method),by in sorted(groups.items()):
        q.require(set(by)==expected,'Incomplete 100 x 3 noise method group')
        for metric in METRICS:
            values=[by[i,s].get(metric) for i in range(100) for s in SEEDS]
            q.require(all(v is not None and math.isfinite(float(v)) for v in values),'Missing/nonfinite registered actual metric '+metric)
            source_means=np.asarray(values,dtype=np.float64).reshape(100,3).mean(1);means[snr,method,metric]=source_means
            summary.append(dict(N=1024,snr_db=snr,method=method,metric=metric,**interval(source_means)))
    contrasts=[]
    for snr in SNRS:
        methods={method for s,method in groups if s==snr}
        pairs=[('B3',b) for b in ('B0','B1','B2','B4','strong_baseline','validation_candidate') if b in methods]
        # Same-source/split controls remain separate from independently selected methods.
        pairs += [('matched_B3',b) for b in ('matched_B1','matched_B2') if 'matched_B3' in methods and b in methods]
        for a,b in pairs:
            for metric in METRICS:
                contrasts.append(dict(N=1024,snr_db=snr,method_A=a,method_B=b,metric=metric,delta='A_minus_B',
                                      **interval(means[snr,a,metric]-means[snr,b,metric])))
    return summary,contrasts


def extra_metrics(scorer,native,data,index,images,rows,classifier,record):
    """Independent post-policy validation plus established mismatch diagnostics."""
    from batch_speed import qualified_chunks
    torch=native.torch;source_prediction=classifier.predict(record['pixels'].astype(np.float32)/np.float32(255))
    misindex=native.assets.mismatch_permutation()[index]
    misreference=data['reference'][misindex].to(native.loaded['device']).float()[None]
    offset=0
    for count in qualified_chunks(len(images),8,[1,2,4,8]):
        with torch.inference_mode(),scorer.metric.no_network():
            batch=torch.from_numpy(np.stack(images[offset:offset+count])).to(native.loaded['device'])
            embeddings=scorer.quality.dino_features(native.loaded['dino'],batch)
            mismatch=torch.nn.functional.cosine_similarity(embeddings,misreference.expand_as(embeddings),dim=1).cpu().tolist()
            probability,prediction=scorer.evaluator.models['resnet50'](scorer.metric.preprocess_resnet50(batch)).float().softmax(-1).max(-1)
            probability,prediction=probability.cpu().tolist(),prediction.cpu().tolist()
        for j in range(count):
            row=rows[offset+j];q.require(prediction[j]==row['resnet50_prediction'],'Classifier confidence prediction differs')
            row.update(dino_mismatched=mismatch[j],dino_specificity=row['dino_cosine']-mismatch[j],
                       resnet50_top1_probability=probability[j],confidently_wrong=int(probability[j]>=.5 and row['semantic_error']))
            row.update(classifier.score(images[offset+j],true_label=int(record['class_index']),source_prediction=source_prediction))
            row['convnext_source_prediction_agreement']=int(row['convnext_top1_source_prediction'])
            row['convnext_model_id']='torchvision_ConvNeXt_Tiny_IMAGENET1K_V1'
        offset+=count


def score(native,admission,events,output,scorer,classifier):
    events,output=Path(events).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    reg,done=event_completion(events,admission)
    q.require(native.loaded['identity']==admission['visual_identity'],'Final calibration visual identity changed')
    data=native.data('development');records=data['records']
    q.require([r['image_id'] for r in records]==reg['source_ids'],'Original development source order differs')
    source_identity=[dict(source_index=i,source_id=r['image_id'],preprocessing_id=r['preprocessing_id']) for i,r in enumerate(records)]
    q.require(source_identity==reg['source_identity'],'Development preprocessing differs from actual link')
    bindings=dict(admission['input_bindings'],**{str(events/'registration.json'):q.sha(events/'registration.json'),
                 str(events/'completion.json'):q.sha(events/'completion.json')})
    clean_states={}
    for profile in admission['codebook']['entries']:
        m,k=profile['m'],profile['K'];name=q.state_id(m,k);clean_states[name]=dict(state_id=name,m=m,K=k)
        if profile['G']==2:
            j=profile['j'];name=q.state_id(j,0);clean_states[name]=dict(state_id=name,m=j,K=0)
    clean_states=q.validate_states(list(clean_states.values()))
    registration=dict(status='UEP_ACTUAL_SCORING_REGISTERED',population='development',sources=100,source_identity=source_identity,
        input_bindings=bindings,source_bindings=dict(native.driver_bindings,**{str(Path(__file__).resolve()):q.sha(__file__),
            str(Path(validation.__file__).resolve()):q.sha(validation.__file__)}),
        selected_policies_sha256=admission['selected_policies_sha256'],codebook_sha256=q.identity(admission['codebook']['entries']),
        visual_identity=native.loaded['identity'],metric_identity=scorer.identity,independent_classifier=classifier.identity,
        decoder='Dc',generation=q.GENERATION,receiver_input='actual accepted payload only',label_conditioned=False,
        expected_method_rows=done['method_rows'],training_updates=0,selection_updates=0,numerical_runtime=native.flags,
        development_source_Q=dict(diagnostic_only=True,policy_reselection_allowed=False,
            development_Q_used_for_selection=False,states=clean_states,receivers=['VAR','gray'],
            state_scope='only full/prefix states of final frozen public codebook',metric=q.PRIMARY_METRIC))
    q.seal(output/'registration.json',registration);binding=q.identity(registration)
    outputs={};allrows=[];allclean=[];started=time.monotonic();receiver=q.ReceivedRenderer(native.loaded,native.receiver)
    initial_classifier_state=native.assets.old.b.state_sha256(classifier.model)
    profiles={int(p['profile_id']):p for p in admission['codebook']['entries']}
    for index,record0 in enumerate(records):
        if driver.boundary(native) or (output/'STOP').exists():
            q.write(output/'status.json',dict(status='PAUSED_AT_SOURCE_BOUNDARY',sources=index));native.frozen();return None
        cp_path=events/'source_checkpoints'/('%04d.json'%index);cp=q.read(cp_path)
        q.require(cp['binding']==q.identity(reg) and cp['payload_sha256']==q.identity({k:v for k,v in cp.items() if k!='payload_sha256'}),
                  'Actual receive checkpoint seal differs')
        raw=frame_rows(cp,index,record0['image_id'],admission['codebook'],reg['labels_by_snr'])
        destination=output/'source_checkpoints'/('%04d.json'%index)
        if destination.exists():
            saved=q.read(destination)
            q.require(saved['binding']==binding and saved['input_checkpoint_sha256']==q.sha(cp_path)
                      and saved['payload_sha256']==q.identity({k:v for k,v in saved.items() if k!='payload_sha256'}), 'Scored source seal differs')
            proof=saved['float_reconstructions'];q.require(q.sha(proof['path'])==proof['sha256'],'Scored float archive changed')
        else:
            began=time.monotonic();record=dict(record0,source_index=index);target=record['pixels'].astype(np.float32)/np.float32(255)
            received_cache={};physical={};images=[];image_map={};rows=[];slots=[];clean_rows=[]
            # Diagnostic Q uses only already frozen endpoints. A reconstruction
            # may be reused by the real receiver solely after its actual
            # accepted-payload key matches; no CRC correctness oracle is used.
            clean_renderer=q.SourceRenderer(native.loaded,native.receiver,data['T'][index].cpu().numpy(),clean_states,use_snapshots=True)
            try:
                for state in clean_states+[dict(state_id='gray',m=0,K=0)]:
                    if state['state_id']=='gray':
                        image=np.full((3,256,256),.5,np.float32);kind='gray';key='gray'
                    else:
                        result=clean_renderer.render(state,'VAR');image=result['image'];kind='VAR';key=result['accepted_payload_key']
                        received_cache[key]=result
                    digest=q.rgb_sha(image)
                    if digest not in image_map:image_map[digest]=len(images);images.append(image)
                    clean_rows.append(dict(source_index=index,source_id=record['image_id'],preprocessing_id=record['preprocessing_id'],
                        state_id=state['state_id'],receiver=kind,m=state['m'],K=state['K'],image_slot=image_map[digest],
                        image_sha256=digest,reference_sha256=q.rgb_sha(target),accepted_payload_key=key,
                        population='development',diagnostic_only=True,policy_reselection_allowed=False,
                        development_Q_used_for_selection=False,label_conditioned=False))
            finally:
                clean_renderer.close()
            for frame,method,state in raw:
                fid=frame['physical_frame_id'];spec=frame['spec'];event=frame['receiver_event']
                if fid not in physical:
                    if state['kind']=='gray':
                        image=np.full((3,256,256),.5,np.float32);latent=None;accepted_key='gray'
                    else:
                        restored=receiver.render_received(state['prefix'],state['partial_values'],state['m'],state['K'],'VAR',received_cache)
                        image,latent,accepted_key=restored['image'],restored['fhat'],restored['accepted_payload_key']
                    digest=q.rgb_sha(image)
                    if digest not in image_map:image_map[digest]=len(images);images.append(image)
                    errors=native.common.legacy.latent_fields(data['F'][index],latent)
                    errors['F_sq_error_zero_erasure_proxy']=errors['latent_sq_err_final'] if latent is not None else errors['zero_erasure_proxy_sq_error']
                    physical[fid]=(image_map[digest],digest,accepted_key,errors)
                slot,digest,accepted_key,errors=physical[fid]
                tx=profiles[int(spec['profile_id'])];energy=float(frame['ledger']['E'])
                all_qpsk=all(g['modulation']=='QPSK' for g in tx['groups'])
                q.require(not all_qpsk or energy==2048.,'QPSK actual energy differs from E=2N')
                row=dict(N=1024,source_index=index,source_id=record['image_id'],preprocessing_id=record['preprocessing_id'],
                    snr_db=spec['snr_db'],noise_seed=spec['noise_seed'],**method_context(frame,method),
                    profile_id=spec['profile_id'],wire_key=spec['wire_key'],physical_frame_id=fid,
                    decoded_profile_id=event.get('profile_id'),header_ok=bool(event['header_ok']),accepted_groups=state['accepted_groups'],
                    received_m=state['m'],received_K=state['K'],receiver_state=state['kind'],accepted_payload_key=accepted_key,
                    label_conditioned=False,decoder_id='Dc',E=energy,energy_kind='strict_2N_QPSK' if all_qpsk else 'actual_fixed_constellation_QAM_energy',
                    source_group_modulations='/'.join(g['modulation'] for g in tx['groups']),
                    image_sha256=digest,reference_sha256=q.rgb_sha(target),**errors)
                # These diagnostics are copied only after the actual payload has
                # been rendered. They are never provided to ReceivedRenderer.
                truth=frame['offline_truth']
                for field in ('header_correct','header_false_accept','undetected_body_errors','noise_sha256'):
                    if field in truth:row[field]=truth[field]
                row['row_id']=q.identity({k:row[k] for k in ('source_id','snr_db','noise_seed','method')})
                rows.append(row);slots.append(slot)
            native.torch.cuda.synchronize();render_seconds=time.monotonic()-began
            scorer.prepare_source(record,target,index);metric_started=time.monotonic();values=scorer(record,images)
            actual_slots=sorted(set(slots))
            extra_metrics(scorer,native,data,index,[images[j] for j in actual_slots],[values[j] for j in actual_slots],classifier,record)
            for row,slot in zip(rows,slots):
                q.require(not(set(row)&set(values[slot])),'Metric callback changes actual link metadata');row.update(q.jsonable(values[slot]))
            for row in clean_rows:
                for field in ('psnr_db','lpips_alex','dino_cosine','dinov2_vitl14_cosine','clip_image_cosine','dists','dreamsim','ms_ssim'):
                    row[field]=values[row['image_slot']][field]
            native.torch.cuda.synchronize();metric_seconds=time.monotonic()-metric_started
            archive=output/'reconstructions'/('%04d.npz'%index);archive.parent.mkdir(parents=True,exist_ok=True)
            temp=archive.with_name(archive.name+'.tmp')
            with temp.open('wb') as handle:
                np.savez_compressed(handle,images=np.stack(images),source_rgb=target,row_ids=np.asarray([r['row_id'] for r in rows]),
                                    image_slots=np.asarray(slots,dtype=np.int64))
            os.replace(temp,archive)
            saved=dict(binding=binding,source_index=index,source_id=record['image_id'],input_checkpoint_sha256=q.sha(cp_path),
                rows=rows,clean_quality_rows=clean_rows,unique_images=len(images),physical_frames=len(physical),unique_received_payloads=len(received_cache),
                render_total_seconds=render_seconds,metrics_total_seconds=metric_seconds,
                source_total_seconds=time.monotonic()-began,float_reconstructions=dict(path=str(archive),sha256=q.sha(archive)))
            saved['payload_sha256']=q.identity(saved);q.seal(destination,saved)
        q.require([(r['snr_db'],r['noise_seed'],r['method']) for r in saved['rows']]==
                  [(f['spec']['snr_db'],f['spec']['noise_seed'],m['family']) for f,m,_ in raw], 'Scored method row mapping differs')
        q.require([(r['state_id'],r['receiver']) for r in saved['clean_quality_rows']]==
                  [(s['state_id'],'VAR') for s in clean_states]+[('gray','gray')], 'Frozen diagnostic source-Q coverage differs')
        allclean.extend(saved['clean_quality_rows'])
        allrows.extend(saved['rows']);outputs[str(destination)]=q.sha(destination)
        outputs[saved['float_reconstructions']['path']]=saved['float_reconstructions']['sha256']
        q.write(output/'status.json',dict(status='RUNNING',sources=index+1,rows=len(allrows),elapsed_seconds=time.monotonic()-started))
    q.require(len(allrows)==done['method_rows'],'Actual method population is incomplete')
    summary,paired=analyze(allrows);classification,transitions=validation.summarize(allrows)
    for name,rows in [('metrics_per_frame.csv.gz',allrows),('metrics_summary.csv',summary),('metrics_paired.csv',paired),
                     ('convnext_summary.csv',classification),('convnext_transitions.csv',transitions),
                     ('development_source_quality.csv',allclean)]:
        path=output/name;write_csv(path,rows);outputs[str(path)]=q.sha(path)
    native.frozen();q.require(native.assets.old.b.state_sha256(classifier.model)==initial_classifier_state,'Independent classifier weights changed')
    verify(bindings);verify(registration['source_bindings'])
    complete=dict(status='UEP_ACTUAL_SCORING_COMPLETE',population='development',sources=100,rows=len(allrows),
        outputs=outputs,registration_sha256=q.sha(output/'registration.json'),input_bindings=bindings,
        synthetic=False,training_updates=0,selection_updates=0,development_Q_rows=len(allclean),
        diagnostic_only_development_Q=True,development_Q_used_for_selection=False,elapsed_seconds=time.monotonic()-started)
    q.seal(output/'completion.json',complete);q.write(output/'status.json',dict(status='COMPLETE',sources=100,rows=len(allrows)))
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','events','output','policies','quality-bundle','quality-completion','bler','codebook','convnext-weights'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-runtime',type=Path);args=parser.parse_args()
    from evaluate import admit_frozen
    admission=admit_frozen(args.policies,args.quality_bundle,args.quality_completion,args.bler,args.codebook)
    admission['selected_policies_sha256']=q.sha(args.policies)
    event_completion(args.events,admission)
    with driver.single_writer(args.output):
        native=driver.build_native(args.root,args.native_runtime or args.root/'experiments/m1-n2048-full-grid-20261004')
        signal.signal(signal.SIGTERM,driver.stop);signal.signal(signal.SIGINT,driver.stop)
        scorer=driver.MetricScorer(native,args.root,args.output)
        classifier=validation.ConvNeXtValidation(args.convnext_weights,validation.WEIGHTS_SHA256,device=native.loaded['device'],
                    stage='development',policy_path=args.policies,policy_sha256=q.sha(args.policies))
        score(native,admission,args.events,args.output,scorer,classifier)


if __name__=='__main__':main()
