"""Complete original1000 reconstruction and four-metric evaluation primitives.

These functions have no command-line execution path or model imports. Owners
provide immutable closed inputs and finite ledgers. All logical failures remain.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import h800_ep_full_phy_core_v1 as full
import h800_ep_pilot_metrics_v1 as metric

core=full.pilot
require=core.require
POPULATION='original_calibration_full1000'
PROVIDER_SCOPE='original_calibration_pilot'
VISUAL_CAPS=dict(model_load=1,encoder=0,source_tx=0,source_rx=36000,var_render=36000,prior_scale=720000,decoder_forward=36000)
METRIC_CAPS=dict(model_constructions=3,reference_preparations=1000,image_scores=36000,
    dinov2_vitl14_reference=1000,dinov2_vitl14_reconstruction=36000,convnext_reference=1000,
    convnext_reconstruction=36000,lpips_pair=36000,lpips_alexnet_backbone_forward=72000)
FAILURE_FIELDS=('actual_RX_status','actual_received_profile','actual_header_ok','actual_crc_accepted','actual_parser_accepted')


def winners(rows,records,finalists,policy):
    """Mean over three noises per source, then all1000 sources; no bootstrap."""
    expected=full.frames(records,finalists,policy)
    require(len(rows)==len(expected),'Final selection requires the complete full1000 grid')
    grouped={}
    for row,event in zip(rows,expected):
        require(all(row[k]==event[k] for k in ('source_index','source_id','snr_db','noise_seed','candidate_id')),
            'Missing, duplicate, reordered or foreign full1000 metric row')
        for key in full.plan.METRICS:
            require(type(row[key]) in (int,float) and math.isfinite(row[key]),'Finite four-metric row required')
        require(row['convnext_top1_source_prediction'] in (0,1),'Agreement must be a prediction match indicator')
        grouped.setdefault((row['snr_db'],row['candidate_id'],row['source_index']),[]).append(row)
    chosen=full.admitted_finalists(finalists,policy);answer={}
    for snr,cids in chosen.items():
        means={}
        for cid in cids:
            source_means=[]
            for index in range(1000):
                observations=grouped[(snr,cid,index)]
                require([x['noise_seed'] for x in observations]==list(full.SEEDS),'Exactly three registered noise realizations required')
                source_means.append({m:math.fsum(x[m] for x in observations)/3 for m in full.plan.METRICS})
            means[cid]={m:math.fsum(x[m] for x in source_means)/1000 for m in full.plan.METRICS}
        ranked=sorted(cids,key=lambda cid:(-means[cid]['dinov2_vitl14_cosine'],cid))
        winner=ranked[0];candidate=next(x for x in full.plan.candidates() if x['candidate_id']==winner)
        answer[str(snr)]=dict(winner=winner,candidate=candidate,ranking=ranked,means=means,
            original_final_whole_winner=policy['policies']['EC_VAR_WHOLE'][str(snr)]['candidate_id'],
            whole_winner_retained=True,source_count=1000,noise_count=3,logical_frames=3000*len(cids))
    return dict(schema='H800_EP_FULL1000_CALIBRATION_WINNERS_V1',status='FULL1000_CALIBRATION_STRATEGY_FROZEN',
        population=POPULATION,calibration_source_ids=[r['source_id'] for r in records],snrs=answer,
        selection_metric='dinov2_vitl14_cosine',selection_direction='maximize',tie_break='candidate_id_lexicographic',
        averaging='three_noise_mean_per_source_then_1000_source_mean',noise_seeds=list(full.SEEDS),
        holdout_used_for_selection=False,new100_admitted=False,automatic_successor=False,final_strategy_frozen=True)


def image_bytes(pin,expected,g):
    path=g.inside(pin['path']);require(g.sha(path)==pin['sha256'],'Reconstruction archive SHA changed')
    with np.load(path,allow_pickle=False) as archive:image=archive['image'].copy()
    require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and
        (image>=0).all() and (image<=1).all() and metric.array_sha(image)==expected,'Reconstruction pixels changed')
    return image


def check_counts(counts,caps):
    require(counts['caps']==caps and counts['unresolved']==0 and counts['reserved']==counts['completed'] and
        set(counts['completed'])==set(caps) and all(0<=counts['completed'][k]<=v for k,v in caps.items()),
        'Finite actual-call ledger must close without unresolved reservations')


def images(r,backend,codec,partial,ledger,boundary,out,g,read):
    out=Path(out);out.mkdir();(out/'images').mkdir();(out/'logical').mkdir()
    cache={};physical_cache={};pins=[];reuse=dict(closed_pilot=0,same_run=0,new_actual_source_inputs=0)
    policy=read(r['policy']);full.complete_rows(r['logical_events'],r['records'],r['finalists'],policy)
    require(len(r['physical_frames'])==len(r['logical_events']),'Complete actual full PHY frames required')
    for pin in r['reuse_pilot_images']:
        boundary();result=read(pin);packet=read(result['physical_frame']);recon=result['reconstruction']
        key=core.digest(core.source_input(packet['actual_RX'],r['visual_identity']))
        require(key==recon['actual_source_input_key'],'Pilot image is not keyed to the identical actual source input')
        if key in cache:
            require(cache[key]['image_sha256']==recon['image_sha256'],'Conflicting same-input pilot reconstructions');continue
        image_bytes(recon['image_archive'],recon['image_sha256'],g)
        cache[key]=dict(recon,origin='CLOSED_PILOT',closed_pilot_result=pin)
    for index,pin in enumerate(r['physical_frames']):
        boundary();row=read(pin)
        require(row['frame_index']==index and row['logical_event']==r['logical_events'][index],'Full PHY logical identity changed')
        pp=row['physical_frame'];pk=core.canonical(pp)
        if pk not in physical_cache:physical_cache[pk]=read(pp)
        rx=physical_cache[pk]['actual_RX'];key=core.digest(core.source_input(rx,r['visual_identity']))
        if key not in cache:
            backend.source_index=row['logical_event']['source_index']
            image,evidence=core.link.recover_and_render(rx,codec,backend,partial,ledger,boundary)
            ip=out/'images'/f"{reuse['new_actual_source_inputs']:05d}.npz"
            with ip.open('xb') as stream:np.savez(stream,image=image)
            cache[key]=dict(image_archive=core.descriptor(ip),image_sha256=g.image_sha(image),actual_source_input_key=key,
                evidence=evidence,first_physical_frame=pp,origin='THIS_FULL1000')
            reuse['new_actual_source_inputs']+=1;origin='ACTUAL_NEW'
        else:
            origin=cache[key]['origin'];reuse['closed_pilot' if origin=='CLOSED_PILOT' else 'same_run']+=1
        result=dict(frame_index=index,logical_event=row['logical_event'],physical_frame=pp,reconstruction=cache[key],reuse=origin,
            actual_RX_status=rx['status'],actual_received_profile=rx['rx_profile'],actual_header_ok=rx['header']['header_ok'],
            actual_crc_accepted=None if rx['body'] is None else rx['body']['crc_accepted'],
            actual_parser_accepted=None if rx['body'] is None else rx['body']['parser_accepted'],
            reconstruction_evidence_describes_first_cached_decode=True)
        path=out/'logical'/f'{index:05d}.json';g.write(path,result);pins.append(core.descriptor(path))
    counts=ledger.summary();check_counts(counts,VISUAL_CAPS)
    require(len(pins)==sum(reuse.values())==len(r['logical_events']) and counts['completed']['model_load']==1 and
        counts['completed']['decoder_forward']==counts['completed']['var_render'],'Full1000 reconstruction closure incomplete')
    return dict(logical_frames=pins,reuse=reuse,counts=counts,old_pilot_counts_preserved=True,population=POPULATION)


def pilot_pairs(r,evaluator,g,read,boundary):
    """Closed owner identity is bound by registration; bytes and pairs are rechecked here."""
    cert=r['reuse_pilot_pairs'];cache={}
    identity=read(cert['metric_identity'])
    if identity['identity']!=evaluator.identity:return cache
    require(core.digest(identity['metadata'])==evaluator.identity,'Closed metric provider identity differs')
    rows=read(cert['metric_rows']);require(len(rows)==43200,'Complete closed pilot score population required')
    checked=set();references={};verified_images={}
    for row in rows:
        boundary();pin=row['metric_pair_result'];pk=core.canonical(pin)
        if pk in checked:continue
        checked.add(pk);pair=read(pin)
        require(pair['metric_identity']==evaluator.identity and pair['metric_pair_key']==row['metric_pair_key'] and
            all(pair['metrics'][m]==row[m] for m in full.plan.METRICS),'Closed pair row or provider changed')
        recon=read(row['reconstruction_result'])['reconstruction']
        record=r['records'][row['source_index']]
        require(row['source_index']<100 and row['source_id']==record['source_id'] and
            pair['reference_archive']==record['archive'] and pair['reconstruction_archive']==recon['image_archive'],
            'Closed pair original source/reconstruction binding differs')
        if row['source_index'] not in references:
            references[row['source_index']]=metric.array_sha(core.reference_pixels(record))
        ak=core.canonical(recon['image_archive'])
        if ak not in verified_images:
            image_bytes(recon['image_archive'],recon['image_sha256'],g);verified_images[ak]=recon['image_sha256']
        require(verified_images[ak]==recon['image_sha256'],'Conflicting closed reconstruction pixel identity')
        key=core.digest(dict(reference_sha256=references[row['source_index']],reconstruction_sha256=recon['image_sha256'],
            metric_identity=evaluator.identity))
        require(key==pair['metric_pair_key'],'Closed pair must match both actual pixel hashes and provider identity')
        if key in cache:require(cache[key][0]==pair['metrics'],'Conflicting closed pair scores')
        cache[key]=(pair['metrics'],pin,'CLOSED_PILOT')
    return cache


def scores(r,evaluator,ledger,boundary,out,g,read):
    out=Path(out);out.mkdir();(out/'pairs').mkdir();rows=[];references={};verified={};prepared={}
    policy=read(r['policy']);full.complete_rows(r['logical_events'],r['records'],r['finalists'],policy)
    require(len(r['reconstruction_results'])==len(r['logical_events']),'Complete full reconstruction list required')
    cache=pilot_pairs(r,evaluator,g,read,boundary);reuse=dict(closed_pilot=0,same_run=0,new_actual_pairs=0)
    for i,pin in enumerate(r['reconstruction_results']):
        boundary();result=read(pin);event=result['logical_event']
        require(result['frame_index']==i and event==r['logical_events'][i],'Actual reconstruction identity differs')
        index=event['source_index'];record=r['records'][index]
        require(record['source_id']==event['source_id'],'Full reference source ID changed')
        if index not in references:
            target=core.reference_pixels(record);references[index]=(target,metric.array_sha(target))
        target,reference_sha=references[index];recon=result['reconstruction'];archive=recon['image_archive']
        key=core.digest(dict(reference_sha256=reference_sha,reconstruction_sha256=recon['image_sha256'],metric_identity=evaluator.identity))
        ak=core.canonical(archive)
        if ak not in verified:image=image_bytes(archive,recon['image_sha256'],g);verified[ak]=recon['image_sha256']
        else:
            require(verified[ak]==recon['image_sha256'],'Conflicting actual image hash');image=None
        if key not in cache:
            if image is None:image=image_bytes(archive,recon['image_sha256'],g)
            require(metric.pair_cache_key(target,image,evaluator.identity)==key,'Actual metric pair identity differs')
            # Reference model calls occur only for a new pair, never to fill a cap.
            if index not in prepared:prepared[index]=evaluator.prepare(target)
            values=evaluator.score(target,image,record['evaluation_class_index'],prepared[index])
            pp=out/'pairs'/f"{reuse['new_actual_pairs']:05d}.json"
            g.write(pp,dict(metrics=values,metric_pair_key=key,reference_archive=record['archive'],
                reconstruction_archive=archive,metric_identity=evaluator.identity,population=POPULATION,
                compute_provider_scope=PROVIDER_SCOPE,provider_reused_unchanged=True))
            cache[key]=(values,core.descriptor(pp),'THIS_FULL1000');reuse['new_actual_pairs']+=1
        else:reuse['closed_pilot' if cache[key][2]=='CLOSED_PILOT' else 'same_run']+=1
        values,pair,origin=cache[key]
        rows.append(dict(source_index=index,source_id=event['source_id'],snr_db=event['snr_db'],noise_seed=event['noise_seed'],
            candidate_id=event['candidate_id'],**values,metric_pair_key=key,metric_pair_result=pair,reconstruction_result=pin,
            reuse=origin,**{k:result[k] for k in FAILURE_FIELDS}))
        if (i+1)%300==0:g.write(out/f'progress_{i+1:05d}.json',dict(logical_frames=i+1,reuse=reuse,counts=ledger.summary()))
    final=winners(rows,r['records'],r['finalists'],policy);counts=ledger.summary();check_counts(counts,METRIC_CAPS)
    actual=reuse['new_actual_pairs'];completed=counts['completed']
    require(sum(reuse.values())==len(rows) and completed['model_constructions']==3 and
        completed['reference_preparations']==len(prepared) and completed['image_scores']==actual,'Full scoring ledger differs')
    for kind in ('dinov2_vitl14_reference','convnext_reference'):require(completed[kind]==len(prepared),'Reference features incomplete')
    for kind in ('dinov2_vitl14_reconstruction','convnext_reconstruction','lpips_pair'):require(completed[kind]==actual,'Pair scores incomplete')
    require(completed['lpips_alexnet_backbone_forward']==2*actual,'LPIPS actual backbone count differs')
    g.write(out/'metric_rows.json',rows);g.write(out/'full_winners.json',final)
    return dict(metric_rows=core.descriptor(out/'metric_rows.json'),full_winners=core.descriptor(out/'full_winners.json'),
        logical_frames=len(rows),actual_reference_preparations=len(prepared),reuse=reuse,counts=counts,
        population=POPULATION,compute_provider_scope=PROVIDER_SCOPE,provider_reused_unchanged=True,
        old_pilot_and_PHY_counts_preserved=True,final_strategy_frozen=True,new100_admitted=False)
