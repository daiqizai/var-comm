"""Frozen four computations for the independent new100 confirmation population.

No model is constructed on import. No prior score is accepted, no configuration
is selected, and no statistical interval is computed. The cache is local to one
actual score call and keyed by both exact pixel arrays and provider identity.
"""
from __future__ import annotations
import inspect
from pathlib import Path
import types
import numpy as np
import ep_new100_confirmation_core_v1 as plan
import h800_ep_pilot_metrics_v1 as original

SCHEMA='EP_NEW100_FOUR_METRIC_COMPUTATIONS_V1'
POPULATION='independent_confirmation_new100'
CAPS=plan.scientific_caps()['metrics']
METRICS=plan.METRICS
FLAGS=original.FLAGS
ORIGINAL_SHA='6bde19df620a7dbf391f5d406daa3b4c1de2a66d051c990c26ae9a2474c1ec8d'
PLAN_SHA='0e688098e8ecf41daccd5cdc93f91a08bbeac1d4f40b273a42d6b89d05284bc8'
FAILURE_FIELDS=('actual_RX_status','actual_received_profile','actual_header_ok','actual_crc_accepted','actual_parser_accepted',
    'actual_source_status','body_attempted','fixed_gray')
require=plan.require
array_sha=original.array_sha
pair_cache_key=original.pair_cache_key


def bound_assets(config):
    require(config['confirmation_population']==POPULATION and config['policy_selection'] is False and
        config['historical_score_reuse_allowed'] is False,'Independent new100 computation scope required')
    for name in ('confirmation_policy_bundle','confirmation_source_manifest'):plan.pin(config[name])
    policy=original.core.checked(config['confirmation_policy_bundle'])
    manifest=original.core.checked(config['confirmation_source_manifest'])
    require(policy['schema']=='EP_NEW100_FOUR_FROZEN_POLICY_PROJECTION_V1' and
        policy['strategy_selection_uses_new100'] is False and set(policy['policies'])==set(plan.METHODS) and
        all(set(v)=={'4','10','19'} for v in policy['policies'].values()),'Four calibration-frozen confirmation policies required')
    require(manifest['schema']=='EP_NEW100_SOURCE_CONTENT_ASSETS_V1' and manifest['source_count']==100 and
        manifest['policy_bundle']==config['confirmation_policy_bundle'],'Exact independent content source manifest required')
    return original.bound_assets(config)


def private_provider():
    require(original.core.sha(original.__file__)==ORIGINAL_SHA and original.core.sha(plan.__file__)==PLAN_SHA,
        'Frozen metric math or confirmation contract changed')
    namespace=dict(vars(original));namespace.update(__file__=__file__,SCHEMA=SCHEMA,POPULATION=POPULATION,CAPS=CAPS)
    functions={}
    for name,value in vars(original).items():
        if inspect.isfunction(value) and value.__module__==original.__name__:
            clone=types.FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=value.__kwdefaults__;namespace[name]=clone;functions[name]=clone.__code__ is value.__code__
    # Additional confirmation admission runs in __new__; the cloned constructor
    # still calls the exact original asset validator in this private namespace.
    attrs={};methods={}
    for name,value in vars(original.FourMetrics).items():
        if inspect.isfunction(value):
            clone=types.FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=value.__kwdefaults__;attrs[name]=clone;methods[name]=clone.__code__ is value.__code__
    require(all(functions.values()) and all(methods.values()),'Original scientific code objects changed')
    return type('OriginalFourMetricsPrivateNew100Scope',(),attrs),dict(original_adapter_sha256=ORIGINAL_SHA,
        functions_same_code_object=functions,methods_same_code_object=methods,new_population=POPULATION,
        original_globals_modified=False,numerical_forward_functions_changed=False)


class FourMetrics:
    def __new__(cls,config,ledger,guard):
        bound_assets(config);provider,proof=private_provider();value=provider(config,ledger,guard)
        value.evaluator.metadata['metric_selection_usage']='Independent new100 four-arm confirmation only; no policy selection'
        value.metadata.update(population=POPULATION,selection_objective=None,policy_selection=False,
            confirmation_policy_bundle=config['confirmation_policy_bundle'],confirmation_source_manifest=config['confirmation_source_manifest'],
            historical_compute_admission_anchor=config['whole_policy'],historical_score_reuse_allowed=False,
            DINO_L_metadata=value.evaluator.metadata,private_population_adapter_proof=proof,
            adapter_source=original.core.descriptor(__file__),original_compute_adapter=dict(path=original.__file__,sha256=ORIGINAL_SHA))
        value.identity=original.core.digest(value.metadata);guard();return value


def reference_pixels(record):
    # The original uint8 -> float32 /255 expression is called without modification.
    return original.core.reference_pixels(dict(record,archive=record['pixels_archive']))


def reconstruction_pixels(pin,expected,inside):
    path=inside(pin['path']);require(path.is_file() and not path.is_symlink() and original.core.sha(path)==pin['sha256'],
        'Actual received reconstruction archive changed')
    if 'bytes' in pin:require(path.stat().st_size==pin['bytes'],'Reconstruction archive length differs')
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)=={'image'},'Only the exact float32 reconstructed image may be scored');image=z['image'].copy()
    require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and
        ((image>=0)&(image<=1)).all() and array_sha(image)==expected,'Exact float32 reconstruction pixels required')
    return image


def validate_rows(rows,expected,records):
    require(len(records)==100 and [x['source_index'] for x in records]==list(range(100)) and
        len({x['source_id'] for x in records})==100,'Exact common fixed100 reference order required')
    plan.complete_grid(rows,expected,scored=True)
    for row,event in zip(rows,expected):
        require(row['policy']==event['policy'] and row['receiver_contract']==event['receiver_contract'] and
            row['source_id']==records[row['source_index']]['source_id'],'Each scored arm must retain its frozen policy and source')
        require(all(k in row for k in FAILURE_FIELDS),'Every logical receive failure must remain in the metric row')


def validate_completion(result,rows,request):
    validate_rows(rows,request['logical_events'],request['records']);plan.check_ledger(result['counts'],CAPS)
    counts=result['counts']['completed'];actual=result['actual_unique_pairs'];reused=result['reused_pairs']
    require(type(actual) is int and type(reused) is int and actual>=0 and reused>=0 and actual+reused==3600 and
        result['logical_frames']==3600 and result['full_grid_complete'] is True and result['historical_score_reuse']==0 and
        result['bootstrap_calls']==0 and result['policy_selection'] is False and counts['model_constructions']==3,
        'Complete new100 computation scope and exact actual pair accounting required')
    for name in ('reference_preparations','dinov2_vitl14_reference','convnext_reference'):
        require(counts[name]==100,'Complete100 actual reference preparations required')
    for name in ('image_scores','dinov2_vitl14_reconstruction','convnext_reconstruction','lpips_pair'):
        require(counts[name]==actual,'Actual pair closure disagrees with completed model calls')
    require(counts['lpips_alexnet_backbone_forward']==2*actual,'Actual LPIPS backbone closure differs')


def scores(r,evaluator,ledger,boundary,out,g,read):
    records=r['records'];events=r['logical_events'];plan.complete_grid(events,events)
    require(len(records)==100 and [x['source_index'] for x in records]==list(range(100)) and
        len(r['reconstruction_results'])==3600 and r['historical_score_reuse_allowed'] is False,
        'Closed full3600 reconstructions and no historical score cache required')
    out=Path(out);out.mkdir();(out/'pairs').mkdir()
    references={};verified={};cache={};rows=[];reused=0
    for index,pin in enumerate(r['reconstruction_results']):
        boundary();result=read(pin);event=events[index]
        require(result['frame_index']==index and result['logical_event']==event,'Actual reconstruction identity/order changed')
        record=records[event['source_index']];require(record['source_id']==event['source_id'],'Reference/source identity differs')
        if event['source_index'] not in references:
            target=reference_pixels(record)
            references[event['source_index']]=(target,array_sha(target),evaluator.prepare(target))
        target,reference_sha,prepared=references[event['source_index']];recon=result['reconstruction'];archive=recon['image_archive']
        key=plan.digest(dict(reference_sha256=reference_sha,reconstruction_sha256=recon['image_sha256'],metric_identity=evaluator.identity))
        ak=plan.canonical(archive);image=None
        if ak not in verified:
            image=reconstruction_pixels(archive,recon['image_sha256'],g.inside);verified[ak]=recon['image_sha256']
        else:require(verified[ak]==recon['image_sha256'],'One archive cannot claim different pixel hashes')
        if key not in cache:
            if image is None:image=reconstruction_pixels(archive,recon['image_sha256'],g.inside)
            require(pair_cache_key(target,image,evaluator.identity)==key,'Both actual pixel arrays and provider identity must match')
            values=evaluator.score(target,image,record['evaluation_class_index'],prepared)
            path=out/'pairs'/f'{len(cache):04d}.json'
            g.write(path,dict(metrics=values,metric_pair_key=key,reference_archive=record['pixels_archive'],
                reference_pixel_sha256=reference_sha,reconstruction_archive=archive,reconstruction_pixel_sha256=recon['image_sha256'],
                metric_identity=evaluator.identity,population=POPULATION,historical_score_reuse=False))
            cache[key]=(values,original.core.descriptor(path));origin='ACTUAL_NEW_PAIR'
        else:reused+=1;origin='SAME_RUN_IDENTICAL_PIXEL_PAIR'
        values,pair=cache[key]
        rows.append(dict(event,**values,metric_pair_key=key,metric_pair_result=pair,reconstruction_result=pin,
            physical_frame=result['physical_frame'],reuse=origin,**{k:result[k] for k in FAILURE_FIELDS}))
        if (index+1)%300==0:g.write(out/f'progress_{index+1:04d}.json',dict(logical_frames=index+1,actual_pairs=len(cache),reused_pairs=reused,counts=ledger.summary()))
    validate_rows(rows,events,records);counts=ledger.summary();plan.check_ledger(counts,CAPS);c=counts['completed'];actual=len(cache)
    require(c['model_constructions']==3 and c['reference_preparations']==len(references)==100 and
        c['image_scores']==actual and actual+reused==3600,'Finite complete new100 metric closure required')
    for name in ('dinov2_vitl14_reference','convnext_reference'):require(c[name]==100,'All100 source reference features required')
    for name in ('dinov2_vitl14_reconstruction','convnext_reconstruction','lpips_pair'):require(c[name]==actual,'Incomplete actual four-metric pair')
    require(c['lpips_alexnet_backbone_forward']==2*actual,'LPIPS actual backbone accounting differs')
    path=out/'metric_rows.json';g.write(path,rows)
    return dict(metric_rows=original.core.descriptor(path),logical_frames=3600,actual_unique_pairs=actual,reused_pairs=reused,
        counts=counts,population=POPULATION,historical_score_reuse=0,policy_selection=False,bootstrap_calls=0,
        full_grid_complete=True,all_logical_receive_failures_retained=True,statistics_not_computed=True)
