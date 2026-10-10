"""Finite N1024 four-arm confirmation contracts; no CLI or scientific execution.

This module does not select or rank a source pool, open pixels, create models,
generate noise, call a channel/decoder, or bootstrap. Owners still need complete
actual closure, independent admission, device/resource limits and durable ledgers.
Preparing these functions is not a registered or completed confirmation run.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import re
import ep_plan as ep
import ep_source_population as population

SCHEMA='EP_NEW100_N1024_FOUR_ARM_CONFIRMATION_V1'
SNRS=(4,10,19)
SEEDS=(9301,9302,9303)
METHODS=('RAW_WHOLE','RAW_PARTIAL','EC_VAR_WHOLE','EC_VAR_PARTIAL')
CONTRASTS=(('RAW_PARTIAL','RAW_WHOLE'),('EC_VAR_PARTIAL','EC_VAR_WHOLE'))
METRICS=ep.METRICS
SOURCE_COUNT=100
LOGICAL_FRAMES=3600
PACKET_CAP=7200
RX_CAP=600  # Existing ep_plan confirmation cap; never silently raised to1800.
RAW_PLAN_SHA='1ab63b0e0bdde865d677467232c956ebbae746239ac300dd7723e22dbe8c7bce'
RAW_POLICY_SHA='7871ac7f9c619bd21c63ba31157738cd56c0dda9344c993e99b935f60ce70c0c'
WHOLE_POLICY_SHA='08097679bd01e6988fc4574d4f74ab57e1a891b2837a2f7a77713565ace7f70d'
SHA=re.compile(r'[0-9a-f]{64}')
require=ep.require


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def pin(value):
    require(isinstance(value,dict) and isinstance(value.get('path'),str) and value['path'] and
        isinstance(value.get('sha256'),str) and SHA.fullmatch(value['sha256']),'Exact artifact pin required')
    return copy.deepcopy(value)


def scientific_caps():
    old=ep.budgets()['confirmation']
    require(old==dict(new_sources=100,snrs=list(SNRS),noise_count=3,noise_seeds=list(SEEDS),methods=4,
        logical_frames=3600,packet_cap=7200,Encoder_cap=100,TX_VAR_provider_cap=100,independent_RX_VAR_cap=600),
        'Existing confirmation budget changed; independent review required')
    return dict(logical_frames=LOGICAL_FRAMES,packet_decodes=PACKET_CAP,
        content=dict(distinct_original_JPEG_files=100,canonical_preprocess=100,horizontal_flip_hashes=100,
            model_load=0,encoder=0,source_tx=0,source_rx=0,packet_decodes=0),
        source=dict(model_load=1,encoder=100,source_tx=100,source_rx=0,var_render=0,prior_scale=1000,decoder_forward=0),
        visual=dict(model_load=1,encoder=0,source_tx=0,source_rx=RX_CAP,var_render=3600,
            prior_scale=10*RX_CAP+10*3600,decoder_forward=3600),
        metrics=dict(model_constructions=3,reference_preparations=100,image_scores=3600,
            dinov2_vitl14_reference=100,dinov2_vitl14_reconstruction=3600,convnext_reference=100,
            convnext_reconstruction=3600,lpips_pair=3600,lpips_alexnet_backbone_forward=7200),
        statistics=dict(source_count=100,noise_count=3,bootstrap_replicates=10000,bootstrap_seed=2026100701,
            draw_matrices_max=1,summary_intervals_max=48,paired_intervals_max=24),
        source_qualification_RX_calls=0,training_updates=0,new_policy_search=0,automatic_retry=False)


def method_contract(method):
    require(method in METHODS,'Unknown confirmation method')
    raw=method.startswith('RAW_')
    return dict(method=method,representation='raw12' if raw else 'arithmetic',
        catalogue_count=433 if raw else 144 if method=='EC_VAR_WHOLE' else 360,
        PHY_provider='original_raw433' if raw else 'original_t1_phy' if method=='EC_VAR_WHOLE' else 'ep_phy',
        source_decoder='original_raw_KEEP' if raw else 'original_SourceCodec' if method=='EC_VAR_WHOLE' else 'PartialSourceCodec',
        body_failure='KEEP_actual_hard_raw12_tokens' if raw else 'constant_RGB_0.5',
        header_failure='constant_RGB_0.5',software_failure='STOP_preserve_unresolved_no_retry',
        target_K_zero_required=method in ('RAW_WHOLE','EC_VAR_WHOLE'),
        whole_action_allowed=True,received_profile_comes_from_actual_header=True)


def counter(source_index,snr,noise_seed):
    require(type(source_index) is int and 0<=source_index<100 and snr in SNRS and noise_seed in SEEDS,
        'Fixed confirmation source/SNR/noise domain required')
    return (SNRS.index(snr)*100+source_index)*3+SEEDS.index(noise_seed)


def verify_full_calibration(pins,read,validate_registration,recompute_winners):
    """Future owner supplies pinned readers and the unchanged full1000 validators.

    This is called only after the root verifies actual final closure. It recomputes
    the existing complete-grid winner summary, not any image metric or bootstrap.
    Bare winner JSON or a Boolean 'frozen' is insufficient.
    """
    require(set(pins)=={'completion','owner_actual_wait','request'},'Exact fullcal owner/request pins required')
    for item in pins.values():pin(item)
    done=read(pins['completion']);wait=read(pins['owner_actual_wait']);request=validate_registration(pins['request'])
    require(done['schema']=='H800_EP_ORIGINAL1000_FULL_METRICS_V2' and
        done['status']=='PASS_H800_ORIGINAL1000_FULL_FOUR_METRICS_AND_CALIBRATION' and
        done['request_sha256']==pins['request']['sha256'] and done['actual_wait']['success'] is True and
        done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and
        wait['actual_wait'] is True and wait['returncode']==0,'Actual full1000 owner/worker closure required')
    worker=read(pin(done['worker_completion']))
    require(worker['schema']==done['schema'] and worker['status']==done['status'] and
        worker['request_sha256']==done['request_sha256'] and worker['population']=='original_calibration_full1000' and
        worker['final_strategy_frozen'] is True and worker['new100_admitted'] is False,'Actual full1000 worker binding differs')
    result=worker['results'];counts=result['counts']
    require(counts['caps']==request['caps'] and counts['unresolved']==0 and counts['reserved']==counts['completed'] and
        set(counts['completed'])==set(request['caps']) and all(type(counts['completed'][k]) is int and
        0<=counts['completed'][k]<=v for k,v in request['caps'].items()),'Full calibration call ledger not closed')
    expected_caps=dict(model_constructions=3,reference_preparations=1000,image_scores=36000,
        dinov2_vitl14_reference=1000,dinov2_vitl14_reconstruction=36000,convnext_reference=1000,
        convnext_reconstruction=36000,lpips_pair=36000,lpips_alexnet_backbone_forward=72000)
    completed=counts['completed'];actual=result['reuse']['new_actual_pairs'];references=result['actual_reference_preparations']
    require(set(result['reuse'])=={'new_actual_pairs','same_run','closed_pilot'} and
        all(type(v) is int and v>=0 for v in result['reuse'].values()),'Invalid fullcal logical reuse accounting')
    require(request['caps']==expected_caps and completed['model_constructions']==3 and
        completed['reference_preparations']==completed['dinov2_vitl14_reference']==completed['convnext_reference']==references and
        completed['image_scores']==completed['dinov2_vitl14_reconstruction']==completed['convnext_reconstruction']==completed['lpips_pair']==actual and
        completed['lpips_alexnet_backbone_forward']==2*actual,'Actual fullcal model/reference/pair call closure differs')
    rows=read(pin(result['metric_rows']));winners=read(pin(result['full_winners']))
    expected=recompute_winners(rows,request['records'],request['finalists'],read(request['policy']))
    require(winners==expected and 9000<=len(rows)==len(request['logical_events'])==result['logical_frames']<=36000 and
        sum(result['reuse'].values())==len(rows),'Complete-grid actual metrics/final winners differ')
    require(winners['status']=='FULL1000_CALIBRATION_STRATEGY_FROZEN' and winners['final_strategy_frozen'] is True and
        winners['new100_admitted'] is False and not winners['holdout_used_for_selection'] and
        winners['noise_seeds']==[4101,4102,4103] and len(winners['calibration_source_ids'])==1000 and
        len(set(winners['calibration_source_ids']))==1000,'Full original1000 strategy freeze required')
    return dict(schema='EP_NEW100_FULL_CALIBRATION_CLOSURE_PROOF_V1',full_owner=copy.deepcopy(pins),
        worker_completion=done['worker_completion'],metric_rows=result['metric_rows'],winners=result['full_winners'],
        original_entropy_policy=request['policy'],complete_grid_verified=True,content_reads=0,
        model_calls=0,bootstrap_calls=0),winners


def policies(original_pins,winners,closure,read):
    """Project fixed choices only; this does not admit any source access."""
    require(closure['schema']=='EP_NEW100_FULL_CALIBRATION_CLOSURE_PROOF_V1' and closure['complete_grid_verified'] is True,
        'Complete actual fullcal proof required before projecting confirmation policies')
    expected={'raw_plan':RAW_PLAN_SHA,'raw_freeze':RAW_POLICY_SHA,'whole_policy':WHOLE_POLICY_SHA}
    require(set(original_pins)==set(expected) and all(pin(original_pins[k])['sha256']==v for k,v in expected.items()),
        'Exact original raw and whole-entropy policy pins required')
    raw_plan=read(original_pins['raw_plan']);raw_freeze=read(original_pins['raw_freeze']);whole_policy=read(original_pins['whole_policy'])
    require(closure['original_entropy_policy']['sha256']==WHOLE_POLICY_SHA,'Full calibration used a different whole anchor')
    require(raw_plan['schema']=='MAIN_RAW64_UNIFIED500_HOLDOUT_PLAN_V1' and raw_plan['source_count']==500 and
        raw_freeze['status']=='POLICIES_FROZEN_ON_CALIBRATION1000' and raw_freeze['holdout_used'] is False and
        raw_freeze['objective']=='dinov2_vitl14_cosine','Original calibration-only raw policies required')
    require(whole_policy['status']=='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1' and whole_policy['source_count']==1000 and
        whole_policy['holdout_used_for_selection'] is False,'Original entropy whole calibration required')
    require(winners['status']=='FULL1000_CALIBRATION_STRATEGY_FROZEN' and winners['final_strategy_frozen'] is True and
        winners['holdout_used_for_selection'] is False and winners['calibration_source_ids']==whole_policy['calibration_source_ids'] and
        len(set(winners['calibration_source_ids']))==1000 and winners['noise_seeds']==[4101,4102,4103],
        'Same complete original calibration population required')
    raw_winners={(x['snr_db'],x['family']):x['candidate_id'] for x in raw_freeze['winners']}
    raw_schedule={}
    for item in raw_plan['schedule']:
        for family in item['families']:
            key=item['snr_db'],family
            require(key not in raw_schedule,'Duplicate original raw method/SNR point')
            raw_schedule[key]={k:item[k] for k in ('candidate_id','profile_id','wire_key','m','K','modulation')}
    result={m:{} for m in METHODS};legal={c['candidate_id']:c for c in ep.candidates()}
    for snr in SNRS:
        for method,family in [('RAW_WHOLE','WHOLE'),('RAW_PARTIAL','PARTIAL')]:
            choice=raw_schedule[snr,family]
            require(choice['candidate_id']==raw_winners[snr,family] and
                (method!='RAW_WHOLE' or choice['K']==0),'Frozen raw choice changed')
            result[method][str(snr)]=copy.deepcopy(choice)
        old=whole_policy['policies']['EC_VAR_WHOLE'][str(snr)]['candidate_id'];entry=winners['snrs'][str(snr)]
        require(old in legal and legal[old]['target_K']==0 and entry['original_final_whole_winner']==old and
            old in entry['ranking'] and entry['whole_winner_retained'] is True and
            entry['winner']==entry['ranking'][0] and 1<=len(set(entry['ranking']))==len(entry['ranking'])<=4 and
            entry['source_count']==1000 and entry['noise_count']==3 and entry['logical_frames']==3000*len(entry['ranking']) and
            entry['candidate']==legal[entry['winner']],'Complete calibrated finalist winner/whole retention differs')
        result['EC_VAR_WHOLE'][str(snr)]=copy.deepcopy(legal[old])
        result['EC_VAR_PARTIAL'][str(snr)]=copy.deepcopy(legal[entry['winner']])
    return dict(schema='EP_NEW100_FOUR_FROZEN_POLICY_PROJECTION_V1',policies=result,full_calibration=copy.deepcopy(closure),
        original_raw_plan_sha256=RAW_PLAN_SHA,original_raw_policy_sha256=RAW_POLICY_SHA,original_whole_policy_sha256=WHOLE_POLICY_SHA,
        strategy_selection_uses_new100=False,source_access_admitted=False,
        interpretation='EC_VAR_PARTIAL names an allowed-partial policy and may choose an unchanged whole action')


def frame_grid(selection,selection_pin,content_gate,policy_bundle):
    """Construct only the already selected and deduplicated100; never select IDs."""
    pin(selection_pin)
    require(selection['schema']=='EP_NEW100_METADATA_SELECTION_V1' and selection['status']=='EP_SOURCE_IDS_FIXED_NO_PIXEL_ACCESS' and
        selection['source_count']==100 and selection['N']==1024 and selection['SNRs']==list(SNRS) and
        selection['noise_seeds']==list(SEEDS) and selection['methods']==list(METHODS) and
        selection['source_reselection_allowed'] is False,'Frozen exact new100 selection required')
    require(content_gate['schema']=='EP_NEW100_CONTENT_GATE_V1' and content_gate['status']=='EP_NEW100_COMPLETE_CONTENT_DEDUP_PASS' and
        content_gate['selection']==selection_pin and content_gate['registry']==selection['registry'] and
        content_gate['source_count']==100 and content_gate['conflicts']==[] and content_gate['Encoder_calls_allowed'] is True and
        content_gate['source_reselection_allowed'] is False,'Actual complete content gate must precede source calls')
    require(policy_bundle['schema']=='EP_NEW100_FOUR_FROZEN_POLICY_PROJECTION_V1' and
        policy_bundle['strategy_selection_uses_new100'] is False,'Four policies frozen before confirmation required')
    records=selection['records'];require(len(records)==100 and [r['source_index'] for r in records]==list(range(100)) and
        [r['source_id'] for r in records]==selection['source_ids'] and len(set(selection['source_ids']))==100,
        'One exact ordered source manifest for every arm required')
    require(len({population.canonical_id(r['source_id']) for r in records})==100,'Source aliases are forbidden')
    rows=[]
    for record in records:
        for snr in SNRS:
            for seed in SEEDS:
                for method in METHODS:
                    rows.append(dict(frame_index=len(rows),N=1024,source_index=record['source_index'],source_id=record['source_id'],
                        snr_db=snr,noise_seed=seed,method=method,public_frame_counter=counter(record['source_index'],snr,seed),
                        policy=copy.deepcopy(policy_bundle['policies'][method][str(snr)]),receiver_contract=method_contract(method)))
    require(len(rows)==LOGICAL_FRAMES,'Complete3600 frame plan required')
    return rows


def complete_grid(rows,expected,scored=False):
    require(len(expected)==len(rows)==LOGICAL_FRAMES,'All3600 logical outcomes required, including failures')
    keys=('frame_index','N','source_index','source_id','snr_db','noise_seed','method','public_frame_counter')
    seen=set();ids={}
    for result,event in zip(rows,expected):
        require(all(result[k]==event[k] for k in keys),'Missing, duplicated, reordered or foreign confirmation row')
        key=event['source_index'],event['snr_db'],event['noise_seed'],event['method']
        require(key not in seen,'Repeated confirmation cell');seen.add(key)
        require(event['N']==1024 and event['public_frame_counter']==counter(event['source_index'],event['snr_db'],event['noise_seed']) and
            event['receiver_contract']==method_contract(event['method']),'Counter/receiver semantics changed')
        require(ids.setdefault(event['source_index'],event['source_id'])==event['source_id'],'Different method source identity')
        if scored:
            for metric in METRICS:
                require(type(result[metric]) in (int,float) and math.isfinite(result[metric]),'Finite four-metric outcome required')
            require(result[METRICS[-1]] in (0,1),'Prediction agreement is a binary match, not class accuracy')
    require(len(seen)==100*3*3*4 and len(ids)==len(set(ids.values()))==100,'Complete four-method common population required')


def source_vectors(rows,expected):
    """Read-only aggregation for the future once-only statistics owner; no CI."""
    complete_grid(rows,expected,scored=True)
    keyed={(r['method'],r['snr_db'],r['source_index'],r['noise_seed']):r for r in rows}
    means={};paired={}
    for method in METHODS:
        for snr in SNRS:
            for metric in METRICS:
                means[method,snr,metric]=[math.fsum(keyed[method,snr,i,n][metric] for n in SEEDS)/3 for i in range(100)]
    for method,reference in CONTRASTS:
        for snr in SNRS:
            for metric in METRICS:
                paired[method,reference,snr,metric]=[a-b for a,b in zip(means[method,snr,metric],means[reference,snr,metric])]
    return means,paired


def reuse_key(evidence):
    """Actual complete PHY identity only; candidate names or CRC cannot grant reuse."""
    fields=('source_id','tokens_sha256','snr_db','noise_seed','public_frame_counter','payload_sha256','waveform_sha256',
        'standard_noise_sha256','observation_sha256','transmitted_profile','wire_session','receive_catalogue_sha256',
        'receiver_provider_sha256','PHY_runtime_identity_sha256')
    require(set(evidence)==set(fields),'Exactly the actual PHY identity fields are required')
    for key in fields:
        if key.endswith('sha256'):require(isinstance(evidence[key],str) and SHA.fullmatch(evidence[key]),'Invalid observed digest')
    return digest(evidence)


def check_ledger(counts,caps):
    require(counts['caps']==caps and counts['reserved']==counts['completed'] and counts['unresolved']==0 and
        set(counts['completed'])==set(caps) and all(type(counts['completed'][k]) is int and
        0<=counts['completed'][k]<=v for k,v in caps.items()),'Unresolved calls or a frozen finite cap exceeded')


def preparation_summary():
    return dict(schema=SCHEMA,status='IMPLEMENTATION_PREPARED_NOT_REGISTERED_OR_EXECUTED',N=1024,
        methods=[method_contract(m) for m in METHODS],snrs=list(SNRS),noise_seeds=list(SEEDS),caps=scientific_caps(),
        paired_contrasts=[dict(method=m,reference=r,delta='method minus reference') for m,r in CONTRASTS],
        LPIPS_sign_unchanged=True,agreement_internal_unit='fraction',agreement_display_unit='percentage_points_for_deltas',
        stage_order=['actual_full1000_policy_freeze','metadata_selection','content_hash_gate','source100','CPU_PHY','actual_RX_render','four_metrics','once_only_statistics'],
        shared_noise_provider='Original t1_phy.standard_noise(source_id,9301/9302/9303); same1024x2 variates for all four arms',
        counter_rule='(index_in_[4,10,19]*100+confirmation_source_index)*3+index_in_[9301,9302,9303]',
        entropy_wire_session='WCL_T1_ENTROPY_WHOLE_PHY_20261009_V1',raw_wire_session='original actual-body unchanged',
        historical_images_or_scores_reused=False,old500_mean_subtraction=False,
        IDs_selected=0,pool_hash_ranks_computed=0,new_pixel_reads=0,model_calls=0,channel_calls=0,bootstrap_calls=0,
        execution_ready=False,automatic_successor=False,
        blockers=['Actual full1000 metrics closure and root verification must precede metadata selection.',
            'Independent bounded content/source/three-provider PHY/visual/metric/statistics owners remain to be implemented and audited.',
            'Raw433, original T1 whole144 and EP360 must retain separate runtime/catalogue identities.',
            'Existing600 independent source-RX cap is a hard stop, not a guarantee of completion if accepted erroneous inputs exceed it.',
            'One old pool length differs from the official archive (val00019877:244013 versus507596 bytes); no silent replacement/exclusion.',
            'Actual selected source retrieval must validate all original lengths before content/Encoder; no substitute images.'])
