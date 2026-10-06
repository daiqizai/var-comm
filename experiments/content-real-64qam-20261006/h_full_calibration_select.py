"""Prepared pure CPU finalization of the frozen H calibration1000 policies.

Whole policies stay exactly as selected on200. Only the six previously frozen
partial profiles compete, three per SNR. This module neither reads development
nor launches anything. Its caller must separately verify the visual owner exit,
registration, all completion output files and packet-ledger closure.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from fractions import Fraction
from pathlib import PurePosixPath,PureWindowsPath
import re

import h_full_payload_cpu as physical

ARMS=('H16-R','H16-A','H64-R','H64-A')
SNRS=(13,19)
SEEDS=(6101,6102,6103)
COUNT=1000
EPSILON=1e-12
FINAL=('RAW_SOURCE_DECODED','ARITHMETIC_SOURCE_DECODED','WIRE_REJECT_GRAY','ARITHMETIC_SOURCE_INVALID_GRAY')
INPUT_NAMES=('shortlist','selected_whole','partial_reference','catalogue','calibration_registration','protocol','interpretation')


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(data):
    require(isinstance(data,bytes),'Original JSON bytes required');return hashlib.sha256(data).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def is_sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def absolute(path):return isinstance(path,str) and (PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute())
def decode(data):return json.loads(data.decode('utf-8-sig'))


def frozen_inputs(artifacts,receipt):
    require(set(artifacts)==set(INPUT_NAMES),'Exact original calibration input set required')
    values={};proofs={}
    for name,a in artifacts.items():
        require(set(a)=={'path','bytes'} and absolute(a['path']),'Expected absolute original input path and bytes')
        checksum=sha(a['bytes'])
        require(receipt['input_bindings'].get(a['path'])==checksum,'Input differs from actual renderer registration: '+name)
        values[name]=decode(a['bytes']);proofs[name]={'path':a['path'],'sha256':checksum}
    return values,proofs


def protocol_check(p,interpretation):
    require(p['schema']=='H_CODEC_PROTOCOL_V1' and p['status']=='FROZEN_BEFORE_DATA' and p['N']==1024
        and p['header_symbols']==68 and p['body_symbols']==956 and p['main_snrs_db']==list(SNRS)
        and p['population']['initial_and_full_cal_noise_seeds']==list(SEEDS)
        and p['population']['development_noise_seeds']==[6201,6202,6203]
        and p['calibration']['no_development_selection'] is True,'Original H selection protocol required')
    require(interpretation['schema']=='H_PRESCREEN_INTERPRETATION_V1'
        and interpretation['status']=='FROZEN_BEFORE_COARSE_DATA'
        and interpretation['original_protocol_changed'] is False,'Original tie-cost interpretation required')
    matched=p['matched_attribution']
    require((matched['m'],matched['K'],matched['nominal_rate'])==(7,0,'1/2') and matched['arms']==list(ARMS),
        'Fixed attribution point must remain m7/K0/rate1/2')
    require(p['development']['maximum_packet_calls']==13200,'Original development reserve required')


def receiver_row(row,candidate,ids,outputs,full_slot):
    i,seed=row['source_index'],row['noise_seed']
    require(type(i) is int and 0<=i<COUNT and row['source_id']==ids[i] and type(seed) is int and seed in SEEDS,
        'Full calibration source/noise identity differs')
    for name in ('candidate_id','arm','target_m','K','q','nominal_rate','snr_db','slot'):
        require(row[name]==candidate[name],'Frame differs from frozen candidate: '+name)
    phase='whole_calibration' if full_slot<8 else 'partial_calibration'
    require(row['full_slot']==full_slot and row['phase']==phase and row['noise_stage']==physical.NOISE_STAGE
        and row['public_frame_counter']==physical.frame_counter(full_slot,i,seed),'Actual full frame schedule differs')
    mse,psnr=row['mse'],row['psnr_db']
    require(type(mse) in (int,float) and type(psnr) in (int,float) and 0<mse<=1
        and math.isfinite(mse) and math.isfinite(psnr) and abs(psnr+10*math.log10(mse))<=1e-9,
        'Invalid or inconsistent actual frame PSNR/MSE')
    s=row['rx_summary'];state=row['source_status']
    require(state in FINAL and type(row['gray']) is bool and s['status']=='H_ACTUAL_RX_RECONSTRUCTION_COMPLETE'
        and s['source_decode_complete'] is True and s['new_packet_decodes']==0 and s['source_status']==state
        and s['gray'] is row['gray'] and row['gray'] is (state in FINAL[2:]),'Actual receiver state unresolved or inconsistent')
    for name in ('target_image_used_for_reconstruction','truth_correction','cached_clean_image_used'):
        require(s[name] is False,'Actual RX used source truth or proxy')
    for name in ('image_sha256','receiver_view_sha256'):
        require(is_sha(row[name]) and row[name]==s[name],'Actual image/RX provenance differs')
    require(absolute(row['image_archive']) and is_sha(outputs.get(row['image_archive']))
        and isinstance(row['image_key'],str) and row['image_key'],'Actual float32 RGB archive is unsealed')
    for name in ('received_m','received_K','received_mode','received_profile_id'):
        require(row[name]==s[name],'Received source identity differs')
    if state=='ARITHMETIC_SOURCE_DECODED':
        require(s['arithmetic_canonical_attempted'] is True and s['arithmetic_canonical'] is True
            and s['received_mode']=='arithmetic' and s['body_parser_accepted'] is True,'Canonical arithmetic acceptance missing')
    elif state=='ARITHMETIC_SOURCE_INVALID_GRAY':
        require(s['arithmetic_canonical_attempted'] is True and s['arithmetic_canonical'] is False
            and s['canonical_decode_invalid'] is True and s['received_mode']=='arithmetic','Canonical arithmetic rejection missing')
    elif state=='RAW_SOURCE_DECODED':
        require(s['received_mode']=='raw' and s['body_parser_accepted'] is True
            and s['arithmetic_canonical_attempted'] is False,'Actual raw source state differs')
    else:require(s['header_accepted'] is False or s['body_parser_accepted'] is False,'Accepted payload cannot be relabelled gray')
    return i,seed


def tie_key(c):
    require(c['mean_attempted_tx_probability_scales']==0 and c['q']==6,'Frozen partial TX cost/modulation differs')
    return (0,c['q'],Fraction(c['nominal_rate']),c['target_m'],c['K'],c['policy_key'])


def choose_partial(summaries,candidates):
    require(len(summaries)==3,'Exactly three frozen partial candidates per SNR required')
    best=max(s['mean_per_source_psnr_db'] for s in summaries)
    tied=[s for s in summaries if best-s['mean_per_source_psnr_db']<=EPSILON]
    return min(tied,key=lambda s:tie_key(candidates[s['full_slot']])),[s['candidate_id'] for s in tied]


def fixed_controls(catalogue,p,measured_policies=()):
    """Use the protocol's one existing whole-grid point; never fit an m/r here."""
    result=[];m=p['matched_attribution']['m'];rate=p['matched_attribution']['nominal_rate']
    for snr in SNRS:
        for arm in ARMS:
            q=4 if arm.startswith('H16') else 6;wire=dict(arm=arm,q=q,nominal_rate=rate,target_m=m,K=0)
            key=digest(wire);b=[b for b in catalogue['buckets'] if (b['q'],b['nominal_rate'])==(q,rate)]
            row=dict(**wire,snr_db=snr,policy_key=key,candidate_id='HWHOLE:'+key,
                family='FIXED_SOURCE_ATTRIBUTION',source_token_count=155,raw_source_bits=1860,
                selected_using_calibration_metrics=False,measured_in_this_full1000_grid=(key,snr) in measured_policies,
                mean_attempted_tx_probability_scales=float(m if arm.endswith('-A') else 0),
                no_source_truncation_differences=True,required_source_modes=['raw','arithmetic'] if arm.endswith('-A') else ['raw'])
            reason=None;profiles=[]
            if len(b)!=1 or b[0]['admission']!='ADMITTED' or not b[0].get('layout'):
                reason='ORIGINAL_BUCKET_NOT_ACTUAL_ADMITTED'
            elif b[0]['source_capacity']<1860:reason='FIXED_M7_RAW_DOES_NOT_FIT_NO_SUBSTITUTION'
            else:
                for mode in row['required_source_modes']:
                    matches=[x for x in catalogue['profiles'] if (x['q'],x['nominal_rate'],x['m'],x['K'],x['mode'])==(q,rate,m,0,mode)]
                    if len(matches)!=1 or matches[0]['admission']!='ADMITTED' or matches[0]['layout_id']!=b[0]['layout']['layout_id']:
                        reason='ORIGINAL_FIXED_SOURCE_MODE_NOT_ACTUAL_ADMITTED';break
                    profiles.append(dict(mode=mode,profile_id=matches[0]['profile_id'],profile_key=matches[0]['profile_key']))
            row.update(status='NOT_FEASIBLE' if reason else 'EXISTING_FIXED_POINT_FEASIBLE_NOT_EVALUATED',reason=reason,
                public_profiles=profiles if reason is None else [],fallback_permitted=False,
                arithmetic_choice='Frozen strict-shorter choice; raw wins tie; raw155-token payload itself fits')
            result.append(row)
    return result


def finalize_calibration(artifacts,frame_metrics_bytes,*,frame_metrics_path,render_completion):
    """Pure selection core. Accept only the entire sealed42,000-frame output."""
    receipt=render_completion;v,proofs=frozen_inputs(artifacts,receipt);p=v['protocol'];protocol_check(p,v['interpretation'])
    expected=dict(status='H_FULL1000_RX_COMPLETE',source_count=1000,frame_count=42000,
        phase_frame_counts={'whole_calibration':24000,'partial_calibration':18000},images_scored=True,
        source_decode_complete=True,arithmetic_source_decode_complete=True,new_packet_decodes=0,
        policy_selection=False,development_used=False,holdout_used=False)
    for name,value in expected.items():require(receipt.get(name)==value,'Incomplete/wrong full1000 actual-RX receipt: '+name)
    require(is_sha(receipt['registration_sha256']) and absolute(frame_metrics_path)
        and receipt['outputs'].get(frame_metrics_path)==sha(frame_metrics_bytes),'Actual frame metrics raw SHA differs')
    cal=v['calibration_registration'];ids=cal['source_ids']
    require(cal['stage']==cal['calibration_or_development']=='m1_calibration' and len(ids)==len(set(ids))==COUNT
        and ids==receipt['source_ids'],'Original calibration1000 population required; never development')
    schedule=physical.schedule(v['selected_whole'],v['partial_reference'],v['shortlist'],v['catalogue'],ids)
    candidates={e['full_slot']:e['candidate'] for e in schedule}
    require(all(candidates[i]['mean_attempted_tx_probability_scales']==float(candidates[i]['target_m'] if candidates[i]['arm'].endswith('-A') else 0)
        for i in range(14)),'Original TX tie costs changed')
    rows=decode(frame_metrics_bytes);require(isinstance(rows,list) and len(rows)==42000,'Full actual frame count incomplete')
    grouped={slot:{} for slot in candidates};images={}
    for row in rows:
        slot=row['full_slot'];require(type(slot) is int and slot in candidates,'Unregistered candidate slot')
        index,seed=receiver_row(row,candidates[slot],ids,receipt['outputs'],slot)
        require((index,seed) not in grouped[slot],'Duplicate source/noise frame');grouped[slot][index,seed]=row
        ref=row['image_archive'],row['image_key'];require(ref not in images or images[ref]==row['image_sha256'],'Image reference has conflicting hashes')
        images[ref]=row['image_sha256']
    evidence=[];summaries=[];grid={(i,n) for i in range(COUNT) for n in SEEDS}
    for slot,c in candidates.items():
        samples=grouped[slot];require(set(samples)==grid,'Full candidate source/noise Cartesian grid incomplete')
        psnrs=[];mses=[];states={k:0 for k in FINAL}
        for i,sid in enumerate(ids):
            source=[samples[i,n] for n in SEEDS];psnr=math.fsum(x['psnr_db'] for x in source)/3;mse=math.fsum(x['mse'] for x in source)/3
            psnrs.append(psnr);mses.append(mse)
            for x in source:states[x['source_status']]+=1
            evidence.append(dict(full_slot=slot,candidate_id=c['candidate_id'],arm=c['arm'],snr_db=c['snr_db'],
                source_index=i,source_id=sid,mean_noise_psnr_db=psnr,mean_noise_mse=mse,
                noise_samples=[{k:x[k] for k in ('noise_seed','psnr_db','mse','source_status','gray','image_archive','image_key','image_sha256','receiver_view_sha256')} for x in source]))
        mean=math.fsum(mses)/COUNT
        summaries.append(dict(full_slot=slot,phase=schedule[slot]['phase'],candidate_id=c['candidate_id'],arm=c['arm'],snr_db=c['snr_db'],
            source_count=COUNT,frame_count=3000,mean_per_source_psnr_db=math.fsum(psnrs)/COUNT,mean_overall_mse=mean,
            psnr_of_mean_mse_db=-10*math.log10(mean),mean_mse_is_selection_objective=False,final_receiver_status_counts=states,
            whole_policy_reselected=False,calibration_not_independent_test=True))
    selected=[];decisions=[]
    for snr in SNRS:
        family=[s for s in summaries if s['phase']=='partial_calibration' and s['snr_db']==snr]
        choice,tied=choose_partial(family,candidates);c=candidates[choice['full_slot']]
        selected.append(dict(candidate=copy.deepcopy(c),full_slot=choice['full_slot'],mean_per_source_psnr_db=choice['mean_per_source_psnr_db'],
            mean_overall_mse=choice['mean_overall_mse'],selection_scope='ACTUAL_FULL1000_THREE_FROZEN_PARTIAL_CANDIDATES'))
        decisions.append(dict(snr_db=snr,selected_candidate_id=c['candidate_id'],considered_candidate_ids=[r['candidate_id'] for r in family],
            within_epsilon_candidate_ids=tied,epsilon_psnr_db=EPSILON,tie='TX probability scales,q,rational rate,target m,K,lexical profile key'))
    controls=fixed_controls(v['catalogue'],p,{(c['policy_key'],c['snr_db']) for c in candidates.values()})
    return dict(status='H_FULL1000_CALIBRATION_POLICIES_FINALIZED',schema='H_FULL1000_SELECTION_V1',source_count=COUNT,
        source_ids=ids,measured_frames=42000,whole_candidates=copy.deepcopy(v['selected_whole']['selected_candidates']),
        whole_policy_reselected=False,whole_selection_origin='Original actual200 selected_whole bytes preserved; full1000 only validation',
        selected_partial=selected,partial_decisions=decisions,fixed_source_controls=controls,
        all_candidate_summaries=summaries,per_source_evidence=evidence,input_proofs=proofs,
        frame_metrics_proof={'path':frame_metrics_path,'sha256':sha(frame_metrics_bytes)},
        render_registration_sha256=receipt['registration_sha256'],render_completion_canonical_sha256=digest(receipt),
        selection_objective='Per-source mean of all3 actual noise PSNR, then1000 equally weighted sources; all failed frames included',
        search_claim='Only the original three partial profiles per SNR compete; whole winners fixed; no all-K real optimum claim',
        development_plan=dict(status='NOT_LAUNCHED_REQUIRES_MAIN_FULL_CALIBRATION_AND_SEPARATE_EXECUTION_REGISTRATION',
            whole_policy_snr_count=8,partial_policy_snr_count=2,fixed_source_policy_snr_count=8,
            main_references=[dict(modulation=m,snrs_db=list(SNRS),status='PENDING_SEPARATE_FULL_MAIN_CALIBRATION') for m in ('QPSK','16QAM')],
            maximum_total_policy_snr_points=22,development_source_count=100,noise_seeds=[6201,6202,6203],
            maximum_header_plus_body_calls=13200,additional_fixed_points_if_unavailable=False),
        new_candidate_search=False,new_packet_decodes=0,new_visual_inference=0,development_used=False,holdout_used=False,
        H_full_delivery_claimed=False,independent_main_system_complete=False,execution_registration_created=False,
        provenance_boundary='Caller must verify normal visual-owner/worker exits, registration/output SHA closure and unchanged ledger. This core verifies sealed JSON bytes and image/RX references; it does not reread pixels.')
