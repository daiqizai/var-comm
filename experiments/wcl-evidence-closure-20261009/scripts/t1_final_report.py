"""Read-only T1 publication tables from actually closed source/render/metric stages.

No scientific computation, image/model loading, channel calls or resampling.
Missing online timing stays explicitly missing; offline preparation is not timing.
"""
import argparse,csv,json,math,shutil
from collections import Counter
from pathlib import Path
from statistics import mean,median
from t1_entropy_core import OFFSETS,read,require,sha,verify,write,csv_write,choose_arithmetic
from t1_holdout_metadata import FAMILIES,SNRS,SEEDS,RAW_REFS,RAW_DROP,pinned
from t1_holdout_statistics import METRICS,bound_output
from t1_readonly_frame_join import joined_rows

PUBLIC={'EC_STATIC_WHOLE':'Static-entropy whole-scale transmission with VAR completion',
        'EC_VAR_WHOLE':'VAR-entropy whole-scale transmission with VAR completion',
        'RAW64_WHOLE_VAR_COMPLETION':'Raw whole-scale transmission with VAR completion (KEEP)',
        'RAW64_PARTIAL_VAR_COMPLETION':'Raw partial-scale transmission with VAR completion (KEEP)',
        'RAW64_WHOLE_CRC_DROP_VAR_COMPLETION':'Raw whole-scale transmission with VAR completion (CRC-DROP diagnostic)',
        'RAW64_PARTIAL_CRC_DROP_VAR_COMPLETION':'Raw partial-scale transmission with VAR completion (CRC-DROP diagnostic)'}

def csv_read(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def truth(value):
    require(value in ('True','False','',True,False,None),'Invalid boolean receipt')
    return None if value in ('',None) else value in ('True',True)

def interval_direction(metric,low,high):
    require(low<=high,'Interval reversed')
    if low<=0<=high:return 'interval includes zero'
    method_better=(low>0) if metric!='lpips_alex' else (high<0)
    return 'interval favors method' if method_better else 'interval favors reference'

def source_table(completion,role,policy):
    done=read(completion);manifest=read(bound_output(done,'manifest.json'));rows=[]
    for entry in manifest['records']:
        verify(entry['checkpoint'],entry['checkpoint_sha256']);sr=read(entry['checkpoint'])
        require(sr['schema']=='T1_SOURCE_ASSET_V1' and sr['source_role']==role,'Source role differs')
        for family in FAMILIES:
            stream=sr['streams'][family];attempts=sr.get('tx_prefixes',{}).get(family,{})
            lengths={m:(attempts[str(m)]['arithmetic_bits']if str(m)in attempts else stream[str(m)]['payload_bits'])
                for m in range(4,10)if str(m)in attempts or str(m)in stream}
            choices={s:choose_arithmetic(lengths,policy['policies'][family][str(s)],minimum_m=4)for s in SNRS}
            require(all(c['status']=='SOURCE_LENGTH_FITS_LAYOUT_PENDING'for c in choices.values()),'Frozen policy must have complete reachable source lengths')
            for s,c in choices.items():
                saved=sr.get('selected_by_snr',{}).get(family,{}).get(str(s))
                if saved is not None:require(c['actual_m']==saved['actual_m']and c['arithmetic_bits']==saved['arithmetic_bits'],'Reported length fallback differs from actual frozen choice')
            for m in range(4,10):
                e=attempts.get(str(m));cached=stream.get(str(m))
                length=e['arithmetic_bits'] if e else (cached['payload_bits'] if cached else None)
                selected=[s for s in SNRS if choices[s]['actual_m']==m]
                feasibility={}
                for s in SNRS:
                    c=policy['policies'][family][str(s)];chosen=choices[s]
                    feasibility.update({f'source_capacity_SNR{s}':c['source_capacity'],f'fits_frozen_MCS_SNR{s}':None if length is None else length<=c['source_capacity'],
                        f'actual_selected_m_SNR{s}':chosen['actual_m'],f'fallback_SNR{s}':chosen['actual_m']<c['target_m'],
                        f'target_m_SNR{s}':c['target_m'],f'fallback_attempts_SNR{s}':json.dumps(chosen['attempts'],sort_keys=True)})
                rows.append(dict(population=role,source_index=entry['source_index'],source_id=entry['source_id'],family=family,
                    complete_scale_m=m,token_count=OFFSETS[m],raw12_bits=12*OFFSETS[m],arithmetic_bits=length,
                    arithmetic_stream_includes_flush=length is not None,raw_fallback_used=False,
                    independent_RX_cache_available=cached is not None,
                    length_status='ACTUAL_ENCODED_LENGTH' if length is not None else
                        ('NOT_REQUIRED_BECAUSE_M6_ALREADY_FITS_MINIMUM_CAPACITY' if role=='calibration' and m<6 else 'NOT_COMPUTED_ABOVE_FROZEN_TARGET'),
                    selected_at_SNRs=json.dumps(selected),source_checkpoint_sha256=entry['checkpoint_sha256'],**feasibility))
    return rows,manifest

def m9_summary(rows):
    result=[]
    for role in ('calibration','holdout'):
        for family in FAMILIES:
            selected=[x for x in rows if x['population']==role and x['family']==family and x['complete_scale_m']==9]
            for s in SNRS:
                known=[x for x in selected if x['arithmetic_bits']is not None];fits=sum(x[f'fits_frozen_MCS_SNR{s}']for x in known)
                result.append(dict(population=role,family=family,snr_db=s,source_count=len(selected),m9_actual_encoded_count=len(known),
                    m9_uncomputed_count=len(selected)-len(known),m9_fits_frozen_MCS_count=fits,
                    m9_fits_fraction_all_sources=fits/len(selected)if len(known)==len(selected)else None,
                    m9_fits_fraction_known_sources=fits/len(known)if known else None,
                    actual_selected_m9_count=sum(x[f'actual_selected_m_SNR{s}']==9 for x in selected),
                    source_capacity=selected[0][f'source_capacity_SNR{s}'],m9_raw_bits=5088,length_and_CRC_paid=True))
    return result

def read_timing(args,freeze):
    if not args.timing_csv:return {}
    require(args.timing_completion,'Actual timing completion is required with its CSV')
    done=read(args.timing_completion)
    require(done['status']=='T4_FIXED16_ONLINE_TIMING_COMPLETE'and done['total_frames']==1152 and done['measured_frames']==576
        and done['source_count']==16 and done['no_online_tokens_or_output_cache']is True and done['packet_budget']['unresolved_frames']==0,
        'Only completed exclusive uncached online timing can enter quality-cost table')
    verify(args.timing_csv,done['outputs'][str(Path(args.timing_csv).resolve())])
    req=pinned(dict(path=done['request_path'],sha256=done['request_sha256']))
    if req['frozen_policy']!=freeze:
        projection=pinned(req['frozen_policy']);original=pinned(freeze)
        require(projection['original_policy_freeze']==freeze,'Timing must use identical frozen entropy policies')
        calibration=projection['calibration_completion'];cal=pinned(calibration)
        require(original['input_bindings'].get(calibration['path'])==calibration['sha256']and req['calibration_completion']==calibration
            and cal['status']=='T1_FULL_CALIBRATION_COMPLETE'and cal['source_count']==1000 and cal['noise_count']==3 and not cal['holdout_used'],
            'Timing policy projection must bind the same completed full calibration')
        rankings=csv_read(bound_output(cal,'calibration_rankings.csv'))
        rows=projection['rows'];require(len(rows)==6 and {(r['family'],int(r['snr_db']))for r in rows}=={(f,s)for f in FAMILIES for s in SNRS},
            'Complete six-point timing projection required')
        for row in rows:
            selected=original['policies'][row['family']][str(row['snr_db'])]
            winner=[r for r in rankings if r['family']==row['family']and int(r['snr_db'])==int(row['snr_db'])and int(r['rank'])==1]
            require(len(winner)==1 and all(row[k]==selected[k]==winner[0][k]for k in('candidate_id','nominal_rate'))
                and all(int(row[k])==int(selected[k])==int(winner[0][k])for k in('target_m','q')),'Timing projection differs from the actual rank1 frozen winner')
    mapping={'RAW64_WHOLE':RAW_REFS[0],'RAW64_PARTIAL':RAW_REFS[1],**{f:f for f in FAMILIES}}
    values={}
    for row in csv_read(args.timing_csv):
        if row['condition']!='all'or row['metric']not in('TX_seconds','RX_seconds','software_e2e_excluding_channel_seconds'):continue
        require(row['method']in mapping and int(row['snr_db'])in SNRS and int(row['sample_count'])==48 and int(row['source_count'])==16,'Exact fixed16 x3 unconditional timing')
        k=(mapping[row['method']],int(row['snr_db']));group=values.setdefault(k,{})
        require(row['metric']not in group,'Duplicate unconditional timing measurement')
        group[row['metric']]=row
        require(all(math.isfinite(float(row[x]))and float(row[x])>0 for x in('mean','median','p95')),'Actual finite positive timings required')
    expected={(f,s)for f in FAMILIES+RAW_REFS for s in SNRS}
    require(set(values)==expected and all(set(v)=={'TX_seconds','RX_seconds','software_e2e_excluding_channel_seconds'}for v in values.values()),'Complete matched12 method/SNR timing points required')
    return values

def run(args):
    out=Path(args.out).resolve();require(not out.exists(),'Fresh publication output required')
    stats=read(args.statistics_completion);scored=read(args.metric_completion);rendered=read(args.render_completion)
    require(stats['status']=='T1_POSTHOC_COMMON500_NEW_PAIRED_STATISTICS_COMPLETE'
        and scored['status']=='T1_POSTHOC_COMMON500_FOUR_METRICS_COMPLETE'
        and rendered['status']=='T1_POSTHOC_COMMON500_RENDER_COMPLETE','Actual complete T1 scientific inputs required')
    require(stats['source_ids']==scored['source_ids']==rendered['source_ids'] and len(stats['source_ids'])==500
        and stats['freeze']==scored['freeze']==rendered['freeze'],'Final population/freeze differs')
    require(stats['input_bindings'][str(Path(args.metric_completion).resolve())]==sha(args.metric_completion),
        'Statistics must bind the actual score completion')
    metric_request=pinned(dict(path=scored['request_path'],sha256=scored['request_sha256']))
    require(metric_request['render_completion']==dict(path=str(Path(args.render_completion).resolve()),sha256=sha(args.render_completion)),
        'Scores must bind actual rendered images')
    require(rendered['noise_seeds']==SEEDS and rendered['frame_count']==9000 and rendered['ledger']['unresolved']==0,
        'Complete registered6201/6202/6203 entropy grid and closed physical ledger required')
    policy=pinned(stats['freeze']);require(policy['holdout_used_for_selection'] is False,'Calibration-only selection required')
    gate=read(args.gate_completion)
    require(gate['status']=='T1_SOURCE32_REAL_RECOVERY_GATE_COMPLETE' and gate['exact_equal_image_pairs']==gate['static_raw_image_pairs']==96,
        'Actual first32 image recovery qualification required')
    source=read(args.holdout_source_completion)
    require(source['status']=='T1_POSTHOC_COMMON500_SOURCE_PREPARATION_COMPLETE' and source['source_count']==500
        and source['freeze']==stats['freeze'],'Actual frozen holdout source preparation required')
    frames=csv_read(bound_output(rendered,'per_frame.csv'));metrics=csv_read(bound_output(scored,'per_frame.csv'))
    key=lambda r:(r['family'],int(r['snr_db']),int(r['source_index']),int(r['noise_seed']))
    bymetric={key(r):r for r in metrics};require(len(bymetric)==18000 and len(frames)==9000,'Complete joined frames required')
    joined=joined_rows(scored);merged=[x for x in joined if x['family']in FAMILIES]
    raw_drop_rows=[x for x in joined if x['family']in RAW_DROP]
    summaries=csv_read(bound_output(stats,'summary.csv'));paired=csv_read(bound_output(stats,'paired.csv'))
    old=read(bound_output(stats,'raw_reference_summary_unchanged.json'))
    require(len(summaries)==48 and len(paired)==132 and len(old)==24,'Complete registered statistic tables required')
    for row in summaries+paired+old:
        require(int(row['source_count'])==500 and int(row['noise_count'])==3 and int(row['frame_count'])==1500
            and row['metric']in METRICS and int(row['snr_db'])in SNRS,'Exact four metrics,500 sources and3 noises required')
        require(float(row['ci_low'])<=float(row['mean'])<=float(row['ci_high']),'Mean or CI is invalid')
    require(all(x['delta_definition']=='method minus reference'for x in paired),'Original paired difference direction must stay unchanged')
    source_rows,cal_manifest=source_table(args.calibration_source_completion,'calibration',policy)
    holdout_rows,holdout_manifest=source_table(args.holdout_source_completion,'holdout',policy);source_rows+=holdout_rows
    require(cal_manifest['source_count']==1000 and holdout_manifest['source_count']==500
        and holdout_manifest['source_ids']==stats['source_ids'],'Source length population differs')
    breakdown=[];resources=[];config=[]
    for family in FAMILIES:
        for snr in SNRS:
            rows=[x for x in frames if x['family']==family and int(x['snr_db'])==snr];require(len(rows)==1500,'Balanced frame subset required')
            categories=Counter();ms=Counter();error_frames=error_tokens=accepted_tokens=0
            for row in rows:
                ms[int(row['actual_m'])]+=1
                if not truth(row['header_ok']):name='header_reject'
                elif truth(row['body_crc_accept']) is False:name='body_CRC_reject'
                elif truth(row['body_parser_accept']) is False:name='body_parser_reject'
                elif not truth(row['source_canonical_accept']):name='arithmetic_source_reject'
                else:name='source_decoded'
                categories[name]+=1;require(truth(row['gray'])==(name!='source_decoded'),'Failure output rule differs')
                if name=='source_decoded':
                    n=int(row['token_error_count']);error_frames+=n>0;error_tokens+=n;accepted_tokens+=int(row['token_error_denominator'])
                require(truth(row['diagnostic_truth_used_for_output']) is False,'Offline diagnostic entered output decision')
            c=policy['policies'][family][str(snr)]
            breakdown.append(dict(family=family,snr_db=snr,source_count=500,noise_count=3,frame_count=1500,
                **{k:categories[k] for k in ('header_reject','body_CRC_reject','body_parser_reject','arithmetic_source_reject','source_decoded')},
                gray_fraction=1-categories['source_decoded']/1500,source_decoded_frames_with_token_errors=error_frames,
                source_decoded_token_errors=error_tokens,source_decoded_token_count=accepted_tokens,
                source_decoded_token_error_rate=error_tokens/accepted_tokens if accepted_tokens else None,
                actual_m_counts=json.dumps(dict(sorted(ms.items())),sort_keys=True),
                fallback_fraction=sum(int(x['actual_m'])<int(x['target_m']) for x in rows)/1500))
            config.append(dict(c,family=family,public_name=PUBLIC[family],N=1024,snr_db=snr,K=0,
                actual_m_rule='per source; target descending to m4 by actual pure arithmetic length',holdout_used_for_selection=False))
            resources.append(dict(family=family,snr_db=snr,**{k+'_mean':mean(float(x[k]) for x in rows)
                for k in ('arithmetic_bits','actual_k','actual_n','actual_code_rate','header_symbols','body_symbols',
                    'header_energy','body_energy','frame_padding_symbols','known_information_padding_bits','E_frame','rho')}))
    raw_drop_breakdown=[]
    for f in RAW_DROP:
        for s in SNRS:
            rr=[x for x in raw_drop_rows if x['family']==f and int(x['snr_db'])==s]
            require(len(rr)==1500,'Complete raw diagnostic grid')
            counts=Counter(x['failure_state']for x in rr)
            raw_drop_breakdown.append(dict(family=f,snr_db=s,source_count=500,noise_count=3,frame_count=1500,
                **{n:counts[n]for n in('HEADER_REJECT_ORIGINAL_GRAY','BODY_CRC_REJECT_DIAGNOSTIC_GRAY',
                    'CRC_ACCEPTED_WITH_TOKEN_ERRORS_KEEP_UNCHANGED','CRC_ACCEPTED_NO_TOKEN_ERRORS_KEEP_UNCHANGED')},
                gray_frames=sum(x['gray']for x in rr),original_KEEP_gray_frames=sum(x['original_KEEP_gray']for x in rr),
                original_KEEP_token_errors=sum(x['original_KEEP_token_error_count']or 0 for x in rr),
                CRC_undetected_error_frames=sum(x['CRC_undetected_token_error']for x in rr),original_KEEP_outputs_modified=False))
    timing=read_timing(args,stats['freeze'])
    costs=[]
    for row in summaries+old:
        family=row['point_id'].rsplit('_SNR_',1)[0]
        if family not in FAMILIES+RAW_REFS:continue
        t=timing.get((family,int(row['snr_db'])))
        timing_values={}
        for metric,prefix in [('TX_seconds','tx'),('RX_seconds','rx'),('software_e2e_excluding_channel_seconds','software_e2e')]:
            for statistic in ('mean','median','p95'):timing_values[prefix+'_'+statistic+'_ms']=1000*float(t[metric][statistic])if t else ''
        costs.append(dict(row,public_name=PUBLIC[family],**timing_values,timed_frames=48 if t else '',timing_source_count=16 if t else '',
            timing_condition='all',online_timing_status='ACTUAL_T4_ONLINE_NO_CACHE' if t else 'T4_NOT_YET_MEASURED',cached_source_preparation_is_online_timing=False))
    conclusions=[dict(row,interval_direction=interval_direction(row['metric'],float(row['ci_low']),float(row['ci_high']))) for row in paired]
    out.mkdir(parents=True);outputs={}
    def table(name,rows):
        fields=list(dict.fromkeys(k for r in rows for k in r));normalized=[{k:r.get(k,'')for k in fields}for r in rows]
        csv_write(out/name,normalized);outputs[str(out/name)]=sha(out/name)
    for name,rows in [('per_frame.csv',merged),('raw_crc_drop_per_frame.csv',raw_drop_rows),('raw_crc_drop_breakdown.csv',raw_drop_breakdown),
        ('source_lengths.csv',source_rows),('m9_sendable_summary.csv',m9_summary(source_rows)),('summary.csv',summaries),('paired.csv',paired),
        ('original_raw_summary_unchanged.csv',old),('frozen_configurations.csv',config),('failure_breakdown.csv',breakdown),
        ('transmission_resources.csv',resources),('quality_vs_tx_cost.csv',costs),('paired_interpretation.csv',conclusions)]:table(name,rows)
    for name,p in [('frozen_policy.json',stats['freeze']['path']),('entropy_protocol.md',args.entropy_protocol)]:
        shutil.copyfile(p,out/name);outputs[str(out/name)]=sha(out/name)
    rt=dict(status='ACTUAL_ROUNDTRIP_AND_RECOVERY_VALIDATED',first32_gate=dict(path=str(Path(args.gate_completion).resolve()),sha256=sha(args.gate_completion)),
        clean_image_pairs=96,exact_equal_image_pairs=96,holdout_VAR_TX=source['new_VAR_TX_traversals'],
        holdout_VAR_independent_RX=source['new_VAR_independent_RX_calls'],holdout_static_independent_RX=source['new_static_independent_RX_calls'],
        independent_RX_required_for_all_selected_holdout_prefixes=True,old_full_model_and_Dc_frozen=True)
    write(out/'roundtrip_validation.json',rt);outputs[str(out/'roundtrip_validation.json')]=sha(out/'roundtrip_validation.json')
    lines=['# T1: Whole-scale entropy coding at the published common500 sources','',
        'This is a post-hoc same-source supplement to the already reported holdout. Policies were selected only on the original calibration1000, using DINOv2-L; the new experiment does not rewrite the original registration or results.','',
        'Two pure-arithmetic whole-scale families use the frozen tokenizer, null-class VAR, and Dc. Static probabilities use only the original training20k. Both pay the registered header, length and CRC costs, use actual LDPC reception, and fall back by measured bit length to a minimum whole prefix m4. No raw/arithmetic minimum substitution is used.','',
        'N=1024; SNRs 4, 10 and 19 dB; 500 sources and three noises per source. The new entropy namespace uses t1_phy.standard_noise with seeds 6201/6202/6203 fixed before execution. Raw comparisons reuse their original receive records. Pairing is by source; equal numeric seeds do not assert equal received observations across waveforms or implementations.','',
        '## Actual failure and fallback results','',
        '| Family | SNR | Gray frames / 1500 | Fallback fraction | Decoded frames with token errors |',
        '|---|---:|---:|---:|---:|']
    for row in breakdown:
        lines.append('| '+PUBLIC[row['family']]+' | '+str(row['snr_db'])+' | '+str(1500-row['source_decoded'])+' | '+format(row['fallback_fraction'],'.6f')+' | '+str(row['source_decoded_frames_with_token_errors'])+' |')
    lines+=['','Token-error diagnostics are evaluated after the actual output is fixed. They never select or repair a receiver output. Arithmetic parser failures, body CRC rejection and header rejection are retained separately in failure_breakdown.csv. The original raw KEEP outputs remain unchanged; CRC-DROP is a separately identified same-reception diagnostic.','',
        'The raw_crc_drop_breakdown.csv separates original header rejection, body-CRC rejection with diagnostic gray output, accepted erroneous tokens, and accepted correct tokens. Its token errors include both whole-prefix and transmitted partial tokens; accepted CRC-undetected errors retain their original output. raw_crc_drop_per_frame.csv preserves the exact receive proof and all four actual metrics.','',
        '## DINOv2-L paired comparisons','',
        'Each row is method minus reference after averaging the three noises within each source. Every preregistered pair is retained, including negative and zero differences.','',
        '| Method | Reference | SNR | Mean difference | 95% interval | Direction |','|---|---|---:|---:|---|---|']
    for row in conclusions:
        if row['metric']!='dinov2_vitl14_cosine':continue
        a=row['method'].rsplit('_SNR_',1)[0];b=row['reference'].rsplit('_SNR_',1)[0]
        lines.append('| '+PUBLIC[a]+' | '+PUBLIC[b]+' | '+row['snr_db']+' | '+format(float(row['mean']),'.7f')+' | ['+format(float(row['ci_low']),'.7f')+', '+format(float(row['ci_high']),'.7f')+'] | '+row['interval_direction']+' |')
    lines+=['','## Direct implications for the original partial-scale system','',
        'The following comparisons retain the direction entropy whole-scale minus original raw partial-scale. Positive PSNR/DINO and negative LPIPS favor entropy. Agreement differences are shown in percentage points. These statements are read directly from the paired CSV, including unfavorable evidence for the original system.','']
    for family in FAMILIES:
        for s in SNRS:
            group=[x for x in conclusions if x['method']==family+'_SNR_'+str(s)and x['reference']==RAW_REFS[1]+'_SNR_'+str(s)]
            require(len(group)==4,'All four direct mechanism comparisons required')
            pieces=[]
            for x in group:
                scale=100 if x['metric']=='convnext_top1_source_prediction'else 1
                label='ConvNeXt agreement (percentage points)'if scale==100 else x['metric']
                pieces.append(label+' '+format(scale*float(x['mean']),'+.7f')+' ['+format(scale*float(x['ci_low']),'.7f')+', '+format(scale*float(x['ci_high']),'.7f')+']; '+x['interval_direction'])
            lines.append('- '+PUBLIC[family]+' versus original raw partial-scale at '+str(s)+' dB: '+ '; '.join(pieces)+'.')
    lines+=['','The original raw partial-scale method cannot be described as uniformly superior when a paired interval favors the entropy method. A zero agreement difference is retained as zero; it does not cancel differences in other metrics.','']
    lines+=['','All four metrics and all 132 paired rows are in paired.csv. LPIPS retains its negative-is-better sign. ConvNeXt is source-prediction agreement, not classification accuracy; stored values are fractions and paired differences are absolute differences.','',
        'Intervals use the original source-level bootstrap implementation, 10,000 replicates and seed 2026100701. Existing raw intervals are copied; identical vectors reuse existing intervals. Intervals are pointwise, with no multiple-comparison adjustment. Inclusion of zero is not evidence of equivalence.','',
        '## Compression and computational cost','',
        'source_lengths.csv includes every m4–m9 prefix slot for calibration and holdout, per-SNR paid source capacity, actual fit, selected m and full fallback attempts. Actual encoded lengths include flushing. An absent short VAR calibration prefix is marked as unnecessary when m6 already fits the minimum capacity; a holdout prefix above the maximum frozen target is marked uncomputed. Neither is replaced with an estimated length. m9_sendable_summary.csv reports the actual fraction that fits each frozen MCS; missing m9 lengths never count as failures or successes.','',
        ('Actual online TX/RX/e2e mean, median and p95 timings are joined for '+str(len(timing))+' of twelve entropy/raw-anchor method/SNR points, using only condition=all and 48 measured calls from fixed16 sources. Seconds are converted to milliseconds. Conditional success/failure timing is not substituted for all-frame timing.' if timing else
            'T4 online sender measurements are not yet supplied. quality_vs_tx_cost.csv therefore marks all sender timings missing. Offline cached source preparation is not an online complexity measurement, and no speed or efficiency superiority is claimed.'),
        'All timing observations, including long-latency frames, are retained. An empirical mean can exceed the 95th percentile when a few calls are long; that is not a reason to trim them. Fixed16 timing is a small engineering sample and does not establish precise tail latency or universal deployment performance.',
        'A quality advantage at a particular metric/SNR does not establish universal optimality. If entropy improves quality, that improvement must be reported; any quality–compute tradeoff requires the actual T4 measurements.','',
        '## Execution and reuse','',
        '- Actual entropy packet decoder calls: '+str(rendered['ledger']['total'])+'.',
        '- New holdout VAR renders: '+str(rendered['new_VAR_render_calls'])+' plus one exact original-render replay qualification.',
        '- Four-metric execution counts: `'+json.dumps(scored['counts'],sort_keys=True)+'`.',
        '- This reporting command performs zero model, channel or bootstrap calls.','']
    (out/'REPORT_T1.md').write_text('\n'.join(lines),encoding='utf-8');outputs[str(out/'REPORT_T1.md')]=sha(out/'REPORT_T1.md')
    inputs=[args.statistics_completion,args.metric_completion,args.render_completion,args.calibration_source_completion,
        args.holdout_source_completion,args.gate_completion,args.entropy_protocol]+([args.timing_csv,args.timing_completion] if args.timing_csv else [])
    done=dict(status='T1_FINAL_QUALITY_REPORT_COMPLETE' if len(timing)==12 else 'T1_QUALITY_REPORT_COMPLETE_T4_TIMING_PENDING',
        actual_scientific_results_only=True,post_hoc_supplement=True,source_count=500,noise_seeds=SEEDS,
        holdout_used_for_selection=False,new_model_calls=0,new_packet_decodes=0,new_bootstrap_calls=0,
        online_timing_points=len(timing),input_bindings={str(Path(p).resolve()):sha(p) for p in inputs},outputs=outputs)
    write(out/'completion.json',done);return done

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('statistics-completion','metric-completion','render-completion','calibration-source-completion','holdout-source-completion','gate-completion','entropy-protocol','out'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--timing-csv');p.add_argument('--timing-completion');print(run(p.parse_args())['status'])
