"""Read-only final N2048 evidence export; no model, channel or resampling calls.

Consumes actual closed calibration/source/render/score/statistics artifacts.
It copies original-precision results and preserves all failures and zero deltas.
"""
import argparse,csv,json,math,shutil
from collections import Counter
from pathlib import Path
import t6_score as s

read=s.read;require=s.require;sha=s.sha;pin=s.pin;pinned=s.pinned
SNRS=s.SNRS;SEEDS=s.SEEDS;METRICS=s.METRICS

def boolean(value):
    require(value in(True,False,'True','False','',None),'Invalid actual Boolean field')
    return None if value in('',None)else value in(True,'True')

def csv_read(path):
    with Path(path).open(newline='',encoding='utf-8-sig')as f:return list(csv.DictReader(f))

def csv_write(path,rows):
    require(rows,'Empty final evidence table')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for r in rows:w.writerow({k:s.canonical(v)if isinstance(v,(dict,list))else v for k,v in r.items()})

def request_of(done):
    d=done.get('request')or dict(path=done['request_path'],sha256=done['request_sha256'])
    return d,pinned(d)

def verify_outputs(done,seen):
    for p,v in done['outputs'].items():
        if p in seen:require(seen[p]==v,'Conflicting artifact SHA')
        else:s.verify(p,v);seen[p]=v

def wait_receipt(done,request,explicit=None):
    require(done.get('actual_children_waited')is True and done.get('worker_exit_codes')
        and all(x==0 for x in done['worker_exit_codes']),'Actual successful child exits required')
    candidates=[explicit]if explicit else[p for p in done['outputs']if Path(p).suffix=='.json'and
        (Path(p).name=='completion.json'or any(x in Path(p).name for x in('owner','wait','exit')))]
    matches=[]
    for p in candidates:
        require(p in done['outputs'],'Owner receipt is not sealed by scientific completion');s.verify(p,done['outputs'][p]);v=read(p)
        if v.get('actual_children_waited')is True and v.get('request_sha256')==request['sha256']and v.get('worker_exit_codes')==done['worker_exit_codes']:
            matches.append(pin(p))
    require(matches,'Missing actual owner wait/exit receipt; completion prose alone is insufficient')
    return matches

def source_wait(directory,request,source_done):
    path=Path(directory);launch=read(path/'launch.json');exited=read(path/'exit.json')
    started=read(Path(request['path']).parent/'owner_started.json')
    require(launch['owner_script_sha256']==s.OBSERVER_SHA and launch['child_pid']==started['pid']and started['request_sha256']==request['sha256']==source_done['request_sha256']
        and exited['actual_child_waited']is True and exited['exit_code']==0,'Exact new100 source owner must have actually exited0')
    require(launch['argv'][-3:]==['run','--request',request['path']]
        and any(Path(x).name=='t6_confirmation_sources.py'for x in launch['argv']),'Source owner argv/request mismatch')
    return dict(launch=pin(path/'launch.json'),exit=pin(path/'exit.json'))

def intervals(summary,paired,family):
    require(len(summary)==36 and len(paired)==36,'All36 summary and36 paired rows required')
    expected={(m,snr,metric)for m in s.methods(family)for snr in SNRS for metric in METRICS}
    require({(r['method_id'],int(r['snr_db']),r['metric'])for r in summary}==expected,'Complete four-metric method/SNR summary grid')
    pairs={(a,b,snr,metric)for a,b in s.comparisons(family)for snr in SNRS for metric in METRICS}
    require({(r['method_id'],r['reference_method_id'],int(r['snr_db']),r['metric'])for r in paired}==pairs,'Complete preregistered paired grid')
    for row in summary+paired:
        require(int(row['N'])==2048 and int(row['source_count'])==100 and int(row['noise_count'])==3 and int(row['frame_count'])==300,
            'Never mix other populations, budgets or noise counts')
        low,mu,high=(float(row[k])for k in('ci_low','mean','ci_high'))
        require(all(math.isfinite(v)for v in(low,mu,high))and low<=mu<=high,'Finite ordered intervals required')
        factor=100 if row['metric']==METRICS[-1]else 1
        for field in('mean','ci_low','ci_high'):
            require(math.isclose(float(row['display_'+field]),float(row[field])*factor,rel_tol=1e-13,abs_tol=1e-13),'Display-unit conversion changed')
        require(float(row['display_scale'])==factor,'Display scale differs from actual units')
        if row in paired:
            require(row['delta_definition']=='method minus reference'and boolean(row['frame_level_noise_pairing_claimed'])is False,'Paired direction/source-pairing contract changed')
            direction='exact_zero_difference'if low==high==0 else 'interval_includes_zero'if low<=0<=high else 'method_better'if(high<0 if row['metric']=='lpips_alex'else low>0)else'reference_better'
            require(row['interval_direction']==direction,'Paired interval interpretation or LPIPS direction changed')
            require(row['improvement_direction']==('negative'if row['metric']=='lpips_alex'else'positive'),'Paired improvement direction changed')
        if factor==100:require(row['display_unit']==('percentage_points'if row in paired else'percent'),'Agreement units are not relative percentages')

def source_mean_check(means,rows,summary,paired,ids,family):
    """Read-only arithmetic audit; no confidence interval or resampling calculation."""
    require(len(means)==3600,'All3600 sealed source means required')
    frames=s.frame_grid(rows,ids,family,scored=True);keyed={}
    for row in means:
        key=row['method_id'],int(row['snr_db']),row['metric'],int(row['source_index'])
        method,snr,metric,i=key
        require(key not in keyed and method in s.methods(family)and snr in SNRS and metric in METRICS and 0<=i<100,
            'Duplicate or foreign source-mean row')
        require(row['source_id']==ids[i]and int(row['N'])==2048 and int(row['source_count'])==100 and int(row['noise_count'])==3,
            'Source mean population changed')
        actual=math.fsum(s.metric_value(frames[method,snr,i,seed],metric)for seed in SEEDS)/3
        require(math.isclose(float(row['mean']),actual,rel_tol=1e-12,abs_tol=1e-12),'Published source mean differs from three actual scored frames')
        keyed[key]=float(row['mean'])
    for row in summary+paired:
        method,snr,metric=row['method_id'],int(row['snr_db']),row['metric']
        values=[keyed[method,snr,metric,i]for i in range(100)]
        if row in paired:values=[x-keyed[row['reference_method_id'],snr,metric,i]for i,x in enumerate(values)]
        require(math.isclose(float(row['mean']),math.fsum(values)/100,rel_tol=1e-12,abs_tol=1e-12),'Published mean differs from sealed source means')
        if row in paired and all(x==0 for x in values):
            require(float(row['mean'])==float(row['ci_low'])==float(row['ci_high'])==0,'Exact zero source differences must retain zero intervals')

def labels(family):
    a,b,c=s.methods(family)
    return {a:'Complete-scale raw transmission with VAR completion',b:'Partial-scale raw transmission with VAR completion (NeST-Com)',
        c:'VAR-conditional entropy-coded complete-scale transmission with VAR completion'if family=='EC_VAR_WHOLE'else'Static-entropy complete-scale transmission with VAR completion'}

def mechanism_rows(rows,freeze,family):
    whole,partial,entropy=s.methods(family);result=[]
    for method in (whole,partial,entropy):
        group_name={whole:'RAW_WHOLE',partial:'RAW_PARTIAL',entropy:'ENTROPY_WHOLE'}[method]
        for snr in SNRS:
            sub=[r for r in rows if r['method_id']==method and int(r['snr_db'])==snr];require(len(sub)==300,'All100x3 mechanism conditions')
            policy=freeze['policies'][group_name][str(snr)];wc=freeze['policies']['RAW_WHOLE'][str(snr)]
            exact_whole=group_name=='RAW_PARTIAL'and policy['candidate_id']==wc['candidate_id']
            states=Counter((int(r['actual_m']),int(r['K']))for r in sub)
            m10=sum(int(r['actual_m'])==10 and int(r['K'])==0 for r in sub)
            k0=sum(int(r['K'])==0 for r in sub)
            header=sum(boolean(r['header_ok'])is False for r in sub);crc=sum(boolean(r['body_crc_accept'])is False for r in sub)
            parser=sum(boolean(r['parser_accepted'])is False for r in sub);gray=sum(boolean(r['gray'])is True for r in sub)
            canonical=sum(boolean(r['header_ok'])is True and boolean(r['body_crc_accept'])is True and
                boolean(r['parser_accepted'])is True and boolean(r.get('canonical_source_accepted'))is False for r in sub)if group_name=='ENTROPY_WHOLE'else 0
            require(all(math.isclose(float(r['rho']),float(r['E_frame'])/4096,rel_tol=1e-12,abs_tol=1e-12)
                and int(r['header_symbols'])+int(r['body_symbols'])+int(r['padding_symbols'])==2048 for r in sub),'Actual paid symbols/energy changed')
            result.append(dict(method_id=method,snr_db=snr,source_count=100,noise_count=3,frame_count=300,candidate_id=policy['candidate_id'],
                frozen_target_m=policy.get('target_m',policy.get('m')),frozen_K=policy.get('K',0),
                actual_m_K_counts={f'm{m}_K{k}':v for(m,k),v in sorted(states.items())},m10_K0_frames=m10,m10_K0_fraction=m10/300,
                K0_frames=k0,partial_selected_complete_scale=group_name=='RAW_PARTIAL'and int(policy['K'])==0,
                partial_same_policy_as_whole=exact_whole,entropy_length_fallback_frames=sum(int(r['actual_m'])<int(r['target_m'])for r in sub)if group_name=='ENTROPY_WHOLE'else 0,
                header_reject_frames=header,body_CRC_reject_frames=crc,body_parser_reject_frames=parser,entropy_canonical_reject_frames=canonical,
                gray_frames=gray,CRC_undetected_error_frames=sum(boolean(r['undetected_token_error_diagnostic'])is True for r in sub),
                E_frame_mean=math.fsum(float(r['E_frame'])for r in sub)/300,rho_mean=math.fsum(float(r['rho'])for r in sub)/300,
                failure_counts_may_overlap=True,raw_KEEP_not_replaced_by_DROP=True))
    # Identical frozen whole/partial wires must remain identical in measured outputs.
    by={(r['method_id'],int(r['snr_db']),int(r['source_index']),int(r['noise_seed'])):r for r in rows}
    for record in result:
        if not record['partial_same_policy_as_whole']:continue
        snr=record['snr_db']
        for i in range(100):
            for seed in SEEDS:
                a=by[whole,snr,i,seed];b=by[partial,snr,i,seed]
                require(a['received_state_sha256']==b['received_state_sha256']and a['image_sha256']==b['image_sha256']
                    and all(s.metric_value(a,m)==s.metric_value(b,m)for m in METRICS),'Identical whole/partial physical action produced different reported results')
    return result

def calibration_rows(freeze,inputs):
    result=[];details={}
    for branch,field,status in [('raw','raw_calibration_completion','T6_RAW_FULL1000_CALIBRATION_COMPLETE'),
        ('entropy','entropy_calibration_completion','T6_ENTROPY_FULL1000_CALIBRATION_COMPLETE')]:
        done=pinned(freeze[field]);require(done['status']==status and done['source_count']==1000 and done['noise_count']==3
            and done['source_ids']==freeze['calibration_source_ids'],'Complete original1000 full calibration required')
        rd,r=request_of(done);waits=wait_receipt(done,rd);inputs[freeze[field]['path']]=freeze[field]['sha256'];inputs[rd['path']]=rd['sha256']
        prior=pinned(r['pilot_completion']);pd,pr=request_of(prior)
        require(prior['source_count']==100 and prior['noise_count']==1 and prior['source_ids']==done['source_ids'][:100],
            'Pilot must be the exact first100 original calibration sources')
        require(not r['confirmation_images_opened']and not r['holdout_used_for_selection']and not pr['confirmation_images_opened']and not pr['holdout_used_for_selection'],
            'Calibration cannot depend on confirmation content')
        p_wait=wait_receipt(prior,pd);inputs[r['pilot_completion']['path']]=r['pilot_completion']['sha256'];inputs[pd['path']]=pd['sha256']
        details[branch]=dict(full_done=done,full_request=r,pilot_done=prior,pilot_request=pr)
        for phase,d,q,wait in [('pilot',prior,pr,p_wait),('full',done,r,waits)]:
            ledger=d['packet_ledger'];require(ledger['unresolved']==0 and ledger['total']<=ledger['cap']==q['packet_cap'],'Actual finite packet budget not closed')
            for receipt in wait:inputs[receipt['path']]=receipt['sha256']
            result.append(dict(branch=branch,phase=phase,source_count=d['source_count'],noise_count=d['noise_count'],noise_seeds=d['noise_seeds'],
                logical_frames=d['frame_count'],maximum_packet_calls=ledger['cap'],actual_packet_calls=ledger['total'],
                candidate_SNR_points=len(q['schedule']),actual_children_waited=True,worker_exit_codes=d['worker_exit_codes'],
                received_state_cache_reuse_only=True,global_optimum_claimed=False))
    return result,details

def run(a):
    out=Path(a.out).resolve();require(not out.exists(),'Fresh final report output required')
    freeze=read(a.calibration_freeze);source_done=read(a.source_completion);render=read(a.render_completion);score=read(a.score_completion);stats=read(a.statistics_completion)
    source,frozen,selected,duplicate=s.metadata(render)
    require(frozen==freeze and render['calibration_freeze']==pin(a.calibration_freeze),'One exact N2048 policy freeze')
    family=selected['family'];names=labels(family);ids=source['source_ids'];inputs={str(Path(p).resolve()):sha(p)for p in
        (a.calibration_freeze,a.source_completion,a.render_completion,a.score_completion,a.statistics_completion)};seen={}
    require(source_done['status']=='T6_CONFIRMATION100_SOURCE_ASSETS_COMPLETE'and source_done['source_count']==100
        and source_done['source_ids']==ids and source_done['source_manifest']==render['source_manifest']
        and source_done['counts']['Encoder_VQ']==source_done['counts']['VAR_TX']==100,'Actual completed new100 sources required')
    source_request_path=Path(a.source_completion).parent/'request.json';s.verify(source_request_path,source_done['request_sha256'])
    sr=pin(source_request_path);source_request=read(source_request_path)
    require(source_request['calibration_freeze']==pin(a.calibration_freeze),'Source access must follow the same policy freeze')
    waits=dict(source=source_wait(a.source_wait_record,sr,source_done));inputs[sr['path']]=sr['sha256']
    render_request,r=request_of(render);waits['render']=wait_receipt(render,render_request,a.render_owner_receipt)
    require(r['source_completion']==pin(a.source_completion),'Render must consume the exact completed new100 sources')
    ledger=render['actual_packet_ledger']
    require(ledger['unresolved']==0 and ledger['total']<=ledger['cap']==5400,'Actual confirmation packet ledger must close within5400 calls')
    require(score['status']=='T6_CONFIRMATION100_FOUR_METRICS_COMPLETE'and score['actual_children_waited']is True
        and score['worker_exit_codes']==[0]and score['frame_count']==2700,'Actually waited full four-metric execution required')
    score_request,p=pinned(dict(path=score['request_path'],sha256=score['request_sha256'])),pin(score['request_path'])
    require(score_request['render_completion']==pin(a.render_completion),'Metrics must bind the exact completed render')
    launch=pinned(score['parent_launch']);exited=pinned(score['parent_exit'])
    score_started=read(Path(p['path']).parent/'owner_started.json')
    require(launch['owner_script_sha256']==s.OBSERVER_SHA and launch['child_pid']==score_started['pid']
        and score_started['request_sha256']==p['sha256']and exited['actual_child_waited']is True and exited['exit_code']==0
        and launch['argv'][-3:]==['run','--request',p['path']]and any(Path(x).name=='t6_score.py'for x in launch['argv']),
        'Actual score parent exit must correspond to the frozen worker request')
    waits['score']=dict(launch=score['parent_launch'],exit=score['parent_exit'])
    require(stats['status']=='T6_N2048_CONFIRMATION100_SOURCE_PAIRED_STATISTICS_COMPLETE'and stats['source_mean_first']is True
        and stats['source_count']==100 and stats['noise_count']==3 and stats['frame_count']==2700 and stats['summary_rows']==stats['paired_rows']==36 and stats['source_mean_rows']==3600
        and stats['bootstrap_seed']==2026100701 and stats['bootstrap_replicates']==10000 and not stats['multiple_comparison_adjustment'],
        'Actual complete registered source-paired statistics required')
    stat_req=pinned(stats['request']);require(stat_req['score_completion']==pin(a.score_completion),'Statistics must use these completed score rows')
    for done in(score,stats):
        require(done['source_ids']==ids and done['noise_seeds']==SEEDS and done['snrs']==SNRS and done['N']==2048
            and done['family']==family and done['methods']==s.methods(family)and done['comparisons']==s.comparisons(family)
            and done['calibration_freeze']==pin(a.calibration_freeze)and done['source_manifest']==render['source_manifest']
            and not done['selection_used_confirmation']and not done['holdout_used_for_selection'],'One complete frozen confirmation population')
    require(duplicate['duplicate_original_JPEG_count']==0 and not duplicate['source_reselection']
        and duplicate['Encoder_calls_before_check']==0,'New confirmation duplicate gate must precede Encoder and forbid reselection')
    for done in(source_done,render,score,stats):verify_outputs(done,seen)
    rows_path=s.bound_output(score,'per_frame.csv');rows=csv_read(rows_path);s.frame_grid(rows,ids,family,scored=True)
    summary_path=s.bound_output(stats,'summary.csv');paired_path=s.bound_output(stats,'paired.csv')
    summary=csv_read(summary_path);paired=csv_read(paired_path);intervals(summary,paired,family)
    source_means_path=s.bound_output(stats,'source_means.csv')
    source_mean_check(csv_read(source_means_path),rows,summary,paired,ids,family)
    mechanisms=mechanism_rows(rows,freeze,family);calibration,cal_details=calibration_rows(freeze,inputs)
    rawq=pinned(freeze['raw_phy_qualification']);ecq=pinned(freeze['entropy_phy_qualification'])
    rawqr=pinned(rawq['request']);ecqr=pinned(ecq['request']);rawcat=pinned(rawqr['candidate_catalogue']);eccat=ecqr['catalogue']
    require(rawq['status']==ecq['status']=='PASS'and rawq['ledger']['unresolved']==ecq['ledger']['unresolved']==0
        and ecqr['family']==family and ecqr['entropy_family_selection']==freeze['entropy_family_selection'],'Both actual paid PHY qualifications required')
    require(rawq['packet_decode_count']==rawqr['packet_cap']and ecq['packet_decode_count']==ecqr['packet_cap'],
        'All registered actual qualification calls required')
    legal=[dict(wire_family='raw',**x)for x in rawcat['profiles']]+[dict(wire_family='entropy',**x)for x in eccat['profiles']]
    queries=ecqr['admitted_candidates']+ecqr['unsupported_candidate_queries'];require(len(queries)==48,'All entropy resource queries, including unsupported, retained')
    require(set(ids).isdisjoint(freeze['calibration_source_ids']),'Confirmation source IDs overlap policy calibration')
    configs=[]
    for branch,m in zip(('RAW_WHOLE','RAW_PARTIAL','ENTROPY_WHOLE'),s.methods(family)):
        for snr in SNRS:configs.append(dict(freeze['policies'][branch][str(snr)],method_id=m,public_name=names[m],snr_db=snr))
    out.mkdir(parents=True);outputs={}
    def copied(name,path):
        shutil.copyfile(path,out/name);outputs[str(out/name)]=sha(out/name)
    def table(name,values):csv_write(out/name,values);outputs[str(out/name)]=sha(out/name)
    for name,path in [('per_frame.csv',rows_path),('summary.csv',summary_path),('paired.csv',paired_path),
        ('source_means.csv',source_means_path),('failure_breakdown.csv',s.bound_output(stats,'failure_breakdown.csv')),
        ('points.json',s.bound_output(stats,'points.json')),('pairs.json',s.bound_output(stats,'pairs.json')),
        ('frozen_policy.json',a.calibration_freeze),('entropy_family_selection.json',freeze['entropy_family_selection']['path']),
        ('confirmation_registration.json',source['confirmation_registration']['path']),('content_duplicate_check.json',source['content_duplicate_check_completion']['path'])]:copied(name,path)
    table('legal_actions.csv',legal);table('entropy_candidate_admission.csv',queries);table('selected_configurations.csv',configs)
    table('mechanism_and_failure_summary.csv',mechanisms);table('calibration_execution.csv',calibration)
    protocol_sources=[]
    for file in a.protocol:
        name=Path(file).name;require(name not in protocol_sources,'Unique protocol filenames required');protocol_sources.append(name)
        copied('source_'+name,file);inputs[str(Path(file).resolve())]=sha(file)
    protocol=['# N2048 digital-branch protocol','',
        'This preregistered second budget pays 68 header symbols within exactly 2048 symbols. Raw WHOLE and PARTIAL use the same legal public action set and the frozen KEEP rule. PARTIAL may select K=0. Entropy uses one N1024-calibration-selected family, paid 13-bit source length and CRC16, actual LDPC construction and the fixed length-only whole-prefix fallback. No raw substitution or free metadata is used.',
        '',f"The actual qualified catalogues contain {len(rawcat['profiles'])} raw actions and {len(eccat['profiles'])} entropy receive profiles. Of 48 original entropy target/MCS queries, {len(ecqr['admitted_candidates'])} were constructor-supported and {len(ecqr['unsupported_candidate_queries'])} unsupported. Unsupported layouts retain their exact queried k/n; no truncated k is substituted. Source capacity remains min(k-29,8191).",
        '', 'Raw pilot: all actual legal actions on the first100 original calibration sources and one noise. Raw full: the finite per-SNR union of pilot WHOLE top5, PARTIAL top5, all full-budget whole actions and all m10 actions. Entropy pilot: all constructor-supported targets m7/8/9/10; entropy full: top3 per SNR. Both full stages use the original1000 calibration sources and seeds4101/4102/4103. Selection is source-mean DINOv2-L with lexical ties, within these finite sets.',
        '', 'Confirmation: 100 sources fixed by the metadata registration before content access, SNR4/10/19, seeds9201/9202/9203. All three methods and all2700 conditions remain in the results. Source-noise keyed standard Gaussian variates are shared; different transmitted waveforms still yield different observations. Failures follow the frozen receiver policy and enter full-sample means.',
        '', 'All source/decode/model identities, actual child waits and cache proofs are carried by the bound completions. Independent receiver decoding verifies new arithmetic streams; actual accepted received payloads may reuse only exact independent receiver caches. Offline token-error diagnostics cannot change outputs.',
        '', 'Audited source-content exclusion is limited to the registered manifests and available comparable hashes. No global never-seen or global blind-test claim is made. This report does not combine the N2048 confirmation100 with the N1024 common500 population or infer an equal-quality bandwidth saving.',
        '', 'Detailed input protocol documents: '+', '.join('source_'+n for n in protocol_sources)+'.','']
    (out/'N2048_protocol.md').write_text('\n'.join(protocol),encoding='utf-8');outputs[str(out/'N2048_protocol.md')]=sha(out/'N2048_protocol.md')
    text=['# T6: N2048 digital-branch confirmation','',
        'This is the actually completed second-budget experiment: 100 pre-fixed sources, three noise realizations, three SNRs and three frozen methods, totaling 2,700 method-frame conditions. Both calibration branches finished before the confirmation source pipeline. No failures or zero increments are removed.',
        '', '## Scope and source audit','',
        f"The single entropy family is {family}, chosen using N1024 calibration only. The original1000 calibration selected all nine N2048 method/SNR policies; confirmation images and scores did not select policies. The content audit scope is `{duplicate['scope']}`: available_reference_hash_count={duplicate['available_reference_hash_count']}, unavailable_reference_hash_count={duplicate['unavailable_reference_hash_count']}. Missing hash domains are retained in content_duplicate_check.json. No source was replaced after content inspection. This establishes exclusion within the audited coverage, not that these images were never encountered anywhere.",
        '', '## Complete-scale boundaries and receiver outcomes','',
        '| Method | SNR | m10, K=0 frames | K=0 frames | Gray frames | Body CRC rejects | Partial policy identical to WHOLE |',
        '|---|---:|---:|---:|---:|---:|---|']
    for x in mechanisms:text.append('| '+names[x['method_id']]+' | '+str(x['snr_db'])+' | '+str(x['m10_K0_frames'])+'/300 | '+str(x['K0_frames'])+'/300 | '+str(x['gray_frames'])+'/300 | '+str(x['body_CRC_reject_frames'])+'/300 | '+str(x['partial_same_policy_as_whole'])+' |')
    text+=['','Raw CRC rejection and gray fallback are different events under KEEP. The mechanism table separately preserves header, CRC, framing, entropy canonical rejection and accepted token-error diagnostics. Failure counts can overlap; all2700 scored rows remain in per_frame.csv. Entropy source-length fallback and raw PARTIAL choosing K=0 are distinct mechanisms.','',
        'If PARTIAL selects the same full-scale physical action as WHOLE, its exact state, image and metrics must match; zero paired increments are valid boundary results. Complete m10 transmission is reported only where the actually transmitted frame has m=10,K=0. It is never inferred from an average compressed length.','',
        '## Source-paired comparisons: all four metrics','',
        '| Method minus reference | SNR | Metric | Mean difference | 95% interval | Interpretation |',
        '|---|---:|---|---:|---|---|']
    for row in paired:
        metric=row['metric'];unit=' (percentage points)'if metric==METRICS[-1]else''
        text.append('| '+names[row['method_id']]+' minus '+names[row['reference_method_id']]+' | '+row['snr_db']+' | '+metric+unit+' | '+format(float(row['display_mean']),'+.8f')+' | ['+format(float(row['display_ci_low']),'.8f')+', '+format(float(row['display_ci_high']),'.8f')+'] | '+row['interval_direction']+' |')
    text+=['','Deltas remain method minus reference. Negative LPIPS is better; ConvNeXt is source-prediction agreement, not classification accuracy. Canonical CSV agreement values are proportions; displayed paired agreement values are percentage points. Three noise results are averaged within each source before the existing100-source bootstrap (10,000 replicates, seed2026100701). Intervals are pointwise and unadjusted for multiple comparisons. Inclusion of zero is not evidence of equivalence. This export performs no new bootstrap.',
        '', '## Computation and interpretation','',
        'calibration_execution.csv records actual pilot/full logical frames and packet callbacks against their finite preregistered caps. Qualification failures unsupported by the real constructor are documented, not silently replaced. Selection within the pilot/full unions is not a claim of global optimality.',
        '', f"Actual confirmation packet callbacks: {render['actual_packet_ledger']['total']} of maximum5400; unresolved={render['actual_packet_ledger']['unresolved']}. New VAR renders: {render['new_VAR_render_calls']} plus {render['qualification_VAR_calls']} exact replay qualification calls. New entropy source decodes: {render['new_VAR_source_decode_calls']}. Actual score accounting: {s.canonical(score['counts'])}.",
        '', 'N1024 used the previously published common500 source population; N2048 uses the separately registered confirmation100. Their means cannot be treated as a paired cross-budget trajectory. Two budget points do not establish a precise bandwidth saving, an equal-quality crossing or an interpolated optimum. Any m10/full-scale fallback, reduced partial advantage, negative difference or zero difference in this experiment limits the claim rather than motivating another search.',
        '', 'Only the supplied completed artifacts are exported. The source parent, rendering/calibration owners and score parent were checked for actual waits and successful exits. No model, channel, image rendering or resampling is run by this command.','']
    (out/'REPORT_T6.md').write_text('\n'.join(text),encoding='utf-8');outputs[str(out/'REPORT_T6.md')]=sha(out/'REPORT_T6.md')
    audit=dict(status='T6_READ_ONLY_FINAL_PROVENANCE_VERIFIED',wait_receipts=waits,source_input=sr,
        consumed_scientific_output_files=len(seen),actual_packet_ledger=render['actual_packet_ledger'],
        source_counts=source_done['counts'],source_population_scope=duplicate['scope'],available_reference_hash_count=duplicate['available_reference_hash_count'],
        unavailable_reference_hash_count=duplicate['unavailable_reference_hash_count'],selected_entropy_family=family,
        raw_qualification=freeze['raw_phy_qualification'],entropy_qualification=freeze['entropy_phy_qualification'],
        no_global_blindness_claim=True,new_model_calls=0,new_packet_decodes=0,new_bootstrap_calls=0)
    s.write(out/'provenance.json',audit);outputs[str(out/'provenance.json')]=sha(out/'provenance.json')
    complete=dict(status='T6_N2048_FINAL_READ_ONLY_REPORT_COMPLETE',N=2048,source_count=100,noise_count=3,noise_seeds=SEEDS,
        snrs=SNRS,frame_count=2700,summary_rows=36,paired_rows=36,family=family,source_ids=ids,
        calibration_freeze=pin(a.calibration_freeze),source_manifest=render['source_manifest'],actual_waits_verified=True,
        selection_used_confirmation=False,global_optimality_claim=False,bandwidth_saving_inferred=False,
        new_model_calls=0,new_packet_decodes=0,new_bootstrap_calls=0,input_bindings=inputs,outputs=outputs,script_sha256=sha(__file__))
    s.write(out/'completion.json',complete);return complete

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in('calibration-freeze','source-completion','source-wait-record','render-completion','score-completion','statistics-completion','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--protocol',action='append',required=True);p.add_argument('--render-owner-receipt')
    a=p.parse_args();print(run(a)['status'])
