"""Four-arm new100 finite PHY composition with separate raw433/T1-144/EP360 domains.

No runtime is constructed on import. The actual owner supplies separately
admitted original providers; this module never calls a source or image model.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import ep_new100_confirmation_core_v1 as confirmation
import ep_plan as plan
import h800_mismatch_phy_core_v1 as raw_original

SCHEMA='H800_EP_NEW100_ACTUAL_FOUR_ARM_PHY_V1'
GROUPS={'raw':('RAW_WHOLE','RAW_PARTIAL'),'whole':('EC_VAR_WHOLE',),'partial':('EC_VAR_PARTIAL',)}
DOMAIN_COUNTS={'raw':433,'whole':144,'partial':360}
require=confirmation.require
digest=confirmation.digest
save=raw_original.save
readpin=raw_original.readpin
pin=raw_original.pin


def group_for(method):
    for group,methods in GROUPS.items():
        if method in methods:return group
    raise ValueError('Unknown confirmation arm')


def validate_grid(rows):
    confirmation.complete_grid(rows,rows)
    require([r['frame_index'] for r in rows]==list(range(3600)),'Original fixed logical order required')
    for group,methods in GROUPS.items():
        selected=[r for r in rows if r['method'] in methods]
        require(len(selected)==900*len(methods),'Complete per-domain source/SNR/seed coverage required')


def source_tokens(item,inside):
    p=inside(item['source_assets']['path']);require(raw_original.sha(p)==item['source_assets']['sha256'],'Fresh source assets changed')
    with np.load(p,allow_pickle=False) as z:
        require(set(z.files)=={'pixels','tokens','F'},'Exact new100 source asset schema required')
        tokens=z['tokens'].copy()  # Never decode/load the pixels or latent array.
    require(tokens.dtype==np.int64 and tokens.shape==(680,) and np.all((0<=tokens)&(tokens<4096)) and
        hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==item['tokens_sha256'],
        'Actual complete source token identity differs')
    return [tokens[raw_original.PREFIX[i]:raw_original.PREFIX[i+1]] for i in range(10)]


def entropy_choice(event,streams,profiles,group):
    require(group in ('whole','partial') and len(profiles)==DOMAIN_COUNTS[group],'Exact full entropy receive domain required')
    choice=event['policy'];require(group!='whole' or choice['target_K']==0,'Original whole arm must remain whole-only')
    selected=plan.select_stream({key:len(value) for key,value in streams.items()},choice)
    m,K=selected['m'],selected['K'];require(group!='whole' or K==0,'Original whole fallback changed')
    family='EC_VAR_PARTIAL' if K else 'EC_VAR_WHOLE'
    found=[p for p in profiles.values() if (p['family'],p['m'],p['K'],p['q'],p['nominal_rate'])==
        (family,m,K,choice['q'],choice['nominal_rate'])]
    require(len(found)==1,'Unique qualified TX profile required; full RX domain remains intact')
    profile=found[0];require(profile['source_capacity_bits']==choice['source_capacity_bits'],'Frozen length capacity differs')
    bits=streams[m,K].copy();require(bits.dtype==np.uint8 and bits.ndim==1 and np.isin(bits,(0,1)).all(),
        'Previously encoded actual uint8 arithmetic stream required')
    return profile,bits,selected


def raw_choice(event,scales,runtime):
    choice=event['policy'];p=runtime.cat.entry(choice['profile_id'])
    require(p['wire_key']==choice['wire_key'] and p['m']==choice['m'] and p['K']==choice['K'] and
        choice['candidate_id'] in p['alias_candidate_ids'] and (event['method']!='RAW_WHOLE' or p['K']==0),
        'Original raw winner/profile/alias changed')
    return p,runtime.original.serialize_raw(scales,p)


def identity(event,item,group,profile,payload,wave,standard,observed,provider,sha_array):
    require(event['public_frame_counter']==confirmation.counter(event['source_index'],event['snr_db'],event['noise_seed']),
        'New100 common counter changed')
    require(all(a.dtype==np.float64 and a.shape==(1024,2) and np.isfinite(a).all() for a in (wave,standard,observed)),
        'Full1024 float64 waveform/noise/actual observation required')
    require(provider['profile_count']==DOMAIN_COUNTS[group] and provider['group']==group,'Provider receive domain mismatch')
    # Excludes method label deliberately: only truly identical observations in
    # the same complete receive domain may share a finished paid RX result.
    return dict(source_index=event['source_index'],source_id=event['source_id'],tokens_sha256=item['tokens_sha256'],
        source_assets_sha256=item['source_assets']['sha256'],snr_db=event['snr_db'],noise_seed=event['noise_seed'],
        public_frame_counter=event['public_frame_counter'],TX_profile=profile,payload_sha256=sha_array(payload),
        waveform_sha256=sha_array(wave),standard_noise_sha256=sha_array(standard),observation_sha256=sha_array(observed),
        complete_receive_provider=provider,wire_session='actual-body' if group=='raw' else 'WCL_T1_ENTROPY_WHOLE_PHY_20261009_V1')


def verify_entropy_paid(actual,ledger,event_prefix,runtime,observed,snr,counter,phy,phase):
    """Check actual callbacks, full receive domain and CRC/parser failure outcomes."""
    for kind in ('header','body'):
        body=actual[kind]
        if body is None:
            require(kind=='body' and actual['status']=='HEADER_REJECT' and not actual['header']['header_ok'],
                'A missing body is valid only after actual header rejection');continue
        eid=event_prefix+':'+kind
        saved=ledger.db.execute('SELECT event,request,result,status FROM events WHERE event_id=?',(eid,)).fetchone()
        require(saved is not None and saved[3]=='COMPLETE','Missing actual paid entropy callback')
        import json
        event,request,result=(json.loads(x) for x in saved[:3])
        section=observed[:68] if kind=='header' else observed[68:].astype(np.float32)
        require(event['kind']==kind and event['phase']==phase and result==body and request['snr_db']==float(snr) and
            request['received_sha256']==phy.array_sha(section) and request['catalogue_sha256']==runtime.catalogue['catalogue_sha256'],
            'Actual entropy paid result or full-domain observation differs')
        if kind=='body':
            expected_session=phy.PROTOCOL if phase=='new100_original_whole144' else phy.WIRE_PROTOCOL
            require(request['public_frame_counter']==counter and request['session']==expected_session,
                'Original entropy wire session/counter changed')
            received=runtime.profiles[str(actual['rx_profile']['profile_id'])]
            require(received['profile_key']==request['profile_key']==actual['body']['profile_key'],
                'Actual accepted header profile, including wrong/static profiles, must select the body parser')


def execute_group(group,runtime,rows,sources,streams,provider,ledger,boundary,out,whole_phy,inside):
    validate_grid(rows);require(group in GROUPS and len(sources)==100,'Exact source100 and receiver group required')
    require(provider['group']==group and provider['profile_count']==DOMAIN_COUNTS[group],'Qualified full receive domain required')
    if group=='raw':
        require(runtime.cat.digest==provider['catalogue_sha256'] and runtime.cat.aliases_digest==provider['aliases_sha256'],
            'Original raw433 complete catalogue/aliases differ')
    else:
        require(len(runtime.profiles)==DOMAIN_COUNTS[group] and runtime.catalogue['catalogue_sha256']==provider['catalogue_sha256'],
            'Complete original entropy receiver catalogue differs')
    selected=[row for row in rows if row['method'] in GROUPS[group]]
    out=Path(out);require(not out.exists(),'Fresh per-domain output');out.mkdir();physical={};logical=[];source_cache={}
    before=ledger.snapshot();require(before['unresolved']==0 and before['cap']==7200,'Shared finite7200 packet ledger required')
    for event in selected:
        boundary();i=event['source_index'];item=sources[i]
        require(item['source_index']==i and item['source_id']==event['source_id'],'Same fixed source across every arm required')
        counter=event['public_frame_counter'];standard=whole_phy.standard_noise(event['source_id'],event['noise_seed'])
        if group=='raw':
            if i not in source_cache:source_cache={i:source_tokens(item,inside)}
            profile,payload=raw_choice(event,source_cache[i],runtime)
            wave,tx=runtime.rx.transmit_frame(runtime.cat,runtime.backend,runtime.legacy,runtime.header,profile['profile_id'],payload,counter)
            sha_array=runtime.original.array_sha;fallback=None
        else:
            profile,payload,fallback=entropy_choice(event,streams[i],runtime.profiles,group)
            wave,tx=runtime.transmit(profile['profile_id'],payload,counter);sha_array=whole_phy.array_sha
        observed=wave+standard*10**(-event['snr_db']/20)
        ident=identity(event,item,group,profile,payload,wave,standard,observed,provider,sha_array);key=digest(ident)
        if key not in physical:
            event_prefix=SCHEMA+'/'+group+'/'+key;start=ledger.snapshot()['total']
            if group=='raw':
                actual=runtime.receiver.receive(observed,float(event['snr_db']),counter,phase='raw_holdout',event_prefix=event_prefix)
                require(actual['received_sha256']==ident['observation_sha256'] and actual['public_frame_counter']==counter and
                    actual['codebook_sha256']==runtime.cat.digest and actual['aliases_sha256']==runtime.cat.aliases_digest,
                    'Raw receiver did not consume this complete-domain observation')
                require(all(actual[k]==v for k,v in runtime.original.present_actual(actual['header'],actual['body'],runtime.cat).items()),
                    'Actual hard-token KEEP presentation changed')
                raw_original.verify_paid(actual,ledger);summary=raw_original.normalized(actual)
            else:
                phase='new100_original_whole144' if group=='whole' else 'new100_allowed_partial360'
                actual=runtime.receive(observed,event['snr_db'],counter,runtime.profiles,ledger,event_prefix,phase=phase)
                parser=whole_phy if group=='whole' else runtime._new100_phy_module
                verify_entropy_paid(actual,ledger,event_prefix,runtime,observed,event['snr_db'],counter,parser,phase)
                summary=dict(status=actual['status'],source_decoding_performed=False)
            calls=ledger.snapshot()['total']-start;require(calls in (1,2),'Actual header/body paid callback count differs')
            result=dict(schema=SCHEMA,state='complete',provider_group=group,observation_identity=ident,physical_key=key,
                actual_RX=actual,receiver=summary,transmission=tx,fallback=fallback,actual_new_packet_calls=calls,
                historical_reuse_admitted=False,source_truth_supplied_to_RX=False)
            physical[key]=save(out/'physical'/f'{key}.json',result);mode='ACTUAL_NEW'
        else:
            require(readpin(physical[key])['observation_identity']==ident,'Actual same-run reuse changed');mode='EXACT_SAME_DOMAIN_REUSE'
        logical.append(dict(event,physical_key=key,physical_frame=physical[key],reuse=mode,
            common_standard_noise_sha256=whole_phy.array_sha(standard),provider_group=group))
    after=ledger.snapshot();require(after['unresolved']==0 and after['total']<=7200,'Finite shared paid ledger did not close')
    return dict(group=group,logical_rows=save(out/'logical_rows.json',logical),physical_frames=list(physical.values()),
        logical_count=len(logical),new_physical_count=len(physical),actual_new_packet_calls=after['total']-before['total'],
        initial_ledger=before,final_ledger=after,source_RX_calls=0,encoder_calls=0,model_calls=0)


def close_groups(results,rows,read):
    require([r['group'] for r in results]==list(GROUPS),'Exactly three isolated receiver groups required')
    logical=[];last=0
    for result in results:
        group=result['group'];part=read(result['logical_rows'])
        require(len(part)==result['logical_count']==900*len(GROUPS[group]) and
            result['initial_ledger']==dict(total=last,unresolved=0,cap=7200),'Complete joined receiver group required')
        last+=result['actual_new_packet_calls']
        require(result['final_ledger']==dict(total=last,unresolved=0,cap=7200),'Shared paid ledger continuity differs')
        logical.extend(part)
    logical.sort(key=lambda r:r['frame_index']);confirmation.complete_grid(logical,rows)
    shared={}
    for row in logical:
        key=row['source_index'],row['snr_db'],row['noise_seed'];noise=row['common_standard_noise_sha256']
        require(shared.setdefault(key,noise)==noise,'Methods consumed different standard AWGN at the common working point')
        require(row['provider_group']==group_for(row['method']),'Wrong receive catalogue selected')
    require(len(shared)==900 and last<=7200,'Complete common900 observations and bounded7200 decoder calls required')
    return logical,dict(total=last,unresolved=0,cap=7200)
