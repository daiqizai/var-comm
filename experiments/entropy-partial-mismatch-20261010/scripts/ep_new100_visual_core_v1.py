"""New100 actual-receiver reconstruction, without transmitter truth or scoring.

Original raw KEEP and arithmetic source decoders remain distinct. Arithmetic
whole/partial accepted states may share only identical actual input and runtime.
The previously registered600 source-RX cap is a hard stop, including bad streams.
"""
from pathlib import Path
import numpy as np
import ep_new100_confirmation_core_v1 as confirmation
import ep_new100_phy_core_v1 as physical
import h800_ep_pilot_core_v1 as pilot
import h800_mismatch_visual_v1 as raw

require=confirmation.require
CAPS=confirmation.scientific_caps()['visual']
POPULATION='independent_confirmation_new100'
FAILURE_FIELDS=('actual_RX_status','actual_received_profile','actual_header_ok',
                'actual_crc_accepted','actual_parser_accepted','body_attempted','fixed_gray')


def validate_packet(row,packet,providers):
    group=row['provider_group'];ident=packet['observation_identity']
    require(group==physical.group_for(row['method']) and packet['provider_group']==group and
        packet['schema']==physical.SCHEMA and packet['state']=='complete' and
        packet['physical_key']==row['physical_key']==confirmation.digest(ident) and
        packet['historical_reuse_admitted'] is False and packet['source_truth_supplied_to_RX'] is False,
        'Only closed actual new100 physical outcomes are admitted')
    require(all(ident[k]==row[k] for k in ('source_index','source_id','snr_db','noise_seed','public_frame_counter')) and
        ident['complete_receive_provider']==providers[group] and
        ident['standard_noise_sha256']==row['common_standard_noise_sha256'],
        'Actual physical source, observation or complete receive domain differs')
    return packet['actual_RX']


def input_and_status(actual,group,identity,module,catalogue):
    if group=='raw':
        state,tokens=raw.received_tokens(actual,module,catalogue)
        key=confirmation.digest(dict(receiver='original_raw_KEEP',actual_state=state,visual_identity=identity))
        status=dict(actual_RX_status=actual['source_status'],actual_received_profile=dict(
            profile_id=actual['rx_profile_id'],m=state['m'],K=state['K']),actual_header_ok=actual['header_ok'],
            actual_crc_accepted=actual['body_crc_accept'],actual_parser_accepted=None,
            body_attempted=actual['body_decode_complete'],fixed_gray=state['kind']=='gray')
        return key,(state,tokens),status
    require(group in ('whole','partial'),'Unknown actual source receiver')
    key=confirmation.digest(dict(receiver='original_arithmetic_with_partial_extension',
        actual_input=pilot.source_input(actual,identity)))
    body=actual['body']
    status=dict(actual_RX_status=actual['status'],actual_received_profile=actual['rx_profile'],
        actual_header_ok=actual['header']['header_ok'],actual_crc_accepted=None if body is None else body['crc_accepted'],
        actual_parser_accepted=None if body is None else body['parser_accepted'],body_attempted=body is not None,
        fixed_gray=not actual['header']['header_ok'] or body is None or not body['crc_accepted'] or not body['parser_accepted'])
    return key,None,status


def reconstruct(actual,group,prepared,backend,codec,partial,ledger,boundary):
    if group=='raw':
        state,tokens=prepared
        image=raw.render_actual(state,tokens,backend,ledger,boundary)
        evidence=dict(kind=state['kind'],source_status=actual['source_status'],source_decode_called=False,
            render_called=state['kind']!='gray',receiver_semantics='original_raw_KEEP',
            transmitted_CDF_used=False,comparison_truth_used=False)
        return image,evidence
    # Original source decoder constructs a fresh provider. Its arguments are
    # exactly actual accepted family/m/K/bits, with no source/TX CDF witness.
    return pilot.link.recover_and_render(actual,codec,backend,partial,ledger,boundary)


def check_counts(counts):
    require(counts['caps']==CAPS and counts['unresolved']==0 and counts['reserved']==counts['completed'] and
        set(counts['completed'])==set(CAPS) and all(type(counts['completed'][k]) is int and
        0<=counts['completed'][k]<=v for k,v in CAPS.items()),'Actual reconstruction call ledger did not close within frozen caps')
    c=counts['completed']
    require(c['model_load']==1 and c['encoder']==c['source_tx']==0 and c['decoder_forward']==c['var_render'] and
        10*c['var_render']<=c['prior_scale']<=10*(c['var_render']+c['source_rx']),
        'Original receiver and ten-scale renderer call accounting differs')


def images(r,backend,codec,partial,ledger,boundary,out,g,read):
    logical=r['logical_rows'];confirmation.complete_grid(logical,r['logical_events'])
    require(len(logical)==3600 and r['historical_image_reuse_admitted']==0 and r['caps']==CAPS,
        'Complete fixed3600 confirmation outcomes and original caps required')
    out=Path(out);out.mkdir();(out/'images').mkdir();(out/'logical').mkdir()
    module,catalogue=raw.parser(r['raw_binding']);cache={};packets={};pins=[]
    reuse=dict(actual_new_inputs=0,exact_same_input=0)
    for index,row in enumerate(logical):
        boundary();pp=row['physical_frame'];pk=confirmation.canonical(pp)
        if pk not in packets:packets[pk]=read(pp)
        actual=validate_packet(row,packets[pk],r['providers']);group=row['provider_group']
        key,prepared,status=input_and_status(actual,group,r['visual_identity'],module,catalogue)
        if key not in cache:
            backend.source_index=row['source_index']
            image,evidence=reconstruct(actual,group,prepared,backend,codec,partial,ledger,boundary)
            require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and
                ((image>=0)&(image<=1)).all(),'Frozen float32 reconstruction contract differs')
            ip=out/'images'/f'{len(cache):05d}.npz'
            with ip.open('xb') as stream:np.savez(stream,image=image)
            cache[key]=dict(image_archive=pilot.descriptor(ip),image_sha256=g.image_sha(image),
                actual_source_input_key=key,evidence=evidence,first_physical_frame=pp,origin='THIS_NEW100')
            reuse['actual_new_inputs']+=1;mode='ACTUAL_NEW'
        else:reuse['exact_same_input']+=1;mode='EXACT_ACTUAL_INPUT_REUSE'
        # Source parse rejection is this identical decoded bitstream's outcome;
        # header/body failures below always come from this logical observation.
        evidence=cache[key]['evidence'];status['fixed_gray']=evidence['kind']=='gray'
        status['actual_source_status']=evidence['source_status'] if evidence['source_decode_called'] else status['actual_RX_status']
        result=dict(frame_index=index,logical_event=r['logical_events'][index],physical_frame=pp,
            reconstruction=cache[key],reuse=mode,provider_group=group,**status,
            reconstruction_evidence_describes_first_cached_decode=True,
            failed_logical_observations_preserved=True)
        path=out/'logical'/f'{index:04d}.json';g.write(path,result);pins.append(pilot.descriptor(path))
    counts=ledger.summary();check_counts(counts)
    require(len(pins)==sum(reuse.values())==3600,'Complete four-arm confirmation images required')
    return dict(logical_frames=pins,logical_count=3600,physical_count=len(packets),reuse=reuse,counts=counts,
        population=POPULATION,historical_image_reuse_admitted=0,entropy_RX_hard_cap=600,
        cap_exhaustion_action='STOP_PRESERVE_PAID_WORK_NO_RETRY_NO_GRAY_SUBSTITUTION',
        policy_selection_uses_new100=False,source_truth_used=False,TX_CDF_used=False)
