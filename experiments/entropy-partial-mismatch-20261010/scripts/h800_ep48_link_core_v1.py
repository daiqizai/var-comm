"""Bounded EP48 image-link operations; no launch or automatic execution.

CPU PHY and visual GPU callers are separate processes. In particular, receive
decoding never uses the source-gate TX CDF witnesses as an acceptance oracle.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time
import types
import numpy as np

SCHEMA='H800_EP48_IMAGE_LINK_GATE_V1'
PASS='PASS_H800_EP48_ACTUAL_IMAGE_LINK_ONLY'
SNRS=(4,10,19)
CAPS=dict(model_load=1,encoder=0,source_tx=0,source_rx=48,var_render=48,prior_scale=960,decoder_forward=48)
POLICY_SHA='08097679bd01e6988fc4574d4f74ab57e1a891b2837a2f7a77713565ace7f70d'
STATIC_SHA='f74d838372f8e71b00c6bd62c9e37202c868e32ad3f294a7d21043eeee0b8c48'
STATIC_NPZ_SHA='483e1e39fbed2ede7eba372e94b6b7c9d6263b74a2e9e9d6ed2d7059b440abb7'
SOURCE_DRIVER_SHA='b38f8451ca2d94a8ceb4381df3647e327dba38e3d4cf969aa15597bc47c342ce'
OLD_ROOT='/home/liulu/projects/VAR_COMM/outputs/WCL-EVIDENCE-CLOSURE-20261009/T1_entropy_whole/'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def descriptor(path):return dict(path=str(path),sha256=sha(path))


def counter(source_index,snr):
    require(type(source_index) is int and 0<=source_index<4 and snr in SNRS,'Fixed first4/threeSNR only')
    return ([1,4,7,10,13,19].index(snr)*1000+source_index)*3


def plan(policy,records,partial):
    require(policy['status']=='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1' and policy['N']==1024 and
        policy['source_count']==1000 and policy['noise_count']==3 and policy['holdout_used_for_selection'] is False,
        'Original full-calibration whole winners required')
    require(len(records)==4 and [r['source_index'] for r in records]==list(range(4)) and
        [r['source_id'] for r in records]==policy['calibration_source_ids'][:4],'Original first4 source identity/order differs')
    winners=policy['policies']['EC_VAR_WHOLE'];expected={4:(7,2,'2/3'),10:(9,4,'2/3'),19:(9,6,'5/6')}
    rows=[]
    for record in records:
        for snr in SNRS:
            winner=winners[str(snr)];m,q,rate=expected[snr]
            require((winner['target_m'],winner['q'],winner['nominal_rate'])==(m,q,rate),'Published whole winner differs')
            for K in (0,*partial.quarter_counts(m)):
                rows.append(dict(source_index=record['source_index'],source_id=record['source_id'],snr_db=snr,noise_seed=4101,
                    target_m=m,target_K=K,q=q,nominal_rate=rate,source_capacity_bits=winner['source_capacity'],
                    public_frame_counter=counter(record['source_index'],snr),whole_anchor_candidate=winner['candidate_id'],
                    event_id=f"{SCHEMA}/source{record['source_index']}/snr{snr}/seed4101/target_m{m}_K{K}"))
    require(len(rows)==48 and len({r['event_id'] for r in rows})==48,'Fixed48 events required')
    return rows


def static_binding(policy,seed_root,read_pin,inside):
    """Explicit relocation of original metadata; no rewrite of old completion."""
    root=Path(seed_root)/'VAR_COMM/outputs/WCL-EVIDENCE-CLOSURE-20261009/T1_entropy_whole'
    expected=dict(path=OLD_ROOT+'static_train20k_v1/completion.json',sha256=STATIC_SHA)
    require(policy['protocol']['static_completion']==expected,'Original static model policy pin differs')
    path=inside(root/'static_train20k_v1/completion.json');done=read_pin(path,STATIC_SHA)
    old=OLD_ROOT+'static_train20k_v1/static_scale_counts_cdf.npz'
    require(done['status']=='T1_STATIC_TRAIN20K_CDF_COMPLETE' and done['source_count']==20000 and
        done['source_role']=='original_train20k' and done['calibration_used_for_fitting'] is False and
        done['outputs']=={old:STATIC_NPZ_SHA},'Frozen train20k static model metadata differs')
    table=inside(root/'static_train20k_v1/static_scale_counts_cdf.npz')
    require(sha(table)==STATIC_NPZ_SHA,'Relocated static CDF bytes differ')
    return dict(original_completion=expected,completion=descriptor(path),
        original_table=dict(path=old,sha256=STATIC_NPZ_SHA),table=descriptor(table),
        adaptation='Exact original bytes; only explicitly recorded source path relocation',fitting_calls=0)


def bind_static(codec,binding):
    """Same original validation equations, using the explicitly relocated file."""
    require(binding['completion']['sha256']==STATIC_SHA and binding['table']['sha256']==STATIC_NPZ_SHA and
        sha(binding['completion']['path'])==STATIC_SHA and sha(binding['table']['path'])==STATIC_NPZ_SHA,
        'Static source decoder binding changed')
    with np.load(binding['table']['path'],allow_pickle=False) as archive:
        cdf=codec.entropy.validate_cdf(archive['cdf']).copy()
        require(cdf.shape==(10,4097),'Static table scale count differs')
        require(np.array_equal(cdf,codec.entropy.probability_cdf(np.log(archive['counts'].astype(np.float64)+.5))),
            'Static CDF differs from original registered counts and smoothing')
    codec.static_cdf=cdf;codec.static_identity=STATIC_SHA


def bind_independent_provider(codec,backend,math_module,source_driver,g):
    """Original provider, with observation only; no expected CDF/token inputs."""
    require(sha(source_driver)==SOURCE_DRIVER_SHA,'Frozen independent provider source differs')
    namespace=dict(np=np,time=time)
    math_module.definitions(source_driver,{'IndependentProvider'},namespace)
    Original=namespace['IndependentProvider']
    class Observed(Original):
        def cdf(inner):
            result=super().cdf()
            backend.traces.append(dict(source=backend.source_index,role='ACTUAL_RX',m=backend.current[1],
                scale=inner.scale,cdf_sha256=g.image_sha(result)))
            return result
    codec.native=backend.native
    codec.provider_factory=lambda:Observed(backend.native,types.SimpleNamespace(cdf_function=codec.entropy.probability_cdf))


def load_streams(pin,metadata,partial):
    require(sha(pin['path'])==pin['sha256'],'Fresh H800 source-gate archive changed')
    require(metadata['archive']==pin and metadata['old_host_bits_used'] is False and metadata['old_host_CDF_used'] is False,
        'Fresh H800 TX provenance required')
    points=partial.all_endpoints();expected={f'm{m}_K{K}' for m,K in points}
    with np.load(pin['path'],allow_pickle=False) as archive:
        require(set(archive.files)==expected,'Exact24 fresh endpoint arrays required')
        values={(m,K):archive[f'm{m}_K{K}'].copy() for m,K in points}
    entries={(r['m'],r['K']):r for r in metadata['endpoints']}
    require(len(entries)==len(metadata['endpoints'])==24 and set(entries)==set(points),'Endpoint metadata differs')
    for key,bits in values.items():
        require(bits.dtype==np.uint8 and bits.ndim==1 and len(bits)>=2 and np.isin(bits,(0,1)).all() and
            len(bits)==entries[key]['payload_bits'],'Fresh stream shape/length differs')
        a=np.ascontiguousarray(bits)
        digest=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
        require(digest==entries[key]['bits_sha256'],'Fresh stream array digest differs')
    return values


def cpu_frames(runtime,rows,streams,partial,phy,ledger,boundary,out):
    """Actual paid PHY for all fixed48 frames; only length-based TX fallback."""
    out=Path(out);out.mkdir();pins=[]
    require(len(rows)==48,'Fixed48 logical frame plan required')
    lookup={(p['family'],p['m'],p['K'],p['q'],p['nominal_rate']):p for p in runtime.profiles.values()}
    for index,row in enumerate(rows):
        boundary();source=streams[row['source_index']]
        selection=partial.choose_encoded_length({key:len(bits) for key,bits in source.items()},
            row['target_m'],row['target_K'],row['source_capacity_bits'])
        require(selection['status']=='SOURCE_LENGTH_FITS_LAYOUT_PENDING','Gate cannot transmit: '+selection['status'])
        m,K=selection['actual_m'],selection['actual_K'];family=partial.PARTIAL_FAMILY if K else partial.WHOLE_FAMILY
        profile=lookup[family,m,K,row['q'],row['nominal_rate']];bits=source[m,K].copy()
        wave,tx=runtime.transmit(profile['profile_id'],bits,row['public_frame_counter'])
        noise=phy.standard_noise(row['source_id'],row['noise_seed'])*10**(-row['snr_db']/20)
        observed=wave+noise
        rx=runtime.receive(observed,row['snr_db'],row['public_frame_counter'],runtime.profiles,ledger,row['event_id'],phase='image_link_gate')
        packet=dict(schema=SCHEMA,frame_index=index,logical_event=row,fallback=selection,transmission=tx,
            payload_sha256=phy.array_sha(bits),noise_sha256=phy.array_sha(noise),observation_sha256=phy.array_sha(observed),
            actual_RX=rx,full_public_receive_catalogue=True,external_packet_reuse=False,source_truth_supplied_to_RX=False)
        path=out/f'{index:02d}.json';save(path,packet);pins.append(descriptor(path))
    require(ledger.snapshot()['unresolved']==0 and ledger.snapshot()['total']<=96,'PHY ledger not closed within96')
    return pins


def recover_and_render(actual_rx,codec,backend,partial,ledger,boundary):
    """Actual accepted wire fields/payload only; no TX or source truth arguments."""
    boundary();body=actual_rx['body'];profile=actual_rx['rx_profile'];backend.traces=[]
    gray=lambda:np.full((3,256,256),.5,dtype=np.float32)
    if not actual_rx['header']['header_ok'] or body is None or not body['crc_accepted'] or not body['parser_accepted']:
        return gray(),dict(kind='gray',source_status=actual_rx['status'],source_decode_called=False,
            actual_received_profile=profile,received_tokens=None,render_called=False)
    require(profile is not None and profile['family'] in ('EC_STATIC_WHOLE','EC_VAR_WHOLE','EC_VAR_PARTIAL'),
        'Accepted header has no registered received family')
    family,m,K=profile['family'],profile['m'],profile['K'];partial.endpoint(m,K)
    backend.current=('ACTUAL_RX',m)
    def decode():
        try:return dict(decoded=codec.decode(family,np.asarray(body['payload'],dtype=np.uint8),m,K),parse_error=None)
        except partial.InvalidSourceStream as error:return dict(decoded=None,parse_error=str(error))
    outcome=ledger.call('source_rx',decode,source=backend.source_index,m=m,K=K,family=family)
    if outcome['decoded'] is None:
        return gray(),dict(kind='gray',source_status='SOURCE_PARSE_REJECT',source_decode_called=True,
            actual_received_profile=profile,received_tokens=None,render_called=False,parse_error=outcome['parse_error'],
            actual_RX_CDF_trace=copy.deepcopy(backend.traces))
    decoded=outcome['decoded'];tokens=decoded['received_tokens']
    require(decoded['canonical'] is True and decoded['zero_extension_reads']==30 and tokens.shape==(partial.token_count(m,K),),
        'Actual source decoder contract differs')
    boundary()
    def render():
        with backend.native.torch.no_grad():return backend.render_function(backend.native,tokens.copy(),m,K)
    image=ledger.call('var_render',render,source=backend.source_index,m=m,K=K)
    require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and
        ((image>=0)&(image<=1)).all(),'Frozen actual reconstruction contract differs')
    return image,dict(kind='tokens',source_status='ARITHMETIC_SOURCE_DECODED',source_decode_called=True,
        actual_received_profile=profile,received_tokens=tokens.tolist(),render_called=True,
        actual_RX_CDF_trace=copy.deepcopy(backend.traces),transmitted_CDF_used=False,comparison_truth_used=False)


def gpu_frames(packet_pins,codec,backend,partial,ledger,boundary,out):
    out=Path(out);out.mkdir();results=[]
    require(len(packet_pins)==48,'Exact48 actual physical outcomes required')
    for index,pin in enumerate(packet_pins):
        require(sha(pin['path'])==pin['sha256'],'Actual received frame changed')
        packet=json.loads(Path(pin['path']).read_text());require(packet['schema']==SCHEMA and packet['frame_index']==index,
            'Actual frame order/scope differs')
        backend.source_index=packet['logical_event']['source_index']
        image,evidence=recover_and_render(packet['actual_RX'],codec,backend,partial,ledger,boundary)
        path=out/f'{index:02d}.npz'
        with path.open('xb') as stream:np.savez(stream,image=image)
        result=dict(frame_index=index,physical_frame=pin,image_archive=descriptor(path),
            image_sha256=backend.g.image_sha(image),evidence=evidence)
        output=out/f'{index:02d}.json';save(output,result);results.append(descriptor(output))
    return results


def close_counts(counts):
    require(counts['unresolved']==0 and counts['reserved']==counts['completed'],'Unresolved image-gate model calls')
    require(set(counts['completed'])==set(CAPS) and all(0<=counts['completed'][k]<=cap for k,cap in CAPS.items()),
        'Image-gate model call cap exceeded')
    require(counts['completed']['model_load']==1,'Exactly one admitted visual model construction')
    require(counts['completed']['decoder_forward']==counts['completed']['var_render'],'One frozen Dc per actual render')
    return counts
