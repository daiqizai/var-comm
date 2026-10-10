"""Owned, post-freeze entropy preparation from original common500 token caches.

No scheduler or model constructor. A root-owned window supplies native,
boundary, and an attempt/completion accounting callback. Only selected complete
prefixes receive independent decode work; all actual TX lengths stay auditable.
"""
from pathlib import Path
import hashlib
import numpy as np
from t1_entropy_core import OFFSETS,read,require,sha,verify,write,source_tokens,choose_arithmetic
from t1_codec_runtime import SourceCodec,check_native
from t1_source_asset_schema import build_record,token_sha
from t1_holdout_metadata import FAMILIES,SNRS,MANIFEST_SHA


def run_owned(*,request,out,native,boundary,account):
    require(callable(boundary) and callable(account),'Explicit root-owned boundary and call accounting required')
    boundary();check_native(native);r=read(request);out=Path(out)
    require(not out.exists(),'Fresh post-freeze source output required')
    require(r['schema']=='T1_POSTHOC_COMMON500_SOURCE_REQUEST_V1' and r['source_count']==500
        and r['post_hoc_supplement'] and r['holdout_used_for_selection'] is False,'Post-hoc frozen500 source request required')
    for path,digest in r['input_bindings'].items():verify(path,digest)
    verify(r['freeze']['path'],r['freeze']['sha256']);policy=read(r['freeze']['path'])
    require(policy['status']=='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1' and not policy['holdout_used_for_selection'],
        'Actual calibration-only freeze required')
    verify(r['original_source_manifest']['path'],MANIFEST_SHA)
    static=policy['protocol']['static_completion'];verify(static['path'],static['sha256'])
    codec=SourceCodec(r['root'],static_completion=static['path'],native=native)
    out.mkdir(parents=True);(out/'sources').mkdir();(out/'source_checkpoints').mkdir()
    outputs={};records=[];selected_rt={f:0 for f in FAMILIES};source_prior_evaluations=0
    for record in r['records']:
        boundary();index=record['source_index'];sid=record['source_id']
        require(sid==r['source_ids'][index],'Source order differs')
        verify(record['checkpoint'],record['checkpoint_sha256']);cp=read(record['checkpoint'])
        verify(record['archive'],record['archive_sha256'])
        with np.load(record['archive'],allow_pickle=False) as z:
            require(set(z.files)=={'tokens','pixels'},'Original cached source array keys changed')
            tokens=source_tokens(z['tokens']).copy();pixels=z['pixels'].copy()
        require(token_sha(tokens)==record['tokens_sha256']==cp['tokens_sha256']
            and pixels.dtype==np.uint8 and pixels.shape==(3,256,256)
            and hashlib.sha256(pixels.tobytes()).hexdigest()==record['preprocessing_id'],
            'Original cached tokens or preprocessing identity changed')
        arrays={'tokens':tokens};selected={};tx_evidence={};call_evidence={}
        for family in FAMILIES:
            prefix='static' if family=='EC_STATIC_WHOLE' else 'var'
            frozen=policy['policies'][family];maximum=max(c['target_m'] for c in frozen.values())
            modes=tuple(range(4,maximum+1))
            encoded=account(prefix+'_source_tx',lambda:codec.encode(family,tokens,modes=modes),
                stage='posthoc_common500_source',source_index=index,family=family,m=maximum)
            lengths={m:e['arithmetic_bits'] for m,e in encoded.items()};selected[family]={}
            tx_evidence[family]={str(m):dict(bits_key=f'tx_{prefix}_m{m}_bits',
                **{k:e[k] for k in ('raw_bits','arithmetic_bits','flush_bits')}) for m,e in encoded.items()}
            arrays.update({f'tx_{prefix}_m{m}_bits':e['bits'].copy() for m,e in encoded.items()})
            for snr in SNRS:
                choice=choose_arithmetic(lengths,frozen[str(snr)],minimum_m=4)
                require(choice['status']=='SOURCE_LENGTH_FITS_LAYOUT_PENDING','Registeredm4 fallback chain cannot fit')
                selected[family][str(snr)]=dict(candidate_id=frozen[str(snr)]['candidate_id'],
                    target_m=frozen[str(snr)]['target_m'],q=frozen[str(snr)]['q'],nominal_rate=frozen[str(snr)]['nominal_rate'],
                    **choice)
            unique=sorted({choice['actual_m'] for choice in selected[family].values()})
            call_evidence[family]=dict(TX_traversal_max_m=maximum,independent_RX_prefixes=unique)
            for m in unique:
                decoded=account(prefix+'_source_rx',lambda:codec.decode(family,encoded[m]['bits'],m),
                    stage='posthoc_common500_source',source_index=index,family=family,m=m)
                require(np.array_equal(decoded['received_tokens'],tokens[:OFFSETS[m]]),'Actual independent source roundtrip differs')
                arrays[f'{prefix}_m{m}_bits']=encoded[m]['bits'].copy()
                arrays[f'{prefix}_m{m}_received_tokens']=decoded['received_tokens'].copy();selected_rt[family]+=1
            if family=='EC_VAR_WHOLE':source_prior_evaluations+=maximum+sum(unique)
        archive=out/'sources'/f'{index:04d}.npz'
        with archive.open('xb') as f:np.savez(f,**arrays)
        item=build_record(source_index=index,source_id=sid,tokens=tokens,preprocessing_id=record['preprocessing_id'],
            source_assets_checkpoint=dict(path=record['checkpoint'],sha256=record['checkpoint_sha256']),
            archive=archive,arrays=arrays,static_completion_sha=static['sha256'],
            origins={f:'NEW_ACTUAL_FROZEN_HOLDOUT_SELECTED_PREFIX_ROUNDTRIP' for f in FAMILIES},
            upstream_evidence={f:dict(request=str(Path(request).resolve()),request_sha256=sha(request),
                freeze=r['freeze'],source_checkpoint=dict(path=record['checkpoint'],sha256=record['checkpoint_sha256']),
                calls=call_evidence[f]) for f in FAMILIES})
        item.update(source_role='holdout',post_hoc_supplement=True,holdout_used_for_selection=False,
            tx_prefixes=tx_evidence,selected_by_snr=selected,freeze=r['freeze'],
            evaluation_class_index=record['evaluation_class_index'])
        target=out/'source_checkpoints'/f'{index:04d}.json';write(target,item)
        outputs[str(archive)]=sha(archive);outputs[str(target)]=sha(target)
        records.append(dict(source_index=index,source_id=sid,checkpoint=str(target),checkpoint_sha256=sha(target)))
        print(f'post-hoc frozen source {index+1}/500; selected independent RX static={selected_rt[FAMILIES[0]]}, VAR={selected_rt[FAMILIES[1]]}',flush=True)
    boundary();native.frozen()
    mp=out/'manifest.json';write(mp,dict(status='T1_POSTHOC_COMMON500_SOURCE_ASSETS_READY',source_count=500,
        source_ids=r['source_ids'],records=records,freeze=r['freeze'],selected_SNRs=SNRS,
        source_role='holdout',post_hoc_supplement=True,holdout_used_for_selection=False))
    outputs[str(mp)]=sha(mp)
    result=dict(status='T1_POSTHOC_COMMON500_SOURCE_PREPARATION_COMPLETE',request_sha256=sha(request),
        source_count=500,freeze=r['freeze'],new_VAR_TX_traversals=500,new_VAR_independent_RX_calls=selected_rt['EC_VAR_WHOLE'],
        new_static_TX_traversals=500,new_static_independent_RX_calls=selected_rt['EC_STATIC_WHOLE'],
        actual_VAR_source_prior_scale_evaluations=source_prior_evaluations,new_encoder_calls=0,new_image_renders=0,
        new_metric_calls=0,new_packet_decodes=0,training_updates=0,holdout_used_for_selection=False,
        post_hoc_supplement=True,outputs=outputs)
    write(out/'completion.json',result);return result
