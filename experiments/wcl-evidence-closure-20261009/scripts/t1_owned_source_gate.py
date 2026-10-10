"""Bounded real-model source gate called only by an admitted outer GPU owner.

No scheduler/SSH/PHY is provided. Caller supplies a live boundary() function.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from t1_entropy_core import OFFSETS,read,require,sha,verify,write,csv_write
from t1_codec_runtime import SourceCodec,module,check_native
from t1_source_check import load_h_source,H_COMPLETION_SHA

def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def run_owned(*,root,calibration,h_completion,static_completion,source32_completion,out,native,boundary,
              reference_lookup=None,account=None):
    """32 sources: registered m4/m5 streams and clean static/raw render parity.

    VAR m6..m9 exact token roundtrip evidence is reused; no redundant VAR
    probability pass is made for these streams. Their identical recovered-token
    state shares the newly verified reference reconstruction.
    """
    root=Path(root);out=Path(out);require(not out.exists(),'Fresh GPU gate output required')
    def call(kind,operation,index,m):
        return operation() if account is None else account(kind,operation,source_index=index,m=m,stage='source32_gate')
    require(callable(boundary),'An owned GPU safe-boundary callback is required');boundary();check_native(native)
    verify(h_completion,H_COMPLETION_SHA);old=read(h_completion);cal=read(calibration)
    source_done=read(source32_completion)
    require(source_done['status']=='T1_SOURCE32_CPU_STAGE_COMPLETE_WITH_EXPLICIT_FOLLOWUPS'
            and source_done['source_count']==32,'First32 actual CPU source stage required')
    source_dir=Path(source32_completion).parent
    for path in [source_dir/'sample_manifest.json',source_dir/'roundtrip_validation.json']:
        verify(path,source_done['outputs'][str(path)])
    sample=read(source_dir/'sample_manifest.json')
    require(sample['source_ids']==cal['source_ids'][:32] and old['source_ids']==cal['source_ids'],
            'Original first32 source identity differs')
    codec=SourceCodec(root,static_completion=static_completion,native=native)
    original=module(root/'experiments/content-real-64qam-20261006/h_source_driver.py','_wcl_t1_gate_original_source')
    out.mkdir(parents=True);(out/'sources').mkdir();(out/'fallback').mkdir()
    inputs={str(Path(p)):sha(p) for p in (calibration,h_completion,static_completion,source32_completion)}
    outputs={};rows=[];fallback=[];render_calls=0;reference_hits=0
    with native.torch.no_grad():
        for index in range(32):
            boundary();tokens,var_streams,_,cp,bound=load_h_source(old,index,cal);inputs.update(bound)
            source_archive=source_dir/'streams'/f'{index:04d}.npz'
            verify(source_archive,source_done['outputs'][str(source_archive)])
            with np.load(source_archive,allow_pickle=False) as z:
                static_bits={m:z[f'static_m{m}_bits'].copy() for m in (7,8,9)}
            if len(var_streams[6])>927:
                boundary();tx=call('var_source_tx',lambda:codec.encode('EC_VAR_WHOLE',tokens,modes=(4,5)),index,5)
                modes=(4,5);arrays={};records=[]
                for m in modes:
                    rx=call('var_source_rx',lambda:codec.decode('EC_VAR_WHOLE',tx[m]['bits'],m),index,m)
                    require(np.array_equal(rx['received_tokens'],tokens[:OFFSETS[m]]),'VAR fallback independent roundtrip failed')
                    arrays[f'm{m}_bits']=tx[m]['bits'];arrays[f'm{m}_received_tokens']=rx['received_tokens'].copy()
                    records.append({k:tx[m][k] for k in ('raw_bits','arithmetic_bits','flush_bits')}|dict(m=m,zero_extension_reads=30))
                require(tx[4]['arithmetic_bits']<=927,'Registered minimum prefix capacity guarantee failed')
                fp=out/'fallback'/f'{index:04d}.npz'
                with fp.open('xb') as f:np.savez(f,**arrays)
                outputs[str(fp)]=sha(fp)
                fallback.append(dict(source_index=index,source_id=cp['source_id'],tokens_sha256=cp['tokens_sha256'],
                    archive=str(fp),sha256=sha(fp),lengths=records,new_independent_roundtrips=len(modes),
                    m4_profile_required=True,source_m6_bits=len(var_streams[6]),
                    use_rule='only_if_all_longer_whole_arithmetic_prefixes_overflow'))
            images={}
            for m in (7,8,9):
                boundary();rx=call('static_source_rx',lambda:codec.decode('EC_STATIC_WHOLE',static_bits[m],m),index,m)
                require(np.array_equal(rx['received_tokens'],tokens[:OFFSETS[m]]),'Static actual received prefix differs')
                cached=None if reference_lookup is None else reference_lookup(index,m,tokens[:OFFSETS[m]].copy())
                if cached is None:
                    baseline=call('image_render',lambda:original.render_received(native,tokens[:OFFSETS[m]].copy(),m,0),index,m);render_calls+=1
                else:
                    require(cached['source_id']==cp['source_id'] and cached['m']==m and cached['K']==0
                            and np.array_equal(cached['received_tokens'],tokens[:OFFSETS[m]]),
                            'Reference cache receiver input differs')
                    require(cached['models']==native.loaded['identity']['models'], 'Reference cache model identity differs')
                    for path,digest in cached['bindings'].items():verify(path,digest)
                    baseline=np.asarray(cached['image'])
                    require(baseline.dtype==np.float32 and baseline.shape==(3,256,256)
                            and array_sha(baseline)==cached['image_sha256'],'Reference image identity differs')
                    inputs.update(cached['bindings']);reference_hits+=1
                restored=call('image_render',lambda:original.render_received(native,rx['received_tokens'].copy(),m,0),index,m);render_calls+=1
                max_error=float(np.max(np.abs(baseline-restored)))
                require(np.array_equal(baseline,restored),'Same source token/Dc recovery is not exactly equal')
                images[f'm{m}_image']=baseline
                rows.append(dict(source_index=index,source_id=cp['source_id'],m=m,
                    exact_tokens_equal=True,static_vs_raw_image_exact=True,max_abs_error=max_error,
                    raw_reference_reused=cached is not None,
                    image_sha256=array_sha(baseline),VAR_image_reused_from_identical_verified_token_state=True,
                    VAR_token_roundtrip_source_checkpoint=old['records'][index]['checkpoint'],
                    frozen_decoder_state_sha256=native.loaded['identity']['models']['decoder']))
            target=out/'sources'/f'{index:04d}.npz'
            with target.open('xb') as f:np.savez(f,**images)
            outputs[str(target)]=sha(target)
            print(f'owned source gate {index+1}/32; exact static/raw recovery; historical VAR token evidence reused',flush=True)
    boundary();native.frozen()
    csv_write(out/'recovery_equivalence.csv',rows)
    write(out/'fallback_manifest.json',dict(status='T1_SOURCE32_VAR_FALLBACK_COMPLETE',records=fallback,
        source_count=32,original_source_order=cal['source_ids'][:32],minimum_capacity_bits=927,minimum_fallback_m=4,
        m4_profiles_required=any(x['m4_profile_required'] for x in fallback)))
    for p in out.iterdir():
        if p.is_file():outputs[str(p)]=sha(p)
    completion=dict(status='T1_SOURCE32_REAL_RECOVERY_GATE_COMPLETE',source_count=32,whole_prefixes=[7,8,9],
        static_raw_image_pairs=96,exact_equal_image_pairs=96,real_image_renders=render_calls,
        exact_raw_reference_cache_hits=reference_hits,
        historical_VAR_prefix_roundtrips_reused=96,VAR_new_m7_m8_m9_probability_calls=0,
        VAR_new_fallback_roundtrips=sum(r['new_independent_roundtrips'] for r in fallback),
        fallback_source_count=len(fallback),requires_new_m4_paid_profiles=any(r['m4_profile_required'] for r in fallback),
        input_bindings=inputs,outputs=outputs,new_packet_decodes=0,training_updates=0,
        source_scope='first32_original_calibration',holdout_used=False,metrics_for_policy_selection=False,
        full_T1_complete=False)
    write(out/'completion.json',completion);return completion
