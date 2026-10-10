"""First32 original calibration sources: actual static codec and verified H reuse.

No neural inference, channel, LDPC decoding, metric computation or bootstrap.
Historical VAR roundtrips are reused explicitly, never relabelled new decodes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from t1_entropy_core import (CODE_SHA, OFFSETS, candidates, choose_arithmetic, csv_write,
    load_integer_codec, read, require, sha, source_tokens, static_encode_decode, verify, write)

H_COMPLETION_SHA='e4bd19e04b74db88e91baec4539d7a305db4e07ad0d456b1bf399ad0d99605b5'
MODELS={'vae':'6830fc533a34d52c8f765a7b212782a5b5ef6138992099e8a817ce920fe6fe2b',
        'var':'b82cb0855b24e1d0d8d0aacebbb2d2d6728c2dcc6509b667dfa697c9330347d5',
        'decoder':'bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1'}

def load_h_source(done,index,calibration):
    record=done['records'][index]; cp_path=Path(record['checkpoint'])
    require(record['source_index']==index and record['source_id']==calibration['source_ids'][index],
            'Historical source index mismatch')
    require(done['outputs'].get(str(cp_path))==record['sha256'],'Source checkpoint unsealed')
    verify(cp_path,record['sha256']); cp=read(cp_path)
    require(cp['status']=='H_FULL1000_SOURCE_CODEC_SOURCE_COMPLETE'
        and cp['source_index']==index and cp['source_id']==record['source_id']
        and cp['independent_roundtrip'] is True,'Historical source roundtrip identity differs')
    require(cp['registration_sha256']==done['registration_sha256'], 'Historical source registration differs')
    ref=cp['source_assets_checkpoint']; verify(ref['path'],ref['sha256']); asset=read(ref['path'])
    require(asset['source_id']==cp['source_id'] and asset['source_index']==index,
            'Source token asset identity differs')
    require(asset['tokens_sha256']==cp['tokens_sha256'] and asset['preprocessing_id']==cp['preprocessing_id']
            ==calibration['preprocessing_ids'][index], 'Token/pixel preprocessing identity differs')
    archive=Path(asset['archive']); verify(archive,asset['outputs'][str(archive)])
    with np.load(archive,allow_pickle=False) as z:
        tokens=source_tokens(z['tokens']).copy(); pixels=z['pixels']
        require(pixels.shape==(3,256,256) and pixels.dtype==np.uint8,'Source pixels differ')
        require(hashlib.sha256(pixels.tobytes()).hexdigest()==cp['preprocessing_id'],'Source pixel SHA differs')
    token_sha=hashlib.sha256(b'int64:680\0'+np.asarray(tokens,dtype='<i8').tobytes()).hexdigest()
    require(token_sha==cp['tokens_sha256'],'Complete original token SHA differs')
    require(len(cp['outputs'])==1,'Historical codec stream archive mapping differs')
    codec_path,codec_sha=next(iter(cp['outputs'].items()))
    require(done['outputs'].get(codec_path)==codec_sha,'Source codec archive unsealed'); verify(codec_path,codec_sha)
    lengths={r['m']:r for r in cp['lengths']}; require(set(lengths)=={6,7,8,9},'Historical prefix evidence incomplete')
    with np.load(codec_path,allow_pickle=False) as z:
        require(set(z.files)=={f'm{m}_bits' for m in (6,7,8,9)},'Unexpected historical stream arrays')
        streams={m:z[f'm{m}_bits'].copy() for m in lengths}
    for m,bits in streams.items():
        r=lengths[m]
        require(bits.dtype==np.uint8 and bits.ndim==1 and np.isin(bits,(0,1)).all()
                and len(bits)==r['arithmetic_bits'] and r['raw_bits']==12*OFFSETS[m]
                and 2<=r['flush_bits']<=len(bits) and r['zero_extension_reads']==30,
                'Historical actual stream/length/canonical evidence differs')
    binding={str(cp_path):record['sha256'],ref['path']:ref['sha256'],str(archive):asset['outputs'][str(archive)],codec_path:codec_sha}
    return tokens,streams,list(lengths.values()),cp,binding

def run(args):
    root=Path(args.root); out=Path(args.out); require(not out.exists(),'Fresh output required')
    cal=read(args.calibration); verify(args.h_completion,H_COMPLETION_SHA); old=read(args.h_completion)
    require(old['status']=='H_FULL1000_SOURCE_CODEC_COMPLETE' and old['independent_roundtrip'] is True,
            'Completed independent H source evidence required')
    require(old['source_ids']==cal['source_ids'] and len(set(old['source_ids']))==1000,
            'Original calibration1000 identity/order differs')
    for name,expected in MODELS.items():
        require(old['frozen_visual_identity']['models'][name]==cal['identity']['models'][name]==expected,
                'Frozen same-prior model state differs: '+name)
    entropy,Encoder,Decoder=load_integer_codec(root)
    static=read(args.static_completion)
    require(static['status']=='T1_STATIC_TRAIN20K_CDF_COMPLETE' and static['source_count']==20000
            and static['source_role']=='original_train20k' and static['calibration_used_for_fitting'] is False,
            'Original train20k static model required')
    require(not set(static['source_ids'])&set(cal['source_ids']),'Static fit/calibration overlap')
    require(len(static['outputs'])==1,'Static model archive mapping differs')
    static_path,static_sha=next(iter(static['outputs'].items())); verify(static_path,static_sha)
    with np.load(static_path,allow_pickle=False) as z:
        cdfs=entropy.validate_cdf(z['cdf']); counts=z['counts']
        require(cdfs.shape==(10,4097) and counts.shape==(10,4096),'Static scale table shape differs')
        require(np.array_equal(cdfs,entropy.probability_cdf(np.log(counts.astype(np.float64)+0.5))),
                'Static CDF does not match frozen count/smoothing rule')
    out.mkdir(parents=True); (out/'streams').mkdir()
    source_ids=cal['source_ids'][:32]
    write(out/'sample_manifest.json',dict(rule='first32_original_calibration_registration_order',
        source_ids=source_ids,source_indices=list(range(32)),selection_uses_image_quality=False,
        calibration_registration_sha256=sha(args.calibration)))
    inputs={str(Path(p).resolve()):sha(p) for p in [args.calibration,args.h_completion,args.static_completion]}
    inputs[static_path]=static_sha
    rows=[]; feasibility=[]; receipts=[]; outputs={}; catalogue=candidates()
    for index in range(32):
        tokens,var_streams,var_lengths,cp,bindings=load_h_source(old,index,cal); inputs.update(bindings)
        static_streams,static_lengths=static_encode_decode(tokens,cdfs,Encoder,Decoder)
        archive=out/'streams'/f'{index:04d}.npz'
        with archive.open('xb') as f:
            np.savez(f,**{f'static_m{m}_bits':v for m,v in static_streams.items()},
                     **{f'var_m{m}_bits':v for m,v in var_streams.items()})
        outputs[str(archive.resolve())]=sha(archive)
        for family,streams,lengths,reused in [('EC_STATIC_WHOLE',static_streams,static_lengths,False),
                                              ('EC_VAR_WHOLE',var_streams,var_lengths,True)]:
            for r in lengths:
                if r['m'] not in (7,8,9): continue
                rows.append(dict(family=family,source_id=source_ids[index],source_index=index,
                    target_m=r['m'],actual_m=r['m'],raw_bits=int(r['raw_bits']),
                    arithmetic_bits=int(r['arithmetic_bits']),terminal_bits=int(r['flush_bits']),
                    byte_alignment_bits=0,length_field_bits=13,body_crc_bits=16,
                    body_information_bits_before_padding=int(r['arithmetic_bits'])+29,
                    header_source_bits=12,header_crc_bits=16,header_tail_bits=6,header_coded_bits=136,
                    header_symbols=68,zero_extension_reads=30,roundtrip_reused=reused,
                    source_role='calibration_source_gate',stream_file=str(archive.resolve())))
            for candidate in catalogue:
                selected=choose_arithmetic({m:len(v) for m,v in streams.items()},candidate)
                actual=selected['arithmetic_bits']
                feasibility.append(dict(family=family,source_id=source_ids[index],source_index=index,
                    **candidate,status=selected['status'],actual_m=selected['actual_m'],
                    source_bits=actual,information_padding_bits=None if actual is None else candidate['source_capacity']-actual,
                    missing_m=selected['missing_m'],attempts_json=json.dumps(selected['attempts'],separators=(',',':')),
                    raw_substitution=False))
        receipts.append(dict(source_id=source_ids[index],source_index=index,source_tokens_sha256=cp['tokens_sha256'],
            VAR=dict(mode='REUSED_VERIFIED_H_INDEPENDENT_ROUNDTRIP',modes=[6,7,8,9],new_probability_calls=0,
                     new_arithmetic_decodes=0,source_checkpoint=old['records'][index]),
            STATIC=dict(mode='NEW_ACTUAL_CPU_ENCODE_DECODE_EXACT_TOKENS',modes=[5,6,7,8,9],
                        canonical_reencode_passed=True),
            reconstruction_equivalence='NOT_RUN_REQUIRES_IDENTICAL_FROZEN_RECOVERY_CACHE_OR_BOUNDED_RENDER'))
        print(f'source codec gate {index+1}/32; reused VAR streams, actual static roundtrip',flush=True)
    csv_write(out/'source_lengths.csv',rows); csv_write(out/'source_feasibility.csv',feasibility)
    csv_write(out/'finite_candidates.csv',catalogue)
    missing=[r for r in feasibility if r['status']=='BLOCKED_MISSING_PREFIX']
    report=dict(status='SOURCE_COMPONENTS_VERIFIED_VISUAL_GATE_PENDING',source_count=32,
        static_actual_roundtrips=32*5,VAR_roundtrip_receipts_reused=32*4,VAR_new_roundtrips=0,
        requested_m7_m8_m9_rows=len(rows),missing_fallback_candidate_source_rows=len(missing),
        missing_VAR_m5_source_indices=sorted({r['source_index'] for r in missing if r['family']=='EC_VAR_WHOLE'}),
        ldpc_qualification='NOT_RUN_BY_SOURCE_ONLY_GATE',visual_recovery_equivalence='NOT_RUN',
        timings='PREPARATION_ONLY_NOT_ONLINE_TX_RX',new_model_calls=0,new_packet_decodes=0,
        new_bootstrap_calls=0,training_updates=0,holdout_used=False,records=receipts)
    write(out/'roundtrip_validation.json',report)
    for path in out.iterdir():
        if path.is_file(): outputs[str(path.resolve())]=sha(path)
    completion=dict(status='T1_SOURCE32_CPU_STAGE_COMPLETE_WITH_EXPLICIT_FOLLOWUPS',
        code_commit='14b09ecd72984fb39c683d1a62bcd3221c69142f',input_bindings=inputs,outputs=outputs,
        source_count=32,source_length_rows=len(rows),candidate_source_rows=len(feasibility),
        source_only=True,new_model_calls=0,new_packet_decodes=0,training_updates=0,
        T1_fully_complete=False,pending=['real_LDPC_layout_qualification','VAR_m5_for_actual_overflow_sources',
        'identical_recovery_output_gate','calibration_selection_and_frozen_test'],
        missing_fallback_candidate_source_rows=len(missing))
    write(out/'completion.json',completion); return completion

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','calibration','h-completion','static-completion','out'): p.add_argument('--'+name,required=True)
    result=run(p.parse_args()); print(result['status'])

if __name__=='__main__': main()
