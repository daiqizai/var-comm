"""CPU source assets for fixed calibration100/1000, reusing source32 exactly."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from t1_entropy_core import OFFSETS,read,require,sha,verify,write,static_encode_decode,load_integer_codec
from t1_source_check import load_h_source,H_COMPLETION_SHA,MODELS
from t1_source_asset_schema import build_record

def run(args):
    out=Path(args.out);require(not out.exists(),'Fresh output required')
    cal=read(args.calibration);verify(args.h_completion,H_COMPLETION_SHA);old=read(args.h_completion)
    require(old['source_ids']==cal['source_ids'] and len(cal['source_ids'])==1000,'Original calibration order differs')
    for k,v in MODELS.items():require(cal['identity']['models'][k]==old['frozen_visual_identity']['models'][k]==v,'Model state differs')
    static=read(args.static_completion);require(static['status']=='T1_STATIC_TRAIN20K_CDF_COMPLETE','Static fitting not complete')
    p,s=next(iter(static['outputs'].items()));verify(p,s)
    entropy,Encoder,Decoder=load_integer_codec(args.root)
    with np.load(p,allow_pickle=False) as z:
        cdf=entropy.validate_cdf(z['cdf']).copy()
        require(np.array_equal(cdf,entropy.probability_cdf(np.log(z['counts'].astype(np.float64)+.5))),'Changed static CDF')
    first=read(args.source32_completion);first_dir=Path(args.source32_completion).parent
    require(first['status']=='T1_SOURCE32_CPU_STAGE_COMPLETE_WITH_EXPLICIT_FOLLOWUPS','Source32 gate missing')
    require(first['input_bindings'].get(str(Path(args.static_completion).resolve()))==sha(args.static_completion),'Source32 static model differs')
    manifest=read(first_dir/'sample_manifest.json')
    verify(first_dir/'sample_manifest.json',first['outputs'][str(first_dir/'sample_manifest.json')])
    require(manifest['source_ids']==cal['source_ids'][:32],'Source32 ordering differs')
    extra={};extra_inputs={}
    for name in args.fallback_manifest or []:
        value=read(name);extra_inputs[str(Path(name))]=sha(name)
        require(value['status'] in ('T1_SOURCE32_VAR_FALLBACK_COMPLETE','T1_CALIBRATION_VAR_FALLBACK_COMPLETE'),'Fallback stage incomplete')
        for row in value['records']:
            i=row['source_index'];require(i not in extra,'Duplicate fallback source evidence')
            require(row['source_id']==cal['source_ids'][i],'Fallback source order differs');verify(row['archive'],row['sha256']);extra[i]=row
    out.mkdir(parents=True);(out/'sources').mkdir();(out/'source_checkpoints').mkdir()
    inputs={str(Path(x).resolve()):sha(x) for x in [args.calibration,args.h_completion,args.static_completion,args.source32_completion]}
    inputs.update(extra_inputs);outputs={};records=[];pending=[]
    for i in range(args.count):
        require(not (out/'STOP').exists(),'Stopped at source boundary; preserve completed source files')
        tokens,vs,vlengths,cp,bindings=load_h_source(old,i,cal);inputs.update(bindings)
        if i<32:
            stream=first_dir/'streams'/f'{i:04d}.npz';verify(stream,first['outputs'][str(stream)])
            with np.load(stream,allow_pickle=False) as z:ss={m:z[f'static_m{m}_bits'].copy() for m in (5,6,7,8,9)}
            m4,_=static_encode_decode(tokens,cdf,Encoder,Decoder,modes=(4,));ss.update(m4)
            static_origin='EXACT_SOURCE32_STATIC_STREAM_REUSE'
        else:
            ss,_=static_encode_decode(tokens,cdf,Encoder,Decoder,modes=(4,5,6,7,8,9))
            static_origin='NEW_CPU_STATIC_EXACT_ROUNDTRIP'
        if i in extra:
            row=extra[i];require(row['tokens_sha256']==cp['tokens_sha256'],'Fallback source tokens differ')
            with np.load(row['archive'],allow_pickle=False) as z:
                for item in row['lengths']:
                    m=item['m'];bits=z[f'm{m}_bits'];require(m in (4,5) and len(bits)==item['arithmetic_bits'],'Fallback length differs')
                    vs[m]=bits.copy()
        needs=len(vs[6])>927 and (4 not in vs or 5 not in vs)
        if needs:pending.append(dict(source_index=i,source_id=cp['source_id'],tokens_sha256=cp['tokens_sha256'],m6_bits=len(vs[6])))
        target=out/'sources'/f'{i:04d}.npz'
        arrays=dict(tokens=tokens,**{f'static_m{m}_bits':b for m,b in ss.items()},**{f'var_m{m}_bits':b for m,b in vs.items()},
            **{f'static_m{m}_received_tokens':tokens[:OFFSETS[m]].copy() for m in ss},
            **{f'var_m{m}_received_tokens':tokens[:OFFSETS[m]].copy() for m in vs})
        with target.open('xb') as f:np.savez(f,**arrays)
        outputs[str(target.resolve())]=sha(target)
        source_record=build_record(source_index=i,source_id=cp['source_id'],tokens=tokens,
            preprocessing_id=cp['preprocessing_id'],source_assets_checkpoint=cp['source_assets_checkpoint'],archive=target,
            arrays=arrays,static_completion_sha=sha(args.static_completion),
            origins={'EC_STATIC_WHOLE':static_origin,'EC_VAR_WHOLE':'VERIFIED_H_INDEPENDENT_ROUNDTRIP_REUSE'},
            upstream_evidence={'EC_STATIC_WHOLE':dict(static_model=str(Path(args.static_completion).resolve()),
                    source32_completion=str(Path(args.source32_completion).resolve()) if i<32 else None,
                    fresh_m4_roundtrip=True,fresh_remaining_prefixes=i>=32),
                'EC_VAR_WHOLE':dict(source_checkpoint=old['records'][i],short_prefix=extra.get(i))})
        checkpoint=out/'source_checkpoints'/f'{i:04d}.json';write(checkpoint,source_record);outputs[str(checkpoint.resolve())]=sha(checkpoint)
        records.append(dict(source_index=i,source_id=cp['source_id'],tokens_sha256=cp['tokens_sha256'],
            preprocessing_id=cp['preprocessing_id'],source_assets_checkpoint=cp['source_assets_checkpoint'],
            archive=str(target.resolve()),sha256=sha(target),checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),static_origin=static_origin,
            VAR_origin='EXACT_H_CALIBRATION_STREAM_REUSE_PLUS_REGISTERED_SHORT_FALLBACK',
            families={'EC_STATIC_WHOLE':{str(m):len(b) for m,b in ss.items()},'EC_VAR_WHOLE':{str(m):len(b) for m,b in vs.items()}},
            VAR_short_prefix_pending=needs))
        if (i+1)%25==0:print(f'calibration source assets {i+1}/{args.count}; no model or PHY calls',flush=True)
    write(out/'fallback_needed.json',dict(status='EXACT_LENGTH_TRIGGERED_FALLBACK_REQUIREMENTS',source_count=args.count,
        minimum_capacity_bits=927,minimum_fallback_m=4,records=pending,selection_by_quality=False))
    write(out/'manifest.json',dict(status='T1_CALIBRATION_SOURCE_ASSETS_READY_WITH_EXPLICIT_FALLBACKS',source_count=args.count,
        sample_rule=f'first{args.count}_original_calibration_order',source_ids=cal['source_ids'][:args.count],records=records,
        static_model_sha256=sha(args.static_completion),original_H_completion_sha256=H_COMPLETION_SHA))
    for p in out.iterdir():
        if p.is_file():outputs[str(p.resolve())]=sha(p)
    result=dict(status='T1_CALIBRATION_SOURCE_CPU_PREPARATION_COMPLETE',source_count=args.count,
        original_VAR_streams_reused=args.count*4,source32_static_sources_reused=32,
        new_static_roundtrips=(args.count-32)*6+32,VAR_short_fallback_sources_pending=len(pending),
        input_bindings=inputs,outputs=outputs,new_model_calls=0,new_packet_decodes=0,training_updates=0,
        holdout_used=False,quality_selected=False)
    write(out/'completion.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','calibration','h-completion','static-completion','source32-completion','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--count',type=int,choices=(100,1000),required=True)
    p.add_argument('--fallback-manifest',action='append')
    print(run(p.parse_args())['status'])

if __name__=='__main__':main()
