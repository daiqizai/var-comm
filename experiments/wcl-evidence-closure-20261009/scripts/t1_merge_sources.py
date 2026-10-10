"""Seal complete source assets after independently completed VAR short prefixes."""
import argparse
from pathlib import Path
import numpy as np
from t1_entropy_core import OFFSETS,candidates,choose_arithmetic,read,require,sha,verify,write
from t1_source_asset_schema import build_record

def run(args):
    out=Path(args.out);require(not out.exists(),'Fresh merged output required')
    done=read(args.source_completion);base=Path(args.source_completion).parent
    manifest_path=base/'manifest.json';verify(manifest_path,done['outputs'][str(manifest_path)])
    manifest=read(manifest_path);require(done['status']=='T1_CALIBRATION_SOURCE_CPU_PREPARATION_COMPLETE','Source stage incomplete')
    fallback={};inputs={str(Path(args.source_completion)):sha(args.source_completion),str(manifest_path):sha(manifest_path)}
    for name in args.fallback_completion:
        cp=Path(name);fd=read(cp);mp=cp.parent/'fallback_manifest.json'
        require(fd['status'] in ('T1_CALIBRATION_VAR_SHORT_PREFIX_PREPARATION_COMPLETE','T1_SOURCE32_REAL_RECOVERY_GATE_COMPLETE'),
                'Short-prefix execution not complete')
        verify(mp,fd['outputs'][str(mp)]);fm=read(mp);inputs[str(cp)]=sha(cp);inputs[str(mp)]=sha(mp)
        for r in fm['records']:
            i=r['source_index'];require(i not in fallback,'Duplicate source fallback completion')
            require(r['source_id']==manifest['source_ids'][i],'Fallback source identity differs')
            verify(r['archive'],r['sha256']);fallback[i]=r
    out.mkdir(parents=True);(out/'sources').mkdir();(out/'source_checkpoints').mkdir()
    records=[];outputs={};logical_conditions=0
    for original in manifest['records']:
        i=original['source_index'];verify(original['checkpoint'],original['checkpoint_sha256'])
        item=read(original['checkpoint']);verify(item['archive']['path'],item['archive']['sha256'])
        with np.load(item['archive']['path'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        if i in fallback:
            row=fallback[i];require(row['tokens_sha256']==item['source_tokens_sha256'],'Fallback token identity differs')
            with np.load(row['archive'],allow_pickle=False) as z:
                for m in (4,5):
                    bits=z[f'm{m}_bits'];rx=z[f'm{m}_received_tokens']
                    require(np.array_equal(rx,arrays['tokens'][:OFFSETS[m]]),'Independent short-prefix decoded values differ')
                    arrays[f'var_m{m}_bits']=bits.copy();arrays[f'var_m{m}_received_tokens']=rx.copy()
        for family,prefix in [('EC_STATIC_WHOLE','static'),('EC_VAR_WHOLE','var')]:
            lengths={m:len(arrays[f'{prefix}_m{m}_bits']) for m in range(4,10) if f'{prefix}_m{m}_bits' in arrays}
            for candidate in candidates():
                selection=choose_arithmetic(lengths,candidate,minimum_m=4)
                require(selection['status'] not in ('BLOCKED_MISSING_PREFIX','TX_UNENCODABLE'),
                        f'Incomplete registered fallback chain: source{i} {family} {candidate["candidate_id"]}')
                logical_conditions+=1
        path=out/'sources'/f'{i:04d}.npz'
        with path.open('xb') as f:np.savez(f,**arrays)
        record=build_record(source_index=i,source_id=item['source_id'],tokens=arrays['tokens'],
            preprocessing_id=item['preprocessing_id'],source_assets_checkpoint=item['source_assets_checkpoint'],
            archive=path,arrays=arrays,static_completion_sha=item['static_model_completion_sha256'],
            origins={'EC_STATIC_WHOLE':'VERIFIED_SOURCE_CPU_PREPARATION_REUSE',
                     'EC_VAR_WHOLE':'VERIFIED_H_M6_M9_AND_ACTUAL_SHORT_PREFIX_ROUNDTRIPS'},
            upstream_evidence={'EC_STATIC_WHOLE':dict(checkpoint=original['checkpoint'],sha256=original['checkpoint_sha256']),
                'EC_VAR_WHOLE':dict(checkpoint=original['checkpoint'],sha256=original['checkpoint_sha256'],short_prefix=fallback.get(i))})
        cp=out/'source_checkpoints'/f'{i:04d}.json';write(cp,record)
        outputs[str(path)]=sha(path);outputs[str(cp)]=sha(cp)
        records.append(dict(source_index=i,source_id=item['source_id'],checkpoint=str(cp),checkpoint_sha256=sha(cp)))
    mp=out/'manifest.json';write(mp,dict(status='T1_CALIBRATION_SOURCE_ASSETS_READY',source_count=len(records),
        source_ids=manifest['source_ids'],records=records,all_registered_candidates_have_complete_length_fallback=True,
        checked_candidate_source_conditions=logical_conditions,physical_layout_qualification='SEPARATE_REQUIRED_STAGE',minimum_fallback_m=4))
    outputs[str(mp)]=sha(mp)
    result=dict(status='T1_CALIBRATION_COMPLETE_SOURCE_ASSETS_SEALED',source_count=len(records),input_bindings=inputs,
        outputs=outputs,new_model_calls=0,new_packet_decodes=0,new_entropy_encodes=0,training_updates=0,holdout_used=False)
    write(out/'completion.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-completion',required=True)
    p.add_argument('--fallback-completion',action='append',default=[]);p.add_argument('--out',required=True)
    print(run(p.parse_args())['status'])

if __name__=='__main__':main()
