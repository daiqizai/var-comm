"""m4/m5 source asset completion inside an already-owned GPU stage."""
from pathlib import Path
import hashlib
import numpy as np
from t1_codec_runtime import SourceCodec,check_native
from t1_entropy_core import OFFSETS,read,require,sha,verify,write,source_tokens

def run_owned(*,root,source_population_completion,static_completion,out,native,boundary,
              existing_fallback_manifests=(),account=None):
    boundary();check_native(native);out=Path(out);require(not out.exists(),'Fresh output required')
    def call(kind,operation,index,m):
        return operation() if account is None else account(kind,operation,source_index=index,m=m,stage='calibration_short_prefixes')
    done=read(source_population_completion)
    require(done['status']=='T1_CALIBRATION_SOURCE_CPU_PREPARATION_COMPLETE','Calibration source population incomplete')
    manifest_path=Path(source_population_completion).parent/'manifest.json'
    verify(manifest_path,done['outputs'][str(manifest_path)]);manifest=read(manifest_path)
    require(manifest['source_count'] in (100,1000),'Only registered calibration populations allowed')
    codec=SourceCodec(root,static_completion=static_completion,native=native)
    existing={};inputs={str(Path(p)):sha(p) for p in (source_population_completion,manifest_path,static_completion)}
    for path in existing_fallback_manifests:
        value=read(path);inputs[str(Path(path))]=sha(path)
        require(value['status'] in ('T1_SOURCE32_VAR_FALLBACK_COMPLETE','T1_CALIBRATION_VAR_FALLBACK_COMPLETE'),
                'Existing short-prefix evidence incomplete')
        for row in value['records']:
            i=row['source_index'];require(i not in existing,'Duplicate short-prefix source evidence')
            require({x['m'] for x in row['lengths']}=={4,5},'Both registered m4/m5 required')
            verify(row['archive'],row['sha256']);existing[i]=row
    out.mkdir(parents=True);(out/'sources').mkdir();outputs={};records=[];new=0;reused=0
    for item in manifest['records']:
        boundary();i=item['source_index'];require(item['source_id']==manifest['source_ids'][i],'Source index differs')
        if int(item['families']['EC_VAR_WHOLE']['6'])<=927:
            continue  # Every admitted bucket fits m6: shorter fallbacks are unreachable.
        verify(item['archive'],item['sha256'])
        with np.load(item['archive'],allow_pickle=False) as z:tokens=source_tokens(z['tokens']).copy()
        token_sha=hashlib.sha256(b'int64:680\0'+np.asarray(tokens,dtype='<i8').tobytes()).hexdigest()
        require(token_sha==item['tokens_sha256'],'Source tokens changed')
        if i in existing:
            old=existing[i];require(old['source_id']==item['source_id'] and old['tokens_sha256']==token_sha,
                                   'Existing short-prefix/source identity differs')
            records.append(dict(old,origin='EXACT_REGISTERED_SHORT_PREFIX_REUSE'));inputs[old['archive']]=old['sha256'];reused+=1
            continue
        encoded=call('var_source_tx',lambda:codec.encode('EC_VAR_WHOLE',tokens,modes=(4,5)),i,5);arrays={};lengths=[]
        for m in (4,5):
            decoded=call('var_source_rx',lambda:codec.decode('EC_VAR_WHOLE',encoded[m]['bits'],m),i,m)
            require(np.array_equal(decoded['received_tokens'],tokens[:OFFSETS[m]]),'Independent VAR short-prefix roundtrip failed')
            arrays[f'm{m}_bits']=encoded[m]['bits']
            arrays[f'm{m}_received_tokens']=decoded['received_tokens'].copy()
            lengths.append({k:encoded[m][k] for k in ('raw_bits','arithmetic_bits','flush_bits')}|dict(m=m,zero_extension_reads=30))
        require(encoded[4]['arithmetic_bits']<=927,'Registered m4 safeguard unexpectedly overflows; stop before calibration')
        p=out/'sources'/f'{i:04d}.npz'
        with p.open('xb') as f:np.savez(f,**arrays)
        outputs[str(p)]=sha(p);new+=1
        records.append(dict(source_index=i,source_id=item['source_id'],tokens_sha256=token_sha,archive=str(p),sha256=sha(p),
            lengths=lengths,new_independent_roundtrips=2,m4_profile_required=True,
            origin='NEW_ACTUAL_INDEPENDENT_SHORT_PREFIX_ROUNDTRIPS',use_rule='only_on_longer_prefix_length_overflow'))
        if (i+1)%25==0:print(f'VAR short-prefix preparation {i+1}/{manifest["source_count"]}; reused {reused}',flush=True)
    boundary();native.frozen()
    mp=out/'fallback_manifest.json';write(mp,dict(status='T1_CALIBRATION_VAR_FALLBACK_COMPLETE',source_count=manifest['source_count'],
        original_source_order=manifest['source_ids'],minimum_capacity_bits=927,minimum_fallback_m=4,records=records,
        reused_sources=reused,new_sources=new,m4_profiles_required=True))
    outputs[str(mp)]=sha(mp)
    result=dict(status='T1_CALIBRATION_VAR_SHORT_PREFIX_PREPARATION_COMPLETE',source_count=manifest['source_count'],
        reused_sources=reused,new_sources=new,new_independent_roundtrips=2*new,new_image_renders=0,new_packet_decodes=0,
        training_updates=0,holdout_used=False,input_bindings=inputs,outputs=outputs)
    write(out/'completion.json',result);return result
