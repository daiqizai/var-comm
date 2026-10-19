"""Prove the three known entropy-cache metadata schemas without reading pixels."""
from metadata import *

STAGES=(
 ('T1B_NEW_RX','t1b_m10_calibration_20261016_v1','receiver_recovery_v1','registration.json','control_recovery_v1/receiver_recovery_v1',332),
 ('T1C_NEW_RX','t1c_m10_test_20261016_v1','receiver','link_registration.json','receiver_control',75))
OLD_SCHEMA={'image_archive','image_sha256','original_visual_identity','physical_frame','reconstruction_result'}
NEW_SCHEMAS={
 'T1B_NEW_RX':{'evidence','image_archive','image_sha256','origin'},
 'T1C_NEW_RX':{'evidence','image_archive','image_key','image_sha256','image_slot','origin'}}
CRITICAL=('tb_native.py','ep_source_codec.py','h800_ep48_link_core_v1.py','leo_whole_math_v1.py','h_source_driver.py','t6_source_codec.py','partial_receiver.py')
TB_NATIVE_SHA='6de8addc4cda68248474b83e93d1eecc2e198e65d861eae10fb17042ecc6b37a'

def packed(bits):
    return bytes(sum((bits[i+j]if i+j<len(bits)else 0)<<(7-j)for j in range(8))for i in range(0,len(bits),8))
def cache_key(s,c):
    assert s['population']==c['population'] and s['source_id']==c['source_id']
    assert s['codec']==c['codec']==0 and s['endpoint']==c['endpoint']
    return digest(dict(population=s['population'],source_id=s['source_id'],codec=0,endpoint=s['endpoint'],
        length=s['bit_length'],object_sha256=s['object_sha256'],image=c['image']))
def actual_identity(v,routes):
    q=routes['native_qualification']
    assert v['models']==q['model_state_identity'] and v['runtime']==q['numeric_settings']['runtime']
    assert v['numeric_flags']==routes['numeric_flags']
    assert v['sampling']=='unconditional_class1000_greedy_no_CFG' and v['training_updates']==0
    assert v['source_mathematics']=='Unchanged SHA-bound original function/class definitions'
def validate_pair(s,im,label,key,ev,rx):
    assert label in NEW_SCHEMAS and set(im)==NEW_SCHEMAS[label]
    assert im['origin']=='ACTUAL_NEW' and ev['source_input_key']==key
    assert ev['image_archive']==im['image_archive'] and ev['image_sha256']==im['image_sha256']
    if label=='T1C_NEW_RX':
        assert im['image_key']=='image' and im['image_slot']is None
    e=ev['evidence']
    assert e['receiver_truth_used']is False and e['TX_CDF_used']is False
    assert e['source_decode_called']is True and e['render_called']is True and e['canonical']is True
    assert e['source_status']=='ARITHMETIC_SOURCE_DECODED' and (e['m'],e['K'])==(s['m'],s['K'])
    p=rx['rx_profile'];payload=rx['body']['payload']
    assert rx['header']['header_ok']and rx['body']['crc_accepted']and rx['body']['parser_accepted']
    assert digest(dict(kind='ACTUAL_SOURCE_INPUT',family=p['family'],m=p['m'],K=p['K'],payload=payload))==key
    assert (p['m'],p['K'])==(s['m'],s['K']) and len(payload)==s['bit_length']
    assert all(type(x)is int and x in (0,1)for x in payload)
    assert hashlib.sha256(bytes(payload)).hexdigest()==s['bits_sha256']
    assert hashlib.sha256(packed(payload)).hexdigest()==s['object_sha256']

def build():
    # Only JSON and source text are read by this function. All arrays are later
    # verified by the unchanged native.verified_candidate body.
    evidence={}
    def r(p):
        d=pin(p);evidence[str(p)]=d;return read(p)
    def l(d):verify(d);return r(d['path'])
    b0=r(O/'b0_completion.json');assert b0['scientific_calls']==0
    cache=l(next(d for d in b0['runtime_files']if d['path'].endswith('/cache_catalogue.json')))
    sources=l(next(d for d in b0['runtime_files']if d['path'].endswith('/source_catalogue.json')))
    sourceindex={(s['population'],s['source_id'],s['codec'],s['endpoint']):s for s in sources}
    b1=r(O/'b1_registration.json');routes=l(b1['model_routes'])
    oldindex=r(B/'t1b_m10_calibration_20261016_v1/preparation/old_image_index.json')
    oldindexpin=pin(B/'t1b_m10_calibration_20261016_v1/preparation/old_image_index.json')
    oldmembers={canonical(v)for v in oldindex.values()};members={};counts={};proofs=[]
    inline=0;allrows=[c for c in cache if c['origin']=='ORIGINAL_ENTROPY_RX']
    for c in allrows:
        im=c['image']
        if 'original_visual_identity'not in im:continue
        assert set(im)==OLD_SCHEMA and canonical(im)in oldmembers
        for k in NEST_FIELDS:assert im['original_visual_identity'][k]==routes['native_qualification'][k]
        s=sourceindex[c['population'],c['source_id'],c['codec'],c['endpoint']]
        key=cache_key(s,c);assert key not in members
        members[key]=dict(origin='INLINE_ORIGINAL_ENTROPY_RX',original_index=oldindexpin)
        inline+=1
    assert inline==6540
    for label,stage,rxname,regname,control,count in STAGES:
        t=B/stage;o=R/'results'/stage;reg=r(o/regname);closure=r(o/'delivery/closure_audit.json')
        assert reg['model_routes']==b1['model_routes'] and reg['static']==b1['static']
        actual_identity(r(t/rxname/'actual_native_identity.json'),routes)
        copied=r(o/'execution_evidence/copy_manifest.json')['records']
        bindings=[]
        for p in (t/rxname/'completion.json',t/control/'actual_wait.json',t/control/'launch.json'):
            z=next(v for v in copied if v['source']['path']==str(p))
            verify(z['source']);verify(z['copy']);assert z['source']['sha256']==z['copy']['sha256'];bindings.append(z)
        wait=r(t/control/'actual_wait.json');check_wait(wait,hostless=True)
        assert wait['actual_wait']is True
        launch=r(t/control/'launch.json')
        assert all(wait['process'][k]==launch['process'][k]for k in ('pid','start_ticks'))
        assert gone(launch['parent_process'],True)
        done=r(t/rxname/'completion.json')
        assert done['registration_sha256']==pin(o/regname)['sha256']
        assert done['ledger']['unresolved']==0 and done['truth_reads']==done['TX_CDF_reads']==0
        assert not(t/rxname/'failure.json').exists()
        images=l(done['images']);image_members={canonical(v):k for k,v in images.items()}
        phy=r(t/'phy/completion.json');projection=l(phy['rx_projection'])
        received={v['source_input_key']:v['actual_RX']for v in projection}
        code={Path(d['path']).name:d for d in reg['code']}
        for name in CRITICAL:verify(code[name]);evidence[code[name]['path']]=code[name]
        assert code['tb_native.py']['sha256']==TB_NATIVE_SHA
        for name,field in (('ep_source_codec.py','source_codec_SHA'),('h800_ep48_link_core_v1.py','received_render_core_SHA'),('leo_whole_math_v1.py','source_math_SHA')):
            assert code[name]['sha256']==routes['native_qualification'][field]
        if label=='T1B_NEW_RX':
            assert closure['all_stage_and_owner_processes_gone']is True and closure['received_truth_supplied']is False
            recovery=l(closure['recovery_registration'])
            assert recovery['original_registration_sha256']==done['registration_sha256']
            d=next(d for d in recovery['code']if Path(d['path']).name=='tb_receiver_recovery_v1.py');verify(d);evidence[d['path']]=d
        else:
            assert closure['scientific_stages']['receiver']==done
            assert next(d for d in closure['controls']if d['path']==str(t/control/'actual_wait.json'))==pin(t/control/'actual_wait.json')
            z=next(v for v in copied if v['source']['path']==str(t/rxname/'actual_native_identity.json'))
            verify(z['source']);verify(z['copy']);assert z['source']['sha256']==z['copy']['sha256'];bindings.append(z)
            verify(code['tc_receiver.py']);evidence[code['tc_receiver.py']['path']]=code['tc_receiver.py']
        found=0
        stageproof=dict(label=label,registration=pin(o/regname),model_routes=b1['model_routes'],static=b1['static'],
            actual_native_identity=pin(t/rxname/'actual_native_identity.json'),completion=pin(t/rxname/'completion.json'),
            images=done['images'],projection=phy['rx_projection'],actual_wait=pin(t/control/'actual_wait.json'),
            launch=pin(t/control/'launch.json'),published_bindings=bindings,
            runtime_previously_published=(label=='T1C_NEW_RX'),critical_sources=[code[n]for n in CRITICAL])
        proofs.append(stageproof)
        for c in allrows:
            im=c['image']
            if 'original_visual_identity'in im or canonical(im)not in image_members:continue
            s=sourceindex[c['population'],c['source_id'],c['codec'],c['endpoint']]
            oldkey=image_members[canonical(im)];ev=l(im['evidence'])
            validate_pair(s,im,label,oldkey,ev,received[oldkey])
            key=cache_key(s,c);assert key not in members
            members[key]=dict(origin=label,received_input_key=oldkey,stage_label=label,
                per_image_evidence=im['evidence'],image_archive=im['image_archive'],image_sha256=im['image_sha256'])
            found+=1
        assert found==count,(label,found);counts[label]=found
    assert len(members)==len(allrows)==6947
    report=dict(status='THREE_KNOWN_ENTROPY_SCHEMAS_VERIFIED',inline_rows=inline,new_schema_counts=counts,
        model_routes=b1['model_routes'],static=b1['static'],stage_proofs=proofs,members=members,
        evidence=list(evidence.values()),pixel_arrays_opened=0,scientific_calls=0,
        original_records_mutated=False,new_inline_identity_fabricated=False)
    return report

def validate_candidate(c,routes,report,report_descriptor):
    s,meta=c['source'],c['cache'];assert meta['origin']=='ORIGINAL_ENTROPY_RX'
    key=cache_key(s,meta);assert key in report['members'],('UNBOUND_ENTROPY_CACHE',key)
    bound=report['members'][key];im=meta['image']
    if bound['origin']=='INLINE_ORIGINAL_ENTROPY_RX':
        assert set(im)==OLD_SCHEMA
        for k in NEST_FIELDS:assert im['original_visual_identity'][k]==routes['native_qualification'][k]
    else:
        assert bound['origin']in NEW_SCHEMAS and set(im)==NEW_SCHEMAS[bound['origin']]
        assert bound['per_image_evidence']==im['evidence']
        assert bound['image_archive']==im['image_archive'] and bound['image_sha256']==im['image_sha256']
    # Add explicit proof beside the original cache; never fill a missing
    # original_visual_identity with values copied from model_routes.
    c['original_entropy_metadata_revision']=dict(report=report_descriptor,member_key=key,origin=bound['origin'])
