"""Read-only bounded input/asset audit. No scientific modules are imported."""
from pathlib import Path
import base64, csv, hashlib, io, json, subprocess, time
import numpy as np

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
O=R/'results/arpc_external_baseline_20261019_v1'
HEAD='fbc17ae300e46bfdab56c22f989752545a5c9b3c'
pins=[]; exports={}
def raw(p):
    p=Path(p); a=p.stat(); data=p.read_bytes(); z=p.stat()
    assert (a.st_size,a.st_mtime_ns,a.st_ino)==(z.st_size,z.st_mtime_ns,z.st_ino)
    pin=dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    pins.append(pin); return data,pin
def read(p): return json.loads(raw(p)[0])
def checked(d):
    data,pin=raw(d['path']); assert pin['sha256']==d['sha256']
    if 'bytes' in d: assert pin['bytes']==d['bytes']
    return data
def export(name,p):
    data,pin=raw(p); exports[name]=dict(**pin,data_b64=base64.b64encode(data).decode()); return data
def rows(data): return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()==HEAD
assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain','--untracked-files=no'],text=True).strip()
sources=json.loads(export('arpc_sources.json',O/'sources.json'))
cal= sources['cal100']; ev=sources['external_common100_v1']
assert len(cal)==len(ev)==100
export('arpc_cal100_lengths.csv',O/'a2_clean_v1/source_lengths.csv')
export('arpc_cal100_quality.csv',O/'a2_metrics_v1/clean_quality.csv')
export('arpc_metric_identity.json',O/'a2_metrics_v1/metric_identity.json')
cal_lengths=rows(export('nest_cal1000_m10_lengths.csv',R/'results/submission_closure_20261016_v1/m10/m10_lengths.csv'))
export('arpc_cal_clean_completion.json',O/'a2_clean_v1/completion.json')
export('arpc_cal_metrics_completion.json',O/'a2_metrics_v1/completion.json')
export('nest_m10_completion.json',R/'results/submission_closure_20261016_v1/m10/completion.json')
export('nest_test_source_completion.json',B/'t1c_m10_test_20261016_v1/source/completion.json')
calreg=read(R/'results/submission_closure_20261016_v1/source_registration.json')
testreg=read(B/'t1c_m10_test_20261016_v1/source/sources.json')
clean=read(O/'a2_clean_v1/clean_states.json')['states']
evalcat=read(O/'a3_source_v1/stream_catalogue.json')['sources']
calid={x['source_id']:x for x in cal}; evid={x['source_id']:x for x in ev}
calnest={x['source_id']:x for x in calreg['sources'] if x['source_id'] in calid}
evalnest={x['source_id']:x for x in testreg if x['source_id'] in evid and x['population']=='common500'}
assert len(calnest)==len(evalnest)==100
assets=[]; fullbits={}; cohorts={'cal100':calid,'external_common100':evid}
for cohort,ns in [('cal100',calnest),('external_common100',evalnest)]:
    for sid,s in ns.items():
        if cohort=='cal100':
            i=s['source_index']; fixed=read(B/f'submission_closure_20261016_v1/m10_source/streams/{i:04d}_fixed_output.json')
            desc=fixed['archive']
        else: desc=s['current_streams']
        data=checked(desc)
        with np.load(io.BytesIO(data),allow_pickle=False) as z: bits=z['m10_K0'].copy()
        assert bits.ndim==1 and np.isin(bits,(0,1)).all()
        key=hashlib.sha256(bits.astype('uint8').tobytes()).hexdigest();fullbits[(cohort,key)]=sid
        assets.append(dict(cohort=cohort,method='NeST-Com',source_id=sid,endpoint=10,payload_bits=len(bits),wire=desc,raw_bit_bytes_sha256=key,source_archive=s['archive'],full_image=None))
for s in clean:
    if s['endpoint']!=7: continue
    checked(s['wire']);checked(s['image'])
    assets.append(dict(cohort='cal100',method='ARPC',source_id=s['source_id'],endpoint=7,payload_bits=8*s['wire']['bytes'],wire=s['wire'],source_archive=s['source'],full_image=s['image']))
for s in evalcat:
    st=next(x for x in s['streams'] if x['endpoint']==7);checked(st['wire'])
    assets.append(dict(cohort='external_common100',method='ARPC',source_id=s['source_id'],endpoint=7,payload_bits=st['payload_bits'],wire=st['wire'],source_archive=s['source'],full_image=None))
assert len(assets)==400
for cohort in cohorts:
    for sid in cohorts[cohort]:
        pair=[x for x in assets if x['cohort']==cohort and x['source_id']==sid]
        assert len(pair)==2 and pair[0]['source_archive']['sha256']==pair[1]['source_archive']['sha256'],(cohort,sid)
image_scopes=[]
for cohort,stage,rxdir in [('cal100','t1b_m10_calibration_20261016_v1','receiver_recovery_v1'),('external_common100','t1c_m10_test_20261016_v1','receiver')]:
    proj=read(B/stage/'phy/rx_projection.json'); images=read(B/stage/rxdir/'images.json'); found=[]
    for p in proj:
        rx=p['actual_RX']; profile=rx.get('rx_profile') or {}
        if profile.get('m')!=10 or rx.get('status')!='PAYLOAD_PARSED':continue
        payload=rx['body']['payload'];key=hashlib.sha256(bytes(payload)).hexdigest();sid=fullbits.get((cohort,key))
        if sid is None: continue
        im=images[p['source_input_key']]['image_archive'];checked(im)
        row=next(a for a in assets if a['cohort']==cohort and a['source_id']==sid and a['method']=='NeST-Com')
        row['full_image']=im;row['old_exact_RX_input_key']=p['source_input_key'];found.append(sid)
    image_scopes.append(dict(cohort=cohort,method='NeST-Com',scope=str(B/stage/rxdir),exact_full_source_images=len(set(found))))
for cohort,stage in [('cal100','a2b_rx_v1'),('external_common100','a3_rx_v1')]:
    states=read(O/stage/'actual_states.json'); full=[x for x in states if x['endpoint']==7]
    assert not full
    image_scopes.append(dict(cohort=cohort,method='ARPC',scope=str(O/stage),full_e7_images=0))
for rel in ['experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py','experiments/submission-closure-20261016/m10_codec.py','experiments/arpc-external-baseline-20261019/wire.py']:
    raw(R/rel)
print(json.dumps(dict(schema='ARPC_N2048_READ_ONLY_INPUT_AUDIT_V1',basis_HEAD=HEAD,observed_unix=time.time(),exports=exports,assets=assets,image_scopes=image_scopes,input_pins=pins,new_model_calls=0,new_channel_frames=0,new_scores=0,new_bootstrap=0,asset_scope_note='Full streams checked for both cohorts; ARPC cal e7 images and matching existing T1B/T1C m10 RX images checked. Other legacy reconstruction families not exhaustively searched; any reuse still requires the future exact runtime identity.'),ensure_ascii=False))
