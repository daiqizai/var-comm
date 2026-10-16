"""Select fixed predeclared native arrays from verified historical closure."""
from pathlib import Path
import json,hashlib,numpy as np
H=Path(__file__).resolve().parent;W=H.parents[1]
request=W/'.research/main_raw64_20261007/take_over_v1/raw64_fixed16_v2/materials_v2/request.json'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
r=read(request);base=H/'restored_raw';oldroot='/home/liulu/projects/'
def resolve(p):return base/Path(p).relative_to(oldroot)
def checked(p):
 q=resolve(p);assert sha(q)==r['bindings'][p],p;return q
x={k:read(checked(p))for k,p in r['raw_controls'].items()if k!='owner_log'};checked(r['raw_controls']['owner_log'])
outer,spec,cfg,reg,normal,science,end=(x[k]for k in ['outer','spec','config','registration','normal','science','owner_exit'])
assert outer['process_waited']is True and outer['exit_code']==0 and outer['original_closed_batch_called']is True
assert end['process_waited']is True and end['exit_code']==0 and end['capture_or_monitor_error']is None
assert normal['status']=='MAIN_RAW64_DEVELOPMENT_VISUAL_NORMALLY_COMPLETE_V1' and science['status']=='MAIN_RAW64_DEVELOPMENT_SAME_RX_IMAGES_COMPLETE_V1'
assert normal['budget_before']==normal['budget_after']==science['budget_before']==science['budget_after']==reg['budget_before']==outer['budget_before']==outer['budget_after']
assert reg['config_sha256']==normal['config_sha256']==science['config_sha256']==r['bindings'][spec['config']]
assert normal['registration_sha256']==science['registration_sha256']==r['bindings'][cfg['registration']]
assert end['identity']==normal['owner_identity']==outer['owner_identity']
authenticated={}
for d in [reg['source_bindings'],reg['input_bindings'],science['outputs'],normal['outputs'],spec['bindings']]:
 for k,v in d.items():assert k not in authenticated or authenticated[k]==v;authenticated[k]=v
assert all(outer['scientific_evidence_bindings'].get(k)==v for k,v in authenticated.items())
assert r['source_indices']==[0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]
point=[p for p in r['schedule']if p['snr_db']==13 and 'PARTIAL'in p['families']];assert len(point)==1;point=point[0]
out=H/'raw_native_fixed16';out.mkdir(exist_ok=False);result=[]
for i,sid,ref in zip(r['source_indices'],r['source_ids'],r['references']):
 paths=r['raw_sources'][str(i)];cp=read(checked(paths['checkpoint']));rows=read(checked(paths['rows']));arc=checked(paths['archive'])
 assert all(authenticated.get(p)==r['bindings'][p]for p in paths.values())
 assert cp['source_index']==i and cp['source_id']==sid and cp['reference_evidence']==ref
 assert cp['registration_sha256']==science['registration_sha256'] and cp['config_sha256']==science['config_sha256']
 assert cp['outputs']=={paths['archive']:r['bindings'][paths['archive']],paths['rows']:r['bindings'][paths['rows']]}
 assert len(rows)==33 and {(p['development_slot'],p['noise_seed'])for p in rows}=={(p['development_slot'],n)for p in r['schedule']for n in [6201,6202,6203]}
 chosen=[p for p in rows if p['development_slot']==point['development_slot']and p['noise_seed']==6201];assert len(chosen)==1;row=chosen[0]
 assert row['source_id']==sid and row['snr_db']==13 and row['N']==1024 and row['candidate_id']==point['candidate_id']
 im=row['same_RX_images']['VAR_completion'];assert im['image_archive']==paths['archive']and im['image_sha256']==row['same_RX_visual_proof']['image_sha256']['VAR_completion']
 with np.load(arc,allow_pickle=False)as z:a=z['images'][im['image_slot']].copy()
 assert a.dtype==np.float32 and a.shape==(3,256,256) and np.isfinite(a).all()and 0<=a.min()<=a.max()<=1
 assert ash(a)==im['image_sha256'],('array SHA',i,ash(a),im['image_sha256'])
 q=out/f'RAW_PARTIAL_{i:04d}.npz'
 with q.open('xb')as f:np.savez(f,reconstruction=a)
 result.append(dict(method='RAW_PARTIAL',source_index=i,source_id=sid,preprocessing_id=ref['preprocessing_id'],reference_original=ref,
  N=1024,snr_db=13,noise_seed=6201,candidate_id=row['candidate_id'],m=row['m'],K=row['K'],gray=row['gray'],actual_state_sha256=row['actual_state_sha256'],
  original_archive=paths['archive'],original_archive_sha256=r['bindings'][paths['archive']],original_slot=im['image_slot'],
  original_checkpoint_sha256=r['bindings'][paths['checkpoint']],original_rows_sha256=r['bindings'][paths['rows']],
  reconstruction_sha256=ash(a),extracted_archive=dict(path=q.name,bytes=q.stat().st_size,sha256=sha(q))))
v=dict(schema='EXACT_HISTORICAL_RAW_NATIVE_FIXED16_EXTRACTION_V1',request_sha256=sha(request),historical_controls={k:dict(path=p,sha256=r['bindings'][p])for k,p in r['raw_controls'].items()},rows=result,new_reconstruction_calls=0,new_packet_calls=0,new_model_calls=0)
with(out/'manifest.json').open('x',encoding='utf8')as f:json.dump(v,f,indent=2)
print('RAW_NATIVE_EXTRACTED',len(result))
