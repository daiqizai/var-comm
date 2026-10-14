"""Authenticate Pilot50 and admit only exact frozen H800 clean-cache matches."""
from common import *
import numpy as np
def run():
 assert not (O/'assets').exists();(O/'assets').mkdir(parents=True);(T/'assets').mkdir(parents=True)
 scope=read(E/'user_package/scope.json');assert scope['status']=='PLAN_ONLY_NOT_IMPLEMENTED_OR_EXECUTED'
 calpin=dict(path=str(OLD/'r0_reuse/cal200_sources.json'),sha256='afc8a99bb964d20a73e39f552464ff008627022bad39f3cabbcda2cac8b2e00f')
 cal=read(checked(calpin))['records'];assert len(cal)==200 and len({r['source_id'] for r in cal})==200
 sources=sorted(cal,key=lambda r:(digest(['NESTCOM_LAYERED_EC_PILOT50_V1',r['source_id']]),r['source_id']))[:50]
 assert len(scope['source']['endpoint_grid'])==24 and [(x['m'],x['K']) for x in scope['source']['endpoint_grid']]==list(ENDS)
 pixel={};token={}
 for i,r in enumerate(sources):
  with np.load(checked(r['archive']),allow_pickle=False) as z:a=z['pixels'].copy();b=z['tokens'].copy()
  assert a.dtype==np.uint8 and a.shape==(3,256,256) and b.dtype==np.int64 and b.shape==(680,) and np.all((b>=0)&(b<4096))
  assert hashlib.sha256(a.tobytes()).hexdigest()==r['preprocessing_id'] and hashlib.sha256(b'int64:680\0'+b.astype('<i8').tobytes()).hexdigest()==r['tokens_sha256']
  checked(r['checkpoint']);pixel[r['source_id']]=a;token[r['source_id']]=b
  r.update(pilot_slot=i,pilot_selection_sha256=digest(['NESTCOM_LAYERED_EC_PILOT50_V1',r['source_id']]))
 save(O/'assets/pilot50.json',dict(selection_uses_quality=False,cal200=calpin,records=sources,domain='NESTCOM_LAYERED_EC_PILOT50_V1'))
 oldreg=read(OLD/'r0_supplement_preparation/clean_registration.json');assert oldreg['models']==MODELS
 req=read(checked(oldreg['visual_request']));files=req['spec']['files']
 model_files=[r for r in files if r['path'].endswith(('.pt','.pth'))];assert len(model_files)==3
 for r in model_files:checked(r)
 author=[r for r in files if '/models/' in r['path'] and r['path'].endswith('.py')];assert len(author)==7
 for r in author:checked(r)
 authorroot=Path(author[0]['path']).parent.parent
 assert sha(authorroot/'dist.py')=='b55322b4ae4c8c2b600b98870dc98b26435a813b72b3468a4315d7d74771e370'
 candidate=req['spec']['native_dependency_candidate'];audit=read(checked(candidate['audit']));checked(candidate['commit_record'])
 # The candidate dependency was explicitly admitted in the earlier H800
 # qualification. Its immutable audit remains a dependency of this new route.
 save(T/'assets/original_author_dependency_audit.json',audit)
 save(O/'assets/model_routes.json',dict(models=MODELS,files=model_files,author_files=author,author_root=str(authorroot),dist=pin(authorroot/'dist.py'),native_qualification=oldreg['native_identity'],numeric_flags=oldreg['numeric_flags'],metric_config=oldreg['metric_config']))
 # Existing outputs are fixed before this truth-side cache audit. No truth is
 # supplied to any new source receiver, probability provider, or reconstruction.
 candidates={};done=read(L/'r0_clean_supplement_v1/render/completion.json');assert done['status']=='COMPLETE' and done['models']==MODELS
 assert read(L/'r0_clean_supplement_v1/render/actual_wait.json')['returncode']==0
 frames=read(checked(done['frames']))
 scores=list(csv.DictReader((OLD/'r0_clean_supplement_v1/clean_quality.csv').open()));scoreby={int(x['cell_index']):x for x in scores}
 for f in frames:
  sid=f['source_id'];end=(f['m'],f['K'])
  if sid not in token or end not in ENDS:continue
  assert f['source_preprocessing_id']==next(x['preprocessing_id'] for x in sources if x['source_id']==sid)
  prefix=token[sid][:OFFSETS[end[0]]+end[1]];assert asha(prefix)==f['prefix_array_sha256'] and f['model_identity']==digest(MODELS)
  p=checked(f['image_archive'])
  with np.load(p,allow_pickle=False) as z:im=z['image'].copy()
  assert im.dtype==np.float32 and im.shape==(3,256,256) and asha(im)==f['image_array_sha256'] and np.isfinite(im).all()
  q=scoreby[f['cell_index']];assert q['image_array_sha256']==f['image_array_sha256']
  candidates[(sid,*end)]=dict(source_id=sid,m=end[0],K=end[1],image_archive=f['image_archive'],image_key='image',image_array_sha256=f['image_array_sha256'],dino=float(q['dinov2_vitl14_cosine']),metric_identity=q['metric_identity'],origin='REUSED_EXACT_H800_R0_FIXED_OUTPUT')
 # R1's actually received correct clean prefixes are eligible only after exact
 # token hash and frozen output identities. Read metadata, not the whole pixel grid.
 vdone=read(L/'r1_cache_recovery_v1/visual/completion.json');assert vdone['status']=='COMPLETE'
 master=read(OLD/'r1_execution_v1/registered/registration.json');assert master['models']==MODELS and vdone['numeric_flags']==oldreg['numeric_flags']
 checked(vdone['model_bundle']);assert read(L/'r1_cache_recovery_v1/visual_actual_wait.json')['returncode']==0
 inrows=read(checked(vdone['model_inputs']));inputs={f['view_id']:f for f in inrows};del inrows
 sdone=read(L/'r1_cache_recovery_v1/score/completion.json');assert sdone['status']=='COMPLETE' and read(L/'r1_cache_recovery_v1/score_actual_wait.json')['returncode']==0
 scored=read(checked(sdone['scored_rows']));r1_added=0
 for q in scored:
  sid=q['source_id'];f=inputs[q['view_id']];state=f['state'];end=(state.get('m'),state.get('K'))
  if sid not in token or end not in ENDS or state['kind']!='tokens':continue
  assert q['source_preprocessing_id']==next(x['preprocessing_id'] for x in sources if x['source_id']==sid)
  actual=np.asarray([v for scale in state['prefix'] for v in scale]+state['partial_values'],dtype=np.int64)
  if not np.array_equal(actual,token[sid][:OFFSETS[end[0]]+end[1]]):continue
  assert f['model_identity']==vdone['model_identity'] and q['model_identity']==vdone['model_identity'] and q['execution_status']=='COMPLETE'
  key=(sid,*end)
  if key in candidates:
   assert candidates[key]['image_array_sha256']==q['reconstruction_sha256'],'Inconsistent frozen clean output assets'
   continue
  p=checked(q['image_archive'])
  with np.load(p,allow_pickle=False) as z:im=z['image'].copy()
  assert im.dtype==np.float32 and im.shape==(3,256,256) and asha(im)==q['reconstruction_sha256'] and np.isfinite(im).all()
  candidates[key]=dict(source_id=sid,m=end[0],K=end[1],image_archive=q['image_archive'],image_key='image',image_array_sha256=q['reconstruction_sha256'],dino=q['dinov2_vitl14_cosine'],metric_identity=q['metric_identity'],origin='REUSED_EXACT_H800_R1_FIXED_CORRECT_PREFIX',original_view_id=q['view_id'])
  r1_added+=1
 save(T/'assets/admitted_clean_cache.json',list(candidates.values()))
 missing=[dict(source_id=r['source_id'],pilot_slot=r['pilot_slot'],m=m,K=K) for r in sources for m,K in ENDS if (r['source_id'],m,K) not in candidates]
 save(O/'assets/gap.json',dict(status='PASS_SOURCE50_MODEL_FILES_EXACT_FROZEN_CLEAN_CACHE_AUDIT',pairs=1200,reused=len(candidates),R1_added=r1_added,missing_count=len(missing),model_calls=0,channel_calls=0,tokenizer_calls=0,missing=missing))
 print(json.dumps(dict(status='PASS_SOURCE50_MODEL_FILES_AND_EXACT_CACHE_AUDIT',reused=len(candidates),missing=len(missing))),flush=True)
if __name__=='__main__':run()
