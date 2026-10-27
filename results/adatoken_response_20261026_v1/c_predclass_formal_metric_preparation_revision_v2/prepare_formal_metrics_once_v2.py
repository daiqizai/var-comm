"""Prepare two separate metric registrations from fully closed formal RX.

This does not launch a model or compute scores. Run only after common500 and
Kodak RX have closed and passed their post-output audits. The two registrations
allow exact three-metric reuse without falsely reusing legacy mixed-runtime CLIP.
"""
from remote import *
import sys
population=sys.argv[1]
assert population in ('common500','kodak24')
entry=H/'formal_metric_entry_v1'
names=('contracts.py','worker.py','test_contracts.py','README.md','source_delta.json','coordinator_review.json')
files={n:base64.b64encode((entry/n).read_bytes()).decode()for n in names}
code=r'''
from pathlib import Path
import sys,json,base64,copy,csv,sqlite3,fcntl,numpy as np
R=Path(__ROOT__);E=R/'experiments/adatoken-response-20261026';sys.path.insert(0,str(E));from common import *
assert os.uname().nodename=='aibox-r59688baa77c-57c794f5c6-44lm7'
population=__POP__;base='c_predclass_'+population;count=500 if population=='common500'else 24;total=count*9
roles={k:base+'_'+k+'_metrics'for k in('three','clip')}
prep=L/(base+'_metric_preparation');registration_paths={k:O/(v+'_registration.json')for k,v in roles.items()}
lock=(L/'c_predclass_formal_metric_preparation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not prep.exists()and all(not p.exists()for p in registration_paths.values())
for role in roles.values():
 assert not any(p.exists()for p in(L/role,L/(role+'_control'),L/(role+'_initial_control')))
q=L/'c_predclass_formal_metric_cpu_qualification_v1';qw=read(q/'actual_wait.json')
assert sha(q/'actual_wait.json')=='87eadbb37911d4fb0deae88959fc422bc832994b95230727ba65bb047c27ee6f'
assert type(qw['returncode'])is int and qw['returncode']==0 and gone(qw['process'])and gone(qw['parent_process'])
assert not any(qw.get(k,False)for k in('timed_out','terminated','killed'))
assert read(q/'guard_result.json')==dict(tests=16,failures=0,errors=0,blocked_paths=[],blocked_model_imports=[],model_modules=[],status='PASS')
for d in read(q/'source_manifest.json'):
 if 'original'in d:checked(d['original']);checked(d['snapshot'])
 else:checked(d)
raws={n:base64.b64decode(v)for n,v in __FILES__.items()}
for n in('contracts.py','worker.py','test_contracts.py','README.md','source_delta.json'):
 assert(q/'source_snapshot'/n).read_bytes()==raws[n]
review=json.loads(raws['coordinator_review.json']);assert review['qualification']==pin(q/'actual_wait.json')
assert review['status']=='PASS_ENGINEERING_QUALIFICATION_NOT_SCIENTIFIC_EXECUTION'
sys.path.insert(0,str(q/'source_snapshot'));from contracts import validate_plan,validate_closure,validate_metric_identity,METRICS
closures={};admissions={};science=[];all_formal=[]
def closed(role,kind=None,admission=None):
 if role in closures:return closures[role]
 wp=L/(role+'_control/actual_wait.json');cp=L/role/'completion.json';w=read(wp);d=read(cp)
 for suffix in('','_control','_initial_control'):assert not(L/(role+suffix)/'failure.json').exists()
 audit=None;ap=None
 if kind=='RX':
  assert admission is not None
  ap=L/('c_predclass_'+admission['population']+'_rx_audits')/role/'audit.json'
  assert not(ap.parent/'failure.json').exists();audit=read(ap)
  assert audit['actual_wait']==pin(wp)and audit['source_output_manifest']==d['outputs']
  assert audit['logical_manifest']==admission['logical_manifest'];checked(audit['all_checks'])
 # Historical DINO contributes only closed GPU spend, never formal scores.
 # Its distinct completion schema is accepted only for these exact receipts.
 if role=='c_predclass_cal1000_dino':
  assert kind is None and admission is None and audit is None
  assert pin(wp)==dict(path=str(L/'c_predclass_cal1000_dino_control/actual_wait.json'),bytes=1219,sha256='76ccc8c1edcd84d5933d8a36b9c5990119542e8997ab5164bea8a6fc31ee8d3e')
  assert pin(cp)==dict(path=str(L/'c_predclass_cal1000_dino/completion.json'),bytes=2189,sha256='25136a1fa5e820d04111217a015cfd4f81f53f46e91daddc7de7d3b2b5a9345b')
  assert w['actual_wait']is True and type(w['returncode'])is int and w['returncode']==0
  assert gone(w['process'])and gone(w['parent_process'])
  assert all(w[k]is False for k in('timed_out','terminated','killed'))and w['metadata_error']is None
  assert d['status']=='CAL1000_DINO_CLOSED_POLICY_NOT_SELECTED'and d['ledger']['unresolved']==0
  assert d['registration_sha256']==w['registration']['sha256']=='7ae9bb70cce94d4bad5da58330c3bf77617fa07f8d087062758d1aa768f4af95'
  checked(w['registration'])
 else:
  validate_closure(w,d,gone,audit)
 item=dict(role=role,actual_wait=pin(wp),completion=pin(cp),elapsed_seconds=w['elapsed_seconds'])
 if kind:item['kind']=kind
 if audit is not None:item['audit']=pin(ap)
 closures[role]=item;return item

# Fail before reading pixels or writing preparation if any formal RX is active.
for pop in('common500','kodak24'):
 bb='c_predclass_'+pop;a=read(O/(bb+'_rx_admission.json'));admissions[pop]=a
 assert a['population']==pop and a['frozen_policy']['sha256']=='68d12e9b19006e8509b3ae5a62132e73044b88ad29ca041f96b436144923ae8c'
 for suffix,kind in(('_source','SOURCE'),('_link','PHY')):
  c=closed(bb+suffix,kind);all_formal.append(c)
  if pop==population:science.append(c)
 for stage in a['shards']:
  c=closed(stage['role'],'RX',a);d=cr(c['completion'])
  assert d['registration_sha256']==stage['registration']['sha256']and d['received_outcomes']==stage['inputs']
  assert cr(c['audit'])['registration']==stage['registration'];all_formal.append(c)
  if pop==population:science.append(c)
a=admissions[population];link=cr(a['PHY_registration']);ld=cr(closed(base+'_link')['completion'])
assert ld['status']=='COMPLETE_PHY_AWAITING_SOURCE_RX'and ld['logical_events']==ld['physical_events']==total
assert ld['population']==population and ld['source_count']==count
assert a['scope']==link['scope']and a['revision']==link['revision']and a['frozen_policy']==link['frozen_policy']
for d in(a['scope'],a['revision'],a['frozen_policy']):checked(d)
assert a['revision']['sha256']=='fb80b863dad31fa37cac9cc2e8d8ee4a7d5248071b3a73fdcc74c704517fc254'
sd=cr(closed(base+'_source')['completion'])
assert sorted(cr(sd['sources']),key=lambda v:v['source_index'])==cr(link['sources'])
sources=[]
for s in sorted(cr(link['sources']),key=lambda v:v['source_index']):
 with np.load(checked(s['archive']),allow_pickle=False)as z:x=z['pixels'].copy()
 assert x.dtype==np.uint8 and x.shape==(3,256,256)and hashlib.sha256(x.tobytes()).hexdigest()==s['preprocessing_id']
 sources.append({k:s[k]for k in('source_id','source_index','archive','preprocessing_id')}|dict(reference_sha256=asha(x.astype(np.float32)/np.float32(255))))
assert len(sources)==count;sindex={s['source_id']:s for s in sources}
outputs={};output_origins={}
for stage in a['shards']:
 for v in cr(cr(closed(stage['role'])['completion'])['outputs']):
  assert v['key']not in outputs;outputs[v['key']]=v;output_origins[v['key']]=stage['role']
for reused in cr(a['closed_reuse']):
 v=reused['output'];assert v['key']not in outputs
 rr=cr(reused['original_registration']);donor_role=Path(reused['actual_wait']['path']).parent.name.removesuffix('_control')
 donor_ad=None
 for pp,aa in admissions.items():
  if any(s['role']==donor_role for s in aa['shards']):donor_ad=aa
 if donor_ad is None:
  ca=read(O/'c_predclass_cal1000_rx_admission.json')
  if any(s['role']==donor_role for s in ca['shards']):donor_ad=ca|dict(population='cal1000')
 c=closed(donor_role,'RX',donor_ad)if donor_ad else closed(donor_role)
 assert c['actual_wait']==reused['actual_wait']and rr['revision']==a['revision']
 assert reused['contract_sha256']==a['cache_contract_sha256']
 if donor_ad:science.append(c)
 outputs[v['key']]=v;output_origins[v['key']]=donor_role
for v in outputs.values():
 assert v['source_truth_used']is False and v['TX_CDF_used']is False
 assert v['status']in('ACCEPTED_SOURCE','FIXED_GRAY_SOURCE_REJECT')
 p=checked(v['image']);assert read(p.with_suffix('.json'))==v
 with np.load(p,allow_pickle=False)as z:y=z['image'].copy()
 assert y.dtype==np.float32 and y.shape==(3,256,256)and np.isfinite(y).all()and y.min()>=0 and y.max()<=1 and asha(y)==v['pixel_sha256']
gray_sha=asha(np.full((3,256,256),.5,np.float32));frames=[]
for j in cr(a['logical_manifest']):
 s=sindex[j['source_id']];assert s['source_index']==j['source_index'];f=cr(j['frame'])
 assert f['event']['key']==j['event_key']and f['event']['population']==population
 assert all(f['event'][k]==j[k]for k in('source_id','source_index','snr_db','noise_seed'))
 if j['RX_key']is None:
  assert j['origin']=='FIXED_GRAY'and f['source_projection']['status']=='FIXED_GRAY'
  image=None;pixel=gray_sha;status='FIXED_GRAY_PHY_REJECT';prefix=0
 else:
  v=outputs[j['RX_key']];assert f['source_projection']['status']=='SOURCE_INPUT'
  assert v['prefix']==f['source_projection']['prefix']and v['payload_sha256']==f['source_projection']['payload_sha256']
  image=v['image'];pixel=v['pixel_sha256'];status=v['status'];prefix=v['prefix']
 frames.append({k:j[k]for k in('event_key','source_id','source_index','snr_db','noise_seed','wire_status','RX_key')}|dict(population=population,method='ADA_PREDCLASS_FIXEDFRAME',received_prefix=prefix,source_status=status,accepted=status=='ACCEPTED_SOURCE',image=image,pixel_sha256=pixel,reference_sha256=s['reference_sha256'],reuse_old=False,old_values={},old_metric_identity_verified=False))
assert set(outputs)=={j['RX_key']for j in frames if j['RX_key']is not None}
plan_base=dict(population=population,frozen_policy=a['frozen_policy'],sources=sources,frames=frames)
assert validate_plan(plan_base,population)==total

old=read(O/'a_metrics_registration.json');resources={
 'three':dict(torch='2.11.0+cu128',cuda='12.8',pci='0000:a3:00.0'),
 'clip':dict(torch='2.10.0+cu130',cuda='13.0',pci='0000:62:00.0')}
feature_maps={k:{}for k in roles};feature_provenance={k:{}for k in roles};identities={k:[]for k in roles};donor_closures=[];pair_values={};scalar_evidence={}
# Only a role's own completed forwards qualify as feature donors. Its inherited
# external source cache is not silently included. Neither old baseline is scored.
for short in('b','a'):
 for kind in('three','clip'):
  donor=short+'_'+kind+'_metrics';c=closed(donor);donor_closures.append(c);identity=pin(L/donor/'metric_identity.json')
  validate_metric_identity(cr(identity),resources[kind],old['frozen_metric_registration'],old['config']['clip']['loader']);identities[kind].append(identity)
  con=sqlite3.connect('file:'+str(L/donor/'ledger.sqlite')+'?mode=ro',uri=True)
  assert con.execute("SELECT COUNT(*) FROM calls WHERE status!='COMPLETE'").fetchone()[0]==0
  entries={(k,key):json.loads(v)for k,key,v in con.execute('SELECT kind,id,receipt FROM calls')};con.close()
  for (k,key),d in entries.items():
   if k!='image_embedding':continue
   p=checked(d)
   with p.open('rb')as f:z=np.load(f,allow_pickle=False)
   assert z.dtype==np.float32 and z.ndim==2 and z.shape[0]==1 and np.isfinite(z).all()
   if key in feature_maps[kind]:assert feature_maps[kind][key]['sha256']==d['sha256']
   else:feature_maps[kind][key]=d;feature_provenance[kind][key]=dict(role=donor,identity=identity,actual_wait=c['actual_wait'])
  if kind!='three':continue
  # Scalars are reusable only when their three actual calls have complete
  # receipts under this exact runtime. Reused historical scores are excluded.
  oldplan=read(L/(short+'_metric_preparation/plan.json'));sid={s['source_id']:s for s in oldplan['sources']}
  full=list(csv.DictReader((O/(short+'_delivery/frame_metrics.csv')).open()))
  for row in full:
   s=sid[row['source_id']];key=s['reference_sha256']+'_'+row['pixel_sha256']
   triples=[('psnr','psnr_db'),('lpips_pair','lpips_alex'),('cosine_pair','dreamsim')]
   if not all((k,key)in entries for k,_ in triples):continue
   values={m:float(row[m])for m in METRICS};proof=[]
   for k,m in triples:
    d=entries[k,key];v=cr(d);assert v['kind']==k and v['id']==key and v['value']==values[m];proof.append(d)
   if key in pair_values:assert {m:pair_values[key][m]for _,m in triples}=={m:values[m]for _,m in triples}
   else:pair_values[key]=values;scalar_evidence[key]=dict(role=donor,identity=identity,calls=proof,verified_columns=[m for _,m in triples],unused_legacy_clip_value=True)

# Recount actual C GPU spend, including the old qualification and all formal RX.
prior=cr(admissions['common500']['budget'])['gpu_closed'];gpu_roles={v['role']for v in prior}
gpu_roles.update(c['role']for c in all_formal if c['kind']!='PHY')
metric_predecessors=[]
for pop in('common500','kodak24'):
 for kind in('three','clip'):
  role='c_predclass_'+pop+'_'+kind+'_metrics'
  if role in roles.values():continue
  if any(p.exists()for p in(L/role,L/(role+'_control'),L/(role+'_initial_control'))):
   c=closed(role);metric_predecessors.append(c);gpu_roles.add(role)
spent=sum(closed(role)['elapsed_seconds']for role in sorted(gpu_roles))
assert spent+4*7200<=96*3600,'Reserve remaining formal metric stages within the original GPU budget'

# All checks above are read-only. Deploy only qualified bytes, then create
# exclusive preparation and role registrations. No model/PHY/test execution.
target=E/'c_predclass_formal_metrics_v1'
if target.exists():
 assert set(p.name for p in target.iterdir())==set(raws)
 for n,b in raws.items():assert(target/n).read_bytes()==b
else:
 target.mkdir(exist_ok=False)
 for n,b in raws.items():
  with(target/n).open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
prep.mkdir(exist_ok=False)
save(prep/'clip_inherited_reference_cache.json',[])
save(prep/'input_identity.json',dict(population=population,formal_admission=pin(O/(base+'_rx_admission.json')),sources=link['sources'],native_RX_outputs={k:dict(image=v['image'],pixel_sha256=v['pixel_sha256'],role=output_origins[k])for k,v in outputs.items()},logical_manifest=a['logical_manifest'],source_archive_to_float='uint8/float32(255); original pixels and SHA preserved',new_model_calls=0))
save(prep/'cache_evidence.json',dict(donor_closures=donor_closures,feature_provenance=feature_provenance,scalar_evidence=scalar_evidence,legacy_CLIP_source_cache_reused=False,legacy_CLIP_scalar_reused=False))
save(prep/'budget.json',dict(GPU_closed_seconds=spent,GPU_ceiling_seconds=96*3600,conservative_reserved_remaining_metric_seconds=4*7200,gpu_closures=[closed(role)for role in sorted(gpu_roles)],per_stage_seconds=7200,all_formal_closures=all_formal))
summary=[]
for kind,role in roles.items():
 plan=copy.deepcopy(plan_base)
 for j in plan['frames']:
  key=j['reference_sha256']+'_'+j['pixel_sha256']
  if kind=='three'and key in pair_values:
   j.update(reuse_old=True,old_values=pair_values[key],old_metric_identity_verified=True,verified_metric_columns=['psnr_db','lpips_alex','dreamsim'])
 plan['evaluated_columns']=['psnr_db','lpips_alex','dreamsim']if kind=='three'else['clip_image_cosine']
 plan['unused_legacy_clip_values_are_not_scored_or_reported']=True
 assert validate_plan(plan,population)==total
 pairs={(j['reference_sha256'],j['pixel_sha256'])for j in plan['frames']if not j['reuse_old']}
 needed={h for pair in pairs for h in pair};cache={h:d for h,d in feature_maps[kind].items()if h in needed}
 for d in cache.values():checked(d)
 new_features=needed-set(cache);source_hashes={s['reference_sha256']for s in sources}
 save(prep/(kind+'_plan.json'),plan);save(prep/(kind+'_closed_features.json'),cache)
 limits=dict(image_embedding=len(new_features),cosine_pair=len(pairs))
 if kind=='three':limits.update(load_lpips=1,load_dreamsim=1,psnr=len(pairs),lpips_pair=len(pairs))
 else:limits.update(load_clip=1)
 reg={k:copy.deepcopy(old[k])for k in('scope','config','frozen_metric_registration','original_four','prior_clip_identity','persistent_ledger')}
 stage=copy.deepcopy(old['stages']['a_'+kind+'_metrics']);stage['argv']=[stage['argv'][0],'-B',str(target/'worker.py'),role];stage['max_seconds']=7200
 reg.update(schema='C_PREDCLASS_FORMAL_METRIC_ROLE_V1',population=population,hostname=os.uname().nodename,
  revision=a['revision'],frozen_policy=a['frozen_policy'],plan=pin(prep/(kind+'_plan.json')),
  clip_cache=pin(prep/'clip_inherited_reference_cache.json'),science_closures=list({c['role']:c for c in science}.values()),
  feature_cache={role:pin(prep/(kind+'_closed_features.json'))},cache_metric_identities={role:identities[kind]},
  resources={role:resources[kind]},limits={role:limits},metric_kinds={role:kind},max_seconds=7200,stages={role:stage},
  qualified_cpu_actual_wait=pin(q/'actual_wait.json'),coordinator_review=pin(target/'coordinator_review.json'),
  input_identity=pin(prep/'input_identity.json'),cache_evidence=pin(prep/'cache_evidence.json'),budget=pin(prep/'budget.json'),
  new_source_features=len(new_features&source_hashes),new_reconstruction_features=len(new_features-source_hashes),
  new_pair_computations=len(pairs),reused_logical_rows=sum(j['reuse_old']for j in plan['frames']),
  pairing='Source-paired only. No claim of identical cross-method noise.',no_new_bootstrap_indices=True,
  no_reconstruction=True,no_PHY=True,all_logical_failures_retained=True,code=[])
 assert reg['scope']==a['scope'];seen=set()
 for d in[pin(E/n)for n in('common.py','control.py')]+[pin(target/n)for n in raws]+[reg['persistent_ledger'],reg['original_four']]:
  checked(d)
  if d['path']not in seen:reg['code'].append(d);seen.add(d['path'])
 save(registration_paths[kind],reg)
 summary.append(dict(role=role,registration=pin(registration_paths[kind]),limits=limits,logical_pairs=total,new_pairs=len(pairs),new_source_features=reg['new_source_features'],new_reconstruction_features=reg['new_reconstruction_features'],reused_logical_rows=reg['reused_logical_rows'],feature_cache_hits=len(cache),launched=False))
save(prep/'completion.json',dict(status='REGISTERED_NOT_LAUNCHED',population=population,registrations=summary,GPU_closed_seconds=spent,new_model_calls=0,new_PHY_calls=0))
print(json.dumps(dict(preparation=pin(prep/'completion.json'),registrations=summary)))
'''.replace('__ROOT__',repr(R)).replace('__POP__',repr(population)).replace('__FILES__',repr(files))
compile(code,'<closed formal four-metric preparation>','exec')
if __name__=='__main__':print(run(code,'prepare_formal_'+population+'_metrics_v2',host=PRO,timeout=600).decode())
