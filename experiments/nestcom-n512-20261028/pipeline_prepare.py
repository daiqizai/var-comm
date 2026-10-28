"""Stage-specific immutable preparations; no model, PHY, bootstrap or retry."""
from n512_common import *
from n512_phy import ENDS,candidates

def closed(role):
 w=read(L/'controls'/role/'actual_wait.json')
 assert w['actual_wait'] and w['returncode']==0 and not w['timed_out'] and gone(w['process'])and gone(w['owner']),role
 assert not (L/'controls'/role/'failure.json').exists() and not (L/role/'failure.json').exists(),role
 return pin(L/'controls'/role/'actual_wait.json')

def source_closure():
 root=L/'qualified_sources';root.mkdir(exist_ok=False);orig=read(L/'assets/completion.json');donors=read(L/'missing_source_preparation/existing100.json');new=read(L/'route_first_source/sources.json');waits=[closed('route_first_source')]
 for i in range(8):
  role=f'route_missing_source_{i:02d}';waits.append(closed(role));done=read(L/role/'completion.json');assert done['ledger']['unresolved']==0;new.extend(cr(done['sources']))
 assert len(new)==1400 and len({x['source_id']for x in new})==1400
 allrows={x['source_id']:x for x in new+donors};assert len(allrows)==1500;descs={}
 for pop in ('cal1000','common500'):
  source=cr(orig['sources'][pop]);rows=[]
  for s in source:
   v=allrows[s['source_id']];assert s['source_index']==v['source_index']
   with np.load(checked(v['streams']),allow_pickle=False)as z:streams={f'm{m}_K{K}':z[f'm{m}_K{K}'].copy()for m,K in ENDS}
   for a in streams.values():assert a.dtype==np.uint8 and a.ndim==1 and len(a)>=2 and np.isin(a,[0,1]).all()
   t=dict(s);t.update(nest_streams=v['streams'],nest_cdf_trace=v['trace'],nest_lengths={k:len(a)for k,a in streams.items()},nest_payload_hashes={k:asha(a)for k,a in streams.items()},nest_numeric_route='original_models_FP32_no_grad_PRO6000_same_type',nest_historical_H800_streams=s['nest_streams'])
   rows.append(t)
  save(root/(pop+'.json'),rows);descs[pop]=pin(root/(pop+'.json'))
 save(root/'completion.json',dict(status='COMPLETE_SOURCE_ROUTE_CACHE',sources=descs,new_source_probability=1400,previous_PRO_source_reuse=100,new_tokenizer=0,actual_waits=waits,source_audits=[pin(O/'existing_PRO25_source_audit.json'),pin(O/'first_PRO25_source_audit.json')],visual_review=pin(O/'route_visual_review.json')))
 return descs

def seal_protocol():
 assert not(O/'protocol.json').exists();qualified=read(L/'qualified_sources/completion.json');assets=read(L/'assets/completion.json');cat=read(O/'public_catalogue.json');assert len(cat['layouts'])==12
 pilot=cr(assets['pilot_request']);ids=sorted({s['source_id']for s in pilot['records']});assert len(ids)==100
 audit=read(O/'qualification_metadata_audit.json');assert audit['status']=='ADA_OUTPUTS_VERIFIED_NEST_ROUTE_BLOCKED'
 assert read(O/'first_PRO25_source_audit.json')['status']=='PASS'
 r=dict(schema='NESTCOM_N512_INDEPENDENT_CALIBRATION_V1',base_commit=BASE,N=512,header_symbols=68,body_symbols=444,nominal_Es=2,total_nominal_energy=1024,snrs=[4,10,19],methods=['NEST','WHOLE','ADA'],sources=qualified['sources'],pilot_source_ids=ids,catalogue=pin(O/'public_catalogue.json'),candidates=candidates([(v['q'],v['rate'])for v in cat['layouts']]),pilot_noise=[4101],full_noise=[4101,4102,4103],formal_noise=[6201,6202,6203],calibration_metric='dinov2_vitl14_cosine',source_aggregation='average three outcomes first; equally weight source means',finalists='top3 plus q2 r1/2 maximum endpoint anchor; unique candidates',tie_break='candidate_id lexical ascending',source_route=pin(O/'source_route_amendment_v2.json'),nest_model_routes=assets['nest_models'],nest_static=assets['nest_static'],bootstrap=assets['bootstrap'],formal_metrics=['psnr_db','lpips_alex','dreamsim','clip_image_cosine'],caps=dict(pilot_logical=118800,full_logical=108000,formal_logical=13500,PHY_packets=484132,source_probability=1400,tokenizer=0,GPU_device_hours=96,CPU_process_hours=96),budget_interaction='all four metrics and SNRs; source paired change in NeST-Ada gap, separately calibrated systems',planning_code=[pin(E/p)for p in ['n512_common.py','n512_phy.py','plan_stage.py','pipeline_prepare.py']],qualification=[pin(O/'first_PRO25_source_audit.json'),pin(O/'existing_PRO25_source_audit.json'),pin(O/'qualification_metadata_audit.json'),pin(L/'phy_qualification/completion.json')],stage_status='REGISTERED_BEFORE_PILOT',reference_implementation_commit=BASE)
 assert len(r['candidates'])==396;save(O/'protocol.json',r);return pin(O/'protocol.json')

def phy_reuse(stage):
 prior={'pilot':[],'full':['pilot'],'formal':['pilot','full']}[stage];by={}
 for name in prior:
  for i in range(8):
   role=f'{name}_phy_{i:02d}';closed(role);done=read(L/role/'completion.json')
   for p in cr(done['frames']):
    row=cr(p);key=row['physical_identity']
    if key in by:assert by[key]==p
    else:by[key]=p
 return by

def prepare_rx(stage):
 root=L/(stage+'_rx_plan');root.mkdir(exist_ok=False);frames={};jobs={'NEST':{},'ADA':{}};reuse={}
 for source_role in ['route_existing25_rx','route_first25_rx','qualification_rx_ada']:
  if source_role!='qualification_rx_ada':closed(source_role)
  for row in read(L/source_role/'outputs.json'):
   assert row['status']=='ACCEPTED_SOURCE';reuse[row['key']]=dict(record=row,origin=source_role)
 for prior in {'pilot':[],'full':['pilot'],'formal':['pilot','full']}[stage]:
  d=read(L/(prior+'_rx_plan')/'completion.json')
  for rp in d['worker_registrations']:
   reg=cr(rp);role=Path(reg['output']).name;closed(role)
   for row in read(L/role/'outputs.json'):reuse[row['key']]=dict(record=row,origin=role)
 for i in range(8):
  role=f'{stage}_phy_{i:02d}';closed(role);d=read(L/role/'completion.json')
  for p in cr(d['frames']):
   row=cr(p);key=row['physical_identity'];frames[key]=p;proj=row['projection'];k=row['received_key'];assert digest(proj)==k
   if proj['status']=='SOURCE_INPUT'and k not in reuse:jobs[proj['codec']][k]=dict(key=k,projection=proj)
 save(root/'frames.json',frames);save(root/'reuse.json',reuse)
 paths={}
 for codec in jobs:
  paths[codec]=[];ordered=sorted(jobs[codec].values(),key=lambda j:j['key'])
  for i in range(4):
   p=root/f'{codec}_{i:02d}.json';save(p,ordered[i::4]);paths[codec].append(pin(p))
 save(root/'plan.json',dict(stage=stage,frames=pin(root/'frames.json'),reuse=pin(root/'reuse.json'),shards=paths,new_unique={k:len(v)for k,v in jobs.items()},prior_unique=len(reuse)))
 return read(root/'plan.json')

def pair_plan(stage):
 root=L/(stage+'_metric_plan');root.mkdir(exist_ok=False);p=read(L/(stage+'_rx_plan')/'completion.json');rx={k:v['record']for k,v in cr(p['reuse']).items()}
 for desc in p['worker_registrations']:
  r=cr(desc);role=Path(r['output']).name;closed(role)
  for row in read(L/role/'outputs.json'):rx[row['key']]=row
 protocol=read(O/'protocol.json');sources=cr(protocol['sources']['common500'if stage=='formal'else'cal1000']);sm={s['source_id']:s for s in sources};pairs={};physical={}
 for key,desc in cr(p['frames']).items():
  f=cr(desc);s=sm[f['job']['source_id']];proj=f['projection'];rk=f['received_key']
  if proj['status']=='FIXED_GRAY':image=None;pixel=asha(np.full((3,256,256),.5,np.float32));status=proj['reason']
  else:
   result=rx[rk];image=result['image'];pixel=result['pixel_sha256'];status=result['status']
  pk=digest([s['reference_sha256'],pixel]);pairs.setdefault(pk,dict(pair_key=pk,source_id=s['source_id'],source_index=s['source_index'],image=image,pixel_sha256=pixel,reference_sha256=s['reference_sha256']))
  physical[key]=dict(pair_key=pk,source_status=status,received_key=rk,image=image,pixel_sha256=pixel)
 used=set(v['source_id']for v in pairs.values());save(root/'pairs.json',dict(sources=[s for s in sources if s['source_id']in used],pairs=list(pairs.values())))
 save(root/'physical.json',physical);save(root/'completion.json',dict(status='READY_FOR_METRICS',stage=stage,pairs=pin(root/'pairs.json'),physical=pin(root/'physical.json'),unique_pairs=len(pairs),logical=pin(L/(stage+'_plan')/'logical.json'),source_count=len(used)))
 return read(root/'completion.json')

if __name__=='__main__':
 action=sys.argv[1]
 if action=='source_closure':print(source_closure())
 elif action=='seal_protocol':print(seal_protocol())
 elif action=='prepare_rx':print(prepare_rx(sys.argv[2]))
 elif action=='pair_plan':print(pair_plan(sys.argv[2]))
