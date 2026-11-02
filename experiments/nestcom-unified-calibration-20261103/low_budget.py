"""N512/10 matched grids; first-only stages, exact reuse and source pairing."""
from study_common import *
from grid_codec import endpoints

def prepare():
 out=L/'low_plan';out.mkdir(exist_ok=False)
 sets=read(F/'source_sets.json');allss=cr(read(OLD/'frozen_final.json')['common500_sources']);sm={s['source_id']:s for s in allss};ss=[sm[x]for x in sets['control100']];assert len(ss)==100
 base=read(OLD/'final_phy_512_registration.json');cat=cr(base['catalogue']);profiles=[p for p in cat['profiles']if(p['q'],p['nominal_rate'])==(4,'2/3')]
 jobs=cr(base['jobs']);oldjobs={j['key']:j for j in jobs};core=read(FL/'inventory_core/logical.json')
 canonical={(v['source_id'],int(v['channel_repeat'])):oldjobs[v['physical_key']]for v in core if v['method']=='final_NeST'and int(v['N'])==512 and int(v['snr_db'])==10}
 fi={v['physical_key']:v['frame']for v in read(OL/'historical_regression_inputs/physical_index.json')};fi.update({v['physical_key']:v['frame']for v in read(OL/'final_phy_512/frames.json')})
 streams={v['source_id']:v for v in read(OL/'final_D_source/outputs.json')};physical={};logical=[];cap=profiles[0]['k']-29;target=(7,0);limit=OFFSETS[7]
 for s in ss:
  v=streams[s['source_id']];assert v['source_pixels_sha256']==s['preprocessing_id']
  with np.load(checked(v['streams']),allow_pickle=False)as z:bits={k:z[k].copy()for k in z.files}
  for grid in ['whole','quarter','eighth','token']:
   es=sorted([e for e in endpoints('token'if grid=='whole'else grid)if OFFSETS[e[0]]+e[1]<=limit and(grid!='whole'or e[1]==0)],key=lambda e:OFFSETS[e[0]]+e[1],reverse=True)
   feasible=[(i,e,bits['m%d_K%d'%e])for i,e in enumerate(es,1)if len(bits['m%d_K%d'%e])<=cap];assert feasible
   rank,e,b=feasible[0];checks=len(es);pr=next(p for p in profiles if(p['m'],p['K'],p['family'])==(*e,'EC_VAR_PARTIAL'if e[1]else'EC_VAR_WHOLE'))
   for repeat in range(3):
    c=canonical[s['source_id'],repeat];assert c['noise_seed']==6201+repeat and c['snr_db']==10
    identity={k:c[k]for k in ['N','source_id','source_index','snr_db','noise_seed','counter']};identity.update(profile_id=pr['profile_id'],payload_sha256=asha(b));key=digest(identity)
    old=fi.get(key);job=dict(key=key,**identity,payload=b.tolist(),profile=pr,old=old,public_base=None)
    if grid=='token':assert all(job[k]==c[k]for k in ['key','payload','profile','counter','noise_seed'])and old is not None
    physical.setdefault(key,job);logical.append(dict(source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],N=512,snr_db=10,method=grid,channel_repeat=repeat,noise_seed=c['noise_seed'],physical_key=key,m=e[0],K=e[1],target=list(target),q=4,rate='2/3',source_bits=len(b),capacity_bits=cap,utilization=len(b)/cap,candidate_count=len(es),actual_length_comparisons=checks))
 assert len(logical)==1200
 save(out/'sources.json',[source_only(s)for s in ss]);save(out/'physical.json',list(physical.values()));save(out/'logical.json',logical)
 save(out/'completion.json',dict(status='COMPLETE_PLAN',logical=1200,physical=len(physical),old_candidates=sum(j['old']is not None for j in physical.values()),target=list(target),N=512,q=4,rate='2/3',source_sets=pin(F/'source_sets.json')))
 r={k:base[k]for k in ['N','catalogue','original_catalogue','n512_qualification','qualification','code']};r.update(jobs=pin(out/'physical.json'),cpu_affinity=[24,26]);launch('low_phy',r,'phy',CPY,3600)

def receiver():
 closed('low_phy');out=L/'low_rx_plan';out.mkdir(exist_ok=False)
 jobs={j['key']:j for j in read(L/'low_phy/rx_inputs.json')};reuse={};lineage=[];donor=read(OL/'final_rx/actual_native_identity.json')
 sys.path.insert(0,str(R/'experiments/nestcom-perceptual-followup-20261030'));from prepare_e2_receivers import wait_for
 candidates=[(FL,x)for x in ['E3_rx','E2_nest']]+[(OL,'final_rx'),(B/'nestcom_7db_common500_20261030_v1','rx_nest')]
 candidates += [(B/'nestcom_n512_20261028_v1',f'{p}_rx_nest_{i:02d}')for p in ['formal','full','pilot']for i in range(4)]
 for root,role in candidates:
  ix=root/role/'outputs.json';ip=root/role/'actual_native_identity.json'
  if not(ix.exists()and ip.exists()):continue
  identity=read(ip)
  if not all(identity[k]==donor[k]for k in ['models','runtime','numeric_flags','source_mathematics','sampling','GPU_name']):continue
  wait=wait_for(root,role)
  if wait is None:continue
  used=0
  for v in read(ix):
   key=v['key']
   if key not in jobs or key in reuse:continue
   p=jobs[key]['projection'];assert key==digest(p)
   if v['image']is not None:
    with np.load(checked(v['image']),allow_pickle=False)as z:assert asha(z['image'])==v['pixel_sha256']and asha(z['received_tokens'])==v['received_tokens_sha256']and len(z['received_tokens'])==OFFSETS[p['m']]+p['K']
   else:assert p['status']!='SOURCE_INPUT'
   reuse[key]=v;used+=1
  if used:lineage.append(dict(index=pin(ix),identity=pin(ip),wait=wait,reused=used))
 save(out/'jobs.json',list(jobs.values()));save(out/'reuse.json',reuse);save(out/'completion.json',dict(status='COMPLETE_PLAN',unique=len(jobs),reuse=len(reuse),lineage=lineage))
 old=read(F/'E3_rx_registration.json');r={k:old[k]for k in ['code','model_routes','static','expanded_grid']};n=len(jobs)-len(reuse)
 r.update(jobs=pin(out/'jobs.json'),reuse=pin(out/'reuse.json'),cpu_affinity=[28,30],caps=dict(model_load=1,source_decode=n,completion_forward=n,render=n,prior_scale=20*n,decoder_forward=n))
 launch('low_rx',r,'rx',VPY,3600,gpu=2)

def metric_plan():
 closed('low_rx');out=L/'low_metric_plan';out.mkdir(exist_ok=False);sources=read(L/'low_plan/sources.json');sm={s['source_id']:s for s in sources}
 old={v['dists_pair_key']:v for v in read(FL/'E1_join/frames.json')if v['method']in ['final_NeST','original_NeST','WHOLE_final_receiver']}
 physical={v['physical_key']:v for v in read(L/'low_phy/frames.json')};rx={v['key']:v for v in read(L/'low_rx/outputs.json')};frames=[];missing={}
 for v in read(L/'low_plan/logical.json'):
  v=dict(v);f=physical[v['physical_key']];q=rx[f['received_key']];s=sm[v['source_id']];pk=digest([s['reference_sha256'],q['pixel_sha256']])
  v.update(received_key=f['received_key'],frame=f['frame'],image=q['image'],pixel_sha256=q['pixel_sha256'],pair_key=pk,receiver_status=q['status'])
  if pk in old:
   p=old[pk];assert(p['reference_sha256'],p['pixel_sha256'])==(v['reference_sha256'],v['pixel_sha256']);v.update({m:p[m]for m in METRICS});v['metric_origin']='EXACT_E1_NATIVE_PAIR'
  else:
   missing.setdefault(pk,dict(pair_key=pk,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=q['pixel_sha256'],image=q['image']));v['metric_origin']='NEW_FIVE_METRIC_GAP'
  frames.append(v)
 save(out/'logical.json',frames);save(out/'plan.json',dict(sources=sources,pairs=list(missing.values())));save(out/'pairs.json',list(missing.values()));save(out/'completion.json',dict(status='COMPLETE_PLAN',logical=len(frames),unique_pairs=len({v['pair_key']for v in frames}),missing_pairs=len(missing)))

def score(mode):
 p=L/'low_metric_plan';n=read(p/'completion.json')['missing_pairs'];old=read(F/('E3_'+mode+'_registration.json'))
 if mode=='dists':
  r={k:old[k]for k in ['assets','loader','torch','code','caches']};r.update(sources=pin(L/'low_plan/sources.json'),pairs=pin(p/'pairs.json'),cpu_affinity=[32,34],caps=dict(load_dists=1,pair=n),priority_pairs=[]);kind='dists';py=VPY
 else:
  r={k:old[k]for k in ['code','metric_registration','mode','caches']};r.update(plan=pin(p/'plan.json'),cpu_affinity=[32,34],caps=dict(embedding=n+100,pair=n,**(dict(load_clip=1)if mode=='clip'else dict(load_lpips=1,load_dreamsim=1))));kind='metrics';py=APY if mode=='clip'else VPY
 r['caches']=r['caches']+[pin(FL/('E3_'+mode)/'completion.json')]
 launch('low_'+mode,r,kind,py,1800,gpu=1)

if __name__=='__main__':
 action=sys.argv[1]
 if action=='prepare':prepare()
 elif action=='receiver':receiver()
 elif action=='metric_plan':metric_plan()
 elif action=='score':score(sys.argv[2])
