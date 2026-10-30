import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
assert not(L/'fit_rx_plan').exists();out=L/'fit_rx_plan';out.mkdir()
jobs={};frames={}
for N in(512,1024):
 role='fit_phy_'+str(N);w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
 for j in read(L/role/'rx_inputs.json'):jobs.setdefault(j['key'],j)
 for f in read(L/role/'frames.json'):frames[f['physical_key']]=f
logical=read(L/'fit_frame_plan/logical.json');assert len(logical)==1400
for f in logical:f.update(received_key=frames[f['physical_key']]['received_key'])
N=B/'nestcom_n512_20261028_v1';identity=read(L/'p0_replay/actual_native_identity.json');reuse={};lineage=[]
for name in ['route_existing25_rx','route_first25_rx']+[f'{phase}_rx_nest_{i:02d}'for phase in ['pilot','full','formal']for i in range(4)]:
 root=N/name;index=root/'outputs.json';ni=root/'actual_native_identity.json'
 if not(index.exists()and ni.exists()):continue
 old=read(ni)
 assert all(old[k]==identity[k]for k in ['models','runtime','numeric_flags','source_mathematics','sampling'])
 assert old['GPU_name']==identity['GPU_name']
 wa=N/'controls'/name/'actual_wait.json'
 if wa.exists():assert read(wa)['returncode']==0
 else:
  # Qualification roles predate the common controls folder; require the original actual receipt.
  matches=list(N.glob('*'+name+'*/actual_wait.json'))
  if not matches:print('SKIP_UNRESOLVED_OWNER_LINEAGE',name);continue
  assert all(read(p)['returncode']==0 for p in matches)
 used=0
 for row in read(index):
  key=row['key']
  if key not in jobs or key in reuse or row.get('codec')!='NEST' or jobs[key]['projection']['status']!='SOURCE_INPUT':continue
  assert row['status']=='ACCEPTED_SOURCE'and row['canonical']and not row['source_truth_used']and not row['TX_CDF_used']
  with np.load(checked(row['image']),allow_pickle=False)as z:a=z['image'];t=z['received_tokens']
  p=jobs[key]['projection'];assert t.shape==(OFFSETS[p['m']]+p['K'],)and asha(t)==row['received_tokens_sha256']and asha(a)==row['pixel_sha256']
  reuse[key]=row;used+=1
 if used:lineage.append(dict(index=pin(index),identity=pin(ni),wait=pin(wa)if wa.exists()else[pin(p)for p in matches],reused=used))
save(out/'inputs.json',list(jobs.values()));save(out/'reuse.json',reuse);save(out/'logical.json',logical)
save(out/'completion.json',dict(status='PREPARED',logical_count=len(logical),unique_inputs=len(jobs),reused_images=len(reuse),new_baseline_decodes=sum(j['projection']['status']=='SOURCE_INPUT'and k not in reuse for k,j in jobs.items()),cache_lineage=lineage,hardware_route_qualification=pin(L/'p0_replay/completion.json'),variant_qualification=pin(L/'variant_qualification/completion.json')))
print(json.dumps({k:v for k,v in read(out/'completion.json').items()if k!='cache_lineage'}))
save(D/'new200_availability.json',dict(status='BLOCKED',affected_stage='NEW200 final independent validation only',reason='No available auditable unused ImageNet validation pool identified at the registered original source root on either PRO or main host. Current qualified PFS pool contains cal1000/common500 only and is not relabeled independent.',original_root='/home/liulu/projects/VAR-MAP-GATE0/data/imagenet_hf',PRO_root_exists=False,MAIN_root_exists=False,available_qualified_pool=1500,selection_rule='SHA256(nestcom_opt_new200_20261030_v1|canonical_source_id), exclude every project-used source',candidate_quality_seen=False,other_routes_continue=True))
