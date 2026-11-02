"""Admit exact previous receiver identities; only missing projections run."""
from study_common import *
from launch_extra_v2 import launch_extra_v2
from launch_extra import launch_extra
sys.path.insert(0,str(R/'experiments/nestcom-perceptual-followup-20261030'))
from prepare_e2_receivers import wait_for

def donor_wait(root,role):
 paths=[root/(role+'_control/actual_wait.json'),root/(role+'_owner_control/actual_wait.json'),root/'controls'/role/'actual_wait.json']
 p=next((p for p in paths if p.exists()),None)
 if p is None or read(p)['returncode']!=0:return None
 return wait_for(root,role)

def start(stage):
 out=L/('cal_'+stage+'_rx_plan');out.mkdir(exist_ok=False);alljobs={};frames={}
 for role in read(L/('cal_'+stage+'_phy_dispatch.json'))['roles']:
  closed(role)
  for j in read(L/role/'rx_inputs.json'):assert j['key']==digest(j['projection']);alljobs[j['key']]=j
  for row in read(L/role/'frames.json'):frames[row['physical_key']]=row
 jobs={k:{j['key']:j for j in alljobs.values()if j['projection'].get('codec')==k}for k in ['NEST','ADA']};reuse={};lineage=[]
 # Per-input dedup excludes gray inputs from all model loads.
 for key,j in alljobs.items():
  if j['projection']['status']=='FIXED_GRAY':reuse[key]=dict(key=key,image=None,pixel_sha256=asha(np.full((3,256,256),.5,np.float32)),status='FIXED_GRAY',reason=j['projection']['reason'])
 nref=read(OL/'final_rx/actual_native_identity.json');aroot=B/'adatoken_response_20261026_v1';n5=B/'nestcom_n512_20261028_v1'
 npaths=[OL/'final_rx',FL/'E3_rx',FL/'E2_nest',L/'low_rx',B/'nestcom_7db_common500_20261030_v1/rx_nest']+[p for p in n5.glob('*_rx_nest_*')if(p/'outputs.json').exists()]
 npaths += [L/r for prior in ['pilot','full']if prior!=stage and (L/('cal_'+prior+'_rx_plan/completion.json')).exists()for r in read(L/('cal_'+prior+'_rx_plan/completion.json'))['roles']if'_nest_'in r]
 for root in npaths:
  if not(root/'actual_native_identity.json').exists():continue
  identity=read(root/'actual_native_identity.json')
  if not all(identity[k]==nref[k]for k in ['models','runtime','numeric_flags','source_mathematics','sampling','GPU_name']):continue
  w=donor_wait(root.parent,root.name)
  if w is None:continue
  used=0
  for row in read(root/'outputs.json'):
   key=row['key']
   if key not in jobs['NEST']or key in reuse or row['status']!='ACCEPTED_SOURCE':continue
   p=jobs['NEST'][key]['projection']
   with np.load(checked(row['image']),allow_pickle=False)as z:assert asha(z['image'])==row['pixel_sha256']and asha(z['received_tokens'])==row['received_tokens_sha256']and len(z['received_tokens'])==OFFSETS[p['m']]+p['K']
   reuse[key]=row;used+=1
  if used:lineage.append(dict(codec='NEST',root=str(root),wait=w,index=pin(root/'outputs.json'),identity=pin(root/'actual_native_identity.json'),reused=used))
 # Original sequential Ada routes share weight bytes, branch arithmetic and seeds.
 aref=read(B/'nestcom_7db_common500_20261030_v1/rx_ada/numeric_identity.json')
 apaths=[p for p in n5.glob('*rx_ada*')if(p/'outputs.json').exists()]+[p for p in aroot.glob('c_predclass_*_rx_s??')if(p/'outputs.json').exists()]+[B/'nestcom_7db_common500_20261030_v1/rx_ada']
 apaths += [L/r for prior in ['pilot','full']if prior!=stage and (L/('cal_'+prior+'_rx_plan/completion.json')).exists()for r in read(L/('cal_'+prior+'_rx_plan/completion.json'))['roles']if'_ada_'in r]
 targets={}
 sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'));from c_predclass_protocol import generation_seed
 for key,j in jobs['ADA'].items():
  p=j['projection'];targets[(asha(np.array(p['bits'],np.uint8)),p['m'],p['received_class'])]=key
 for root in apaths:
  ip=root/'numeric_identity.json'
  if not ip.exists():continue
  identity=read(ip)
  # Storage moves may alter path strings; model content and every numeric flag remain fixed.
  def norm(v):
   if isinstance(v,dict):return {k:norm(x)for k,x in v.items()if k not in ['path','pci']}
   if isinstance(v,list):return [norm(x)for x in v]
   return v
  if norm(identity)!=norm(aref):continue
  w=donor_wait(root.parent,root.name)
  if w is None:continue
  used=0
  for row in read(root/'outputs.json'):
   key=row['key']if row['key']in jobs['ADA']else targets.get((row.get('payload_sha256'),row.get('prefix'),row.get('received_class')))
   if key is None or key in reuse or row.get('status')!='ACCEPTED_SOURCE':continue
   p=jobs['ADA'][key]['projection'];assert row['generation_seed']==generation_seed(np.array(p['bits'],np.uint8),p['m'],p['received_class'])
   with np.load(checked(row['image']),allow_pickle=False)as z:assert asha(z['image'])==row['pixel_sha256']and asha(z['received_tokens'])==row['received_tokens_sha256']and len(z['received_tokens'])==p['m']
   reuse[key]=dict(row,key=key);used+=1
  if used:lineage.append(dict(codec='ADA',root=str(root),wait=w,index=pin(root/'outputs.json'),identity=pin(ip),reused=used))
 save(out/'reuse.json',reuse);save(out/'frames.json',frames);save(out/'all_jobs.json',alljobs);roles=[];counts={}
 for codec,gpus in [('NEST',[4,5]),('ADA',[2,3,6,7])]:
  missing=sorted([v for k,v in jobs[codec].items()if k not in reuse],key=lambda v:v['key']);counts[codec]=dict(unique=len(jobs[codec]),new=len(missing),reused=len(jobs[codec])-len(missing))
  for i,gpu in enumerate(gpus):
   subset=missing[i::len(gpus)]
   if not subset:continue
   role=f'cal_{stage}_{codec.lower()}_{i:02d}';jp=out/(codec+str(i)+'.json');save(jp,subset);empty=out/(codec+str(i)+'_reuse.json');save(empty,{})
   if codec=='NEST':
    old=read(F/'E3_rx_registration.json');r={k:old[k]for k in ['code','model_routes','static','expanded_grid']};n=len(subset);r.update(jobs=pin(jp),reuse=pin(empty),cpu_affinity=[54+4*i,56+4*i],caps=dict(model_load=1,source_decode=n,completion_forward=n,render=n,prior_scale=20*n,decoder_forward=n))
    # Existing worker dispatch remains untouched; this role imports the exact original receiver.
    from study_common import launch
    launch(role,r,'rx',VPY,7200,gpu=gpu)
   else:
    old=read(R/'results/nestcom_n512_20261028_v1/qualification_rx_ada_registration.json');native=read(R/'results/nestcom_timing_fairness_20261102_v1/paired_prefill_child_v4/registration.json')['worker_template']['native']
    r={k:v for k,v in old.items()if k in ['code','codec','weights','upstream_code']};r.update(weights=native['weights'],upstream_code=native['upstream_code'],inputs=pin(jp),cpu_affinity=[64+4*i,66+4*i],caps=dict(model_load=1,receive_render=len(subset)))
    sys.path.insert(0,str(R/'tools'));import checkpoint_relocation_20261028 as relocation
    r=relocation.relocate_value(r);r['code']+=[pin(R/'tools/checkpoint_relocation_20261028.py'),pin(relocation.RECEIPT)]
    launch_extra_v2(role,r,'ada_original_rx.py',VPY,14400 if stage=='full' else 7200,gpu=gpu)
   roles.append(role)
 save(out/'completion.json',dict(status='COMPLETE_DISPATCH',roles=roles,counts=counts,reuse=pin(out/'reuse.json'),frames=pin(out/'frames.json'),lineage=lineage))
 print('RX_DISPATCH',stage,counts,flush=True)
if __name__=='__main__':start(sys.argv[1])
