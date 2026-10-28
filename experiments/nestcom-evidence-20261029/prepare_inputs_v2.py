"""CPU-only identity join. Never imports Torch or executes a model/channel."""
from evidence_common import *
import csv,collections,subprocess
NL=B/'nestcom_n512_20261028_v1';AL=B/'adatoken_response_20261026_v1'
METRICS=['psnr_db','lpips_alex','dreamsim','clip_image_cosine']
SIZES=(1,2,3,4,5,6,8,10,13,16)
def csvread(p):return list(csv.DictReader(Path(p).open()))
def main():
 out=L/'input_inventory_v2';out.mkdir(parents=True,exist_ok=False);assert O.is_dir()
 def save_existing(p,v):
  if p.exists():assert read(p)==v
  else:save(p,v)
 inputs={};verified={};native_identities={};outputs={};jobs={};logical=[]
 def bind(p):
  d=pin(p);inputs[str(p)]=d;return read(p)
 manifest=bind(R/'results/arpc_external_baseline_20261019_v1/a3_source_v1/sources.json')
 bank=bind(R/'results/arpc_external_baseline_20261019_v1/a3_metrics_v1/bootstrap_identity.json')
 members={s['source_id']:s for s in manifest['new_sources']};order=bank['source_ids'];assert len(order)==100 and set(order)==set(members)
 checked(bank['indices']);ix=np.load(bank['indices']['path'],allow_pickle=False);assert ix.shape==(10000,100)and ix.min()==0 and ix.max()==99
 full=bind(NL/'formal_metric_plan/pairs.json');sources={s['source_id']:s for s in full['sources']};assert len(sources)==500
 # Preregister figure IDs before reading reconstruction files or metric tables.
 old_ids=['n07860988/ILSVRC2012_val_00010848_n07860988','n02105641/ILSVRC2012_val_00018762_n02105641']
 assert not set(old_ids)&set(sources),'A matching legacy ID requires its complete original pixel/preprocessing audit'
 common=bind(R/'results/arpc_common500_extension_20261019_v1/population.json')['common500']
 selected=common[:2];assert len({s['source_id']for s in selected})==2
 save_existing(O/'figure3_selection.json',dict(rule='Retain exact legacy identities if present; otherwise first unused source in original common500 manifest order. Both legacy full IDs absent.',legacy_ids=old_ids,selected=[{k:s[k]for k in ('source_id','source_index','preprocessing_id','archive')}for s in selected],N=512,snr_db=10,noise_seed=6201,selected_before_reconstruction_quality=True,original_manifest=inputs[str(R/'results/arpc_common500_extension_20261019_v1/population.json')]))
 source_rows=[]
 for sid in order+[s['source_id']for s in selected if s['source_id']not in members]:
  s=sources[sid];p=checked(s['archive'])
  if sid in members:
   old=members[sid];assert old['source_index']==s['source_index'] and old['preprocessing_id']==s['preprocessing_id'] and old['archive']==s['archive']
  with np.load(p,allow_pickle=False)as z:a=z['pixels'].copy()
  assert a.dtype==np.uint8 and a.shape==(3,256,256)and hashlib.sha256(a.tobytes()).hexdigest()==s['preprocessing_id']
  assert asha(a.astype(np.float32)/np.float32(255))==s['reference_sha256']
  source_rows.append({k:s[k]for k in ('source_id','source_index','preprocessing_id','archive','reference_sha256')})
 save_existing(O/'source_manifest.json',dict(population='external_common100_v1',source_order=order,sources=source_rows[:100],manifest=inputs[str(R/'results/arpc_external_baseline_20261019_v1/a3_source_v1/sources.json')],source_bank=dict(original_identity=inputs[str(R/'results/arpc_external_baseline_20261019_v1/a3_metrics_v1/bootstrap_identity.json')],indices=bank['indices'],source_ids=order,note='Only original resampling indices reused; comparisons are specified by this new protocol.')))
 def native(image,expected):
  if image is None:return None
  if image['path']not in verified:
   p=checked(image)
   with np.load(p,allow_pickle=False)as z:im=z['image'].copy();t=z['received_tokens'].copy()
   assert asha(im)==expected and im.dtype==np.float32 and im.shape==(3,256,256)
   side=p.with_suffix('.json')
   if side.exists():v=bind(side)
   else:
    role=p.parent.parent
    if str(role)not in outputs:outputs[str(role)]={x['image']['path']:x for x in bind(role/'outputs.json')}
    v=outputs[str(role)][str(p)]
   assert v['pixel_sha256']==expected and v['image']==image and asha(t)==v['received_tokens_sha256']
   role=p.parent.parent;ni=role/('numeric_identity.json'if v.get('codec')=='ADA' or 'received_class'in v else 'actual_native_identity.json')
   if not ni.exists():ni=role/'numeric_identity.json'
   identity=bind(ni);native_identities[str(ni)]=identity
   verified[image['path']]=dict(image=image,pixel_sha256=expected,received_tokens=t.tolist(),received_tokens_sha256=asha(t),record=v,identity=pin(ni),identity_value=identity)
  assert verified[image['path']]['pixel_sha256']==expected
  return verified[image['path']]
 phys=bind(NL/'formal_metric_plan/physical.json');frames=bind(NL/'formal_rx_plan/frames.json')
 tablepath=R/'results/nestcom_n512_20261028_v1/delivery/frame_scores.csv';inputs[str(tablepath)]=pin(tablepath)
 figure=[]
 for r in csvread(tablepath):
  sid=r['source_id'];snr=int(r['snr_db']);method=r['method'];isfig=sid in {s['source_id']for s in selected}and snr==10 and int(r['noise_seed'])==6201
  if not(isfig or(sid in members and snr in (4,10))):continue
  p=phys[r['physical_key']];assert p['pixel_sha256']==r['pixel_sha256'] and r['reference_sha256']==sources[sid]['reference_sha256']
  v=native(p['image'],r['pixel_sha256']);wire=cr(frames[r['physical_key']]);assert wire['job']['source_id']==sid and int(wire['job']['noise_seed'])==int(r['noise_seed'])and wire['received_key']==r['received_key']
  row=dict(N=512,snr_db=snr,source_id=sid,source_index=int(r['source_index']),noise_seed=int(r['noise_seed']),source_status=r['receiver_status'],fixed_gray=r['fixed_gray']=='True',image=p['image'],pixel_sha256=r['pixel_sha256'],reference_sha256=r['reference_sha256'],wire=frames[r['physical_key']],received_key=r['received_key'],metrics={k:float(r[k])for k in METRICS})
  if isfig:figure.append(dict(row,method=method))
  if sid not in members:continue
  if method in ('ADA','NEST'):
   logical.append(dict(row,experiment='e1',arm='Ada-sampling'if method=='ADA'else'NeST'))
   if method=='ADA':
    change=dict(row,experiment='e1',arm='Ada-argmax',reuse_kind='GRAY'if row['fixed_gray']else'FULL_PREFIX'if len(v['received_tokens'])==256 else'NEW_ARGMAX')
    if change['reuse_kind']=='NEW_ARGMAX':
     pr=wire['projection'];assert pr['codec']=='ADA' and pr['m']==len(v['received_tokens'])
     j=dict(kind='e1',received_tokens=v['received_tokens'],received_tokens_sha256=v['received_tokens_sha256'],received_class=pr['received_class'],generation_seed=v['record']['generation_seed'],numeric_identity=v['identity'],donor_image=v['image'],donor_pixel_sha256=v['pixel_sha256'],prefix=pr['m'])
     key=digest({k:(j[k]['sha256']if k=='numeric_identity'else j[k])for k in ('kind','received_tokens_sha256','received_class','generation_seed','numeric_identity')});jobs.setdefault(key,dict(j,key=key));change['new_key']=key
    logical.append(change)
  elif method=='WHOLE':
   logical.append(dict(row,experiment='e2',arm='R1-VAR-completed'))
   change=dict(row,experiment='e2',arm='R0-coarse-only',reuse_kind='GRAY'if row['fixed_gray']else'NEW_COARSE')
   if not row['fixed_gray']:
    pr=wire['projection'];assert pr['K']==0 and len(v['received_tokens'])==sum(x*x for x in SIZES[:pr['m']])
    j=dict(kind='e2',m=pr['m'],K=0,received_tokens=v['received_tokens'],received_tokens_sha256=v['received_tokens_sha256'],numeric_identity=v['identity'],donor_image=v['image'],donor_pixel_sha256=v['pixel_sha256'])
    key=digest({k:(j[k]['sha256']if k=='numeric_identity'else j[k])for k in ('kind','m','K','received_tokens_sha256','numeric_identity')});jobs.setdefault(key,dict(j,key=key));change['new_key']=key
   logical.append(change)
 # N1024: retain original noise identities. Join full IDs, never row offsets.
 ada_csv=R/'results/adatoken_response_20261026_v1/c_predclass_formal_delivery_v1/common500/frame_metrics.csv';inputs[str(ada_csv)]=pin(ada_csv)
 ada_frames={(x['source_id'],x['snr_db'],x['noise_seed']):x for x in bind(AL/'c_predclass_common500_rx_preparation/logical_frames.json')}
 ada_outputs={x['key']:x for role in('c_predclass_common500_rx_s00','c_predclass_common500_rx_s01')for x in bind(AL/role/'outputs.json')}
 for r in csvread(ada_csv):
  if r['source_id']not in members or int(r['snr_db'])!=10:continue
  sid=r['source_id'];f=ada_frames[sid,10,int(r['noise_seed'])];v=ada_outputs[f['RX_key']];verified_v=native(v['image'],r['pixel_sha256']);assert v['status']==r['source_status']
  row=dict(experiment='e1',N=1024,snr_db=10,source_id=sid,source_index=int(r['source_index']),noise_seed=int(r['noise_seed']),source_status=r['source_status'],fixed_gray=r['accepted']!='True',image=v['image'],pixel_sha256=r['pixel_sha256'],reference_sha256=sources[sid]['reference_sha256'],wire=f['frame'],received_key=f['RX_key'],metrics={k:float(r[k])for k in METRICS})
  logical.append(dict(row,arm='Ada-sampling'));j=dict(kind='e1',received_tokens=verified_v['received_tokens'],received_tokens_sha256=v['received_tokens_sha256'],received_class=v['received_class'],generation_seed=v['generation_seed'],numeric_identity=verified_v['identity'],donor_image=v['image'],donor_pixel_sha256=v['pixel_sha256'],prefix=v['prefix'])
  assert len(j['received_tokens'])==j['prefix'] and not row['fixed_gray']
  if j['prefix']==256:
   logical.append(dict(row,arm='Ada-argmax',reuse_kind='FULL_PREFIX'));continue
  key=digest({k:(j[k]['sha256']if k=='numeric_identity'else j[k])for k in ('kind','received_tokens_sha256','received_class','generation_seed','numeric_identity')});jobs.setdefault(key,dict(j,key=key));logical.append(dict(row,arm='Ada-argmax',new_key=key,reuse_kind='NEW_ARGMAX'))
 three_path=R/'results/n1024_consistency_20261019_v1/repair_v1/delivery/frame_scores_9000.csv';clip_path=B/'clip_i_common500_20261020_v1/score/frame_scores.csv'
 inputs[str(three_path)]=pin(three_path);inputs[str(clip_path)]=pin(clip_path)
 three={(r['source_id'],int(r['noise_seed'])):r for r in csvread(three_path)if r['population']=='common500'and r['family']=='PARTIAL'and int(r['snr_db'])==10}
 inventory=bind(B/'clip_i_common500_20261020_v1/inventory/frames.json');native_nest={(x['source_id'],int(x['noise_seed'])):x for x in inventory if x['method']=='EC_VAR_PARTIAL'and x['snr_db']==10}
 for r in csvread(clip_path):
  if r['method']!='EC_VAR_PARTIAL'or r['source_id']not in members or int(r['snr_db'])!=10:continue
  sid=r['source_id'];f=three[sid,int(r['noise_seed'])];v=native_nest[sid,int(r['noise_seed'])];assert f['image_sha256']==r['pixel_sha256']==v['pixel_sha256']and r['reference_sha256']==sources[sid]['reference_sha256']
  with np.load(checked(v['archive']),allow_pickle=False)as z:im=z[v['image_key']];im=im[v['image_slot']]if v['image_slot']is not None else im;assert asha(im)==r['pixel_sha256']
  logical.append(dict(experiment='e1',arm='NeST',N=1024,snr_db=10,source_id=sid,source_index=int(r['source_index']),noise_seed=int(r['noise_seed']),source_status=r['receiver_status'],fixed_gray=r['fixed_gray']=='True',image_locator=v,pixel_sha256=r['pixel_sha256'],reference_sha256=r['reference_sha256'],metrics={**{k:float(f[k])for k in METRICS[:3]},'clip_image_cosine':float(r['clip_image_cosine'])}))
 counts=collections.Counter((x['experiment'],x['arm'],x['N'],x['snr_db'])for x in logical)
 assert len(logical)==3900 and all(n==300 for n in counts.values()),counts
 for group in set(counts):
  rows=[x for x in logical if(x['experiment'],x['arm'],x['N'],x['snr_db'])==group];assert set(x['source_id']for x in rows)==set(order)and all(n==3 for n in collections.Counter(x['source_id']for x in rows).values())
 save(out/'logical.json',logical);save(out/'jobs.json',list(jobs.values()));save(out/'figure_frames.json',figure);save(out/'sources.json',source_rows)
 save(O/'cache_identity.json',dict(inputs=list(inputs.values()),verified_native_files=[{k:v[k]for k in ('image','pixel_sha256','received_tokens_sha256','identity')}for v in verified.values()],numeric_identities=native_identities,new_PHY=0,new_model_calls=0))
 summary=dict(status='INPUT_IDENTITY_COMPLETE',logical_counts=[dict(experiment=e,arm=a,N=n,snr_db=s,count=c)for(e,a,n,s),c in counts.items()],new_unique_by_kind=dict(collections.Counter(j['kind']for j in jobs.values())),native_receivers_verified=len(verified),figure_sources=[s['source_id']for s in selected],figure_frames=len(figure),bootstrap_indices=bank['indices'],new_scientific_calls=0)
 save(out/'completion.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
