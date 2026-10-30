from pathlib import Path
import json,hashlib,numpy as np,subprocess,csv,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm');L=B/'nestcom_opt_20261030_v1';D=R/'results/nestcom_opt_20261030_v1'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for a in iter(lambda:f.read(8<<20),b''):h.update(a)
 return h.hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(v,f,indent=2)
def asha(a):return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='9d55aacfcd740034ce85014ae65f20a33893b59c'
assert not L.exists() and not D.exists()
L.mkdir();D.mkdir();(L/'inputs').mkdir()
scale=read(R/'results/progressive_rd_scaleup_20261022_v1/registration.json');train=read(B/'progressive_rd_scaleup_20261022_v1/train_index.json')['sources']
parent=read('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/baseline_strength_20261011/parents/p/registration.json')
trainids=set(parent['train_ids']);assert len(trainids)==20000 and len(train)==19968
assert all(x['source_id']in trainids for x in train)
selected=sorted(train,key=lambda x:hashlib.sha256(('nestcom_opt_train_20261030_v1|'+x['source_id']).encode()).digest())[:10000]
assert len(selected)==10000
cal=read(B/'nestcom_n512_20261028_v1/qualified_sources/cal1000.json');common=read(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json')
exclude={'old_decoder_pilot100':set(scale['prior_calibration100_ids']),'scaleup_validation200':{x['source_id']for x in scale['validation_sources']},'7db_calibration100':{x['source_id']for x in read(R/'results/nestcom_7db_20261030_v1/calibration_sources.json')},'old32':{x['source_id']for x in read(R/'results/nextscale_selective_blocks_20261028_v1/source_manifest.json')},'training':trainids}
dev=read(R/'results/nestcom_7db_20261030_v1/development_sources.json');exclude['display']={x['source_id']for x in dev}
exc=set().union(*exclude.values());eligible=[x for x in cal if x['source_id']not in exc]
trainhashes={x['source_pixels_sha256']for x in train};excludedhashes={x['preprocessing_id']for x in cal+common if x['source_id']in exc}|trainhashes
eligible=[x for x in eligible if x['preprocessing_id']not in excludedhashes]
eligible.sort(key=lambda x:hashlib.sha256(('nestcom_opt_split_20261030_v1|'+x['source_id']).encode()).digest());assert len(eligible)>=200
fit,confirm=eligible[:100],eligible[100:200]
aud=[]
for x in cal+common:
 p=Path(x['archive']['path']);assert sha(p)==x['archive']['sha256']
 with np.load(p,allow_pickle=False)as z:a=np.ascontiguousarray(z['pixels'])
 assert a.dtype==np.uint8 and a.shape==(3,256,256)
 assert hashlib.sha256(a.tobytes()).hexdigest()==x['preprocessing_id']
 assert asha(a.astype(np.float32)/np.float32(255))==x['reference_sha256']
 aud.append(dict(source_id=x['source_id'],source_index=x['source_index'],native_uint8_sha256=x['preprocessing_id'],native_float_sha256=x['reference_sha256'],archive=x['archive']))
save(L/'inputs/train10000.json',selected);save(L/'inputs/fit100.json',fit);save(L/'inputs/confirm100.json',confirm)
save(D/'source_manifest.json',dict(train=pin(L/'inputs/train10000.json'),FIT=pin(L/'inputs/fit100.json'),CONFIRM=pin(L/'inputs/confirm100.json'),eligible_calibration_count=len(eligible),exclusions={k:sorted(v)for k,v in exclude.items()},native_source_audit=aud,original_training_ids_count=len(trainids),new200_status='PREPARED_SEARCH_PENDING_NO_EVALUATION'))
print('SPLIT',len(fit),len(confirm),len(eligible),{k:len(v)for k,v in exclude.items()})
# Predetermined replay donors: select by receiver-input key, never quality.
N=B/'nestcom_n512_20261028_v1';projections={};outs={}
for plan in ['route_qualification_preparation','qualification_rx_preparation','pilot_rx_plan','full_rx_plan','formal_rx_plan']:
 for p in (N/plan).glob('*.json'):
  v=read(p)
  if isinstance(v,list):
   for x in v:
    if isinstance(x,dict) and 'projection'in x and x.get('key'):projections.setdefault(x['key'],(x,p))
for p in N.glob('*/outputs.json'):
 if any(s in p.parent.name for s in ['ada','metric']):continue
 v=read(p)
 if isinstance(v,list):
  for x in v:
   if x.get('codec')=='NEST':outs.setdefault(x['key'],(x,p))
donors={}
for k in sorted(projections):
 if k not in outs:continue
 x,ip=projections[k];o,op=outs[k];p=x['projection']
 typ='gray'if p['status']!='SOURCE_INPUT' else ('full'if p['m']==10 else ('partial'if p['K'] else 'whole'))
 if typ in donors:continue
 assert sha(o['image']['path'])==o['image']['sha256']
 with np.load(o['image']['path'],allow_pickle=False)as z:a=z['image'];t=z['received_tokens']
 assert asha(a)==o['pixel_sha256'] and asha(t)==o['received_tokens_sha256']
 donors[typ]=dict(input=x,donor=o,input_file=pin(ip),output_file=pin(op))
save(L/'inputs/p0_donors.json',donors);print('P0_DONORS', {k:dict(key=v['input']['key'],m=v['input']['projection'].get('m'),K=v['input']['projection'].get('K'),origin=v['donor']['image']['path'])for k,v in donors.items()})
save(D/'preparation_receipt.json',dict(created_unix=time.time(),source_manifest=pin(D/'source_manifest.json'),donors=pin(L/'inputs/p0_donors.json'),scientific_calls=0))
