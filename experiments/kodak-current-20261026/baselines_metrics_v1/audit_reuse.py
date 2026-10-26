"""CPU-only native-array and existing-score identity audit; no inference."""
from pathlib import Path
import csv,hashlib,json,os,time,traceback
import numpy as np
from PIL import Image
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
O=R/'results/kodak_current_20261026_v1/baselines_metrics_v1'
L=B/'kodak_current_20261026_v1/baselines_metrics_v1'
PINS={
 'results/kodak_current_20261026_v1/population.json':'a1d255e7491a8ae7303ccdf481a4abf48ff2133e9b03c56da8b756af657a941f',
 'results/kodak_current_20261026_v1/scope_registration.json':'3b5fe30a7d56acc4564beaa6dfaa75745d691def12df5ce63629875989d93b20',
 'results/generalization_kodak_final_20261013/frames.json':'8bc7fca5a25c6f5bc9c0b25cd058e3c56b1435d999669936ee615ae8854f40fc',
 'results/generalization_kodak_20261009/evidence/methods/BPG_ADAPTIVE/frames.json':'626f9246877a2d00284aa163070c3ec13fa2591df5b46776d6acb3343873adf3',
 'results/generalization_kodak_final_20261013/metrics_final_v1/rows.csv':'c0aaf249e0e38be39d4a1d6ae02ab5c18f2754dade91493d11763dc8ce9c4136',
 'results/generalization_kodak_20261009/metrics_v1/rows.csv':'8d5287944435afc16614e46bc5c566c19858aad48e6d81522347e78f2afdb098',
 'results/generalization_kodak_final_20261013/anchor_reuse_audit.json':'babd39302c7813132fb8e966c6fb07e582d52c10209d72765d6c979fcdb1e11d',
 'results/generalization_kodak_final_20261013/metrics_final_v1/metric_math_identity_proof.json':'d9bc21811de74fd4810ece3fae9cec7500e2a7000d561f9ef1268ecc3bd07c2f'}
CHECKPOINTS={'P100K':'059888fec9fb1d3548e2b3c34d681030d3df2882163c26d5b59c69c96dcdc4d6','SWIN200K':'572d7acaceed2712d885ebd351f65ef6042f7716082d81f63bf45fbf8aadbd45'}
METHODS={'P100K':'P100000','SWIN200K':'Swin200000','BPG_ADAPTIVE':'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+np.ascontiguousarray(x).tobytes()).hexdigest()
def desc(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def key(r):return (r['method'],int(r['source_index']),int(r['snr_db']),int(r['noise_seed']))
def main():
 out=L/'reuse_audit_v1';out.mkdir(parents=True,exist_ok=False);O.mkdir(parents=True,exist_ok=True)
 save(out/'launch.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],time=time.time(),scientific_calls=0))
 try:
  for rel,h in PINS.items():assert sha(R/rel)==h,rel
  pop=read(R/'results/kodak_current_20261026_v1/population.json');sources={int(s['source_index']):s for s in pop['sources']};assert set(sources)==set(range(24))
  refs={}
  for i,s in sources.items():
   p=Path(s['png']['path']);assert sha(p)==s['png']['sha256']
   x=np.ascontiguousarray(np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8).transpose(2,0,1))
   assert x.shape==(3,256,256) and hashlib.sha256(x.tobytes()).hexdigest()==s['pixels_chw_uint8_sha256']
   refs[i]=x.astype(np.float32)/np.float32(255)
  restore=L/'restored_assets_v1/restore_manifest.json';assert sha(restore)=='c3d9c93694f2c9f66c9aa5bbafc4a7a6e620d82fc2eb6e2bfc208d6764fc87cf'
  restored=read(restore)['objects'];frames=read(R/'results/generalization_kodak_final_20261013/frames.json')+read(R/'results/generalization_kodak_20261009/evidence/methods/BPG_ADAPTIVE/frames.json')
  oldscores={}
  for rel in ['results/generalization_kodak_final_20261013/metrics_final_v1/rows.csv','results/generalization_kodak_20261009/metrics_v1/rows.csv']:
   for row in csv.DictReader((R/rel).open()):
    if row['method'] not in METHODS or (row['method']!='BPG_ADAPTIVE' and 'final_' not in rel):continue
    k=key(row)+(row['metric'],);assert k not in oldscores;oldscores[k]=row
  expected={(m,i,s,n) for m in METHODS for i in range(24) for s in (4,10,19) for n in (2001,2002,2003)}
  assert {key(f) for f in frames}==expected and len(frames)==648
  verified={};rows=[];evidence=[]
  for f in frames:
   k=key(f);i=k[1];s=sources[i];assert f['source_id']==s['source_id'] and f['preprocessing_id']==s['preprocessing_id']
   d=f['reconstruction'];p=Path(d['path']) if f['method']!='BPG_ADAPTIVE' else Path(restored[d['sha256']]['path'])
   if str(p) not in verified:
    assert sha(p)==d['sha256'],str(p)
    with np.load(p,allow_pickle=False) as z:
     y=z['rgb'];assert y.dtype==np.float32 and y.shape==(3,256,256) and np.isfinite(y).all() and y.min()>=0 and y.max()<=1
     h=ah(y)
     if f['method']=='BPG_ADAPTIVE':assert np.array_equal(z['source_rgb'],refs[i])
     else:
      for name,target in [('wave','wave_sha256'),('observed','observation_sha256'),('standard_noise','standard_noise_sha256')]:assert ah(z[name])==f[target],target
    verified[str(p)]=h
   h=verified[str(p)]
   if f['method'] in CHECKPOINTS:
    assert f['checkpoint_sha256']==CHECKPOINTS[f['method']] and f['pixels_chw_uint8_sha256']==s['pixels_chw_uint8_sha256'] and f['reconstruction_sha256']==h
    assert f['N']==1024 and not f['source_or_hidden_TX_information_to_receiver']
   else:assert f['frozen_method_identity']==METHODS['BPG_ADAPTIVE'] and f['profile_id']=={4:3002,10:3007,19:3013}[k[2]]
   vals={}
   for metric in ('psnr_db','lpips_alex'):
    old=oldscores[k+(metric,)]
    assert old['source_id']==s['source_id'] and old['reference_sha256']==ah(refs[i]) and old['reconstruction_archive_sha256']==d['sha256'] and old['reconstruction_sha256']==h
    assert old['status']==f['status'] and old['N']=='1024' and old['preprocessing_id']==s['preprocessing_id']
    assert np.isfinite(float(old['value']));vals[metric]=old['value']
   rows.append(dict(method=METHODS[f['method']],source_index=i,source_id=s['source_id'],snr_db=k[2],noise_seed=k[3],rx_status=f['status'],image_path=str(p),image_key='rgb',image_file_sha256=d['sha256'],image_array_sha256=h,array_hash_convention='dtype_shape_cbytes',reference_array_sha256=ah(refs[i]),existing_psnr_db=vals['psnr_db'],existing_lpips_alex=vals['lpips_alex'],original_method=f['method'],original_image_path=d['path'],reuse='existing_closed_native_and_scores'))
   evidence.append(dict(key=list(k),file=desc(p),canonical_array_sha256=h,old_receiver_record=f,old_score_psnr=oldscores[k+('psnr_db',)],old_score_lpips=oldscores[k+('lpips_alex',)]))
  target=O/'frame_manifest.csv'
  with target.open('x',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);fp.flush();os.fsync(fp.fileno())
  save(out/'array_and_score_identity.json',evidence)
  save(O/'reuse_audit.json',dict(status='COMPLETE',rows=648,unique_native_archives=len(verified),per_method={v:216 for v in METHODS.values()},existing_psnr_reused=648,existing_lpips_reused=648,new_channel_events=0,new_decodes=0,new_reconstructions=0,new_metric_calls=0,frame_manifest=desc(target),evidence=desc(out/'array_and_score_identity.json'),parent=desc(R/'results/kodak_current_20261026_v1/scope_registration.json'),population=desc(R/'results/kodak_current_20261026_v1/population.json'),input_pins=PINS,restore=desc(restore),checkpoints=CHECKPOINTS,bpg_native_and_scoring_hashes='Different serializers; file SHA and canonical score hash both verified',noise_pairing='Source pairing only; method noise namespaces remain historical'))
  save(out/'completion.json',dict(status='COMPLETE',audit=desc(O/'reuse_audit.json'),scientific_calls=0))
  print(json.dumps(read(O/'reuse_audit.json')),flush=True)
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),scientific_calls=0));raise
if __name__=='__main__':main()
