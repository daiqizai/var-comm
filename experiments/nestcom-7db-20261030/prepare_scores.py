"""Join fixed development sources to independently received native arrays."""
from g import *
from dev_dispatch import dependency

def main():
 sources={s['source_id']:s for s in read(D/'development_sources.json')};outs={};jobs=[]
 for role in ['dev_rx_nest','dev_rx_ada','dev_rx_arpc']:
  dependency(role);done=read(L/role/'completion.json');assert done['ledger']['unresolved']==0
  for v in cr(done['outputs']):outs[v['key']]=v
 def source_fields(s):
  with np.load(checked(s['archive']),allow_pickle=False)as z:x=z['pixels'].astype(np.float32)/np.float32(255)
  return dict(source_id=s['source_id'],source_index=s['source_index'],source=s['archive'],source_array_sha256=asha(x),preprocessing_id=s['preprocessing_id'])
 for method in ['NEST','WHOLE','ADA','ARPC']:
  role='dev_phy_'+method.lower();dependency(role)
  frames=cr(read(L/role/'completion.json')['frames']);assert len(frames)==2
  for f in frames:
   s=sources[f['source_id']]
   if f['rx_key']:v=outs[f['rx_key']]
   else:
    p=L/('gray_'+method+'_%04d.npz'%s['source_index']);y=np.full((3,256,256),.5,np.float32)
    with p.open('xb')as file:np.savez(file,image=y)
    v=dict(image=pin(p),pixel_sha256=asha(y),status='PHY_REJECT')
   jobs.append(dict(**source_fields(s),method=method,image=v['image'],pixel_sha256=v['pixel_sha256'],noise_seed=f['noise_seed'],receive_status=v['status'],prefix=f['endpoint'],frame=pin(L/role/(f['key']+'.json'))))
 for method in ['BPG','P100000','Swin200000']:
  role='dev_'+method.lower()+'_rx';dependency(role)
  for v in cr(read(L/role/'completion.json')['outputs']):
   jobs.append(dict(**source_fields(sources[v['source_id']]),method=method,image=v['image'],pixel_sha256=v['pixel_sha256'],noise_seed=2001,receive_status=v['status'],prefix='BPG adaptive complete stream'if method=='BPG'else'continuous JSCC',frame=pin(L/role/('s7_i%04d.json'%v['source_index']))))
 assert len(jobs)==len({(j['source_id'],j['method'])for j in jobs})==14
 for j in jobs:
  with np.load(checked(j['image']),allow_pickle=False)as z:assert asha(z['image'])==j['pixel_sha256']
 save(D/'development_score_jobs.json',jobs);print('FOURTEEN_NATIVE_SCORE_JOBS_READY')
if __name__=='__main__':main()
