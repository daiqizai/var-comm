"""Document original frozen policy fields absent in some scalar metric rows."""
from followup_common import *
from frame_plan import policy
def main():
 a=read(OLD/'protocol_resolved_p0.json');data=[]
 for N,snr in [(1024,x)for x in [4,7,10,19]]+[(512,x)for x in [4,10,19]]:
  for method in ['NEST','WHOLE']:
   v=policy(N,snr,method,a);data.append(dict(N=N,snr_db=snr,method=method,**{k:v[k]for k in ['target','q','rate']}))
 cats={str(n):cr(pin(OLD/f'grid_catalogues/N{n}.json'))for n in [512,1024]}
 counts={n:dict(profiles=len(c['profiles']),max_profile_id=max(p['profile_id']for p in c['profiles']),fits_12_bits=all(0<=p['profile_id']<4096 for p in c['profiles']))for n,c in cats.items()}
 assert all(v['fits_12_bits']for v in counts.values())
 save(D/'e1/frozen_policy_metadata.json',dict(policies=data,policy_files={k:a[k]for k in ['policy_N512','policy_N1024','policy_7db']},source=pin(OLD/'protocol_resolved_p0.json'),catalogue_counts=counts,new_model_calls=0,new_PHY=0,note='Metadata joins only. Original scalar E1 missing fields and files remain unchanged.'))
 print(json.dumps(data))
if __name__=='__main__':main()
