"""Freeze E decisions before running the corresponding real channel events."""
from opt_common import *

def main():
 w=read(L/'fit_E_score_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
 selections=read(L/'fit_E_score/selection.json');original=read(L/'fit_frame_plan/tx_candidates.json');tx={(r['source_id'],r['N'],r['snr_db']):r for r in original}
 base={(r['source_id'],r['N'],r['snr_db']):r for r in read(L/'fit_frame_plan/logical.json')if r['method']=='NEST'}
 a=read(D/'protocol_resolved_p0.json');cats={7:read(B/'nestcom_7db_20261030_v1/cal_phy_nest/catalogue.json'),10:cr(a['catalogue_N1024'])}
 old={v['physical_key']:v['frame']for v in read(L/'fit_phy_1024/frames.json')};oldphysical={x['key']:x for x in read(L/'fit_frame_plan/physical_1024.json')}
 out=L/'fit_E_frame_plan';out.mkdir(exist_ok=False);physical={};logical=[]
 for s in selections:
  key=(s['source_id'],s['N'],s['snr_db']);v=tx[key]['candidates'][s['selected_rank']];assert v['endpoint']==s['endpoint']
  b=base[key];m,K=v['endpoint'];p=next(p for p in cats[s['snr_db']]['profiles']if(p['family'],p['m'],p['K'],p['q'],p['nominal_rate'])==(v['family'],m,K,b['q'],b['rate']))
  counter=17000000+4096*s['source_index']+p['profile_id']if s['snr_db']==7 else([1,4,7,10,13,19].index(s['snr_db'])*1000+s['source_index'])*3
  bits=np.asarray(v['payload'],np.uint8);identity=dict(N=s['N'],source_id=s['source_id'],source_index=s['source_index'],snr_db=s['snr_db'],noise_seed=4101,counter=counter,profile_id=p['profile_id'],payload_sha256=asha(bits));pk=digest(identity)
  physical.setdefault(pk,dict(key=pk,**identity,payload=bits.tolist(),profile=p,old=old.get(pk),public_base=counter-p['profile_id']if s['snr_db']==7 else None))
  # Baseline actual event is reused as well, with its full event proof.
  if b['physical_key']not in physical:physical[b['physical_key']]=dict(oldphysical[b['physical_key']],old=old[b['physical_key']])
  logical.append(dict(b,variant='original'))
  logical.append(dict(b,variant='E',physical_key=pk,endpoint=[m,K],source_bits=len(bits),utilization=len(bits)/b['capacity_bits'],selected_rank=s['selected_rank']))
 save(out/'physical_1024.json',list(physical.values()));save(out/'logical.json',logical)
 save(out/'completion.json',dict(status='PREPARED',selection=pin(L/'fit_E_score/selection.json'),selection_before_new_channel=True,logical=len(logical),unique_physical=len(physical),old_exact_candidates=sum(bool(v['old'])for v in physical.values())))
 print(read(out/'completion.json'))
if __name__=='__main__':main()
