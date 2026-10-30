"""E3 receiver cache reuse requires actual decoded inputs and identical model route."""
from followup_common import *
from opt_common import OFFSETS
from prepare_e2_receivers import wait_for
def main():
 closed('E3_phy');closed('E2_nest');out=L/'E3_receiver_plan';out.mkdir(exist_ok=False)
 jobs={j['key']:j for j in read(L/'E3_phy/rx_inputs.json')};reuse={};lineage=[]
 donor=read(OL/'final_rx/actual_native_identity.json')
 candidates=[(L,'E2_nest'),(OL,'final_rx'),(B/'nestcom_7db_common500_20261030_v1','rx_nest')]
 candidates += [(B/'nestcom_n512_20261028_v1',f'{phase}_rx_nest_{i:02d}')for phase in ['formal','full','pilot']for i in range(4)]
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
 save(out/'jobs.json',list(jobs.values()));save(out/'reuse.json',reuse)
 save(out/'completion.json',dict(status='PREPARED',unique_inputs=len(jobs),reused=len(reuse),new=len(jobs)-len(reuse),lineage=lineage,new_model_calls=0,new_PHY=0));print({k:v for k,v in read(out/'completion.json').items()if k!='lineage'})
if __name__=='__main__':main()
