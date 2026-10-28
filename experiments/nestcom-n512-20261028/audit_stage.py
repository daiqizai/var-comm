"""Truth is consulted only after all independent RX outputs are immutable."""
from n512_common import *
from n512_phy import SIZES
from pipeline_prepare import closed
from rx_lineage import registrations
from collections import Counter

def main(stage):
 assert stage in ('pilot','full','formal')
 root=L/(stage+'_post_output_audit');root.mkdir(exist_ok=False)
 plan=read(L/(stage+'_rx_plan')/'completion.json');rx={k:v['record']for k,v in cr(plan['reuse']).items()};waits=[]
 for p in registrations(stage):
  r=cr(p);role=Path(r['output']).name;waits.append(closed(role))
  assert read(L/role/'completion.json')['ledger']['unresolved']==0
  for row in read(L/role/'outputs.json'):rx[row['key']]=row
 sources=cr(read(O/'protocol.json')['sources']['common500'if stage=='formal'else'cal1000']);sm={s['source_id']:s for s in sources}
 truth={};verified={};audited=set();counts=Counter();errors=[]
 for pk,fp in cr(plan['frames']).items():
  f=cr(fp);s=sm[f['job']['source_id']];proj=f['projection'];rk=f['received_key'];assert digest(proj)==rk
  if proj['status']=='FIXED_GRAY':counts['PHY_'+proj['reason']]+=1;continue
  row=rx[rk];counts[row['status']]+=1
  if rk not in verified:
   with np.load(checked(row['image']),allow_pickle=False)as z:im=z['image'].copy();tokens=z['received_tokens'].copy()
   assert asha(im)==row['pixel_sha256']and asha(tokens)==row['received_tokens_sha256']
   assert not row['source_truth_used']and not row['TX_CDF_used']and im.dtype==np.float32 and im.shape==(3,256,256)and np.isfinite(im).all()
   if row['status']!='ACCEPTED_SOURCE':assert np.all(im==.5)
   verified[rk]=tokens
  if row['status']!='ACCEPTED_SOURCE':continue
  key=(s['source_id'],rk)
  if key in audited:continue
  codec=proj['codec'];tk=(s['source_id'],codec)
  if tk not in truth:
   if codec=='NEST':
    with np.load(checked(s['archive']),allow_pickle=False)as z:truth[tk]=z['tokens'].copy()
   else:truth[tk]=np.load(checked(s['ada_tokens']),allow_pickle=False)
  n=sum(v*v for v in SIZES[:proj['m']])+proj['K']if codec=='NEST'else proj['m']
  exact=bool(np.array_equal(verified[rk],truth[tk][:n]));cdf=True
  if codec=='ADA':cdf=row['cdf_hashes']==cr(s['ada_cdf_hashes'])[:n]
  if not(exact and cdf):errors.append(dict(source_id=s['source_id'],received_key=rk,physical_key=pk,token_exact=exact,cdf_exact=cdf))
  audited.add(key)
 result=dict(status='PASS'if not errors else'BLOCKED_TOKEN_OR_CDF_MISMATCH',stage=stage,physical_frames=sum(counts.values()),status_counts=dict(counts),unique_saved_outputs=len(verified),accepted_source_stream_checks=len(audited),errors=errors,actual_waits=waits,RX_truth_used=False,truth_access='post-output only',new_model_calls=0,new_PHY=0,code=pin(Path(__file__)),frames=plan['frames'])
 save(root/'audit.json',result);save(O/(stage+'_post_output_audit.json'),result);print(json.dumps(result));assert not errors

if __name__=='__main__':main(sys.argv[1])
