"""One finite dependent science chain. Unknown failures stop new launches."""
from g import *
from dispatch import dispatch
def wait(roles):
 while True:
  ready=True
  for role in roles:
   c=L/(role+'_control')
   if (c/'failure.json').exists()or(L/role/'failure.json').exists():raise RuntimeError('Preserved failure '+role)
   if not(c/'actual_wait.json').exists():ready=False;continue
   w=read(c/'actual_wait.json');assert w['returncode']==0
   if not gone(read(c/'started.json')['process']):ready=False
  if ready:
   for role in roles:closed(role)
   return
  assert not(L/'STOP').exists();time.sleep(10)
def main():
 out=L/'chain';out.mkdir();save(out/'owner.json',dict(process=ident(os.getpid()),protocol=pin(D/'protocol.json')))
 try:
  # Only the six original finite first stages; each has an exclusive role claim.
  for method in ('NEST','ADA','ARPC'):print(dispatch('phy',method),flush=True)
  for method in ('BPG','P100000','Swin200000'):print(dispatch('aux',method,'tx'),flush=True)
  wait(['phy_nest','phy_ada','phy_arpc'])
  for method in ('NEST','ADA','ARPC'):print(dispatch('rx',method),flush=True)
  for method in ('P100000','Swin200000'):
   wait([method.lower()+'_tx']);print(dispatch('aux',method,'rx'),flush=True)
  wait(['bpg_tx']);print(dispatch('aux','BPG','phy'),flush=True)
  wait(['bpg_phy']);print(dispatch('aux','BPG','rx'),flush=True)
  wait(['rx_nest','rx_ada','rx_arpc','p100000_rx','swin200000_rx','bpg_rx'])
  save(out/'completion.json',dict(status='ALL_SIX_RECEIVERS_CLOSED',protocol=pin(D/'protocol.json'),next='Independent metrics preparation, scoring and summary; no science relaunch.'))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
