"""Explicit one-time registration of previously unexecuted Ada RX work."""
from n512_common import *
from dispatch import launch
from stage_operations_v2 import budget

def main():
 stage='pilot';target=O/(stage+'_rx_pci_spelling_amendment.json');assert not target.exists()
 rows=read(L/(stage+'_rx_plan')/'completion.json')['worker_registrations'];missing=[]
 for d in rows:
  r=cr(d);role=Path(r['output']).name
  if r['codec']!='ADA':continue
  w=read(L/'controls'/role/'actual_wait.json');assert w['returncode']==1 and not w['timed_out']and gone(w['process'])and gone(w['owner'])
  f=read(L/role/'failure.json');assert f['ledger']['counts']==[]and f['ledger']['unresolved']==0
  assert "assert b.value.decode().lower()==reg['pci']"in f['traceback']and f['traceback'].endswith('AssertionError\n')
  assert not(L/role/'outputs.json').exists()and not list((L/role/'images').glob('*.npz'))
  assert r['pci']!=r['pci'].lower();missing.append((d,r,role))
 assert len(missing)==4;seconds,spent=budget(True,4);replacements={}
 for d,r,role in missing:
  new=dict(r);new.update(pci=r['pci'].lower(),output=str(L/(role+'_pci_v1')),max_seconds=seconds,GPU_device_seconds_already_spent=spent,replaces_zero_call_registration=d)
  new['code']=r['code']+[pin(E/'recover_zero_call_pci.py'),pin(E/'rx_lineage.py')]
  p=O/(role+'_pci_v1_registration.json');save(p,new);replacements[role]=pin(p)
 save(target,dict(scope='zero-model PCI spelling only',original_exits_preserved=1,original_model_calls=0,original_receive_calls=0,science_repeated=0,algorithm_changed=False,replacements=replacements,code=pin(Path(__file__))))
 for i,(d,r,role)in enumerate(missing):print(launch(role+'_pci_v1',Path(replacements[role]['path']),E/'rx_worker_v2.py',gpu=i+4),flush=True)

if __name__=='__main__':main()
