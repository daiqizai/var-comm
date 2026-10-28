"""Resolve only the four proven zero-model PCI spelling failures."""
from n512_common import *

def registrations(stage):
 rows=read(L/(stage+'_rx_plan')/'completion.json')['worker_registrations']
 amendment=O/(stage+'_rx_pci_spelling_amendment.json')
 if not amendment.exists():return rows
 a=read(amendment);assert stage=='pilot'and a['scope']=='zero-model PCI spelling only'
 result=[]
 for d in rows:
  r=cr(d);role=Path(r['output']).name
  if role not in a['replacements']:result.append(d);continue
  w=read(L/'controls'/role/'actual_wait.json');assert w['returncode']==1 and not w['timed_out']and gone(w['process'])and gone(w['owner'])
  f=read(L/role/'failure.json');assert f['ledger']['counts']==[]and f['ledger']['unresolved']==0
  assert "assert b.value.decode().lower()==reg['pci']"in f['traceback']and f['traceback'].endswith('AssertionError\n')
  p=a['replacements'][role];new=cr(p)
  assert new['inputs']==r['inputs']and new['caps']==r['caps']and new['pci']==r['pci'].lower()
  ignored={'code','output','pci','max_seconds','GPU_device_seconds_already_spent','replaces_zero_call_registration'}
  assert {k:v for k,v in new.items()if k not in ignored}=={k:v for k,v in r.items()if k not in ignored}
  assert new['replaces_zero_call_registration']==d;result.append(p)
 return result
