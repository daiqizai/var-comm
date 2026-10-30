from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
for name in ['experiments/five-method-timing-20261020/native.py','experiments/five-method-timing-20261020/library_recovery_v1/adapter.py','experiments/entropy-partial-mismatch-20261010/scripts/ep_source_codec.py','experiments/kodak-current-20261026/nest_v1/kn_native.py']:
 p=R/name;s=p.read_text();lines=s.splitlines();print('SOURCE',name)
 if 'native.py'in name and 'kn_native'not in name:print('\n'.join(lines[:260]))
 elif 'adapter.py'in name:print('\n'.join(lines[:200]))
 else:
  for i,line in enumerate(lines):
   if 'def encode_endpoints'in line or 'def tokenizer'in line:print('\n'.join(lines[i:i+100]))
