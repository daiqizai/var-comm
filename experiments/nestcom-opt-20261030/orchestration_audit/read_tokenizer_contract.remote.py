from pathlib import Path
p=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/content-real-64qam-20261006/h_online_components.py');s=p.read_text().splitlines()
for i,l in enumerate(s):
 if l.startswith('def native_encoder'):print('\n'.join(s[i:i+60]));break
