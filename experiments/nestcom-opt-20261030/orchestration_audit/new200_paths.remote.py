from pathlib import Path
import json
r=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-data-index')
for p in r.glob('*/*'):print('INDEX',p,p.is_dir(),p.stat().st_size)
p=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/assets/ep_new100_source_code_seed_v1/common500_source_assets.py');print(p.read_text()[:6500])
