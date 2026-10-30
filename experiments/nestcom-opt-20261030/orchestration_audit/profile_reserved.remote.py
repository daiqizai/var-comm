from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
p=R/'experiments/arpc-external-baseline-20261019/phy_v1/link.py';t=p.read_text();print(t[:10000])
for p in (R/'experiments').glob('**/*bpg*'):
 if p.is_file()and p.suffix=='.py':
  lines=p.read_text(errors='ignore').splitlines();hits=[f'{i+1}:{s}'for i,s in enumerate(lines)if ('profile_id'in s or '3000'in s)];
  if hits:print(str(p),'\n'.join(hits[:10]))
