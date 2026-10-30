from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
for name in ['tools/verify_repository.py','tools/run_cpu_checks.py']:
 p=R/name;print(name,p.read_text()[:14000])