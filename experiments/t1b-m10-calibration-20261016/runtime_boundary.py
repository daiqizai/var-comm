"""Exact read-only frozen Python library roots; source assets remain forbidden."""
import os,sys
from pathlib import Path
RT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010')
LIBRARIES=tuple(RT/p for p in(
 'assets/frozen_namespace_v1/VAR_COMM/outputs/UNIFIED-METRICS-20261002/environment/lib/python3.10/site-packages',
 'assets/frozen_namespace_v1/external/home/liulu/.local/lib/python3.10/site-packages',
 'assets/frozen_namespace_v1/external/usr/lib/python3/dist-packages'))
def library_read(path,mode,flags):
 p=Path(os.path.abspath(os.fsdecode(path)))
 if not any(p.is_relative_to(r)for r in LIBRARIES):return False
 if isinstance(mode,str)and any(c in mode for c in('w','a','x','+')):return False
 if isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):return False
 # Resolve aliases before allowing access: no symlink escapes to scientific data.
 real=p.resolve();return any(real.is_relative_to(r.resolve())for r in LIBRARIES)
def firewall(out,allowed,sensitive):
 allowed={str(Path(p).absolute())for p in allowed};out=str(Path(out).absolute())+'/'
 def audit(event,args):
  if event!='open' or not args or isinstance(args[0],int):return
  p=os.path.abspath(os.fsdecode(os.fspath(args[0])))
  if p.startswith(out):return
  if library_read(p,args[1]if len(args)>1 else None,args[2]if len(args)>2 else None):return
  if any(p==str(x)or p.startswith(str(x)+'/')for x in sensitive)and p not in allowed:raise PermissionError('RX truth/storage boundary: '+p)
 sys.addaudithook(audit)
