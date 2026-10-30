from remote import *
code="""from pathlib import Path
import json
roots=[Path('/home/liulu/projects/VAR-MAP-GATE0/data/imagenet_hf'),Path('/home/liulu/projects/VAR-MAP-GATE0/data'),Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-data-index')]
for r in roots:
 print(str(r),r.exists(),[(p.name,p.is_dir())for p in r.iterdir()][:25]if r.is_dir()else[])
"""
print(t.run(code,'new200_main_inventory_'+str(time.time_ns()),host=MAIN,python=CPY,timeout=45).decode())
