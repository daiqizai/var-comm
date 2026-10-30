import sys,sqlite3
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
a=read(D/'protocol_resolved_p0.json');print('NOISE',a['noise_seeds']);print('BSEED',a['B_generation_seed_recipe'])
for role in ['c_train_primary','fit_B']:
 print(role,(L/(role+'_control')/'worker.log').read_text()[-250:])
 p=L/role/'failure.json'
 if p.exists():print(read(p))
 db=sqlite3.connect('file:'+str(L/role/'calls.sqlite')+'?mode=ro',uri=True);print(db.execute('SELECT kind,status,COUNT(*)FROM calls GROUP BY kind,status').fetchall());db.close()
