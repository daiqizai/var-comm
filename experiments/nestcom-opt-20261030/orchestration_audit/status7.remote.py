import sys,sqlite3
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for role in ['c_cache','fit_B']:
 print(role)
 for p in [L/(role+'_control')/'actual_wait.json',L/role/'failure.json',L/role/'completion.json']:
  if p.exists():print(p.name,read(p))
 print((L/(role+'_control')/'worker.log').read_text()[-250:])
 db=sqlite3.connect('file:'+str(L/role/'calls.sqlite')+'?mode=ro',uri=True);print(db.execute('SELECT kind,status,COUNT(*)FROM calls GROUP BY kind,status').fetchall());db.close()
