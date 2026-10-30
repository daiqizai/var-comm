import sys,sqlite3
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for role in ['c_train_primary','fit_B','fit_E_phy_1024']:
 print(role)
 for p in [L/(role+'_control')/'actual_wait.json',L/role/'failure.json',L/role/'completion.json']:
  if p.exists():print(p.name,read(p))
 print((L/(role+'_control')/'worker.log').read_text()[-400:])
 db=L/role/'calls.sqlite'
 if db.exists():
  c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);print(c.execute('select kind,status,count(*)from calls group by kind,status').fetchall());c.close()
print('RESOLVED_KEYS',list(read(D/'protocol_resolved_p0.json')))
