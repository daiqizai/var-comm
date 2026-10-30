import sys,ast
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
p=B/'nestcom_n512_20261028_v1/phy_qualification/completion.json';print('N512QUAL',read(p))
for rel in ['experiments/nestcom-7db-common500-20261030/phy.py','experiments/t1c-m10-test-20261016/tc_plan.py']:
 p=R/rel
 if not p.exists():print('MISSING',p);continue
 text=p.read_text();print('FILE',rel)
 for n,line in enumerate(text.splitlines(),1):
  if any(k in line for k in ['counter','base','seed','620','410','plan','physical']):print(n,line)
print('T1C_FILES',[p.name for p in(R/'experiments/t1c-m10-test-20261016').glob('*.py')])
print('7REGS',[p.name for p in(R/'results/nestcom_7db_common500_20261030_v1').glob('*registration*')])
