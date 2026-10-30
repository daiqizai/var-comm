import sys,ast
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for rel in ['experiments/t1c-m10-test-20261016/prepare_link_v2.py','experiments/t1c-m10-test-20261016/tc_protocol.py','experiments/t1c-m10-test-20261016/tc_phy.py']:
 p=R/rel;s=p.read_text();print('FILE',rel)
 for n,line in enumerate(s.splitlines(),1):
  if any(k in line for k in ['counter','public','620','410','NOISE','noise_seed']):print(n,line)
