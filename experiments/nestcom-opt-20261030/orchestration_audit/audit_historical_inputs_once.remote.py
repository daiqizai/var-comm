import sys,os
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from historical_regression_inputs import main
os.sched_setaffinity(0,{24,25})
rp=D/'historical_regression_inputs_registration.json'
save(rp,dict(sources=pin(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json'),output='historical_regression_inputs',code=pin(E/'historical_regression_inputs.py'),purpose='Read-only identity audit of historical formal records; no new scientific inference or final candidate evaluation'))
main(rp)
