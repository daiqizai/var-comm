import sys,os
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from historical_regression_inputs_v5 import main
os.sched_setaffinity(0,{24,25})
rp=D/'historical_regression_inputs_v5_registration.json'
save(rp,dict(sources=pin(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json'),output='historical_regression_inputs',code=pin(E/'historical_regression_inputs_v5.py'),predecessor=pin(D/'historical_regression_inputs_v4_registration.json'),revision='Read-only audit joins the actual registered 7dB method ID ADA (not its presentation label). All prior schema audits stopped before outputs with zero science. Source identity, native arrays, full denominators and native noise IDs preserved.'))
main(rp)
