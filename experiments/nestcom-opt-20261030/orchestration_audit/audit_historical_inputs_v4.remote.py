import sys,os
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from historical_regression_inputs_v4 import main
os.sched_setaffinity(0,{24,25})
rp=D/'historical_regression_inputs_v4_registration.json'
save(rp,dict(sources=pin(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json'),output='historical_regression_inputs',code=pin(E/'historical_regression_inputs_v4.py'),predecessor=pin(D/'historical_regression_inputs_v3_registration.json'),revision='Read-only TX identity audit uses pinned original terminated source streams for all old 7dB payload hashes, including CRC-rejected frames; no RX repair. All prior read-only schema audits stopped before outputs with zero science calls. Method-native 2001/6201 repeat identities are preserved.'))
main(rp)
