import sys,os
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from historical_regression_inputs_v3 import main
os.sched_setaffinity(0,{24,25})
rp=D/'historical_regression_inputs_v3_registration.json'
save(rp,dict(sources=pin(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json'),output='historical_regression_inputs',code=pin(E/'historical_regression_inputs_v3.py'),predecessor=pin(D/'historical_regression_inputs_v2_registration.json'),revision='Read-only audit also recognizes the historical 7dB literal FIXED_GRAY image-path sentinel. Both prior audit attempts stopped before output creation and made zero model/metric/PHY calls; all rejection rows and original hashes retained.'))
main(rp)
