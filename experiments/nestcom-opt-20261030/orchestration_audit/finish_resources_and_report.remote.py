import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
role='final_unlaunched_metrics_v2';w=read(L/(role+'_control/actual_wait.json'));o=read(L/(role+'_control/owner.json'));assert w['returncode']==0 and ident(w['child']['pid'])is None and ident(o['owner']['pid'])is None
assert read(D/'final_common500_delivery/statistics_identity.json')['status']=='COMPLETE'
from resource_delivery_v2 import main as resources
resources()
from build_delivery_v3 import main as report
report()
from validate_delivery import main as validate
validate()
