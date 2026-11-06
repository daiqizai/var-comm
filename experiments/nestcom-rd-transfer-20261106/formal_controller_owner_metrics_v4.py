"""Actual outer wait for the sole adaptive-resource post-training test chain."""
import json,time
from runtime import *
setup()
from owned_process import run_role
F=L/'formal500_v1'
r=json.loads((F/'protocol_metrics_v4.json').read_text())
run_role(F/'outer_roles_metrics_v4','chain',[PY,str(S/'formal_chain_metrics_v4.py')],env(),[20,22],[],r['deadline']-time.time(),r['bindings'])
