"""Exact publication route; retain original CPU supervisor and limits."""
import importlib.util
import pathlib
import sys
E=pathlib.Path(__file__).parent.parent
sys.path.insert(0,str(E))
from a1_owner import desc
path=E/'final_cpu_execution_v1/execute.py'
assert desc(path)['sha256']=='2e1156408f607102de1d22567811d2f275d8d6de24682ad73e7ed53989a130c3'
assert sys.argv[1]=='publication'
spec=importlib.util.spec_from_file_location('frozen_final_cpu_execution',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROLES['publication']=('final_delivery_route_v1/publish.py',2400)
if __name__=='__main__':module.main()
