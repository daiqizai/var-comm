"""Publish the exact metadata-name repair alongside original final delivery."""
import importlib.util
import pathlib
import sys
E=pathlib.Path(__file__).parent.parent
sys.path.insert(0,str(E))
from a1_owner import desc
path=E/'final_publication_v1/publish.py'
assert desc(path)['sha256']=='aec83acc71aee6016078ba84a11b90b1dc2519511c03901c5b0b1ae9c5cfdae8'
spec=importlib.util.spec_from_file_location('frozen_final_publication',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.CODE_DIRS += ['a3_metric_names_v1','final_delivery_route_v1']
module.SCOPES += ['a3_metric_names_qualify_control_v1','a3_metric_names_prepare_control_v1']
if __name__=='__main__':module.main()
