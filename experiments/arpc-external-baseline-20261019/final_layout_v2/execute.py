"""Original finite CPU supervisor with explicit deterministic layout revision."""
import importlib.util
import pathlib
import sys
E=pathlib.Path(__file__).parent.parent
sys.path.insert(0,str(E))
from a1_owner import desc
path=E/'final_cpu_execution_v1/execute.py'
assert desc(path)['sha256']=='2e1156408f607102de1d22567811d2f275d8d6de24682ad73e7ed53989a130c3'
spec=importlib.util.spec_from_file_location('frozen_final_cpu_execution',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROLES={'figure_revision':('final_layout_v2/render.py',600),
              'report':('final_layout_v2/report.py',300),
              'publication':('final_layout_v2/publish.py',2400)}
if __name__=='__main__':module.main()
