"""New two-source 4 dB registration, unchanged production Ada receiver."""
from pathlib import Path
import sys,importlib.util
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/adatoken-response-20261026/c_predclass_production_rx_v1'
sys.path.insert(0,str(E));sys.path.insert(0,str(E.parent))
spec=importlib.util.spec_from_file_location('bound_ada_receiver',E/'worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.L=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/nestcom_fig3_4db_20261029_v1')
if __name__=='__main__':m.main(*sys.argv[1:])
