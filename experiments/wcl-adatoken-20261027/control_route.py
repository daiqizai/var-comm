"""Reuse the frozen finite process controller in the independent output root."""
from pathlib import Path
import sys
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'))
import control
control.L=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/wcl_adatoken_20261027_v1')
if __name__=='__main__':control.run(sys.argv[1],sys.argv[2],sys.argv[3])
