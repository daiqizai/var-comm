"""Namespace-only entry; reuse qualified numerical worker and proven child fix."""
from pathlib import Path
import sys,runpy
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/nestcom-perceptual-followup-20261030'
sys.path.insert(0,str(E))
from stdlib_statistics_v1 import bind_stdlib_statistics
bind_stdlib_statistics()
# Torch propagates sys.path to compiler subprocess PYTHONPATH. This existing
# sitecustomize binds stdlib statistics there before historical paths shadow it.
sys.path.insert(0,str(E/'arpc_child_stdlib_v1'))
runpy.run_path(str(Path(__file__).with_name('ada_paired_prefill_headers_v2.py')),run_name='__main__')
