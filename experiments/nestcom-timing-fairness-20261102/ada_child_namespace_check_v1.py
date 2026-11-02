"""Import-only qualification of the actual Torch2.11 compiler child process."""
from pathlib import Path
import sys,os,json
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/nestcom-perceptual-followup-20261030'
sys.path.insert(0,str(E))
from stdlib_statistics_v1 import bind_stdlib_statistics
identity=bind_stdlib_statistics()
sys.path.insert(0,str(R/'experiments/five-method-timing-20261020'))
import native
native.paths()
sys.path.insert(0,str(E/'arpc_child_stdlib_v1'))
import torch
assert torch.__version__=='2.11.0+cu128'
from torch._inductor.compile_worker.subproc_pool import SubprocPool
from arpc_child_namespace_probe_v1 import probe
pool=SubprocPool(1)
try:
    child=pool.submit(probe).result(timeout=90)
    assert child['pid']!=os.getpid()
finally:
    pool.shutdown()
print(json.dumps(dict(status='PASS',torch=torch.__version__,parent_statistics=identity,
    actual_compiler_child=child,model_calls=0,PHY_calls=0,timing_calls=0,GPU_calls=0)))
