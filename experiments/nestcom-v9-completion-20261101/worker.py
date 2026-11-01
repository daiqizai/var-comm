"""V9 isolated entry: reuse immutable engines in a new bounded namespace."""
from pathlib import Path
import importlib, sys
R = Path(__file__).resolve().parents[2]
L = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/nestcom_v9_completion_20261101_v1')
sys.path.insert(0, str(R/'experiments/nestcom-opt-20261030'))
role = sys.argv[1]
assert role in ('grid_source', 'extended_frame_worker_v3', 'original_receiver', 'metrics', 'dists_worker')
if role == 'dists_worker':
    sys.path.insert(0, str(R/'experiments/nestcom-perceptual-followup-20261030'))
engine = importlib.import_module(role)
engine.L = L  # Private process only; historical files and controllers unchanged.
engine.main(*sys.argv[2:])
