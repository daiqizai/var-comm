"""Original DINO scorer with explicit, hash-checked checkpoint storage relocation."""
from g import *
sys.path[:0]=[str(R/'experiments/entropy-partial-mismatch-20261010/scripts'),str(R/'tools')]
import h800_ep_pilot_metrics_v1 as metrics
from checkpoint_relocation_20261028 import metric_assets
metrics.bound_assets=lambda config:metric_assets(metrics,config)
import dino_worker as original
original.L=L
if __name__=='__main__':original.main(*sys.argv[1:])
