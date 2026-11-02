"""Original quality implementation; no compiled timing substitution."""
from study_common import *
sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'))
import rx_worker_v2
rx_worker_v2.L=L
rx_worker_v2.main(*sys.argv[1:])
