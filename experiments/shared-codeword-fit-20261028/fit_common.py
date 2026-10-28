"""Finite independent study. Old NO_GO registrations are immutable."""
import sys
from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/nextscale-selective-blocks-20261028'))
from block_common import *
E=R/'experiments/shared-codeword-fit-20261028'
O=R/'results/shared_codeword_fit_20261028_v1'
L=B/'shared_codeword_fit_20261028_v1'
OLD=R/'results/nextscale_selective_blocks_20261028_v1'
BASE='2f13240b9338c8e79f14d6087e221714d3259c6b'
