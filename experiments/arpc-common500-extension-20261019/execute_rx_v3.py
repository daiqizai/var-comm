"""Explicit preparation-v3 routing; original RX run and dispatch directory."""
import hashlib
from common import *
import execute_rx_v2 as prior
PRIOR_SHA='70bdfd0213aede98748b5e17c503a5b0ed82ae7e0ce0a247d2915f242b244a86'
def adapt(raw):
    assert desc(E/'execute_rx_v2.py')['sha256']==PRIOR_SHA
    original=prior.adapt(raw)
    s=original.replace('_prepare_control_v2','_prepare_control_v3').replace('rx_prepare_v2.py','rx_prepare_v3.py')
    assert s.replace('_prepare_control_v3','_prepare_control_v2').replace('rx_prepare_v3.py','rx_prepare_v2.py')==original
    return s
if __name__=='__main__':
    exec(compile(adapt((E/'execute_stage_v1.py').read_bytes()),str(E/'execute_stage_v1.py')+'::rx_prepare_v3','exec'),{'__name__':'__main__','__file__':__file__})
