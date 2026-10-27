"""Separate two-development-image output route for frozen C source/RX code."""
from pathlib import Path
import argparse, importlib.util, sys
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
OLD=R/'experiments/adatoken-response-20261026'
sys.path.insert(0,str(OLD))
from common import checked,read,sha,cr
L=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/wcl_adatoken_20261027_v1')

def main(role,path,expected):
    assert sha(path)==expected
    r=read(path);scope=cr(r['development_scope'])
    assert scope['source_indices']==[95,21] and scope['snr_db']==10
    assert r['route'] in ('source','rx') and r['max_seconds']<=900
    for d in r['code']:checked(d)
    if r['route']=='source':
        assert len(cr(r['sources']))==2
        import c_predclass_source as worker
    else:
        assert len(cr(r['inputs']))<=2
        parent=OLD/'c_predclass_production_rx_v1';sys.path.insert(0,str(parent))
        spec=importlib.util.spec_from_file_location('frozen_production_rx',parent/'worker.py')
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    worker.L=L
    worker.main(role,path,expected)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True)
    a=p.parse_args();main(a.role,a.registration,a.expected_registration)
