"""Reject accidental calibration population and altered resource ledger."""
import pytest
from tools.publish_16qam_development import verify_row, REL
def fixture():
    c=dict(method='digital/raw/16QAM/N2048/m9',N=2048,N_header=70,N_data=1978,
           coded_slots=7912,m_requested=9,mcs='16QAM',family='raw')
    r=dict(c,source_id='source',source_index=0,context_sha256='context',
        run_id='TOKEN-CHANNEL-EFFICIENCY-20260923/'+'digital_grid_v1/16QAM/calibration',
        noise_namespace='digital/raw/16QAM/N2048',noise_seed=4101,
        noise_id='digital/raw/16QAM/N2048|source|4101',
        energy_constraint='fixed_constellation_average_2_per_symbol',
        E=4008.,payload_bits=5088,mother_bits=10220,source_overflow_erasure=False)
    return r,c,dict(context_sha256='context')

def test_realistic_puncturing_and_variable_energy_in_development():
    row,candidate,reg=fixture()
    row['run_id']='TOKEN-CHANNEL-EFFICIENCY-20260923/'+REL
    row['noise_seed']=2001
    row['noise_id']='digital/raw/16QAM/N2048|source|2001'
    verify_row(row,candidate,'source',0,reg)

def test_calibration_row_cannot_enter_development():
    row,candidate,reg=fixture()
    with pytest.raises(RuntimeError):verify_row(row,candidate,'source',0,reg)

from tools.publish_16qam_development import registered_source

def test_development_uses_indexed_rgb_bindings():
    reg={'sources':{'a':'rgb-a','b':'rgb-b'},'image_bindings':[
        {'index':0,'rgb_sha256':'rgb-a'}, {'index':1,'rgb_sha256':'rgb-b'}]}
    assert registered_source(reg,1)=='b'
    reg['image_bindings'][1]['rgb_sha256']='rgb-a'
    with pytest.raises(RuntimeError):registered_source(reg,1)
