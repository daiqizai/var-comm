"""Reject publication identities and preserve punctured FEC resource accounting."""
import pytest
from tools.publish_16qam_calibration import verify_row, REL

def fixture():
    c=dict(method='digital/raw/16QAM/N2048/m9',N=2048,N_header=70,N_data=1978,
           coded_slots=7912,m_requested=9,mcs='16QAM',family='raw')
    r=dict(c,source_id='source',source_index=0,context_sha256='context',
        run_id='TOKEN-CHANNEL-EFFICIENCY-20260923/'+REL,
        noise_namespace='digital/raw/16QAM/N2048',noise_seed=4101,
        noise_id='digital/raw/16QAM/N2048|source|4101',
        energy_constraint='fixed_constellation_average_2_per_symbol',
        E=4008.,payload_bits=5088,mother_bits=10220,source_overflow_erasure=False)
    return r,c,dict(context_sha256='context')

def test_punctured_mother_code_and_variable_energy_are_valid():
    r,c,reg=fixture()
    assert r['mother_bits']>r['coded_slots'] and r['E']!=2*r['N']
    verify_row(r,c,'source',0,reg)

@pytest.mark.parametrize('field,value',[
    ('noise_id','wrong'),('context_sha256','other'),('N_header',68),
    ('source_index',1),('mother_bits',10000),('payload_bits',7910),
    ('energy_constraint','per_frame_2N')])
def test_reject_mixed_identity_or_invalid_ledger(field,value):
    r,c,reg=fixture();r[field]=value
    with pytest.raises(RuntimeError):verify_row(r,c,'source',0,reg)
