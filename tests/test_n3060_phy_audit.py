"""Synthetic assertions for auditing real artifacts, not quality evaluation."""
import numpy as np
import pytest
from tools.audit_n3060_reference_phy import waveform_check,digital_ledger


def test_single_noise_and_finite_energy_are_required():
    tx=np.ones((3060,2));z=np.random.default_rng(23).normal(size=tx.shape)
    rx=tx+z/np.sqrt(10**.7)
    energy,error=waveform_check(tx,rx,z,7.,1e-12)
    assert energy==6120 and error==0
    with pytest.raises(ValueError):waveform_check(tx,tx+2*z/np.sqrt(10**.7),z,7.,1e-12)
    with pytest.raises(ValueError):waveform_check(tx*2,rx,z,7.,1e-12)
    with pytest.raises(ValueError):waveform_check(tx,np.full_like(rx,np.nan),z,7.,1e-12)


def test_float32_archived_waveform_rounding_is_bounded():
    tx=np.ones((1,3060,2),np.float32);z=np.random.default_rng(1).normal(size=(3060,2))
    rx=tx+z.astype(np.float32)[None]*np.float32(10**(-1./20))
    energy,error=waveform_check(tx,rx,z,1.,2e-6)
    assert energy==6120 and 0<error<2e-6
    with pytest.raises(ValueError):waveform_check(tx,rx+1e-3,z,1.,2e-6)


def test_source_control_fec_and_channel_uses_are_not_interchangeable():
    row={'family':'raw','header_uses':68,'data_uses':2992,'complex_uses':3060,'total_energy':6120,
         'actual_payload_bits':1860,'raw_payload_bits':1860,'header_payload_bits':12,'header_crc_bits':16,
         'header_tail_bits':6,'header_coded_bits':136,'data_crc_bits':16,'data_tail_bits':6,
         'data_mother_coded_bits':3764,'data_coded_bits':5984}
    assert digital_ledger(row)==(68,1860)
    for field,value in [('complex_uses',1860),('header_uses',0),('data_mother_coded_bits',3720),('header_payload_bits',25)]:
        with pytest.raises(ValueError):digital_ledger({**row,field:value})
