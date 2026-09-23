import numpy as np
import pytest
from short_prefix.protocol import allocation,transmit_digital,decode_frame,channel,standard_noise,SIZES
from var_comm.progressive import decode_header

@pytest.mark.parametrize('m',[6,7,8])
def test_noiseless_actual_fec_and_energy(m):
    ledger=allocation(m);source=[np.arange(s*s,dtype=np.int64)%4096 for s in SIZES]
    digital=transmit_digital(source,23,ledger)
    whole=np.concatenate((digital,np.ones((ledger['NA'],2))))
    result=decode_frame(whole,100.,allowed_modes=(m,))
    assert result['header_ok'] and result['body_crc_ok']
    assert result['decoded_label']==23 and result['decoded_mode']==m
    np.testing.assert_array_equal(np.concatenate(result['prefix']),np.concatenate(source[:m]))
    assert whole.shape==(4084,2) and (whole**2).sum()==8168
    if m==6:assert not decode_header(digital[:68],100.)['accepted']

@pytest.mark.parametrize('m,N,nd,na',[(6,4084,1200,2816),(7,4084,1968,2048),(8,4084,2992,1024),(6,3060,1200,1792),(7,3060,1968,1024),(6,4084,1456,2560),(7,4084,2224,1792)])
def test_registered_resources(m,N,nd,na):
    assert allocation(m,N,nd)['NA']==na

def test_mode_dispatch_uses_received_header_not_sender():
    ledger=allocation(7);source=[np.zeros(s*s,dtype=np.int64) for s in SIZES]
    whole=np.concatenate((transmit_digital(source,17,ledger),np.ones((ledger['NA'],2))))
    assert not decode_frame(whole,100.,allowed_modes=(6,))['header_ok']
    actual=decode_frame(whole,100.)
    assert actual['ledger']['m']==7 and actual['observation'].shape==(2048,2)

def test_noise_is_once_and_explicitly_length_scoped():
    ledger=allocation(6);d=np.ones((1268,2));a=np.ones((2816,2))
    y=channel(d,a,'source',4101,7,ledger)
    expected=np.concatenate((d+standard_noise('source',4101,ledger,'digital')*10**(-7/20),a+standard_noise('source',4101,ledger,'continuous')*10**(-7/20)))
    np.testing.assert_array_equal(y,expected)
    assert not np.array_equal(standard_noise('source',4101,ledger,'digital')[:100],standard_noise('source',4101,allocation(7),'digital')[:100])
    with pytest.raises(ValueError):channel(d,a*np.nan,'source',1,7,ledger)
