import numpy as np,pytest,torch
from token_efficiency.phy import modulate,soft_half_llr,packet,decode_packet,transmit,receive,channel,PAM,dimensions
from token_efficiency.codec import SIZES,write_stream
from var_comm.scale_channel import indices_to_bits,encode_packet

def payloads(seed=1):
    rng=np.random.default_rng(seed)
    return {m:{'raw':rng.integers(0,2,12*sum(s*s for s in SIZES[:m]),dtype=np.uint8),'payload':rng.integers(0,2,700+100*m,dtype=np.uint8),'length_field':700+100*m} for m in range(6,11)}

def test_qam_gray_energy_soft_likelihood():
    bits=np.array([[i>>3&1,i>>2&1,i>>1&1,i&1] for i in range(16)],dtype=np.uint8).ravel();y=modulate(bits,'16QAM')
    assert np.mean(np.sum(y*y,axis=1))==pytest.approx(2)
    assert np.min(np.sum(y*y,axis=1))<2<np.max(np.sum(y*y,axis=1))
    np.testing.assert_array_equal((soft_half_llr(y,19,'16QAM')<0).astype(np.uint8),bits)
    # Independent direct Gaussian mixture, nonconstellation observations.
    z=np.array([[.13,-.77],[1.12,.31]]);gamma=10**(.7);actual=soft_half_llr(z,7,'16QAM').reshape(-1,2)
    for i,v in enumerate(z.ravel()):
        density=np.exp(-(v-PAM)**2*gamma/2)
        for b in range(2):
            labels=(np.arange(4)>>(1-b))&1
            assert actual[i,b]==pytest.approx(.5*np.log(density[labels==0].sum()/density[labels==1].sum()))

@pytest.mark.parametrize('mcs',['QPSK','16QAM'])
def test_real_fec_soft_roundtrip_and_reject(mcs):
    rng=np.random.default_rng(29);bits=rng.integers(0,2,511,dtype=np.uint8);wave=packet(bits,700,mcs)
    decoded,crc,_=decode_packet(wave,len(bits),13,mcs)
    assert crc;np.testing.assert_array_equal(decoded,bits)
    if mcs=='QPSK':np.testing.assert_array_equal(wave,encode_packet(bits,700)['symbols'])
    noisy=wave+np.random.default_rng(30).normal(0,.05,wave.shape)
    decoded,crc,_=decode_packet(noisy,len(bits),26,mcs);assert crc;np.testing.assert_array_equal(decoded,bits)
    with pytest.raises(ValueError):packet(np.ones(5000,dtype=np.uint8),700,mcs)
    with pytest.raises(ValueError):soft_half_llr(np.array([[np.nan,0]]),1,mcs)
    with pytest.raises(ValueError):modulate(np.array([0.,.4,0.,1.]),mcs)

@pytest.mark.parametrize('N',[2048,3060,4084])
@pytest.mark.parametrize('mcs',['QPSK','16QAM'])
@pytest.mark.parametrize('family',['raw','arithmetic'])
def test_paid_header_modes_and_actual_resource(N,mcs,family):
    encoded=payloads()
    for m in range(6,11):
        nh,nd,slots=dimensions(N,family,mcs)
        if family=='raw' and len(encoded[m]['raw'])+22>slots:
            with pytest.raises(ValueError):transmit(encoded,17,m,N,family,mcs)
            continue
        wave,ledger=transmit(encoded,17,m,N,family,mcs)
        assert wave.shape==(N,2) and ledger['N_header']==nh and ledger['N_data']==nd
        rx=receive(wave,19,N,family,mcs)
        assert rx['header_ok'] and rx['body_crc_ok'] and rx['decoded_label']==17 and rx['decoded_mode']==m
        np.testing.assert_array_equal(rx['source_bits'],encoded[m]['raw' if family=='raw' else 'payload'])
        if mcs=='QPSK':assert ledger['E']==2*N
        y=channel(wave,7,'source',2001,N,family,mcs)
        np.testing.assert_array_equal(y,channel(wave,7,'source',2001,N,family,mcs))
        assert not np.array_equal(y,wave)

def test_overflow_paid_lower_mode_and_full_erasure():
    encoded=payloads();encoded[10]['payload']=np.ones(7900,dtype=np.uint8);encoded[10]['length_field']=7900
    wave,ledger=transmit(encoded,20,10,2048,'arithmetic','QPSK');rx=receive(wave,19,2048,'arithmetic','QPSK')
    assert ledger['overflow_lower_m'] and rx['decoded_mode']==9 and rx['body_crc_ok']
    for e in encoded.values():e['payload']=np.ones(7900,dtype=np.uint8);e['length_field']=7900
    wave,ledger=transmit(encoded,20,10,2048,'arithmetic','QPSK');rx=receive(wave,19,2048,'arithmetic','QPSK')
    assert ledger['source_overflow_erasure'] and ledger['E']==4096 and rx['source_overflow_erasure'] and not rx['header_ok']

def test_written_bits_preserve_nonbyte_length(tmp_path):
    bits=np.array([1,0,1,1,0,1,0,0,1],dtype=np.uint8);rec=write_stream(tmp_path/'stream.bin',bits)
    assert rec['meaningful_bits']==9 and rec['storage_bytes']==2 and rec['storage_padding_bits']==7 and not rec['storage_padding_transmitted']
