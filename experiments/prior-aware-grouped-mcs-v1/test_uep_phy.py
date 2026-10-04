import numpy as np
from uep_phy import append_crc_batch,crc_accept_batch,crc16_batch,continuous_prefix,mask_batch

def reference(bits):
    v=65535
    for b in bits:
        f=((v>>15)&1)^int(b);v=(v<<1)&65535
        if f:v^=0x1021
    return v

def test_crc_matches_original_bit_recursion_and_detects_flips():
    r=np.random.default_rng(2)
    for n in (1,4,7,8,12,360,1860,3060,5088):
        a=r.integers(0,2,(8,n),dtype=np.uint8)
        assert crc16_batch(a).tolist()==[reference(x) for x in a]
        packet=append_crc_batch(a);assert crc_accept_batch(packet).all()
        packet[:,n//2]^=1;assert not crc_accept_batch(packet).any()

def test_first_failed_group_cannot_be_rescued_by_later_success():
    a=np.array([1,0]);b=np.array([0,1])
    assert continuous_prefix([a,b],[False,True])==[]
    out=continuous_prefix([a,b],[True,False]);assert len(out)==1 and np.array_equal(out[0],a)
    assert len(continuous_prefix([a,b],[True,True]))==2

def test_public_mask_uses_counter_not_payload_and_separates_groups():
    a=mask_batch(200,[10,11],1,'test')
    assert np.array_equal(a,mask_batch(200,[10,11],1,'test'))
    assert not np.array_equal(a[0],a[1]) and not np.array_equal(a,mask_batch(200,[10,11],2,'test'))
