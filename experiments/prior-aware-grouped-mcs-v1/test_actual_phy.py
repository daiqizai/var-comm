import numpy as np
from actual_phy import indices_to_bits,bits_to_indices,serialize_groups,accepted_state

def fixture():
    sizes=(1,2,3,4,5,6,8,10,13,16)
    scales=[np.arange(s*s,dtype=np.int64)+index for index,s in enumerate(sizes)]
    profile=dict(N=1024,G=2,m=5,K=9,j=4,groups=[dict(source_bits=360),dict(source_bits=(25+9)*12)])
    return scales,profile

def test_roundtrip_partial_serialization_and_continuous_prefix():
    scales,p=fixture();pos=np.array([2,4,6,8,10,12,14,16,18]);groups=serialize_groups(scales,p,pos)
    full=accepted_state(p,groups,[True,True]);assert full['m']==5 and full['K']==9
    assert full['prefix']==[s.tolist() for s in scales[:5]] and full['partial_values']==scales[5][pos].tolist()
    prefix=accepted_state(p,groups,[True,False]);assert prefix['m']==4 and prefix['K']==0
    assert prefix['prefix']==[s.tolist() for s in scales[:4]]
    assert accepted_state(p,groups,[False,True])['kind']=='gray'

def test_accepted_wrong_values_are_not_silently_replaced_by_truth():
    scales,p=fixture();groups=serialize_groups(scales,p,np.arange(9));groups[0][0]^=1
    state=accepted_state(p,groups,[True,False])
    assert state['prefix'][0][0]!=int(scales[0][0])

def test_raw_indices_use_exact12bits():
    t=np.array([0,1,4095,17]);b=indices_to_bits(t)
    assert len(b)==48 and np.array_equal(bits_to_indices(b),t)
