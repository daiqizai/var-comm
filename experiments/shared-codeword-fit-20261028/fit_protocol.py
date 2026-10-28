"""Paid compact directory. No global padding/LDPC convergence acceptance gate."""
from fit_common import *
from protocol import packet_module,OFFSETS,SIZES
P=packet_module()
VERSION='SHARED_LONG_COMPACT_V1'
KS=(512,544,576,608,637)

def endpoints():
    # Complete m4, every legal next-scale prefix, complete m5, through complete m6.
    return [(4,k)for k in range(25)]+[(5,k)for k in range(36)]+[(6,0)]

def catalogue(ks=KS):
    rows=[]
    def add(mode,m,K,k,null=False):
        rows.append(dict(profile_id=len(rows),version=VERSION,mode=mode,m=m,K=K,k=k,n=1912,q=2,null=null,N=1024,header_symbols=68,body_symbols=956,Es=2))
    for m,K in endpoints():add('S_FILL',m,K,637)
    add('S_FILL',4,0,637,True)
    for m,K in [(4,0),(4,6),(4,12),(4,18),(5,0),(5,9),(5,18),(5,27)]:
        for k in ks:add('S_PROTECT',m,K,k)
    for k in ks:add('S_PROTECT',4,0,k,True)
    add('SHARED_OLD',4,12,636)
    add('SHARED_OLD',4,0,636,True)
    for K in range(26):add('COMPACT',4,K,636)
    add('COMPACT',4,0,636,True)
    assert len(rows)<4096
    return {str(p['profile_id']):p for p in rows}

def context(pid,role,length):
    return np.concatenate((P.intbits(int(pid),12),P.intbits(int(role),2),P.intbits(int(length),10)))

def pack_compact(pid,streams,k=636):
    streams=[P.binary(x)for x in streams];lengths=list(map(len,streams))
    assert 2<=lengths[0] and all(x<1024 for x in lengths)
    assert all(x==0 or x>=2 for x in lengths[1:]) and sum(lengths)+94<=k
    ls=np.concatenate([P.intbits(x,10)for x in lengths])
    directory=np.concatenate((ls,P.intbits(P.crc16(np.concatenate((P.intbits(int(pid),12),ls))),16)))
    parts=[directory]
    for role,x in enumerate(streams):
        parts.extend((x,P.intbits(P.crc16(np.concatenate((context(pid,role,len(x)),x))),16)))
    parts.append(np.zeros(k-94-sum(lengths),np.uint8))
    return np.concatenate(parts)

def parse_compact(pid,K,information,k=636):
    def reject(reason):return dict(status=reason,streams=[None,None,None],lengths=None)
    try:a=P.binary(information)
    except AssertionError:return reject('MALFORMED_BINARY')
    if len(a)!=k:return reject('LENGTH_REJECT')
    if P.crc16(np.concatenate((P.intbits(int(pid),12),a[:30])))!=P.bitsint(a[30:46]):return reject('DIRECTORY_CRC_REJECT')
    lengths=[P.bitsint(a[s:s+10])for s in(0,10,20)]
    counts=[1,int(K)//2,int(K)-int(K)//2]
    if sum(lengths)+94>k or any((n==0 and l!=0)or(n>0 and l<2)for n,l in zip(counts,lengths)):return reject('DIRECTORY_RANGE_REJECT')
    pos=46;streams=[];states=[]
    for role,(n,l)in enumerate(zip(counts,lengths)):
        x=a[pos:pos+l];crc=P.bitsint(a[pos+l:pos+l+16]);pos+=l+16
        if n==0:streams.append(np.empty(0,np.uint8));states.append('DISABLED');continue
        ok=P.crc16(np.concatenate((context(pid,role,l),x)))==crc
        streams.append(x.copy()if ok else None);states.append('LOCAL_CRC_PASS'if ok else'LOCAL_CRC_REJECT')
    return dict(status='DIRECTORY_PASS',streams=streams,lengths=lengths,local_status=states,padding_bits=k-pos)

def modes(c,a,b,K):
    if not c:return dict(ALL='GRAY',SEQ='GRAY',SEL='GRAY')
    enabled=[K//2>0,K-K//2>0];valid=[a,b]
    allok=all(v for v,e in zip(valid,enabled)if e)
    use_seq=[];use_sel=[];cont=True
    for role,e,v in zip(('A','B'),enabled,valid):
        if not e:continue
        if v:use_sel.append(role)
        if cont and v:use_seq.append(role)
        else:cont=False
    return dict(ALL='C'+''.join(r for r,e in zip(('A','B'),enabled)if e)if allok else'GRAY',SEQ='C'+''.join(use_seq),SEL='C'+''.join(use_sel))
