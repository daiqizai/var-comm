"""One arithmetic state across 4096-way coarse and binary refinement CDFs."""
import numpy as np
TOTAL=1<<24
def positions(order):
    assert order in('scalar','progressive')
    return[(i,d)for d in range(4)for i in range(64)]if order=='progressive'else[(i,d)for i in range(64)for d in range(4)]
def binary_cdf(probabilities,prefix,depth):
    p=np.asarray(probabilities,dtype=np.float64);assert p.shape==(16,)and np.all(p>0)and np.isfinite(p).all()
    lo=int(prefix)<<(4-depth);mid=lo+(1<<(3-depth));hi=lo+(1<<(4-depth))
    zero=float(p[lo:mid].sum()/p[lo:hi].sum());n0=int(np.floor(zero*TOTAL+.5));n0=max(1,min(TOTAL-1,n0))
    return np.asarray([[0,n0,TOTAL]],dtype=np.int64)
def encode_candidates(Encoder,coarse_values,coarse_tables,values,probabilities,order):
    e=Encoder()
    for v,cdf in zip(coarse_values,coarse_tables):e.encode(np.asarray(v,np.int64),cdf)
    result={0:np.asarray(e.finish(),np.uint8)};emitted={0:len(e.bits)}
    for j,(i,d)in enumerate(positions(order),1):
        v=int(values[i]);cdf=binary_cdf(probabilities[i],v>>(4-d)if d else 0,d)
        e.encode(np.asarray([(v>>(3-d))&1],np.int64),cdf)
        result[j]=np.asarray(e.finish(),np.uint8);emitted[j]=len(e.bits)
    return result,emitted
def decode_residual(decoder,canonical,j,probabilities,order):
    prefix=np.zeros(64,np.int64);depth=np.zeros(64,np.int64)
    for i,d in positions(order)[:j]:
        assert depth[i]==d
        cdf=binary_cdf(probabilities[i],prefix[i],d);v=decoder.decode(cdf)
        canonical.encode(v,cdf);prefix[i]=(prefix[i]<<1)|int(v[0]);depth[i]+=1
    u=np.where(depth>0,-1+(2*prefix+1)/(2.**depth),0).astype(np.float32)
    return u,depth
