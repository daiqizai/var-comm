"""Same ten-scale traversal; CPU timers never synchronize per endpoint."""
import time
import numpy as np

SIZES=(1,2,3,4,5,6,8,10,13,16)
OFFSETS=tuple(sum(s*s for s in SIZES[:i]) for i in range(11))

def endpoints(divisor):
    assert divisor in (4,8)
    return tuple((m,k) for m in range(4,10) for k in (0,*(SIZES[m]**2*j//divisor for j in range(1,divisor))))+((10,0),)

def encode(codec,tokens,grid,synchronize):
    ends=endpoints(grid);encoder=codec.Encoder();out={};cdfs=[]
    parts=dict(ac_encode_ms=0.,endpoint_finish_ms=0.,prior_cdf_context_ms=0.)
    counts=dict(prior_scale_calls=0,encoded_token_count=0,finish_calls=0,encode_segment_calls=0)
    def measured(name,fn):
        begin=time.perf_counter();v=fn();parts[name]+=(time.perf_counter()-begin)*1000;return v
    def flush(m,k):
        out[m,k]=measured('endpoint_finish_ms',lambda:np.asarray(encoder.finish(),dtype=np.uint8))
        counts['finish_calls']+=1
    def ac(values,cdf):
        measured('ac_encode_ms',lambda:encoder.encode(values,cdf))
        counts['encoded_token_count']+=len(values);counts['encode_segment_calls']+=1
    inf=codec._inference('EC_VAR_WHOLE');ctx=codec._context('EC_VAR_WHOLE')
    begin=time.perf_counter();inf.__enter__();provider=ctx.__enter__();parts['prior_cdf_context_ms']+=(time.perf_counter()-begin)*1000
    try:
        for scale in range(10):
            values=tokens[OFFSETS[scale]:OFFSETS[scale+1]]
            cdf=measured('prior_cdf_context_ms',lambda:codec._cdf('EC_VAR_WHOLE',provider,scale))
            counts['prior_scale_calls']+=1;cdfs.append(cdf);start=0
            for k in sorted(k for m,k in ends if m==scale and k>0):
                ac(values[start:k],cdf[start:k]);flush(scale,k);start=k
            ac(values[start:],cdf[start:])
            measured('prior_cdf_context_ms',lambda:provider.advance(values.copy()))
            if (scale+1,0) in ends or scale==9:flush(scale+1,0)
    finally:
        begin=time.perf_counter();ctx.__exit__(None,None,None);inf.__exit__(None,None,None);synchronize()
        parts['prior_cdf_context_ms']+=(time.perf_counter()-begin)*1000
    assert set(out)==set(ends) and counts['encoded_token_count']==680
    parts['ac_and_finish_ms']=parts['ac_encode_ms']+parts['endpoint_finish_ms']
    return out,cdfs,parts,counts

def select(streams,grid,profiles):
    eligible=[p for p in profiles if p['q']==4 and p['nominal_rate']=='2/3']
    capacity=eligible[0]['source_capacity_bits'];checks=0
    for e in reversed(endpoints(grid)):
        bits=streams[e];checks+=1
        if len(bits)<=min(capacity,8191):
            p=next(p for p in eligible if p['family'] in ('EC_VAR_WHOLE','EC_VAR_PARTIAL') and (p['m'],p['K'])==e)
            return p,bits,checks
    raise RuntimeError('No feasible endpoint: preserve diagnostic; no new NULL qualification needed')
