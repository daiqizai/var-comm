"""m6-m10 actual whole-stream source coding; no legacy mode mutation."""
import numpy as np,torch
from var_comm.whole_entropy import ArithmeticEncoder,ArithmeticDecoder,VarScaleStream
from var_comm.entropy import probability_cdf,validate_bits
from var_comm.scale_channel import indices_to_bits,bits_to_indices
SIZES=(1,2,3,4,5,6,8,10,13,16)

@torch.no_grad()
def cumulative(vae,scales,m):
    if tuple(vae.quantize.v_patch_nums)!=SIZES or m not in range(6,11) or len(scales)<m:raise ValueError('official ten-scale mapping')
    f=vae.quantize.embedding.weight.new_zeros(1,32,16,16)
    for i in range(m):
        t=np.asarray(scales[i])
        if t.shape!=(SIZES[i]**2,) or not np.issubdtype(t.dtype,np.integer) or np.any(t<0) or np.any(t>=4096):raise ValueError('tokens')
        e=vae.quantize.embedding(torch.as_tensor(t[None],device=f.device)).transpose(1,2).reshape(1,32,SIZES[i],SIZES[i])
        f,_=vae.quantize.get_next_autoregressive_input(i,10,f,e)
    return f

@torch.no_grad()
def encode_all(vae,var,source,label,device):
    if len(source)!=10:raise ValueError('complete source schedule')
    encoder=ArithmeticEncoder();result={}
    with VarScaleStream(vae,var,int(label),device) as stream:
        for i in range(10):
            encoder.encode(source[i],probability_cdf(stream.log_probs()));stream.advance(source[i])
            m=i+1
            if m<6:continue
            attempted=encoder.finish();raw=indices_to_bits(np.concatenate(source[:m]))
            fallback=not (2<=len(attempted)<len(raw) and len(attempted)<8192)
            result[m]={'raw':raw,'arithmetic':attempted,'payload':raw.copy() if fallback else attempted.copy(),'length_field':0 if fallback else len(attempted),'raw_fallback':fallback,'finish_flush_bits':len(attempted)-len(encoder.bits)}
    return result

@torch.no_grad()
def decode_arithmetic(bits,m,label,vae,var,device):
    if m not in range(6,11):raise ValueError('source mode')
    decoder=ArithmeticDecoder(bits);prefix=[]
    with VarScaleStream(vae,var,int(label),device) as stream:
        for i in range(m):
            value=decoder.decode(probability_cdf(stream.log_probs()));prefix.append(value);stream.advance(value)
    return prefix,max(0,decoder.position-len(decoder.bits))

def write_stream(path,bits):
    bits=validate_bits(bits);packed=np.packbits(bits,bitorder='big')
    path.write_bytes(packed.tobytes())
    restored=np.unpackbits(np.frombuffer(path.read_bytes(),dtype=np.uint8),bitorder='big')[:len(bits)]
    np.testing.assert_array_equal(restored,bits)
    return {'meaningful_bits':len(bits),'storage_bytes':len(packed),'storage_padding_bits':8*len(packed)-len(bits),'storage_padding_transmitted':False}
