"""17-byte paid mode/endpoint/length/CRC wrapper around native top-k rANS.

Source-packet qualification only: no new radio PHY result is claimed. Public mode
fixes resolution and budget. Independent RX has no source pixels or TX logits.
"""
import struct,zlib
import torch
from model import mode_spec,shapes_for
from tokenizer.tokenizer_image.codec.real.causal_ar_loop import CausalARState
from tokenizer.tokenizer_image.codec.entropy_coding.codecs.topk_tensor_rans import TopkTensorRansCodec,TopkTensorRansConfig,EncodedTopkTensorStreams
HEADER=struct.Struct('>2sBBBII')
def pack(mode,endpoint,top=b'',escape=b''):
    b=HEADER.pack(b'MR',1,mode,endpoint,len(top),len(escape))+top+escape
    return b+struct.pack('>I',zlib.crc32(b))
def unpack(wire):
    assert len(wire)>=17
    magic,version,mode,ep,nt,ne=HEADER.unpack(wire[:13])
    assert magic==b'MR' and version==1 and 0<=ep<=60
    size,budget=mode_spec(mode)
    assert len(wire)==17+nt+ne and len(wire)*8<=budget
    assert zlib.crc32(wire[:-4])==struct.unpack('>I',wire[-4:])[0]
    assert ep or nt+ne==0
    return mode,ep,wire[13:13+nt],wire[13+nt:-4]
@torch.no_grad()
def encode_packet(model,x,mode):
    assert len(x)==1 and x.shape[-1]==mode_spec(mode)[0]
    model.eval();_,ids,_,_=model.encode(x,torch.tensor([mode],device=x.device))
    coder=TopkTensorRansCodec(TopkTensorRansConfig(topk=1024,precision=16,padding_token=0))
    state=CausalARState(model.ar,shapes_for(x.shape[-1]),1,model.codebook,0,None).setup()
    best=pack(mode,0);lengths=[];selected=0
    try:
        for s in range(60):
            logits=state.logits_for_slice(s);target=state.target_slice(ids,s)
            recovered=coder.encode_slice(logits,target,None);state.write_slice(s,recovered)
            streams=coder.finish_encoding();wire=pack(mode,s+1,streams.top_stream,streams.residual_stream)
            lengths.append(len(wire)*8)
            if len(wire)*8<=mode_spec(mode)[1]:best=wire;selected=s+1
    finally:state.destroy()
    keep=model.mask(x.shape[-1])<selected
    return best,dict(mode=mode,endpoint=selected,bits=len(best)*8,candidate_bits=lengths,received_ids=ids[:,keep].cpu())
@torch.no_grad()
def receive_packet(model,wire):
    mode,ep,top,escape=unpack(wire);size,_=mode_spec(mode);model.eval();device=model.codebook.device
    if ep==0:return torch.zeros(1,3,size,size,device=device),torch.empty(1,0,dtype=torch.long),dict(null=True,endpoint=0)
    coder=TopkTensorRansCodec(TopkTensorRansConfig(topk=1024,precision=16,padding_token=0))
    coder.begin_decoding(EncodedTopkTensorStreams(top,escape,0,0))
    state=CausalARState(model.ar,shapes_for(size),1,model.codebook,0,None).setup()
    try:
        for s in range(ep):
            current=coder.decode_slice(state.logits_for_slice(s),None);state.write_slice(s,current)
        indices=state.all_indices().clone()
    finally:state.destroy()
    keep=model.mask(size)[None]<ep
    quant=model.codebook[indices]*keep[...,None]
    raw=model.reconstruct(quant,torch.tensor([mode],device=device),keep,size)
    return raw,indices[:,keep[0]].cpu(),dict(null=False,endpoint=ep)
