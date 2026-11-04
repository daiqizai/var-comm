"""RDVQ adapter preserving native causal slice semantics and tensor-rANS.

Model code is pinned upstream, not copied or hot patched. Receiver functions
accept only a SourcePacket and public model/configuration.
"""
from pathlib import Path
import hashlib,time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from packet import SourcePacket

SHAPES=[(4,4),(8,8),(16,16)]

def numerical_route(seed=20261104):
    torch.set_num_threads(2)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

def tensor_sha(t):
    a=t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()

def load_models(path):
    from tokenizer.tokenizer_image.models.vq_model import VQ_models
    numerical_route()
    model=VQ_models['VQ-16-32-64_quant_once'](codebook_size=4096,codebook_embed_dim=32,use_predictor=True,wo_attn=True,patch_size=256,tau=.1)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    state=checkpoint['model'] if 'model' in checkpoint else checkpoint
    keys=list(state)
    if all(k.startswith('module.') for k in keys):state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    model=model.cuda().float().eval()
    ar=model.quantize.condition_entropy_small
    # Official real_codec_inference.py separates AR to avoid an extra teacher pass.
    model.quantize.condition_entropy_small=None;model.quantize.use_predictor=False
    codebook=F.normalize(model.quantize.embedding.weight,p=2,dim=-1).detach()
    return model,ar.eval(),codebook

def source_image(path):
    """Mechanism preprocessing, explicitly not official full-resolution replication."""
    from torchvision.transforms import functional as TF
    im=Image.open(path).convert('RGB');native=np.asarray(im).copy()
    im=TF.resize(im,256,interpolation=TF.InterpolationMode.BICUBIC,antialias=True)
    im=TF.center_crop(im,[256,256]);pixels=np.asarray(im).copy()
    x=torch.from_numpy(pixels).permute(2,0,1).float().div(127.5).sub(1).unsqueeze(0).cuda()
    ident=dict(native_size=list(native.shape),native_rgb_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
               preprocessed_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
               preprocessing='PIL RGB; shorter side 256 bicubic antialias; center crop 256; /127.5-1')
    return x,pixels,ident

@torch.no_grad()
def encode_all_prefixes(vq,ar,codebook,x,endpoints=range(1,29)):
    """One E call and one causal AR pass; independently finish each requested prefix."""
    from tokenizer.tokenizer_image.codec.latent_io import encode_pixels_to_latents
    from tokenizer.tokenizer_image.codec.real.causal_ar_loop import CausalARState
    from tokenizer.tokenizer_image.codec.entropy_coding.codecs.topk_tensor_rans import TopkTensorRansCodec,TopkTensorRansConfig
    torch.cuda.synchronize();begin=time.perf_counter()
    latents=encode_pixels_to_latents(vq,x)
    torch.cuda.synchronize();encode_seconds=time.perf_counter()-begin
    assert [tuple(s) for s in latents.shape_list]==SHAPES and latents.mask_padded is None and latents.indices.shape==(1,336)
    coder=TopkTensorRansCodec(TopkTensorRansConfig(topk=1024,precision=16,padding_token=0))
    state=CausalARState(ar,SHAPES,1,codebook,0,None).setup()
    records=[];packets={};cum_est=0.;cum_symbols=0;begin=time.perf_counter()
    mask=state.mask_all.detach().cpu().clone()
    try:
        for s in range(28):
            logits=state.logits_for_slice(s);targets=state.target_slice(latents.indices,s)
            cum_est+=float(ar.cross_entropy_log2(logits.reshape(-1,4096),targets.reshape(-1),reduction='none').sum())
            current=coder.encode_slice(logits,targets,None)
            state.write_slice(s,current);cum_symbols+=targets.numel()
            if s+1 in endpoints:
                t0=time.perf_counter();streams=coder.finish_encoding()
                p=SourcePacket(s+1,streams.top_stream,streams.residual_stream);packets[s+1]=p
                torch.cuda.synchronize()
                records.append(dict(slices=s+1,transmitted_tokens=cum_symbols,estimated_likelihood_bits=cum_est,
                    top_symbols=streams.top_symbols,escape_symbols=streams.residual_symbols,
                    termination_seconds=time.perf_counter()-t0,cumulative_all_candidate_seconds=time.perf_counter()-begin,
                    **p.accounting()))
    finally:state.destroy()
    torch.cuda.synchronize()
    return packets,records,latents.indices.detach().cpu(),mask,dict(encoder_seconds=encode_seconds,all_candidates_seconds=time.perf_counter()-begin,encoder_calls=1,causal_ar_slice_calls=28,independent_terminations=len(records))

@torch.no_grad()
def receive(vq,ar,codebook,wire):
    """No source pixels, true indices, TX logits, or TX state enter this API."""
    from tokenizer.tokenizer_image.codec.entropy_coding.codecs.topk_tensor_rans import EncodedTopkTensorStreams
    from tokenizer.tokenizer_image.codec.real.causal_pipeline import decode_all_slices
    p=SourcePacket.from_bytes(wire)
    if p.slices==0:return None,np.full((3,256,256),.5,np.float32),dict(null=True,ar_slice_calls=0,decoder_calls=0,rx_seconds=0.)
    # Symbol counts are accounting metadata; the native decoder derives its counts.
    streams=EncodedTopkTensorStreams(p.top,p.escape,0,0)
    torch.cuda.synchronize();begin=time.perf_counter()
    indices,count=decode_all_slices(ar,shape_list=SHAPES,streams=streams,transfer_slices=p.slices,batch_size=1,
        valid_token_mask=None,codebook=codebook,padding_token=0,topk=1024,precision=16,sample_logits=False,temperature=1.,top_k=-1,top_p=1.)
    features={};start=0
    for name,(h,w) in zip(vq.scale,SHAPES):
        end=start+h*w;features['quant_'+name]=codebook[indices[:,start:end]].permute(0,2,1).reshape(1,32,h,w);start=end
    raw=vq.decode(features)
    image=raw.add(1).mul(.5).clamp(0,1)[0].detach().cpu().float().numpy()
    torch.cuda.synchronize()
    return indices.detach().cpu(),image,dict(null=False,decoded_tokens=count,ar_slice_calls=28,decoder_calls=1,rx_seconds=time.perf_counter()-begin)

