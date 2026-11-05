"""Frozen post-training evaluation adapters; no training code is modified."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from runtime import setup,sha
setup()
from codec_adapter import numerical_route,load_models

METRICS=('psnr_db','lpips_alex','dists','dreamsim')
BUDGETS=(512,768,1024,1536,2048,3072,4096,8192)

def digest_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def source_pixels(source,size):
    from torchvision.transforms import functional as TF
    assert sha(source['path'])==source['sha256']
    with Image.open(source['path']) as im:
        im=im.convert('RGB');native=np.asarray(im).copy()
        assert digest_array(native)==source['native_rgb_sha256']
        im=TF.center_crop(TF.resize(im,size,interpolation=TF.InterpolationMode.BICUBIC,antialias=True),[size,size])
        pixels=np.asarray(im).copy()
    assert digest_array(pixels)==source['preprocessed_sha256'][str(size)]
    return pixels

def source_tensor(pixels):return torch.from_numpy(pixels).permute(2,0,1)[None].cuda().float()/127.5-1

@torch.no_grad()
def original512_tx(vq,ar,cb,x):
    """Official 256-patch representation on a 512 image: four patches in one stream."""
    from packet import SourcePacket
    from tokenizer.tokenizer_image.codec.latent_io import encode_pixels_to_latents
    from tokenizer.tokenizer_image.codec.real.causal_ar_loop import CausalARState
    from tokenizer.tokenizer_image.codec.entropy_coding.codecs.topk_tensor_rans import TopkTensorRansCodec,TopkTensorRansConfig
    lat=encode_pixels_to_latents(vq,x)
    assert [tuple(v) for v in lat.shape_list]==[(4,4),(8,8),(16,16)]
    assert lat.mask_padded is None and tuple(lat.indices.shape)==(4,336)
    coder=TopkTensorRansCodec(TopkTensorRansConfig(topk=1024,precision=16,padding_token=0))
    state=CausalARState(ar,lat.shape_list,4,cb,0,None).setup();packets={0:SourcePacket(0,b'',b'').to_bytes()}
    try:
        mask=state.mask_all.detach().cpu().numpy()
        for t in range(28):
            got=coder.encode_slice(state.logits_for_slice(t),state.target_slice(lat.indices,t),None)
            state.write_slice(t,got);streams=coder.finish_encoding()
            packets[t+1]=SourcePacket(t+1,streams.top_stream,streams.residual_stream).to_bytes()
    finally:state.destroy()
    return packets,lat.indices.cpu().numpy(),mask

@torch.no_grad()
def original512_rx(vq,ar,cb,wire):
    from packet import SourcePacket
    from tokenizer.tokenizer_image.codec.entropy_coding.codecs.topk_tensor_rans import EncodedTopkTensorStreams
    from tokenizer.tokenizer_image.codec.real.causal_pipeline import decode_all_slices
    from tokenizer.tokenizer_image.codec.latent_io import EncodedLatents,restore_indices_to_multiscale_features
    p=SourcePacket.from_bytes(wire)
    if p.slices==0:return np.full((3,512,512),.5,np.float32),np.empty((4,0),np.int64)
    shapes=[(4,4),(8,8),(16,16)]
    indices,count=decode_all_slices(ar,shape_list=shapes,streams=EncodedTopkTensorStreams(p.top,p.escape,0,0),
        transfer_slices=p.slices,batch_size=4,valid_token_mask=None,codebook=cb,padding_token=0,topk=1024,precision=16,
        sample_logits=False,temperature=1.,top_k=-1,top_p=1.)
    lat=EncodedLatents(indices=None,shape_list=shapes,mask_padded=None,pad_infos=None,padded_shapes=None,batch_patches=4)
    features=restore_indices_to_multiscale_features(vq,indices,lat,codebook=cb,transfer_slices=p.slices,image_hw=(512,512),patch_batch=1)
    raw=vq.decode(features)
    from autoregressive.models.mask_generation import generate_MS_ck_mask
    _,mask=generate_MS_ck_mask(shapes,device=indices.device,use_patch_ck_ar=True,k=4)
    return raw.add(1).mul(.5).clamp(0,1)[0].cpu().float().numpy(),indices[:,mask<p.slices].cpu().numpy()

@torch.no_grad()
def score_native(suite,reference,reconstruction):
    assert reference.shape==reconstruction.shape and reference.shape in [(3,256,256),(3,512,512)]
    for a in [reference,reconstruction]:assert a.dtype==np.float32 and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
    if reference.shape[-1]==256:return suite.score(reference,reconstruction)
    xt=torch.from_numpy(reference[None]).cuda();yt=torch.from_numpy(reconstruction[None]).cuda()
    mse=float(np.mean((reference.astype(np.float64)-reconstruction.astype(np.float64))**2))
    return dict(psnr_db=float(-10*np.log10(max(mse,1e-12))),lpips_alex=float(suite.lp_pair(suite.lp,yt,xt)[0].item()),
        dists=float(suite.dists(xt,yt,require_grad=False,batch_average=False).reshape(-1)[0]),
        dreamsim=float((1-F.cosine_similarity(suite.feature(xt),suite.feature(yt),dim=-1))[0]))
