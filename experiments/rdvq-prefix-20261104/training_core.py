"""Matched FP32 training with native RDVQ slices and deployed argmax completion.

The predictor's upstream training wrapper forces BF16; fp32_logits preserves
its query construction and dropout while making the authorized FP32 route
explicit. This does not patch the pinned upstream files.
"""
from pathlib import Path
import contextlib,hashlib,importlib.util,json,math,random
from unittest import mock
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from codec_adapter import load_models,SHAPES

BUDGETS=(384,512,768,1024,1536,2048,3072,4096)

def verified(spec):
    p=Path(spec['path']);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    assert h.hexdigest()==spec['sha256'],p
    return p

def load_training_lpips(asset_spec):
    """Same VGG/linear weights and loss forward as the pinned official RDVQ."""
    from torchvision import models
    from tokenizer.tokenizer_image.training.losses import lpips as official
    vgg=verified(asset_spec['dists']['vgg16_weights']);linear=verified(asset_spec['lpips_vgg'])
    factory=models.vgg16
    def local_vgg(*args,**kwargs):
        net=factory(weights=None);net.load_state_dict(torch.load(vgg,map_location='cpu',weights_only=True),strict=True);return net
    with mock.patch.object(models,'vgg16',local_vgg),mock.patch.object(official,'get_ckpt_path',lambda *a,**k:str(linear)):
        loss=official.LPIPS().cuda().float().eval()
    return loss.requires_grad_(False)

def fp32_logits(ar,quant):
    from autoregressive.models.mask_generation import generate_MS_ck_mask,rerank_input
    B,L,_=quant.shape;start=ar.start_token.expand(B,-1,-1);x=ar.input_proj(quant)
    if not ar.use_patch_ck_ar:x=torch.cat([start,x[:,:-1,:]],dim=1)
    _,mask=generate_MS_ck_mask(input_shape=SHAPES,device=x.device,use_patch_ck_ar=ar.use_patch_ck_ar,k=ar.num_ar_per_scale)
    query=rerank_input(x,start,SHAPES,mask,k=ar.num_ar_per_scale) if ar.use_patch_ck_ar else x
    with torch.autocast('cuda',enabled=False):out=ar.transformer(ranked_input=query.float(),shape_list=SHAPES)
    assert out.dtype==torch.float32
    return out.view(B,L,ar.V_size)

class PrefixModel(nn.Module):
    def __init__(self,checkpoint):
        super().__init__();self.vq,self.ar,codebook=load_models(checkpoint)
        self.vq.quantize.embedding.weight.requires_grad_(False)
        self.register_buffer('codebook',codebook)
        from autoregressive.models.mask_generation import generate_MS_ck_mask
        _,mask=generate_MS_ck_mask(SHAPES,device=codebook.device,use_patch_ck_ar=True,k=4)
        self.register_buffer('slice_mask',mask.long())
        self.register_buffer('counts',torch.tensor([int((mask<t).sum()) for t in range(29)],device=codebook.device))
        self.register_buffer('rate_correction',torch.zeros(29,device=codebook.device))

    def encode_differentiable(self,x):
        h=self.vq.encoder(x);z=torch.cat([self.vq.quant_conv_list[s](h[s]).flatten(2).transpose(1,2) for s in self.vq.scale],dim=1)
        z=F.normalize(z,p=2,dim=-1)
        d=z.square().sum(-1,keepdim=True)+self.codebook.square().sum(-1)-2*torch.einsum('blc,vc->blv',z,self.codebook)
        idx=d.argmin(-1);hard=self.codebook[idx];quant=z+(hard-z).detach()
        soft=F.softmax(-d/.1,dim=-1)
        commitment=.25*F.mse_loss(hard.detach(),z)
        # The original codebook loss has zero gradient when the codebook is fixed.
        codebook_loss=F.mse_loss(hard,z.detach())
        return quant,idx,soft,commitment,codebook_loss

    @torch.no_grad()
    def rate_estimates(self,quant,idx):
        was=self.ar.training;self.ar.eval()
        try:
            logits=fp32_logits(self.ar,quant.detach());bits=-F.log_softmax(logits,dim=-1).gather(-1,idx[...,None]).squeeze(-1)/math.log(2)
            # 72 wrapper + up to two 32-bit rANS states. Entropy discrepancy is
            # measured on the same 128 audit sources, not assumed to vanish.
            rates=torch.stack([torch.zeros_like(bits[:,0])+72]+[bits[:,self.slice_mask<t].sum(-1)+136 for t in range(1,29)],dim=1)
            return rates
        finally:self.ar.train(was)

    def choose_endpoints(self,arm,rates,generator):
        B=rates.shape[0]
        if arm=='R2':
            # Uniform token-count cutoffs in two halves, rounded down to the
            # native slice grid. No image-quality-dependent choice.
            low=torch.randint(1,169,(B,),generator=generator,device='cpu').to(rates.device)
            high=torch.randint(169,337,(B,),generator=generator,device='cpu').to(rates.device)
            return [torch.where(self.counts[None]<=n[:,None],torch.arange(29,device=rates.device),0).max(1).values for n in [low,high]]
        assert arm in ('R3','R3_estimated')
        corrected=rates+(self.rate_correction[None] if arm=='R3' else 0)
        choices=[]
        for pool in (BUDGETS[:4],BUDGETS[4:]):
            ix=torch.randint(len(pool),(B,),generator=generator,device='cpu')
            budget=torch.tensor(pool)[ix].to(rates.device)
            choices.append(torch.where(corrected<=budget[:,None],torch.arange(29,device=rates.device),0).max(1).values)
        return choices

    @torch.no_grad()
    def complete_prefix(self,received_indices,endpoints):
        """Only received positions enter; unsent indices must already be erased."""
        from tokenizer.tokenizer_image.codec.real.causal_ar_loop import CausalARState
        keep=self.slice_mask[None]<endpoints[:,None]
        assert torch.equal(received_indices[~keep],torch.zeros_like(received_indices[~keep]))
        was=self.ar.training;self.ar.eval()
        state=CausalARState(self.ar,SHAPES,len(endpoints),self.codebook,0,None).setup()
        try:
            for s in range(28):
                logits=state.logits_for_slice(s);pos=state.slice_positions(s)
                token=torch.where((endpoints>s)[:,None],received_indices[:,pos],logits.argmax(-1))
                state.write_slice(s,token)
            return state.all_indices().clone()
        finally:state.destroy();self.ar.train(was)

    def decode_embeddings(self,z):
        features={};lo=0
        for s,(h,w) in zip(self.vq.scale,SHAPES):
            hi=lo+h*w;features['quant_'+s]=z[:,lo:hi].transpose(1,2).reshape(-1,32,h,w);lo=hi
        return self.vq.decode(features)

    def objective(self,x,arm,perceptual,generator):
        quant,idx,soft,commit,cb=self.encode_differentiable(x)
        rates=self.rate_estimates(quant,idx)
        endpoints=[] if arm=='R1' else self.choose_endpoints(arm,rates,generator)
        # Rollouts execute before teacher-forced gradient computation so mutable
        # KV buffers cannot be retained by the autograd graph.
        completed=[]
        if endpoints:
            ee=torch.cat(endpoints);ii=idx.detach().repeat(2,1)
            keep=self.slice_mask[None]<ee[:,None];received=torch.where(keep,ii,0)
            generated=self.complete_prefix(received,ee)
            for ep,gg in zip(endpoints,generated.chunk(2)):
                k=self.slice_mask[None]<ep[:,None]
                completed.append((ep,torch.where(k[...,None],quant,self.codebook[gg].detach())))
        logits=fp32_logits(self.ar,quant)
        softbits=-(soft*F.log_softmax(logits,dim=-1)).sum(-1)/math.log(2)
        branches=[(torch.full((len(x),),28,device=x.device),quant)]+completed
        terms=[];records=[]
        for ep,z in branches:
            recon=self.decode_embeddings(z)
            # A valid NULL still produces the registered constant gray image.
            recon=torch.where((ep>0)[:,None,None,None],recon,torch.zeros_like(recon))
            mse=(recon-x).square().flatten(1).mean(1)
            lp=perceptual(recon,x).flatten(1).mean(1)
            rate=(softbits*(self.slice_mask[None]<ep[:,None])).sum(1)/(256*256)
            terms.append((mse+lp+4.8*rate).mean())
            records.append(dict(endpoint=ep.detach().cpu().tolist(),mse=float(mse.detach().mean()),lpips_vgg=float(lp.detach().mean()),soft_bpp=float(rate.detach().mean())))
        loss=torch.stack(terms).mean()+commit+cb
        return loss,dict(branches=records,commitment=float(commit.detach()),codebook=float(cb.detach()),arm=arm)

def rng_state(generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),endpoint_generator=generator.get_state())

def restore_rng(state,generator):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda']);generator.set_state(state['endpoint_generator'])

def state_sha(model):
    h=hashlib.sha256()
    for n,p in sorted(model.state_dict().items()):h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
