"""Budget-conditioned discrete codec, with trainable E/Q/P/D and stride-8 detail.

One deterministic conditional reconstruction route. The decoder sees only paid
quantized values, their mask and the public paid mode. It never sees E features.
RDVQ supplies initialization and the causal entropy implementation, not frozen
targets. New latent transformers are active residual refiners, not unused size.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from runtime import setup
setup()
from codec_adapter import load_models

BUDGETS=(512,768,1024,1536,2048,3072,4096,8192)
def mode_spec(mode):
    assert 0<=int(mode)<16
    return (256 if int(mode)<8 else 512),BUDGETS[int(mode)%8]*(1 if int(mode)<8 else 4)
def shapes_for(size):return [(size//64,size//64),(size//32,size//32),(size//16,size//16),(size//8,size//8)]
def ck(module,x):
    return checkpoint(module,x,use_reentrant=False) if torch.is_grad_enabled() and module.training else module(x)

class Block(nn.Module):
    def __init__(self,d,heads):
        super().__init__();self.heads=heads;self.n1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d)
        self.proj=nn.Linear(d,d);self.n2=nn.LayerNorm(d);self.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x):
        b,n,d=x.shape;q,k,v=self.qkv(self.n1(x)).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.).transpose(1,2).reshape(b,n,d)
        x=x+self.proj(y);return x+self.mlp(self.n2(x))

class LatentRefiner(nn.Module):
    def __init__(self,d,layers,heads,with_mask=False):
        super().__init__();self.with_mask=with_mask
        self.input=nn.Linear(33 if with_mask else 32,d);self.mode=nn.Embedding(16,d)
        self.position=nn.Linear(6,d);self.blocks=nn.ModuleList([Block(d,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(d);self.output=nn.Linear(d,32)
        nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)
    def forward(self,z,modes,shapes,mask=None):
        pos=[]
        for i,(h,w) in enumerate(shapes):
            yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=z.device),torch.linspace(-1,1,w,device=z.device),indexing='ij')
            pos.append(torch.stack([xx,yy,xx.square(),yy.square(),torch.ones_like(xx)*i/3,torch.ones_like(xx)*math.log2(h)/6],-1).reshape(-1,6))
        inp=torch.cat([z,mask[...,None].to(z.dtype)],-1) if self.with_mask else z
        x=self.input(inp)+self.mode(modes)[:,None]+self.position(torch.cat(pos))[None]
        for block in self.blocks:x=ck(block,x)
        return self.output(self.norm(x))

class MultiRateCodec(nn.Module):
    def __init__(self,checkpoint_path):
        super().__init__();self.vq,self.ar,_=load_models(checkpoint_path)
        torch.manual_seed(20261105)
        self.encoder_refiner=LatentRefiner(768,12,12)
        self.decoder_refiner=LatentRefiner(1024,24,16,True)
        self.fine_encoder=nn.Conv2d(256,32,3,padding=1)
        self.fine_decoder=nn.Conv2d(32,512,3,padding=1)
        nn.init.normal_(self.fine_decoder.weight,std=.001);nn.init.zeros_(self.fine_decoder.bias)
        self.requires_grad_(True);self.cuda().float()
        self.mask_cache={}
    @property
    def codebook(self):return F.normalize(self.vq.quantize.embedding.weight.float(),p=2,dim=-1)
    def mask(self,size):
        if size not in self.mask_cache:
            from autoregressive.models.mask_generation import generate_MS_ck_mask
            _,m=generate_MS_ck_mask(shapes_for(size),device=self.codebook.device,use_patch_ck_ar=True,k=4)
            self.mask_cache[size]=m.long()
        return self.mask_cache[size]
    def encode_features(self,x):
        e=self.vq.encoder;h=e.conv_in(x);inputs={};fine=None
        for level in range(e.num_resolutions):
            for j in range(e.num_res_blocks):
                h=ck(e.down[level].block[j],h)
                if len(e.down[level].attn):h=e.down[level].attn[j](h)
            if level==3:fine=h
            if level in e.scales_id:inputs[e.scales[e.scales_id.index(level)]]=h
            if level!=e.num_resolutions-1:h=e.down[level].downsample(h)
        hs={}
        for s in e.scales:
            v=ck(e.mid[s].block_1,inputs[s]);v=e.mid[s].attn_1(v);v=ck(e.mid[s].block_2,v)
            hs[s]=e.conv_out[s](F.silu(e.norm_out[s](v)))
        if e.enc_residual:
            for i,s in enumerate(e.scales):
                for j in range(i):hs[s]=hs[s]-F.interpolate(hs[e.scales[j]],scale_factor=2**(i-j),mode='bicubic',align_corners=False)
        return [self.vq.quant_conv_list[s](hs[s]) for s in e.scales]+[self.fine_encoder(fine)]
    def encode(self,x,modes):
        shapes=shapes_for(x.shape[-1]);z=torch.cat([v.flatten(2).transpose(1,2) for v in self.encode_features(x)],1)
        z=F.normalize((z+self.encoder_refiner(z,modes,shapes)).float(),dim=-1)
        cb=self.codebook;d=z.square().sum(-1,keepdim=True)+cb.square().sum(-1)-2*z@cb.T
        ids=d.argmin(-1);hard=cb[ids];quant=z+(hard-z).detach()
        soft=F.softmax(-d/.1,-1)
        vqloss=.25*F.mse_loss(z,hard.detach())+F.mse_loss(hard,z.detach())
        return quant,ids,soft,vqloss
    def entropy_logits(self,quant,size):
        from autoregressive.models.mask_generation import rerank_input
        # Upstream reranking includes FP32 bicubic interpolation. Keep this
        # probability path in FP32, matching real rANS TX/RX arithmetic.
        with torch.autocast('cuda',enabled=False):
            ar=self.ar;x=ar.input_proj(quant.float());start=ar.start_token.expand(len(quant),-1,-1)
            query=rerank_input(x,start,shapes_for(size),self.mask(size),k=4)
            return ar.transformer(ranked_input=query,shape_list=shapes_for(size)).float()
    def reconstruct(self,received_quant,modes,keep,size):
        # Enforce erasure before any reconstructor input. Missing codebook ID 0
        # is never substituted for absence of information.
        z=received_quant*keep[...,None]
        z=z+self.decoder_refiner(z,modes,shapes_for(size),keep)
        branches=[];lo=0
        for h,w in shapes_for(size):
            n=h*w;branches.append(z[:,lo:lo+n].transpose(1,2).reshape(-1,32,h,w));lo+=n
        d=self.vq.decoder;mid={}
        for i,s in enumerate(d.scales):
            v=d.conv_in[s](self.vq.post_quant_conv_list[s](branches[i]));v=ck(d.mid[s].block_1,v)
            v=d.mid[s].attn_1(v);mid[s]=ck(d.mid[s].block_2,v)
        h=0
        for level in reversed(range(d.num_resolutions)):
            if level in d.scales_id:h=h+mid[d.scales[d.scales_id.index(level)]]
            if level==3:h=h+self.fine_decoder(branches[3])
            for j in range(d.num_res_blocks+1):
                h=ck(d.up[level].block[j],h)
                if len(d.up[level].attn):h=d.up[level].attn[j](h)
            if level:h=d.up[level].upsample(h)
        out=d.conv_out(F.silu(d.norm_out(h)))
        return torch.where(keep.any(1)[:,None,None,None],out,torch.zeros_like(out))
    def forward(self,x,modes,full=False,perceptual=None):
        size=x.shape[-1];quant,ids,soft,vqloss=self.encode(x,modes)
        logits=self.entropy_logits(quant,size);logp=F.log_softmax(logits,-1)/math.log(2)
        hardbits=-logp.detach().gather(-1,ids[...,None]).squeeze(-1)
        mask=self.mask(size);cost=torch.stack([hardbits[:,mask<t].sum(1)+200 for t in range(1,61)],1)
        budget=torch.tensor([mode_spec(int(m))[1] for m in modes],device=x.device)
        endpoint=torch.where(cost<=budget[:,None],torch.arange(1,61,device=x.device),0).max(1).values
        if full:endpoint=torch.full_like(endpoint,60)
        keep=mask[None]<endpoint[:,None]
        rec=self.reconstruct(quant,modes,keep,size)
        mse=(rec.float()-x.float()).square().mean();lp=mse*0 if perceptual is None else perceptual(rec.float(),x.float()).mean()
        bits=-(soft*logp).sum(-1);rate=(bits*keep).sum(1)/(size*size)
        # The differentiable proxy is supervised by actual byte audits; only
        # encode_packet performs the exact hard budget check at deployment.
        excess=F.relu((bits*keep).sum(1)+200-budget)/(size*size)
        loss=mse+lp+rate.mean()+4*excess.mean()+vqloss
        return loss,dict(mse=mse.detach(),lpips_vgg=lp.detach(),soft_bpp=rate.detach().mean(),vq=vqloss.detach(),
                         endpoints=endpoint.detach(),full=full),rec
    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    components={n:sum(p.numel() for p in m.parameters()) for n,m in self.named_children()})
