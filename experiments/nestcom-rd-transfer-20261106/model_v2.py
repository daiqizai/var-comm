"""Original VAR hard forward plus explicit soft-rate / straight-through estimators.

No image gradient is claimed through argmax completion. The prior learns from
soft-rate and hard-token cross entropy; image gradients update E and D_c.
"""
import copy,json,math,sys
from pathlib import Path
from runtime import R,L,SIZES,definitions,sha,setup
setup()
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from resize_gradient_v2 import resize

def tensor_hash(model):
    ns=dict(torch=torch,np=np,hashlib=__import__('hashlib'))
    definitions(R/'src/var_comm/next_scale_prior.py',{'state_sha256'},ns)
    return ns['state_sha256'](model)

class Transfer(nn.Module):
    def __init__(self,routes_path,verify=True):
        super().__init__();routes=json.loads(Path(routes_path).read_text())
        if verify:
            for item in routes['files']+routes['author_files']+[routes['dist']]:assert sha(item['path'])==item['sha256'],item['path']
        paths={'var_source':routes['author_root']}
        for item in routes['files']:
            key='vae_checkpoint' if item['path'].endswith('vae_ch160v4096z32.pth') else 'var_checkpoint' if item['path'].endswith('var_d16.pth') else 'decoder_checkpoint'
            paths[key]=item['path']
        ns=dict(torch=torch,Path=Path,sys=sys,PATCH_NUMS=SIZES,VOCAB_SIZE=4096)
        definitions(R/'src/var_comm/next_scale_prior.py',{'load_models'},ns)
        self.vae,self.prior=ns['load_models'](paths,torch.device('cuda',torch.cuda.current_device()))
        ns=dict(torch=torch,nn=nn,copy=copy)
        definitions(R/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py',{'ContinuousDecoder'},ns)
        self.decoder=ns['ContinuousDecoder'](self.vae).cuda()
        state=torch.load(paths['decoder_checkpoint'],map_location='cpu',weights_only=True)
        self.decoder.load_state_dict(state['decoder'],strict=True);del state
        self.initial_hashes={k:tensor_hash(v) for k,v in [('vae',self.vae),('var',self.prior),('decoder',self.decoder)]}
        assert self.initial_hashes==routes['models']
        self.vae.requires_grad_(False)
        self.vae.encoder.requires_grad_(True);self.vae.quant_conv.requires_grad_(True)
        self.prior.requires_grad_(True);self.decoder.requires_grad_(True)
        # Native decoder uses nearest-neighbor upsampling; keep its forward exact.
        for module in self.decoder.modules():
            if type(module).__name__=='Upsample':
                import types
                def up(this,x):return this.conv(resize(x,(2*x.shape[-2],2*x.shape[-1]),'nearest'))
                module.forward=types.MethodType(up,module)
        self.eval() # disables dropout, not gradients; coding and training share deterministic modules.
        ns=dict(torch=torch,np=np,SIZES=SIZES)
        definitions(R/'experiments/scale-causal-partial-residual-20261002/partial_receiver.py',{'_Prior'},ns)
        self.Prior=ns['_Prior']
    def groups(self):
        return {'encoder':list(self.vae.encoder.parameters())+list(self.vae.quant_conv.parameters()),'prior':list(self.prior.parameters()),'decoder':list(self.decoder.parameters())}
    def counts(self):
        return {k:sum(p.numel() for p in v) for k,v in self.groups().items()}
    def contribution(self,s,embedding):
        pn=SIZES[s];h=embedding.transpose(1,2).reshape(-1,32,pn,pn)
        if s<9:h=resize(h,(16,16),'bicubic')
        return self.vae.quantize.quant_resi[s/9](h)
    def quantize(self,x,soft=True):
        f=self.vae.quant_conv(self.vae.encoder(x));q=self.vae.quantize
        rest=f.detach().clone();full=torch.zeros_like(rest);ids=[];prob=[];ste=[];commit=[]
        book=q.embedding.weight
        for s,pn in enumerate(SIZES):
            # Hard residual updates use the exact original numerical order.
            z=(F.interpolate(rest,size=(pn,pn),mode='area') if s<9 else rest).permute(0,2,3,1).reshape(-1,32)
            with torch.no_grad():
                if q.using_znorm:idx=(F.normalize(z,dim=-1)@F.normalize(book,dim=-1).T).argmax(1)
                else:
                    d=z.square().sum(1,keepdim=True)+book.square().sum(1)
                    d.addmm_(z,book.T,alpha=-2,beta=1);idx=d.argmin(1)
                idx=idx.reshape(x.shape[0],pn*pn);hard=q.embedding(idx)
                h=self.contribution(s,hard);full=full+h;rest=rest-h
            ids.append(idx)
            if soft:
                # Frozen hard previous-scale contributions; no detached encoder in this branch.
                zr=f-(f.detach()-rest-h).detach()
                zr=(resize(zr,(pn,pn),'area') if s<9 else zr).flatten(2).transpose(1,2)
                if q.using_znorm:dist=-F.normalize(zr,dim=-1)@F.normalize(book,dim=-1).T
                else:dist=zr.square().sum(-1,keepdim=True)+book.square().sum(-1)-2*zr@book.T
                p=F.softmax(-dist/0.1,dim=-1);prob.append(p)
                # Hard values at forward time, identity local STE for image distortion.
                ste.append(hard+(zr-zr.detach()))
                commit.append(F.mse_loss(zr,hard.detach()))
        return ids,prob,ste,full,torch.stack(commit).mean() if soft else f.new_zeros(())
    def teacher_logits(self,ids):
        v=self.prior
        with torch.no_grad():inputs=self.vae.quantize.idxBl_to_var_input(ids)
        B=ids[0].shape[0];cond=v.class_emb(torch.full((B,),1000,dtype=torch.long,device=ids[0].device))
        sos=cond[:,None].expand(B,v.first_l,-1)+v.pos_start
        h=torch.cat([sos,v.word_embed(inputs)],1)+v.lvl_embed(v.lvl_1L)+v.pos_1LC
        c=v.shared_ada_lin(cond);bias=v.attn_bias_for_masking
        for block in v.blocks:
            def apply(x,cc,block=block):return block(x=x,cond_BD=cc,attn_bias=bias)
            h=checkpoint(apply,h,c,use_reentrant=False)
        return v.get_logits(h,cond)
    @torch.no_grad()
    def receive(self,received):
        """Receiver context from accepted tokens only. No source or unsent IDs."""
        a=np.asarray(received,dtype=np.int64);pr=self.Prior(self.vae,self.prior)
        out=[];offset=0
        try:
            for s,pn in enumerate(SIZES):
                y=pr.logits(s)[0].argmax(-1).cpu().numpy();n=min(max(len(a)-offset,0),pn*pn)
                if n:y[:n]=a[offset:offset+n]
                offset+=n;pr.advance(y,s);out.append(torch.as_tensor(y,device='cuda')[None])
            assert offset==len(a)
            return out,pr.fhat.clone()
        finally:pr.close()
    def forward(self,x,capacity,perceptual,rate_encoder=True):
        from packet import encode_fit,decode
        assert x.shape[0]==1
        ids,soft,ste,full,commit=self.quantize(x)
        # Actual terminated arithmetic lengths select the endpoint. Independently decode.
        packet,record=encode_fit(self,ids,capacity)
        recovered=decode(self,packet)
        flat=torch.cat(ids,1)[0];assert np.array_equal(recovered,flat[:len(recovered)].detach().cpu().numpy())
        chosen,rxlatent=self.receive(recovered)
        # quant_resi has bias. A derivative-only difference retains exact hard values.
        partial=rxlatent.detach();full_ste=full.detach();pos=0
        for s,pn in enumerate(SIZES):
            hard=self.vae.quantize.embedding(ids[s]);n=min(max(len(recovered)-pos,0),pn*pn);pos+=pn*pn
            mask=hard.new_zeros((1,pn*pn,1));mask[:,:n]=1
            h=self.contribution(s,ste[s]);full_ste=full_ste+(h-h.detach())
            h=self.contribution(s,hard+(ste[s]-hard)*mask);partial=partial+(h-h.detach())
        full_img=self.decoder(full_ste);partial_img=self.decoder(partial) if len(recovered) else x.new_full(x.shape,.5)
        target=(x+1)/2
        fm=F.mse_loss(full_img,target);pm=F.mse_loss(partial_img,target)
        fl=perceptual(full_img*2-1,x).mean();pl=perceptual(partial_img*2-1,x).mean()
        logits=self.teacher_logits(ids);lp=F.log_softmax(logits,dim=-1)
        probs=torch.cat(soft,1);probs=probs if rate_encoder else probs.detach()
        n=len(recovered);rate=-(probs[:,:n]*lp[:,:n]).sum()/math.log(2)/(256*256)
        ce=F.cross_entropy(logits.reshape(-1,4096),flat)
        loss=pm+.1*pl+fm+.1*fl+.05*rate+.01*ce+.25*commit
        return loss,dict(full_mse=float(fm.detach()),partial_mse=float(pm.detach()),full_lpips_vgg=float(fl.detach()),partial_lpips_vgg=float(pl.detach()),soft_bpp=float(rate.detach()),hard_ce=float(ce.detach()),commitment=float(commit.detach()),**record)
