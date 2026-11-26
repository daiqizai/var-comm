"""Budget-constrained replica choice. The public receive call accepts a received waveform only."""
from dataclasses import fields, replace
from runtime import CFG,torch,vec_to_real_ring

def collect(channels,frames,users):
    cls = type(channels[0])
    values = {field.name:torch.stack([getattr(c,field.name) for c in channels],dim=1)[frames,users]
              for field in fields(channels[0])}
    return cls(**values)

def clean_channel(ch,z):
    norm = z/ch.pwr.clamp_min(1.e-8).sqrt()[:,None,None]
    return replace(ch,u_restored=z,u_mmse_restored=z,u_norm=norm,z_norm=norm)

class Router:
    def __init__(self,model,schedule,rule,track=False):
        self.model,self.schedule,self.rule = model,schedule,rule
        self.track = track
        self.logs = []
        self.weights = {}
    def observe(self,u,w,ch,cr):
        self.weights[u]=w

    def choices(self,step,stages,clean,gr,gi,active,k):
        b,u = active.shape
        device = gr.device
        groups = torch.arange(b,device=device)
        power = (gr.square()+gi.square()).mean((2,3))
        weak = power.masked_fill(active<.5,torch.inf).argmin(1)
        allowed = active>.5
        allowed[groups,weak] = False
        if self.rule in CFG["adaptive_rules"]:
            w = torch.stack([self.weights[i] for i in range(u)],dim=1)[groups,weak]
            phi = vec_to_real_ring(gr.permute(0,1,3,2),gi.permute(0,1,3,2))
            priority=(w[:,None]@phi).square().mean((2,3,4))
            if self.rule=="coupling_innovation":
                inn = [(cl-st[0].u_restored).square().mean((1,2))/
                       st[0].u_restored.square().mean((1,2)).clamp_min(1.e-8) for cl,st in zip(clean,stages)]
                priority=priority*torch.stack(inn,dim=1)
            ranks=torch.argsort(priority.masked_fill(~allowed,-torch.inf),dim=1,descending=True,stable=True)
            selected=ranks[:,:k]
        else:
            ranks=torch.argsort(power.masked_fill(~allowed,-torch.inf),dim=1,descending=True,stable=True)[:,:u-1]
            shift={"power":0,"second":1,"rotate":step}[self.rule]
            selected=torch.roll(ranks,-shift,dims=1)[:,:k]
        indices=groups[:,None].expand_as(selected)
        eligible=allowed[indices,selected]
        return indices[eligible],selected[eligible]

    def __call__(self,step,stages,clean,gr,gi,active):
        z=torch.stack(clean,dim=1)
        b,u=active.shape
        k = 4 if self.rule=="full" else CFG["schedules"][self.schedule][step]
        if k==0:
            if self.track: self.logs.append(torch.zeros_like(active,dtype=torch.int64))
            return z
        if self.rule=="full":
            pairs=torch.arange(b*u,device=z.device)
            frames,users=pairs//u,pairs%u
        else:
            frames,users=self.choices(step,stages,clean,gr,gi,active,k)
        if self.track:
            mask=torch.zeros_like(active,dtype=torch.int64)
            mask[frames,users]=1
            self.logs.append(mask)
        if frames.numel():
            ch=collect([s[0] for s in stages],frames,users)
            ch=clean_channel(ch,z[frames,users])
            rgb=self.model._decode_from_channel_out(ch,ch.c_r_db).clamp(0,1)
            stem=self.model.extract_encoder_stem(rgb)
            # The slot chooses a Walsh signature only in the discarded waveform output.
            _,encoded=self.model.encode_single_user_stem(stem,0)
            z[frames,users]=encoded
        return z

def demix(model,y,spec,track=False):
    schedule,rule=spec
    router=None if rule=="latent" else Router(model,schedule,rule,track)
    model.modem.replica_router=router
    model.modem.combiner_observer=router.observe if rule in CFG["adaptive_rules"] else None
    try:
        outs,_,aux=model.modem.blind_demix_users(y,4,model.latent_channels)
    finally:
        model.modem.replica_router=None
        model.modem.combiner_observer=None
    log=None
    if track:
        log=torch.stack(router.logs,dim=1) if router else y.new_zeros((len(y),3,4),dtype=torch.int64)
    return outs,aux["active_hat"],log

def decode_all(model,outs):
    return torch.stack([model._decode_from_channel_out(ch,snr).clamp(0,1) for ch,snr in outs],dim=1)

def weak_only(model,outs,weak):
    f=torch.arange(len(weak),device=weak.device)
    ch=collect([o[0] for o in outs],f,weak)
    return model._decode_from_channel_out(ch,ch.c_r_db).clamp(0,1)
