"""Matched original-NeST versus fixed step5000 clean-packet evaluation.

Four bounded streams retain the existing 12+13+16-bit packet format. The full
representation is a separate, unconstrained canonical arithmetic stream, not a
fixed-budget wireless frame. No TX-only information enters receiver contexts.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from runtime import SIZES,sha,S
sys.path.insert(0,str(S))
from packet import ENT,Encoder,Decoder,pack,decode

CAPACITIES=(591,1273,1911,2548)
METRICS=('psnr_db','lpips_alex','dists','dreamsim','clip_i')

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def pixels(row):
    from torchvision.transforms import functional as TF
    assert sha(row['path'])==row['sha256']
    with Image.open(row['path']) as im:
        im=im.convert('RGB');native=np.array(im)
        assert digest(native)==row['native_rgb_sha256']
        im=TF.center_crop(TF.resize(im,256,interpolation=TF.InterpolationMode.BICUBIC,antialias=True),[256,256])
        a=np.array(im)
    if 'preprocessed_sha256' in row:assert digest(a)==row['preprocessed_sha256']['256']
    assert a.shape==(256,256,3)
    return a

def source_tensor(a):return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).float().cuda()[None]/127.5-1

def load_model(r,method):
    from model_v2 import Transfer,tensor_hash
    model=Transfer(r['model_routes'])
    if method=='trained5000':
        ck=r['checkpoint'];assert sha(ck['path'])==ck['sha256']
        state=torch.load(ck['path'],map_location='cpu',weights_only=False,mmap=True)
        assert state['step']==5000
        model.load_state_dict(state['model'],strict=True);del state
    else:assert method=='original_nest'
    assert tensor_hash(model.vae.quantize)==r['frozen_quantizer_sha256']
    return model.eval().requires_grad_(False)

@torch.no_grad()
def encode_all(model,ids):
    pr=model.Prior(model.vae,model.prior);enc=Encoder();count=0;previous=0
    best={c:(0,np.array([0,0],dtype=np.uint8)) for c in CAPACITIES};cdfhash=hashlib.sha256()
    try:
        for s,pn in enumerate(SIZES):
            table=ENT.probability_cdf(torch.log_softmax(pr.logits(s),-1)[0].cpu().numpy())
            cdfhash.update(table.tobytes());q=ids[s][0].cpu().numpy()
            for j,tok in enumerate(q):
                enc.encode(np.array([tok]),table[j:j+1]);count+=1
                length=len(enc.bits)+enc.pending+2;assert length>=previous;previous=length
                feasible=[c for c in CAPACITIES if count>=30 and length<=c]
                if feasible:
                    bits=enc.finish().copy();assert len(bits)==length
                    for c in feasible:best[c]=(count,bits)
            pr.advance(q,s)
        assert count==680
        full=enc.finish().copy();assert len(full)==previous
        out=[]
        for c,(n,bits) in best.items():
            wire=pack(n,bits);assert len(wire)<=c+41
            out.append((str(c),wire,dict(condition=str(c),source_capacity=c,packet_budget_bits=c+41,
                packet_bits=len(wire),source_bits=len(bits),tokens=n,null=n==0,kind='bounded_packet')))
        out.append(('full',full,dict(condition='full',source_capacity=None,packet_budget_bits=None,
                packet_bits=None,source_bits=len(full),tokens=680,null=False,kind='unconstrained_full_stream')))
        return out,cdfhash.hexdigest()
    finally:pr.close()

@torch.no_grad()
def decode_full(model,bits):
    pr=model.Prior(model.vae,model.prior);dec=Decoder(bits);enc=Encoder();values=[]
    try:
        for s,pn in enumerate(SIZES):
            table=ENT.probability_cdf(torch.log_softmax(pr.logits(s),-1)[0].cpu().numpy())
            q=dec.decode(table);enc.encode(q,table);values.extend(q.tolist());pr.advance(q,s)
        assert len(values)==680 and np.array_equal(enc.finish(),bits),'Full stream is noncanonical'
        return np.array(values,dtype=np.int64)
    finally:pr.close()

@torch.no_grad()
def receive(model,wire,kind):
    ids=decode_full(model,wire) if kind=='unconstrained_full_stream' else decode(model,wire)
    if ids.size:
        _,latent=model.receive(ids);rgb=model.decoder(latent)[0].cpu().float().numpy()
    else:rgb=np.full((3,256,256),.5,dtype=np.float32)
    assert rgb.dtype==np.float32 and rgb.shape==(3,256,256) and np.isfinite(rgb).all()
    assert rgb.min()>=0 and rgb.max()<=1
    return np.ascontiguousarray(rgb),ids

class FiveMetrics:
    def __init__(self,spec):
        from metrics_v2 import MetricSuite
        self.base=MetricSuite(spec)
        with self.base.met.no_network():self.clip=self.base.met.load_clip(spec['clip'],torch.device('cuda:0'))
        self.clipcache={}
    @torch.no_grad()
    def clip_feature(self,a):
        key=digest(a)
        if key not in self.clipcache:
            x=torch.from_numpy(a[None]).cuda()
            self.clipcache[key]=self.clip.encode_image(self.base.met.preprocess_clip(x)).float()
        return self.clipcache[key]
    @torch.no_grad()
    def score(self,x,y):
        v=self.base.score(x,y)
        v['clip_i']=float(torch.nn.functional.cosine_similarity(self.clip_feature(x),self.clip_feature(y),dim=-1)[0])
        assert all(np.isfinite(v[k]) for k in METRICS)
        return v
