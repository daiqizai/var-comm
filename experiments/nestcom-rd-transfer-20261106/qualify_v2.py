"""Finite real-weight checks; never resumes qualification updates as main training."""
from pathlib import Path
import argparse,json,hashlib,time
from runtime import *
setup()
import torch,numpy as np
from model_v2 import Transfer,tensor_hash
from packet import encode_fit,decode,pack,unpack
from data import image
from perceptual import load as load_lp
def digest(x):return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['tx','rx']);a=ap.parse_args()
    old_paused();numerical();torch.cuda.set_device(0)
    out=L/'qualification_v2';out.mkdir(exist_ok=True);
    from resize_gradient_v2 import check
    derivatives=check()
    write(out/('resize_'+a.mode+'.json'),dict(passed=True,checks=derivatives))
    r=json.loads((L/'protocol_v2.json').read_text())
    model=Transfer(L/'model_routes.json');lp=load_lp() if a.mode=='tx' else None
    if a.mode=='rx':
        rec=json.loads((out/'rx_checkpoint.json').read_text());assert sha(rec['path'])==rec['sha256']
        model.load_state_dict(torch.load(rec['path'],map_location='cpu',weights_only=False)['model'],strict=True)
        rows=[]
        # This process receives checkpoint + public packets; never reads source images,
        # source tokens, TX CDFs, or TX expectation arrays.
        for name in ['partial','full','null']:
            packet=np.load(out/'public'/f'{name}.npy');received=decode(model,packet)
            if len(received):_,latent=model.receive(received);y=model.decoder(latent)
            else:y=torch.full((1,3,256,256),.5,device='cuda')
            rows.append(dict(name=name,tokens=len(received),token_sha256=hashlib.sha256(received.tobytes()).hexdigest(),image_sha256=digest(y)))
        write(out/'rx_pass.json',dict(passed=True,rows=rows,unsent_truth_inputs=False,checkpoint_sha256=rec['sha256']));return
    rows=json.loads((L/'qualification_sources.json').read_text())['sources'];x=image(rows[0])[None].cuda()
    with torch.no_grad():
        ids,_,_,full,_=model.quantize(x,soft=False);old=model.vae.img_to_idxBl(x)
        assert all(torch.equal(a,b) for a,b in zip(ids,old))
        h=[model.vae.quantize.embedding(q).transpose(1,2).reshape(1,32,p,p) for p,q in zip(SIZES,old)]
        ref=model.vae.quantize.embed_to_fhat(h,all_to_max_scale=True,last_one=True)
        assert torch.equal(ref,full),(ref-full).abs().max()
        assert torch.equal(model.decoder(ref),model.decoder(full))
    # Prove the new rate gradient enters the encoder and that detach disables it.
    ids,soft,_,_,_=model.quantize(x);logits=model.teacher_logits(ids);logp=torch.log_softmax(logits,-1)
    p=torch.cat(soft,1);rate=-(p[:,:250]*logp[:,:250]).sum()/65536
    eparam=model.groups()['encoder'][0];prparam=model.prior.head.weight
    eg,pg=torch.autograd.grad(rate,[eparam,prparam],retain_graph=True)
    detached=-(p[:,:250].detach()*logp[:,:250]).sum()/65536
    noeg=torch.autograd.grad(detached,eparam,allow_unused=True)[0]
    assert eg.isfinite().all() and eg.norm()>0 and pg.isfinite().all() and pg.norm()>0
    assert noeg is None or torch.count_nonzero(noeg)==0
    gradients=dict(rate_encoder_norm=float(eg.norm()),rate_prior_norm=float(pg.norm()),detached_rate_encoder_zero=True)
    del ids,soft,logits,logp,p,rate,eg,pg,detached,noeg
    qhash=tensor_hash(model.vae.quantize)
    opt=torch.optim.AdamW([{'params':ps,'lr':r['learning_rates'][k]} for k,ps in model.groups().items()],betas=(.9,.95),weight_decay=.01)
    history=[]
    for i,row in enumerate(rows):
        xx=image(row)[None].cuda();opt.zero_grad(set_to_none=True);tick=time.time()
        loss,rec=model(xx,r['source_capacities'][i],lp);assert loss.isfinite();loss.backward()
        norms={k:float(torch.sqrt(sum(p.grad.square().sum() for p in ps if p.grad is not None))) for k,ps in model.groups().items()}
        assert all(np.isfinite(v) and v>0 for v in norms.values())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
        assert tensor_hash(model.vae.quantize)==qhash
        history.append(dict(loss=float(loss.detach()),gradients=norms,seconds=time.time()-tick,**rec))
    CK.mkdir(parents=True,exist_ok=True);ck=CK/'qualification_v2_only.pt';assert not ck.exists()
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),qualification_updates=2,not_main_initialization=True),ck)
    write(out/'rx_checkpoint.json',dict(path=str(ck),sha256=sha(ck)))
    public=out/'public';public.mkdir(exist_ok=False);expected=[]
    with torch.no_grad():
        ids,*_=model.quantize(x,soft=False)
        for name,cap in [('partial',r['source_capacities'][1]),('full',7000),('null',0)]:
            packet,rec=encode_fit(model,ids,cap);received=decode(model,packet)
            assert np.array_equal(received,torch.cat(ids,1)[0,:len(received)].cpu().numpy())
            if len(received):_,f=model.receive(received);y=model.decoder(f)
            else:y=torch.full_like(x,.5)
            if name=='full':assert len(received)==680
            np.save(public/f'{name}.npy',packet)
            expected.append(dict(name=name,tokens=len(received),token_sha256=hashlib.sha256(received.tobytes()).hexdigest(),image_sha256=digest(y)))
    write(out/'tx_pass.json',dict(passed=True,hard_token_parity=True,hard_latent_and_pixel_parity=True,initial_model_hashes=model.initial_hashes,gradients=gradients,updates=history,frozen_quantizer_sha256=qhash,expected=expected,parameters=model.counts(),numeric_route='FP32, TF32 off, math SDPA, deterministic, batch1',cuda_name=torch.cuda.get_device_name(),torch=torch.__version__,max_memory_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':main()
