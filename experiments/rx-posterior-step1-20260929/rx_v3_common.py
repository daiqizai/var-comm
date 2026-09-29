"""RX v3 primitives. New diagnostic only; original training remains stopped."""
import run_preflight as env
import probe as b
import os,sys,time,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as fn
ROOT=b.ROOT;OUT=b.OUT/'revision_v3';DESIGN=Path(__file__).with_name('design_v3.json')
read=b.read;sha=b.digest;write=b.write_json
def seal(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('immutable cell already exists '+str(path))
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
    Path(str(path)+'.sha256').write_text(sha(path)+'\n')
def cell(path):
    path=Path(path)
    if not path.exists():return None
    assert Path(str(path)+'.sha256').read_text().strip()==sha(path),('seal',str(path))
    return read(path)
def status(stage,**kw):write(OUT/(stage+'_status.json'),dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))
def verify_registration(name):
    r=read(OUT/(name+'_registration.json'))
    for p,h in r['bindings'].items():assert sha(p)==h,('bound file changed',p)
    return sha(OUT/(name+'_registration.json'))
def load_stats():
    meta=read(b.OUT/'calibration_statistics.json')
    assert sha(b.OUT/'calibration_statistics.pt')==meta['sha256']
    return torch.load(b.OUT/'calibration_statistics.pt',map_location='cpu',weights_only=True)
def start():
    b.configure_runtime();b.require_available();b.SAFETY=b.Safety()
    vae,var=b.load_models(b.model_paths(),torch.device('cuda:0'))
    model=read(b.OUT/'models.json')
    assert b.state_sha256(vae)==model['vae_state_sha256'] and b.state_sha256(var)==model['var_state_sha256']
    stats=load_stats();static=[b.frequency_log_probs(x,.5).float().cuda() for x in stats['counts']]
    return vae,var,stats,static
def unit_noise(source,seed):return b.noise(source,seed)
def observe(F,stats,source,seed,eta):return F+(eta*stats['sigma']*unit_noise(source,seed)).to(F.device)
def variances(stats,eta,beta=0):return eta**2*stats['unit_noise_var'].double()+beta*stats['s2'].double()
def scores_to_metrics(score,truth):
    post=fn.log_softmax(score,dim=-1);prob=post.exp();confidence,idx=prob.max(-1)
    correct=idx.eq(truth);bins=(confidence*15).long().clamp(max=14)
    return dict(acc=float(correct.double().mean()),logp_true=float(post.gather(-1,truth[...,None]).double().mean()),
        entropy=float(-(prob.double()*post.double()).sum(-1).mean()),
        bins=[dict(count=int((bins==j).sum()),confidence_sum=float(confidence[bins==j].double().sum()),
                   correct_sum=int(correct[bins==j].sum())) for j in range(15)])
@torch.no_grad()
def infer(vae,var,Z,truth,prior,mode,variance,static,lam=1,clean=None,details=True,diag_variance=None,diag_lam=None):
    q=vae.quantize;rest=Z.clone();fhat=torch.zeros_like(Z)
    # The clean residual is an offline scoring branch and never enters MAP or the VAR prefix.
    clean_rest=clean.clone() if clean is not None else None
    pr=b.Prior(var) if prior=='V' else None;tokens=[];metrics=[];diags=[];logits=[]
    try:
        for k,pn in enumerate(b.PATCH_NUMS):
            d,_=b.distance(q,rest,k);lp=0
            if pr:
                lg=pr.logits(k);assert torch.isfinite(lg).all();logits.append(lg);lp=fn.log_softmax(lg,-1)
            elif prior=='A2':lp=static[k][None,None]
            v=float(variance[k]);assert v>=0
            if v==0:
                # Exact noise-only likelihood limit, including deterministic nearest-codeword ties.
                idx=d.argmin(-1);score=None
            else:
                score=-d/(2*v)+lam*lp;idx=score.argmax(-1)
            tokens.append(idx)
            if details:
                assert score is not None,'zero-noise limit has no finite posterior log score'
                m=scores_to_metrics(score,truth[k])
                if clean_rest is not None:
                    clean_d,_=b.distance(q,clean_rest,k)
                    m['path_accuracy']=float(idx.eq(clean_d.argmin(-1)).double().mean())
                metrics.append(m)
                if diag_variance is not None:
                    ds=-d/(2*float(diag_variance[k]))+float(diag_lam)*lp
                    diags.append(scores_to_metrics(ds,truth[k]))
            use=truth[k] if mode=='TF' else idx
            h=b.contribution(q,use,k);rest.sub_(h)
            if clean_rest is not None:clean_rest.sub_(h)
            fhat,nxt=q.get_next_autoregressive_input(k,10,fhat,b.embedded(q,use,k))
            if pr:pr.advance(nxt,k)
    finally:
        if pr:pr.close()
    return dict(fhat=fhat,tokens=tokens,metrics=metrics,diagnostic_metrics=diags,logits=logits,
        latent_error=float((clean.double()-fhat.double()).square().sum()) if clean is not None else None)
def load_quality():
    import yaml
    from var_comm.quality import load_quality_models
    cfg=yaml.safe_load((ROOT/'configs/progressive_channel.yaml').read_text())['quality']
    for key in ['dino_checkpoint','alexnet_checkpoint']:assert sha(cfg[key])==cfg[key+'_sha256']
    lp,dino,linear=load_quality_models(cfg,torch.device('cuda:0'))
    return lp,dino,dict(config=cfg,linear_weights_sha256=sha(linear),
                      lpips_state=b.state_sha256(lp),dino_state=b.state_sha256(dino))
def quality(source,images,lp,dino):
    from var_comm.quality import quality_metrics
    return quality_metrics(source,images,lp,dino,torch.device('cuda:0'))[0]
def dev_source(vae,record):
    pixels=record['pixels'];F=vae.quant_conv(vae.encoder(torch.as_tensor(pixels,device='cuda',dtype=torch.float32)[None]/127.5-1))
    truth=b.split(torch.as_tensor(record['tokens'],device='cuda',dtype=torch.long)[None])
    official=vae.quantize.f_to_idxBl_or_fhat(F,to_fhat=False)
    assert all(torch.equal(x,y) for x,y in zip(official,truth))
    return F,truth,b.cumulative(vae.quantize,truth)
def fusion(Fq,Z,stats,rho,eta):
    mu=stats['mu'].to(Z);sigma=stats['sigma'].to(Z);rho=rho.to(Z)
    assert torch.isfinite(rho).all() and (rho>=0).all()
    xq=(Fq-mu)/sigma;z=(Z-mu)/sigma
    # Algebraically equivalent to precision fusion; stable when rho is zero.
    return mu+sigma*((eta**2*xq+rho*z)/(eta**2+rho))
