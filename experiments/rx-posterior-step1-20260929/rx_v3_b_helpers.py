"""Frozen v3 B scoring and calibration selection helpers; no development tuning."""
import rx_v3_common as c
from rx_v3_common import torch,np,Path,read,sha,seal,cell,OUT
ALGORITHM=Path(__file__).with_name('B_algorithm_v3.json')
PRIORS=['A1','A2','V']
def algorithm():return read(ALGORITHM)
def canonical(vae,var,truth):
    logits=var(torch.tensor([1000],device='cuda'),vae.quantize.idxBl_to_var_input(truth)).float()
    return [torch.log_softmax(x,-1) for x in c.b.split(logits)]
def profiles(selection):
    return [(p,v,1. if v=='fixed' or p=='A1' else selection['decision'][p]['lambda'])
            for p in PRIORS for v in (['decision'] if p=='A1' else ['decision','fixed'])]
def key(prior,version):return prior+'_'+version
def select_arrays(grid,proper,cfg):
    # grid: source,level,noise,prior,lambda,scale,(accuracy,logp)
    assert grid.ndim==7 and grid.shape[3:]==(3,4,10,2)
    means=grid.mean(axis=(0,2));pmeans=proper.mean(axis=(0,2))
    out=[]
    for li,level in enumerate(cfg['levels']):
        a1=means[li,0,2,:,0];band=np.flatnonzero((a1>=.2-1e-12)&(a1<=.95+1e-12)).tolist() # Inclusive endpoints tolerate float64 reduction roundoff only.
        rec=dict(**level,ambiguity_scales_1based=[k+1 for k in band],A1_TF_accuracy=a1.tolist(),decision={},probability_only={})
        for pi,prior in enumerate(PRIORS):
            vals=np.asarray([means[li,pi,j,band,0].mean() for j in range(4)]) if band else np.zeros(4)
            j=int(np.argmax(vals)) if band and prior!='A1' else 2
            rec['decision'][prior]=dict(beta=0,lambda_=cfg['lambda_grid'][j],calibration_accuracy_by_lambda=vals.tolist())
            rec['decision'][prior]['lambda']=rec['decision'][prior].pop('lambda_')
            if prior!='A1':
                flat=int(np.argmax(pmeans[li,pi-1]));bi,lj=np.unravel_index(flat,(4,4))
                rec['probability_only'][prior]=dict(beta=cfg['probability_only_beta_grid'][bi],lambda_=cfg['lambda_grid'][lj],calibration_true_logp=float(pmeans[li,pi-1,bi,lj]))
                rec['probability_only'][prior]['lambda']=rec['probability_only'][prior].pop('lambda_')
        out.append(rec)
    return out
@torch.no_grad()
def calibration_grid(vae,var,F,truth,sid,stats,static,cfg):
    lpv=canonical(vae,var,truth);L=len(cfg['levels']);grid=np.zeros((L,3,3,4,10,2));proper=np.zeros((L,3,2,4,4))
    residuals=[];r=F.clone()
    for k in range(10):residuals.append(r.clone());r.sub_(c.b.contribution(vae.quantize,truth[k],k))
    for li,level in enumerate(cfg['levels']):
        eta=level['eta'];vs=c.variances(stats,eta)
        for ni,seed in enumerate(cfg['calibration_noise_seeds']):
            noise=(eta*stats['sigma']*c.unit_noise(sid,seed)).cuda()
            for k in range(10):
                d,_=c.b.distance(vae.quantize,residuals[k]+noise,k);tr=truth[k]
                for pi,prior in enumerate(PRIORS):
                    lp=0 if prior=='A1' else (static[k][None,None] if prior=='A2' else lpv[k])
                    for lj,lam in enumerate(cfg['lambda_grid']):
                        score=-d/(2*float(vs[k]))+lam*lp
                        grid[li,ni,pi,lj,k]=[float(score.argmax(-1).eq(tr).double().mean()),float(torch.log_softmax(score,-1).gather(-1,tr[...,None]).double().mean())]
                    if k>=6 and prior!='A1':
                        for bi,beta in enumerate(cfg['probability_only_beta_grid']):
                            v=float(vs[k]+beta*stats['s2'][k])
                            for lj,lam in enumerate(cfg['lambda_grid']):
                                score=-d/(2*v)+lam*lp
                                proper[li,ni,pi-1,bi,lj]+=float(torch.log_softmax(score,-1).gather(-1,tr[...,None]).double().mean())/4
    return grid,proper
@torch.no_grad()
def tf_metrics(vae,var,F,truth,Z,stats,static,selection,canonical_lp=None):
    lpv=canonical_lp if canonical_lp is not None else canonical(vae,var,truth)
    r=Z.clone();records={key(p,v):dict(prior=p,version=v,lambda_=lam,scales=[],probability_only=[]) for p,v,lam in profiles(selection)}
    vs=c.variances(stats,selection['eta'])
    for k in range(10):
        d,_=c.b.distance(vae.quantize,r,k)
        for p,v,lam in profiles(selection):
            lp=0 if p=='A1' else (static[k][None,None] if p=='A2' else lpv[k])
            rec=records[key(p,v)]
            rec['scales'].append(c.scores_to_metrics(-d/(2*float(vs[k]))+lam*lp,truth[k]))
            if p!='A1':
                dg=selection['probability_only'][p];vd=float(vs[k]+dg['beta']*stats['s2'][k])
                rec['probability_only'].append(c.scores_to_metrics(-d/(2*vd)+dg['lambda']*lp,truth[k]))
        r.sub_(c.b.contribution(vae.quantize,truth[k],k))
    return records
