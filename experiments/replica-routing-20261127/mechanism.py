"""Matched-gate intervention with receive-only observation and an offline oracle scorer."""
import csv
from dataclasses import replace
import math
from runtime import *

class Trace(ReceiverTrace):
    def __init__(self,model,arm,gates):
        super().__init__(model,{"linear":"linear_fixed_gate","latent":"latent","rgb":"rgb_reencode"}[arm],gates)
        self.states = []
    def observe(self,u,w,ch,cr):
        if u == 3:
            self.states.append((w,ch,cr))
    def gate(self,step,user,value):
        key = step,user
        assert key not in self.gates
        self.gates[key] = (value if self.reference_gates is None else self.reference_gates[key]).detach()
        return self.gates[key]

def power(x):
    return x.abs().square().flatten(1).mean(1)

def ratio(a,b):
    return 10*torch.log10((power(a)/power(b).clamp_min(1.e-12)).clamp_min(1.e-12))

def signals(tx,gain,noise):
    s = torch.view_as_complex((tx / tx.square().sum(-1).mean((2,3),keepdim=True).clamp_min(1.e-8).sqrt().unsqueeze(-1)).contiguous())
    h = torch.complex(gain["g_true_r"],gain["g_true_i"])
    w = build_walsh_16_basis(12,384,device=s.device,dtype=s.real.dtype).to(s.dtype)
    def strip(z):
        return z-torch.einsum("brmk,kn->brmn",torch.einsum("brmn,kn->brmk",z,w),w)
    return (strip(torch.einsum("burm,bumn->brmn",h[:,:3],s[:,:3])),
            strip(h[:,3,:,:,None]*s[:,3,None,:,:]),
            strip(torch.view_as_complex(noise.cuda().contiguous())/math.sqrt(2.)))

def comb(w,x):
    return w@torch.cat((x.real,x.imag),dim=1).permute(0,2,1,3)

def metrics(model,trace,oracle,latent):
    interf,desired,noise = oracle
    result = []
    for (w,ch,cr),(re,im) in zip(trace.states[1:],trace.subtractions):
        err = interf-torch.complex(re,im)
        clean_norm = latent/ch.pwr.clamp_min(1.e-8).sqrt()[:,None,None]
        counter = replace(ch,u_restored=latent,u_mmse_restored=latent,u_norm=clean_norm,z_norm=clean_norm)
        front = model.blind_dec.forward_frontend
        actual = front(ch,cr,base_c0_db=model.base_c0_db)[0]
        ideal = front(counter,cr,base_c0_db=model.base_c0_db)[0]
        result.append(torch.stack([ratio(err,interf),ratio(comb(w,err),comb(w,interf)),
            ratio(comb(w,err+noise),comb(w,desired)),ratio(ch.u_restored-latent,latent),
            ratio(actual-ideal,ideal)],dim=1).cpu().tolist())
    return result

def anchors(seed,cond):
    filename = f"common500_{cond}_{seed}.csv"
    ref = {}
    with (PARENT/"events"/filename).open() as f:
        for z in csv.DictReader(f):
            if z["slot"]=="3" and z["arm"] in ("linear_fixed_gate","latent"):
                key = "linear" if z["arm"].startswith("linear") else "latent"
                ref[key,int(z["group"])] = (z["source"],*[float(z[k]) for k in ("psnr_db","dists","interference_nmse_db_r3")])
    with (PARENT/"gate_events"/filename).open() as f:
        for z in csv.DictReader(f):
            ref["rgb",int(z["group"])] = (z["source"],*[float(z[k]) for k in ("weak_psnr_db","weak_dists","interference_nmse_db_r3")])
    assert len(ref)==375
    return ref

@torch.inference_mode()
def main():
    out,bb,start = init("mechanism")
    images,ids,stems = source_set(bb,"common500",out)
    evaluator = OfflineVisualEvaluator(torch.device("cuda:0"),include_lpips=False)
    errs=[0.,0.,0.]
    total=0
    for seed in CFG["seeds"]:
        model = weight(bb,seed)
        for cond,levels in CFG["conditions"].items():
            bank = make_bank(125,seed,levels,"original")
            ref = anchors(seed,cond)
            results = []
            for st in range(0,125,12):
                en=min(st+12,125)
                y,tx,gain,lat = transmit(model,stems,bank,st,en)
                snapshot = y.clone()
                target = images[4*st+3:4*en:4].cuda().float()/255.
                oracle = signals(tx,gain,bank[2][st:en])
                gates=None
                weights=None
                for name in ("latent","linear","rgb"):
                    trace=Trace(model,name,gates)
                    model.modem.study_trace=trace
                    model.modem.combiner_observer=trace.observe
                    try:
                        stages,_,_=model.modem.blind_demix_users(y,4,model.latent_channels)
                    finally:
                        model.modem.study_trace=None
                        model.modem.combiner_observer=None
                    assert torch.equal(y,snapshot) and len(trace.states)==4 and len(trace.subtractions)==3 and len(trace.gates)==12
                    if name=="latent":
                        gates = trace.gates
                        weights = [s[0] for s in trace.states]
                    else:
                        assert all(torch.equal(value,gates[k]) for k,value in trace.gates.items())
                        assert all(torch.equal(a,b[0]) for a,b in zip(weights,trace.states))
                    data = metrics(model,trace,oracle,lat[3])
                    rgb = model._decode_from_channel_out(*stages[3]).clamp(0,1)
                    scores = compute_psnr_batch(target,rgb).tolist()
                    ds=evaluator.dists_model(target,rgb,require_grad=False,batch_average=False).reshape(-1).tolist()
                    for j,g in enumerate(range(st,en)):
                        key = ids[4*g+3]
                        old = ref[name,g]
                        assert old[0]==key
                        for i,(new,oldv) in enumerate(zip((scores[j],ds[j],data[2][j][0]),old[1:])):
                            errs[i]=max(errs[i],abs(new-oldv))
                        for round_idx,t in enumerate(data):
                            pre,post,totalerr,rawz,decin=t[j]
                            results.append(dict(seed=seed,condition=cond,group=g,weak_source=key,replica=name,round=round_idx+1,
                                antenna_db=pre,combiner_db=post,interference_noise_db=totalerr,latent_db=rawz,
                                frontend_db=decin,weak_psnr=scores[j] if round_idx==2 else "",
                                weak_dists=ds[j] if round_idx==2 else ""))
            assert errs[0]<.003 and errs[1]<.00015 and errs[2]<.00015,errs
            rows(out/f"{cond}_{seed}.csv",results)
            total+=len(results)
            print(cond,seed,len(results),errs,flush=True)
    assert total==6750
    finish(out,start,round_rows=total,distinct_observations=750,weak_reconstructions_and_scores=2250,
           offline_frontend_pairs=6750,anchor_max_abs_psnr_dists_nmse=errs)

if __name__=="__main__":
    main()
