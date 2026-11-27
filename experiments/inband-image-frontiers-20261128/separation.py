"""Cross transmit subspace geometry with three spatial combiners on one frozen 4-user codec."""
import csv
import hashlib
import json
from types import MethodType
import torch
from study import CFG, PRIOR, setup, source, save, rows, finish, scorer, compute_psnr_batch
from access import prior, waveform, load_external

GEOMETRIES=("learned","orthogonal","coincident")
FILTERS=("matched","full_mmse","overlap_mmse")


def geometry(tx,rx,which):
    # P is a 12 x 48 complex row map. Right unitaries preserve each user's singular values.
    users,rank,width=tx.shape
    frames=torch.linalg.qr(tx.to(torch.complex128).mH.contiguous(),mode="complete")[0]
    gen=torch.Generator(device="cpu").manual_seed(281151)
    dummy=torch.randn(width,width,dtype=torch.complex128,generator=gen).to(tx.device)
    shared=torch.linalg.qr(dummy).Q
    rot=[]
    for u in range(users):
        start=u*rank if which=="orthogonal" else 0
        target=torch.roll(shared,-start,dims=1)
        unit=frames[u]@target.mH
        rot.append(unit if which!="learned" else torch.eye(width,device=tx.device,dtype=unit.dtype))
    r=torch.stack(rot).to(tx.dtype)
    p=tx@r
    q=r.mH@rx
    assert (p@q-tx@rx).abs().max()<2.e-5
    if which=="learned":
        p,q=tx,rx
    basis=torch.linalg.qr(p.mH.contiguous(),mode="reduced")[0]
    overlaps=torch.stack([torch.stack([(basis[u].mH@basis[v]).abs().square().sum()/rank for v in range(users)]) for u in range(users)])
    if which=="orthogonal":
        assert (overlaps-torch.eye(users,device=tx.device)).abs().max()<2.e-5
    if which=="coincident":
        assert (overlaps-1).abs().max()<2.e-5
    return p.contiguous(),q.contiguous(),overlaps.clamp(0,1)


def combiner(overlap):
    def demix(self,yr,yi,gr,gi,amp,sigma,channel_error,nu,active,latent_ch):
        estimate=torch.complex(gr,gi)
        y=torch.complex(yr,yi)
        b,u,nr,m=gr.shape
        h=estimate.permute(0,3,1,2)
        energy=1-self.eta_ch-self.eta_sc
        uncertainty=sigma[:,None,None]+nr*channel_error[:,:,None]
        eye=torch.eye(nr,device=y.device,dtype=y.dtype)
        outer=h.unsqueeze(-1)*h.conj().unsqueeze(-2)
        cov=torch.einsum("uv,bmvij->bmuij",overlap.to(y.dtype),outer)
        cov=cov+uncertainty[:,None,:,:,None]*eye/energy
        weights=torch.linalg.solve(cov,h.unsqueeze(-1)).squeeze(-1).permute(0,2,3,1)
        out=(weights.conj().unsqueeze(-1)*y).sum(2)*((1-12/384)**.5/energy.sqrt())
        tag=10*torch.log10((estimate.abs().square().sum(2)*energy/uncertainty.clamp_min(1.e-6)).mean(-1).clamp_min(1.e-5))
        tag=tag.clamp(-6,30)
        return [(out[:,i],tag[:,i]) for i in range(u)]
    return demix


def configure_filter(modem,original,name,overlap):
    if name=="matched":
        modem._demix_single_stage=original
    else:
        weights=torch.ones_like(overlap) if name=="full_mmse" else overlap
        modem._demix_single_stage=MethodType(combiner(weights),modem)


def historic(condition,seed):
    cond={"spread14":"nearfar_14db","spread18":"nearfar_18db_ood"}[condition]
    with (PRIOR/"evaluate/group_scores.csv").open() as f:
        return {int(r["group"]):r for r in csv.DictReader(f) if r["method"]=="pnoma_q2" and r["condition"]==cond and int(r["draw"])==seed}


@torch.inference_mode()
def main():
    out=setup("separation")
    model=load_external()
    orig_tx=torch.view_as_complex(model.projections.tx).detach().clone()
    orig_rx=torch.view_as_complex(model.projections.rx).detach().clone()
    configs={s:geometry(orig_tx,orig_rx,s) for s in GEOMETRIES}
    save(out/"overlaps.json",{s:x[2].cpu().tolist() for s,x in configs.items()})
    dists=scorer()
    total_rx=0
    tx_frames=0
    reuse=0
    solo_reuse=0
    qual={}
    for split in ("val256","common500"):
        im,ids=source(split,out)
        n=len(ids)//4
        dev=split=="val256"
        records=[]
        solo=[]
        seeds=CFG["multi"]["seeds"][:1] if dev else CFG["multi"]["seeds"]
        levels=CFG["multi"]["conditions"]
        conds=["spread14"] if dev else list(levels)
        banks={(s,c):prior.bank(n,s,levels[c],"development" if dev else "rotated") for s in seeds for c in conds}
        if not dev:
            bank_ids=json.loads((PRIOR/"evaluate/paired_inputs.json").read_text())["banks"]
            for (seed,c),bank in banks.items():
                label=("nearfar_14db" if c=="spread14" else "nearfar_18db_ood")+"_"+str(seed)
                got={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in zip(("channel_real","channel_imag","noise_iq","assigned_db"),bank)}
                assert got==bank_ids[label]
            with (PRIOR/"solo/paired_groups.csv").open() as file:
                solo_baseline={(r["condition"],int(r["group"])):r for r in csv.DictReader(file) if r["method"]=="pnoma_q2"}
            assert len(solo_baseline)==250
        for seed in seeds:
            modem=prior.modem_for(seed)
            original=modem._demix_single_stage
            refs={} if dev else {c:historic(c,seed) for c in conds}
            for name,(p,q,overlap) in configs.items():
                model.projections.tx.copy_(torch.view_as_real(p))
                model.projections.rx.copy_(torch.view_as_real(q))
                for st in range(0,n,4):
                    en=min(n,st+4); sl=slice(st,en);b=en-st
                    x=im[4*st:4*en].cuda().reshape(b,4,3,256,256).float()/255.
                    tx=model.transmit(x,modem)
                    assert (tx.square().sum(-1).mean((2,3))-1).abs().max()<3.e-5
                    for c in conds:
                        bank=banks[seed,c]; y=waveform(tx,bank,sl); tx_frames+=b
                        weak=bank[-1][sl].argmin(1).cuda();ii=torch.arange(b,device=x.device)
                        if not qual:
                            configure_filter(modem,original,"matched",overlap);native=model.receive(y,modem)
                            configure_filter(modem,original,"overlap_mmse",torch.eye(4,device=x.device))
                            equal=model.receive(y,modem)
                            error=float((native-equal).abs().max())
                            assert error<.002,error
                            qual={"orthogonal_weight_limit_rgb_max_abs":error,"energy_max_abs":float((tx.square().sum(-1).mean((2,3))-1).abs().max())}
                        for fil in FILTERS:
                            if not dev and name=="learned" and fil=="matched":
                                for j,u in enumerate(weak.tolist()):
                                    r=refs[c][st+j]
                                    assert int(r["weak_slot"])==u and r["weak_source"]==ids[4*(st+j)+u]
                                    records.append(dict(condition=c,draw=seed,group=st+j,geometry=name,receiver=fil,
                                               weak_psnr=float(r["weak_psnr"]),weak_dists=float(r["weak_dists"]),mean_psnr=float(r["mean_psnr"])))
                                reuse+=b
                                continue
                            configure_filter(modem,original,fil,overlap)
                            pred=model.receive(y,modem).clamp(0,1)
                            assert torch.isfinite(pred).all()
                            ps=compute_psnr_batch(x.flatten(0,1),pred.flatten(0,1)).reshape(b,4)
                            d=dists(x[ii,weak],pred[ii,weak],require_grad=False,batch_average=False).reshape(-1)
                            scores=torch.stack((ps[ii,weak],d,ps.mean(1)),1).cpu().tolist()
                            for j,(wp,wd,av) in enumerate(scores):
                                records.append(dict(condition=c,draw=seed,group=st+j,geometry=name,receiver=fil,weak_psnr=wp,weak_dists=wd,mean_psnr=av))
                            total_rx+=b
                        if not dev and seed==seeds[0]:
                            if name=="learned":
                                label="nearfar_14db" if c=="spread14" else "nearfar_18db_ood"
                                for j,u in enumerate(weak.tolist()):
                                    r=solo_baseline[label,st+j]
                                    assert int(r["draw"])==seed and int(r["weak_slot"])==u and r["weak_source"]==ids[4*(st+j)+u]
                                    solo.append(dict(condition=c,draw=seed,group=st+j,geometry=name,weak_psnr=float(r["solo_psnr"])))
                                solo_reuse+=b
                                continue
                            only=torch.zeros(b,4,device=x.device).scatter_(1,weak[:,None],1.)
                            obs=waveform(tx,bank,sl,only);tx_frames+=b
                            configure_filter(modem,original,"matched",overlap)
                            single=model.receive(obs,modem).clamp(0,1)
                            pp=compute_psnr_batch(x[ii,weak],single[ii,weak]).cpu().tolist()
                            solo.extend(dict(condition=c,draw=seed,group=st+j,geometry=name,weak_psnr=v) for j,v in enumerate(pp))
                            total_rx+=b
                print(split,seed,name,flush=True)
        rows(out/(split+"_groups.csv"),records)
        if dev:
            rank=[(sum(r["weak_psnr"] for r in records if (r["geometry"],r["receiver"])==(g,f))/n,g,f) for g in GEOMETRIES for f in FILTERS]
            winner=max(rank,key=lambda item:item[0])
            save(out/"freeze.json",{"selected":list(winner[1:]),"decisions":rank,"criterion":"val256 spread14 weak PSNR"})
            save(out/"qualification.json",qual)
        else:
            rows(out/"solo.csv",solo)
    finish(out,new_receiver_group_calls=total_rx,old_group_scores_reused=reuse,old_solo_groups_reused=solo_reuse,channel_frames=tx_frames,
           fit_optimizer_updates=0,qualification=qual)


if __name__=="__main__":
    main()

