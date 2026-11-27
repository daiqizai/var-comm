"""Finite 1→2→4 training from random Q2 weights, with development-only model selection."""
import argparse
from common import *
from torch.nn import functional as F

def save_model(model,path):
    staged=path.with_suffix(".writing")
    torch.save(model.state_dict(),staged)
    staged.replace(path)

def power_for(batch,users,step,gen):
    if step%2==0:
        return torch.full((batch,users),10.,device="cuda")
    levels={1:[10.],2:[18.,4.],4:[18.,13.,8.,4.]}[users]
    p=torch.tensor(levels,device="cuda")
    order=torch.rand(batch,users,device="cuda",generator=gen).argsort(dim=1)
    return p[order]

def images_for(images,users,side,gen):
    count=PROTOCOL["training"]["samples_per_update"]
    # Within each collision all source indices differ.
    ix=torch.randperm(len(images),device="cuda",generator=gen)[:count]
    x=images[ix]
    if side<256:
        offsets=torch.randint(257-side,(count,2),generator=gen,device="cuda")
        coord=torch.arange(side,device="cuda")
        x=x.permute(0,2,3,1)[torch.arange(count,device="cuda")[:,None,None],
                    (offsets[:,0,None]+coord)[:,:,None],(offsets[:,1,None]+coord)[:,None,:]].permute(0,3,1,2)
    return x.reshape(-1,users,3,side,side).float()/255.

@torch.inference_mode()
def validate(model,modem,images):
    model.eval()
    metrics=[]
    for levels in ((10,10,10,10),(18,13,8,4)):
        events=bank(64,1001,levels,"development")
        values=[]
        for st in range(0,64,4):
            x=images[st*4:(st+4)*4].cuda().reshape(4,4,3,256,256).float()/255.
            tx=model.transmit(x,modem)
            pred=model.receive(wave_from_bank(tx,events,st,st+4),modem).clamp(0,1)
            scores=compute_psnr_batch(x.flatten(0,1),pred.flatten(0,1)).reshape(4,4)
            if levels[0]==levels[-1]:
                values.extend(scores.mean(1).tolist())
            else:
                weak=events[-1][st:st+4].argmin(1).cuda()
                values.extend(scores[torch.arange(4,device="cuda"),weak].tolist())
        metrics.append(sum(values)/len(values))
    model.train()
    return metrics

def qualify(model,modem,images):
    import copy
    seed=PROTOCOL["training"]["seed"]
    gen=torch.Generator(device="cuda").manual_seed(seed-1)
    trial=copy.deepcopy(model)
    trial.projections.expand()
    trial.projections.expand()
    tx_map=torch.view_as_complex(trial.projections.tx.detach())
    rx_map=torch.view_as_complex(trial.projections.rx.detach())
    orth=(tx_map.flatten(0,1)@rx_map.permute(1,0,2).reshape(48,48)-torch.eye(48,device="cuda")).abs().max().item()
    assert orth<1.e-5,orth
    x=images_for(images,4,256,gen)[:1]
    packed=F.pixel_shuffle(F.pixel_unshuffle(x.flatten(0,1),4),4)
    assert torch.equal(packed,x.flatten(0,1))
    tx=trial.transmit(x,modem)
    energy=tx.square().sum(-1).mean((-1,-2))
    assert tx.shape==(1,4,32,384,2) and (energy-1).abs().max().item()<2.e-5
    from dir6_mb.overloaded_noma_4x2 import build_walsh_16_basis
    basis=build_walsh_16_basis(device="cuda")
    wp=torch.einsum("bumni,kn->bumki",tx,basis)
    mode_p=wp.square().sum(-1).mean(2)/384
    probe=torch.stack([mode_p[:,u,2+u] for u in range(4)],1)
    scale=torch.stack([mode_p[:,u,6+u] for u in range(4)],1)
    quiet=mode_p[:,:,[0,1,10,11]].max().item()
    assert quiet<1.e-10 and (probe-modem.eta_ch).abs().max()<2e-5 and (scale-modem.eta_sc).abs().max()<2e-5
    y,_=physical_grantfree_4x2_mac(tx,torch.tensor([[18,13,8,4]],device="cuda"),n_rx=2)
    a=trial.receive(y,modem)
    assert torch.equal(a,trial.receive(y.detach().clone(),modem)) and a.shape==x.shape
    loss=(a-x).square().mean()
    loss.backward()
    gradients={}
    for name,p in (("encoder",next(trial.encoder.parameters())),("decoder",next(trial.decoder.parameters())),
                   ("projection_tx",trial.projections.tx),("projection_rx",trial.projections.rx)):
        g=float(p.grad.norm());assert 0<g<float("inf"),(name,g)
        gradients[name]=g
    return dict(max_orthogonality_error=orth,energy_min=float(energy.min().detach()),energy_max=float(energy.max().detach()),
        probe_energy=float(probe.mean().detach()),scale_energy=float(scale.mean().detach()),quiet_power_max=quiet,gradients=gradients,
        initial_validation="lossless pixel packing, channel shape, payload and reference energy, receiver replay, nonzero gradients",
        trainable_parameters_4user=sum(p.numel() for p in trial.parameters()))

def main():
    out=init("train")
    CHECK.mkdir(exist_ok=False)
    begin=time.time()
    train,trainids,trainpixels=source("train20k")
    val,valids,valpixels=source("val256")
    assert len(train)==20000 and len(val)==256 and not set(trainpixels)&set(valpixels)
    write_json(out/"sources.json",dict(train_ids=trainids,train_pixels=trainpixels,val_ids=valids,val_pixels=valpixels))
    train=train.cuda()
    seed=PROTOCOL["training"]["seed"]
    modem=modem_for(1001)
    model=AdaptedPNOMA().cuda()
    write_json(out/"qualification.json",qualify(model,modem,train))
    seed_everything(seed+1)
    gen=torch.Generator(device="cuda").manual_seed(seed+2)
    print("qualified",flush=True)
    best=-float("inf")
    for index,nsteps in enumerate(PROTOCOL["training"]["updates_per_stage"]):
        if index:
            model.projections.expand()
        users=model.users
        opt=torch.optim.Adam(model.parameters(),lr=.0001)
        running=0.
        for step in range(1,nsteps+1):
            if users==4 and step==1201:
                opt.param_groups[0]["lr"]=.00005
            side=256 if users==4 and step%4==0 else 128
            x=images_for(train,users,side,gen)
            power=power_for(len(x),users,step,gen)
            tx=model.transmit(x,modem,True)
            y,_=physical_grantfree_4x2_mac(tx,power,n_rx=2)
            pred=model.receive(y,modem,True)
            loss=(pred-x).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            opt.step()
            running+=float(loss.detach())
            if step%50==0:
                rec=dict(users=users,step=step,train_mse_50=running/50,grad=float(grad),elapsed_s=time.time()-begin)
                append_row(out/"learning.csv",rec)
                print("train",users,step,round(running/50,5),round(rec["elapsed_s"],1),flush=True)
                running=0.
            if users==4 and step%300==0:
                metrics=validate(model,modem,val)
                score=sum(metrics)/2
                rec=dict(step=step,symmetric_psnr=metrics[0],weak14_psnr=metrics[1],selection=score)
                append_row(out/"validation.csv",rec)
                if score>best:
                    best,selected=score,dict(rec)
                    save_model(model,CHECK/"frozen_four.pt")
                print("validation",rec,flush=True)
            if step==nsteps or (users==4 and step%300==0):
                save_model(model,CHECK/f"last_{users}.pt")
        del opt
    write_json(out/"freeze.json",dict(**selected,weight_sha256=sha(CHECK/"frozen_four.pt"),
            validation_sha256=sha(out/"validation.csv"),protocol_sha256=sha(HERE/"protocol.json")))
    write_json(out/"done.json",dict(actual_updates=sum(PROTOCOL["training"]["updates_per_stage"]),
        seconds=time.time()-begin,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        final_indices=[1000,600,1800],source_visits=8*3400))
if __name__=="__main__":
    main()
