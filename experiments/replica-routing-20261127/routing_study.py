"""Select once on validation, evaluate on frozen sources, measure full receiver wall time."""
import itertools
import statistics
from runtime import *
from receiver import decode_all,demix,weak_only

def candidates():
    return list(itertools.product(CFG["schedules"],CFG["fixed_rules"]+CFG["adaptive_rules"]))

def slug(spec):
    return "/".join(spec)

def lookup(out,kind):
    dev=HERE/"development"
    assert (dev/"closed.json").exists()
    frozen=json.loads((dev/"frozen.json").read_text())
    assert frozen["protocol"]==digest(HERE/"protocol.json")
    assert frozen["rows_sha256"]==digest(dev/"weak_scores.csv")
    aliases={k:tuple(v) for k,v in frozen["policies"].items()}
    record(out/"policy_binding.json",dict(freeze_sha256=digest(dev/"frozen.json"),policies=aliases,
           development_rows_sha256=frozen["rows_sha256"]))
    return aliases,list(dict.fromkeys(aliases.values()))

def qualify(model,y):
    ref = ReceiverTrace(model,"rgb_reencode")
    model.modem.study_trace=ref
    try:
        old,_,_=model.modem.blind_demix_users(y,4,model.latent_channels)
    finally:
        model.modem.study_trace=None
    new,_,log=demix(model,y,("-","full"),True)
    dif=float((decode_all(model,old)-decode_all(model,new)).abs().max())
    assert dif<.002 and (log==1).all().item(),dif
    return dif

def develop(out,bb,start):
    images,ids,stems=source_set(bb,"val256",out)
    model=weight(bb,1001)
    records=[]
    specs=candidates()
    qualification=0.
    for cond,levels in CFG["conditions"].items():
        bank=make_bank(64,1001,levels,"development")
        for st in range(0,64,8):
            en=min(st+8,64)
            y=transmit(model,stems,bank,st,en)[0]
            weak=bank[3][st:en].argmin(1).cuda()
            j=torch.arange(en-st,device=y.device)
            target=images[4*st:4*en].reshape(en-st,4,*images.shape[1:]).cuda().float()[j,weak]/255.
            if not records:
                qualification=qualify(model,y[:1])
            for spec in specs:
                outs,activity,log=demix(model,y,spec,True)
                rgb=weak_only(model,outs,weak)
                p=compute_psnr_batch(target,rgb).tolist()
                calls=log.sum((1,2)).tolist()
                for i,v in enumerate(p):
                    records.append(dict(condition=cond,group=st+i,schedule=spec[0],rule=spec[1],rgb_calls=calls[i],weak_psnr=v))
        print("development",cond,len(records),flush=True)
    assert len(records)==64*2*len(specs)
    rows(out/"weak_scores.csv",records)
    scores={s:statistics.mean(r["weak_psnr"] for r in records if (r["schedule"],r["rule"])==s) for s in specs}
    fixed=[s for s in specs if s[1] in CFG["fixed_rules"]]
    adapt=[s for s in specs if s[1] in CFG["adaptive_rules"]]
    best_fixed=max(fixed,key=scores.get)
    adaptive=max(adapt,key=scores.get)
    matched=max((s for s in fixed if s[0]==adaptive[0]),key=scores.get)
    chosen={"latent":("-","latent"),"best_fixed":best_fixed,"matched_fixed":matched,"adaptive":adaptive,"full":("-","full")}
    rows(out/"candidates.csv",[dict(schedule=s[0],rule=s[1],weak_psnr=scores[s]) for s in specs])
    record(out/"frozen.json",dict(protocol=digest(HERE/"protocol.json"),rows_sha256=digest(out/"weak_scores.csv"),
          source_sha256=digest(out/"sources.json"),policies=chosen,full_vectorized_max_abs_rgb=qualification))
    finish(out,start,validation_quartets=64,distinct_observations=128,candidates=len(specs),weak_scores=len(records))

def formal(out,bb,start):
    aliases,specs=lookup(out,"formal")
    images,ids,stems=source_set(bb,"common500",out)
    a=json.loads((out/"sources.json").read_text())
    b=json.loads((HERE/"development/sources.json").read_text())
    assert not set(a["ids"]).intersection(b["ids"]) and not set(a["pixels"]).intersection(b["pixels"])
    evaluator=OfflineVisualEvaluator(torch.device("cuda:0"),include_lpips=False)
    total=0
    for seed in CFG["seeds"]:
        model=weight(bb,seed)
        for cond,levels in CFG["conditions"].items():
            for layout in ("original","rotated"):
                bank=make_bank(125,seed,levels,layout)
                events=[]
                for st in range(0,125,8):
                    en=min(st+8,125)
                    y=transmit(model,stems,bank,st,en)[0]
                    weak=bank[3][st:en].argmin(1).cuda()
                    j=torch.arange(en-st,device=y.device)
                    target=images[4*st:4*en].reshape(en-st,4,*images.shape[1:]).cuda().float()/255.
                    for spec in specs:
                        outs,act,log=demix(model,y,spec,True)
                        pred=decode_all(model,outs)
                        psnr=compute_psnr_batch(target.flatten(0,1),pred.flatten(0,1)).reshape(en-st,4)
                        ds=evaluator.dists_model(target[j,weak],pred[j,weak],require_grad=False,batch_average=False).reshape(-1)
                        mask=log.sum(-1)
                        values=torch.stack([weak,psnr[j,weak],ds,psnr.mean(1),act.sum(1),mask[:,0],mask[:,1],mask[:,2]],dim=1).cpu().tolist()
                        users=log.cpu().tolist()
                        for i,(slot,p,d,mean,active,c0,c1,c2) in enumerate(values):
                            tokens=["".join(str(u) for u in range(4) if rr[u]) or "-" for rr in users[i]]
                            events.append(dict(layout=layout,condition=cond,seed=seed,group=st+i,slot=int(slot),
                                weak_source=ids[4*(st+i)+int(slot)],spec=slug(spec),weak_psnr=p,weak_dists=d,mean_psnr=mean,
                                active_hat=int(active),rgb_calls=int(c0+c1+c2),users_each_round="|".join(tokens)))
                rows(out/f"{layout}_{cond}_{seed}.csv",events)
                total+=len(events)
                print("formal",layout,cond,seed,len(events),flush=True)
    assert total==1500*len(specs)
    finish(out,start,distinct_observations=1500,unique_receivers_per_observation=len(specs),rows=total,
           final_rgb_decodes=total*4,validation_test_id_overlap=0,validation_test_pixel_overlap=0)

def timing(out,bb,start):
    aliases,specs=lookup(out,"timing")
    images,ids,stems=source_set(bb,"common500",out)
    model=weight(bb,1001)
    all_rows=[]
    import subprocess
    def smi():
        return subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,utilization.gpu,memory.used,clocks.sm",
               "--format=csv,noheader"],text=True).strip().splitlines()
    hardware={"pre":smi(),"affinity":sorted(os.sched_getaffinity(0))}
    def receive(y,spec):
        outs,_,_=demix(model,y,spec,False)
        return decode_all(model,outs)
    for cond,levels in CFG["conditions"].items():
        bank=make_bank(125,1001,levels,"rotated")
        waves=[]
        for st in range(0,32,8):
            y=transmit(model,stems,bank,st,st+8)[0]
            waves.extend(y[i:i+1].contiguous() for i in range(8))
        for spec in specs:
            for i in range(3):
                receive(waves[i],spec)
        torch.cuda.synchronize()
        for repeat in range(2):
            for j,y in enumerate(waves):
                off=(j+repeat)%len(specs)
                call=specs[off:]+specs[:off]
                if repeat==1: call=list(reversed(call))
                for order,spec in enumerate(call):
                    torch.cuda.synchronize()
                    t=time.perf_counter_ns()
                    decoded=receive(y,spec)
                    torch.cuda.synchronize()
                    ms=(time.perf_counter_ns()-t)/1.e6
                    assert decoded.shape[:2]==(1,4)
                    all_rows.append(dict(condition=cond,group=j,repeat=repeat,order=order,spec=slug(spec),rx_ms=ms))
        print("timing",cond,len(all_rows),flush=True)
    hardware["post"]=smi()
    record(out/"hardware.json",hardware)
    rows(out/"batch1.csv",all_rows)
    finish(out,start,measured_calls=len(all_rows),warmup_calls=3*2*len(specs),
           batch=1,decoded_images_per_measured_call=4,unique_policies=len(specs))

@torch.inference_mode()
def main():
    stage=sys.argv[1]
    out,bb,start=init(stage)
    {"development":develop,"formal":formal,"timing":timing}[stage](out,bb,start)

if __name__=="__main__":
    main()
