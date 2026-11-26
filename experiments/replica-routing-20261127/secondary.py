"""Validation-frozen test of a ranking that can actually change the chosen interferer."""
import csv
import hashlib
import subprocess
from runtime import *
from receiver import decode_all,demix

def read(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))

def main():
    reg=json.loads((HERE/"nondegenerate_freeze.json").read_text())
    assert reg["protocol"]==digest(HERE/"nondegenerate_protocol.json")
    assert reg["freeze"]==digest(HERE/"development/frozen.json")
    assert reg["candidates"]==digest(HERE/"development/candidates.csv")
    a=reg["adaptive"];b=reg["fixed"]
    specs=[(v["schedule"],v["rule"]) for v in (a,b)]
    frozen=json.loads((HERE/"development/frozen.json").read_text())["policies"]
    assert list(specs[1])==frozen["best_fixed"] and specs[0][0]==specs[1][0]
    assert (HERE/"formal/closed.json").exists() and (HERE/"timing/closed.json").exists()
    out,bb,start=init("secondary")
    record(out/"binding.json",dict(secondary=digest(HERE/"nondegenerate_freeze.json"),formal=digest(HERE/"formal/policy_binding.json"),
                                  timing=digest(HERE/"timing/policy_binding.json"),specs=specs))
    images,ids,stems=source_set(bb,"common500",out)
    assert digest(out/"sources.json")==digest(HERE/"formal/sources.json")
    evaluator=OfflineVisualEvaluator(torch.device("cuda:0"),include_lpips=False)
    results=[]
    original={}
    observed_hashes={}
    for seed in CFG["seeds"]:
        model=weight(bb,seed)
        for cond,levels in CFG["conditions"].items():
            bank=make_bank(125,seed,levels,"rotated")
            parent=HERE/"formal"/f"rotated_{cond}_{seed}.csv"
            fixed={int(r["group"]):r for r in read(parent) if r["spec"]=="/".join(specs[1])}
            original[parent.name]=digest(parent)
            assert len(fixed)==125
            waves=hashlib.sha256()
            for st in range(0,125,8):
                en=min(st+8,125)
                y=transmit(model,stems,bank,st,en)[0]
                waves.update(y.cpu().contiguous().numpy().tobytes())
                weak=bank[3][st:en].argmin(1).cuda()
                j=torch.arange(en-st,device=y.device)
                target=images[4*st:4*en].reshape(en-st,4,*images.shape[1:]).cuda().float()/255.
                states,act,selection=demix(model,y,specs[0],True)
                pred=decode_all(model,states)
                ps=compute_psnr_batch(target.flatten(0,1),pred.flatten(0,1)).reshape(en-st,4)
                ds=evaluator.dists_model(target[j,weak],pred[j,weak],require_grad=False,batch_average=False).reshape(-1)
                vals=torch.stack((weak,ps[j,weak],ds,ps.mean(1),act.sum(1)),dim=1).cpu().tolist()
                for i,(slot,p,d,m,active) in enumerate(vals):
                    g=st+i
                    ref=fixed[g]
                    assert ids[4*g+int(slot)]==ref["weak_source"] and int(ref["slot"])==int(slot)
                    assert int(ref["active_hat"])==int(active)
                    mask=selection[i].cpu().tolist()
                    users="|".join("".join(str(u) for u in range(4) if rr[u]) or "-" for rr in mask)
                    n=sum(map(sum,mask))
                    assert n==int(ref["rgb_calls"])<=3
                    results.append(dict(seed=seed,condition=cond,group=g,weak_source=ref["weak_source"],slot=int(slot),
                                       adaptive_psnr=p,fixed_psnr=ref["weak_psnr"],adaptive_dists=d,fixed_dists=ref["weak_dists"],
                                       adaptive_all_psnr=m,fixed_all_psnr=ref["mean_psnr"],rgb_calls=n,
                                       adaptive_choices=users,fixed_choices=ref["users_each_round"]))
            observed_hashes[parent.name]=waves.hexdigest()
            print(cond,seed,len(results),flush=True)
    assert len(results)==750
    rows(out/"matched.csv",results)
    record(out/"identities.json",dict(fixed_csv_sha256=original,current_received_waveform_sha256=observed_hashes,
          common_source_hash=digest(out/"sources.json"),formal_code=digest(HERE/"routing_study.py"),
          signal_generator=digest(HERE/"runtime.py"),router=digest(HERE/"receiver.py"),
          batch_size=8,offset=91000003,weight_hashes=PARENT_STATS["checkpoint_sha256"],
          setup="Same generator and weights, GPU UUID, FP32 flags, input CSV identity and tuple order as formal."))
    model=weight(bb,1001)
    logs=[]
    def session(y,spec):
        states,_,_=demix(model,y,spec,False)
        return decode_all(model,states)
    def gpu_stat():
        return subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,utilization.gpu,memory.used,clocks.sm",
            "--format=csv,noheader"],text=True).strip().splitlines()
    hw={"pre":gpu_stat()}
    for cond,levels in CFG["conditions"].items():
        bank=make_bank(125,1001,levels,"rotated")
        wave=[]
        for st in range(0,32,8):
            y=transmit(model,stems,bank,st,st+8)[0]
            wave.extend(y[i:i+1].contiguous() for i in range(8))
        for spec in specs:
            for i in range(3):
                session(wave[i],spec)
        torch.cuda.synchronize()
        for rep in range(2):
            for g,y in enumerate(wave):
                ordering=(0,1) if (rep+g)%2==0 else (1,0)
                for pos,i in enumerate(ordering):
                    torch.cuda.synchronize()
                    begin=time.perf_counter_ns()
                    x=session(y,specs[i])
                    torch.cuda.synchronize()
                    ms=(time.perf_counter_ns()-begin)/1.e6
                    assert x.shape[:2]==(1,4)
                    logs.append(dict(condition=cond,group=g,repeat=rep,order=pos,arm=["adaptive","fixed"][i],rx_ms=ms))
    hw["post"]=gpu_stat()
    record(out/"timing_hardware.json",hw)
    rows(out/"batch1.csv",logs)
    finish(out,start,matched_groups=750,new_scored_receivers=750,reused_fixed_quality=750,
           new_final_images_scored=3000,time_calls=len(logs),warmups=12)

if __name__=="__main__":
    with torch.inference_mode():
        main()
