"""Evaluate after the frozen development decision and time both complete receivers on one GPU."""
import csv
import os
import subprocess
from common import *

ALIASES={"late/power":"semsic_three","-/latent":"semsic_latent","-/full":"semsic_full"}

def array_sha(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()

def load_native_row_map(cond,seed):
    p=PRIOR/"formal"/f"rotated_{cond}_{seed}.csv"
    with p.open(newline="") as f:
        rows={(int(row["group"]),ALIASES[row["spec"]]):row for row in csv.DictReader(f) if row["spec"] in ALIASES}
    assert len(rows)==125*3
    return rows,sha(p)

@torch.inference_mode()
def quality(out,model,images,ids,rt):
    import numpy as np
    evaluator=OfflineVisualEvaluator(torch.device("cuda:0"),include_lpips=False)
    events=[]
    banks={}
    inputs={}
    per_source=[]
    for seed in SEEDS:
        modem=modem_for(seed)
        for cond,levels in CONDITIONS.items():
            label=f"{cond}_{seed}"
            reg=bank(125,seed,levels,"rotated")
            parent=rt.make_bank(125,seed,levels,"rotated")
            assert all(torch.equal(a,b) for a,b in zip(reg,parent))
            del parent
            banks[label]={n:array_sha(t) for n,t in zip(("channel_real","channel_imag","noise_iq","assigned_db"),reg)}
            refs,h=load_native_row_map(cond,seed)
            inputs[label]=h
            for st in range(0,125,4):
                en=min(125,st+4)
                x=images[4*st:4*en].cuda().reshape(en-st,4,3,256,256).float()/255.
                tx=model.transmit(x,modem)
                assert tx.shape[1:]==(4,32,384,2)
                powerr=(tx.square().sum(-1).mean(dim=(2,3))-1).abs().max().item()
                assert powerr<2.e-5
                y=wave_from_bank(tx,reg,st,en)
                pred=model.receive(y,modem).clamp(0,1)
                assert pred.shape==x.shape and torch.isfinite(pred).all()
                slot=reg[-1][st:en].argmin(1).cuda()
                arange=torch.arange(en-st,device="cuda")
                p=compute_psnr_batch(x.flatten(0,1),pred.flatten(0,1)).reshape(en-st,4)
                d=evaluator.dists_model(x[arange,slot],pred[arange,slot],require_grad=False,batch_average=False).reshape(-1)
                p_cpu=p.cpu().tolist()
                for i,u in enumerate(slot.tolist()):
                    g=st+i
                    keys=dict(condition=cond,draw=seed,group=g,weak_slot=u,weak_source=ids[4*g+u])
                    events.append(dict(**keys,method="pnoma_q2",weak_psnr=p_cpu[i][u],weak_dists=float(d[i]),mean_psnr=sum(p_cpu[i])/4))
                    per_source.append(dict(**keys,**{f"slot_{j}_psnr":v for j,v in enumerate(p_cpu[i])}))
                    for alias in ALIASES.values():
                        ref=refs[g,alias]
                        assert int(ref["slot"])==u and ref["weak_source"]==keys["weak_source"]
                        events.append(dict(**keys,method=alias,**{k:float(ref[k]) for k in ("weak_psnr","weak_dists","mean_psnr")}))
            print("scored",label,flush=True)
    write_rows(out/"group_scores.csv",events)
    write_rows(out/"external_user_psnr.csv",per_source)
    write_json(out/"paired_inputs.json",dict(banks=banks,local_input_sha256=inputs,groups=125,channel_banks_equal=True))
    return {(e["condition"],e["draw"],e["method"],e["group"]):e for e in events}

@torch.inference_mode()
def timed(out,model,images,records,rt):
    sys.path.insert(0,str(PRIOR))
    from receiver import decode_all,demix
    from dir2_zpac.models import load_swinjscc_backbone
    bb=load_swinjscc_backbone(rt.BACKBONE,model_mode="SwinJSCC_w/_SA",c_bottleneck=96,device=torch.device("cuda:0")).eval()
    native=rt.weight(bb,1001)
    phy=modem_for(1001)
    frozen=json.loads((PRIOR/"development/frozen.json").read_text())
    assert tuple(frozen["policies"]["best_fixed"])==("late","power")
    def run_native(y):
        h,_,_=demix(native,y,("late","power"),False)
        return decode_all(native,h)
    def run_external(y):
        return model.receive(y,phy).clamp(0,1)
    funcs={"pnoma_q2":run_external,"semsic_three":run_native}
    def hardware():
        return subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,utilization.gpu,memory.used,clocks.sm",
            "--format=csv,noheader"],text=True).strip().splitlines()
    hw=dict(pre=hardware(),cores=sorted(os.sched_getaffinity(0)))
    rows=[]
    anchors=[]
    for cond,levels in CONDITIONS.items():
        reg=bank(125,1001,levels,"rotated")
        waves={k:[] for k in funcs}
        for st in range(0,32,4):
            x=images[st*4:(st+4)*4].cuda().reshape(4,4,3,256,256).float()/255.
            a=wave_from_bank(model.transmit(x,phy),reg,st,st+4)
            features=native.extract_encoder_stem(x.flatten(0,1))
            batch_tx=torch.stack([native.encode_single_user_stem(features[u::4],u)[0] for u in range(4)],dim=1)
            b=wave_from_bank(batch_tx,reg,st,st+4)
            for name,w in (("pnoma_q2",a),("semsic_three",b)):
                waves[name].extend(w[i:i+1].contiguous() for i in range(4))
        for name,fun in funcs.items():
            for j in range(3):
                out_rgb=fun(waves[name][j])
                u=int(reg[-1][j].argmin())
                target=images[4*j+u:4*j+u+1].cuda().float()/255.
                actual=float(compute_psnr_batch(target,out_rgb[:,u]))
                expected=records[cond,1001,name,j]["weak_psnr"]
                anchors.append(dict(condition=cond,method=name,group=j,expected=expected,observed=actual))
                assert abs(actual-expected)<.025,(name,actual,expected)
        torch.cuda.synchronize()
        for rep in range(2):
            for j in range(32):
                order=list(funcs)
                if (j+rep)%2:
                    order.reverse()
                for pos,name in enumerate(order):
                    torch.cuda.synchronize()
                    start=time.perf_counter_ns()
                    image=funcs[name](waves[name][j])
                    torch.cuda.synchronize()
                    elapsed=(time.perf_counter_ns()-start)/1.e6
                    assert image.shape==(1,4,3,256,256)
                    rows.append(dict(condition=cond,group=j,repeat=rep,order=pos,method=name,rx_ms=elapsed))
        print("timed",cond,flush=True)
    hw["post"]=hardware()
    write_rows(out/"timing.csv",rows)
    write_rows(out/"timing_anchors.csv",anchors)
    write_json(out/"hardware.json",hw)

def main():
    out=init("evaluate")
    start=time.time()
    os.environ["STUDY_SCRATCH"]=str(SNAP.parent)
    os.environ["STUDY_PARENT"]=str(OLD)
    sys.path.insert(0,str(PRIOR))
    import runtime as rt
    images,ids,pixels=source("common500")
    history=json.loads((HERE/"train/sources.json").read_text())
    overlap=len(set(pixels)&set(history["train_pixels"]))
    assert overlap==0 and not set(pixels)&set(history["val_pixels"])
    write_json(out/"sources.json",dict(ids=ids,pixels=pixels,train_intersection=overlap,val_intersection=0))
    model=chosen_model()
    scores=quality(out,model,images,ids,rt)
    timed(out,model,images,scores,rt)
    write_json(out/"done.json",dict(seconds=time.time()-start,registered_weight=sha(CHECK/"frozen_four.pt"),
        unique_quartets=125,external_frames=750,external_images=3000,reused_local_rows=2250,
        quality_rows=3000,timed_rx_calls=256,warmup_rx_calls=12,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
if __name__=="__main__":
    main()
