"""Can conventional damped parallel cancellation recover the overloaded weak-user gain?"""
import csv
import sys
from pathlib import Path
import numpy as np
import torch
from common import (D4, PRIMARY, PRIOR, STUDY, SWIN, TEMP, digest, include_sources, load_state,
                    read, record, save, single_replace, table, tensor_hash)

ROOT=STUDY/"pic"
SOURCE=TEMP/"pic_source"
GRID=(0.,.25,.5,.75,1.)


def prepare():
    for f in sorted((TEMP/"source"/D4/"src/dir4_rn").glob("*.py")):
        txt=f.read_text()
        if f.name=="noma_semsic_blind.py":
            expr="self.latent_prior(ch_u.u_restored, alp_um, sigp_um, cr_u)"
            txt=single_replace(txt,expr,"(ch_u.u_restored if getattr(self,'pic_mode','full')=='plain' else "+expr+")")
            pos="                p_clean_u = (2.0 * z_clean_u.square()"
            txt=single_replace(txt,pos,"                remod_u = z_clean_u if getattr(self,'pic_mode','full')=='full' else ch_u.u_restored\n"+
                               "                p_clean_u = (2.0 * remod_u.square()")
            txt=single_replace(txt,"self.project_data_perp(z_clean_u / p_clean_u","self.project_data_perp(remod_u / p_clean_u")
            op="0.96 * torch.sigmoid(self.sic_gate_mlp(gate_in).squeeze(-1)) * alp_um.sqrt()"
            txt=single_replace(txt,op,"((self.pic_scale * alp_um.sqrt()).clamp(0.0, 0.96) if getattr(self,'pic_mode','full')=='plain' else "+op+")")
        compile(txt,str(f),"exec")
        save(SOURCE/"dir4_rn"/f.name,txt.encode())
    record(ROOT/"protocol.json",{
        "trigger":"Follow-up after the matched-solve factorial showed a weak-user benefit from waveform subtraction.",
        "factorial_source":digest(TEMP/"source"/D4/"src/dir4_rn/noma_semsic_blind.py"),
        "variant_source":digest(SOURCE/"dir4_rn/noma_semsic_blind.py"),
        "arms":{"full":"Learned denoised latent used for interferer waveform and published learned gate.",
            "raw_gate":"Use the per-user linear estimate for the interfering waveform. Preserve the trained gate and its learned prior disagreement input, recomputed along this arm's trajectory.",
            "plain":"Use raw per-user linear estimates for interfering waveforms. No learned denoiser or learned gate in the cancellation loop. Damping is clip(c*sqrt(alpha),0,0.96), two rounds with the same residual-covariance rule."},
        "common":"Same frozen image transmitter/decoder, blind channel estimates, RF observation, power, three LMMSE solves. The plain arm is a model-based cancellation stage inside the shared learned image system.",
        "selection":{"coefficient_grid":list(GRID),"validation":"val256, one new common draw 6112312, 13/8/4 dB, 3x2; maximize weak-user mean PSNR, smaller coefficient on an exact tie. Freeze before test.",
                     "image_weights":"Previously frozen full 4B checkpoint in parent protocol. No retraining."},
        "test":"Original common500 and Kodak24 tuples, draws 3001/3002/3003, each source scored for PSNR and DISTS. Reuse native scores only after image and noise hashes and first-batch replay at each split. Fully report both new arms, all users, interactions conditional on this same trained receiver.",
        "uncertainty":"5000 paired bootstrap draws over original tuple anchors; also report structurally disjoint triples for the weak user, 166 and 8 triples. No new source selection on performance.",
    })


@torch.inference_mode()
def run():
    include_sources()
    sys.path.insert(0,str(SOURCE))
    from dir4_rn import noma_semsic_blind as nm
    from dir2_zpac.datasets import load_split_u8
    from dir2_zpac.hygiene import configure_runtime_env,configure_torch_determinism,seed_everything
    from dir2_zpac.metrics import OfflineVisualEvaluator,compute_psnr_batch
    from dir2_zpac.models import load_swinjscc_backbone
    assert digest(Path(nm.__file__))==read(ROOT/"protocol.json")["variant_source"]
    configure_runtime_env(cache_root=TEMP/"cache/pic",num_threads=4)
    seed_everything(20261124)
    configure_torch_determinism()
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    bb=load_swinjscc_backbone(SWIN,model_mode="SwinJSCC_w/_SA",c_bottleneck=96,device="cuda:0")
    m=nm.GrantFreeNOMATransceiver(bb,method="semsic_noma",n_rx=2,num_rbs=32,hidden_ch=160)
    cp=read(STUDY/"protocol.json")["references"]["frozen_4b"]
    assert digest(cp["path"])==cp["sha256"]
    load_state(m,cp["path"])
    m.cuda().eval()
    metric=OfflineVisualEvaluator(device="cuda:0",include_lpips=False)
    ident=read(PRIOR/"evaluation/noma/identity.json")
    validation={c:[] for c in GRID}
    replay=[]
    counts={"new_validation_outputs":0,"new_test_outputs":0,"native_reused":0,"native_qualification_outputs":0}
    for split in ("val256","common500","kodak24"):
        cal=split=="val256"
        ds=load_split_u8(split,allow_holdout=not cal)
        ids=ds.source_ids
        n=len(ids)
        assert len(set(ids))==n and ds.is_holdout != cal
        targets=ds.images_u8.float()/255
        stems=torch.cat([m.extract_encoder_stem(x.cuda()).cpu() for x in targets.split(32)])
        if not cal:
            assert read(PRIOR/"evaluation/noma"/(split+"_source_manifest.json"))==[
                {"source_id":k,"chw_u8_sha256":tensor_hash(x)} for k,x in zip(ids,ds.images_u8)]
            archived={(int(z["seed"]),z["tuple_anchor"],z["role"],z["source_id"]):[float(z[k]) for k in PRIMARY]
                      for z in csv.DictReader((STUDY/"noma"/(split+"_frames.csv")).open()) if z["arm"]=="both"}
            assert len(archived)==n*9
            coef=read(ROOT/"frozen_damping.json")["selected"]
        rows=[]
        for seed in ((6112312,) if cal else (3001,3002,3003)):
            gen=torch.Generator().manual_seed(seed*2089+3*97+130)
            hr,hi=nm.sample_multiuser_uplink_channels(n,3,2,32,.85,generator=gen)
            eps=torch.randn((n,2,32,384,2),dtype=torch.float32,generator=gen)
            if not cal:
                assert [tensor_hash(x) for x in (hr,hi,eps)]==next(z["tensor_hashes"] for z in
                    ident["events"][split]["over3x2_13_8_4"]["noise"] if z["seed"]==seed)
            for a in range(0,n,25):
                b=min(a+25,n)
                indices=[(np.arange(a,b)+s)%n for s in (0,n//3,2*n//3)]
                tx=torch.stack([m.encode_single_user_stem(stems[idx].cuda(),u)[0] for u,idx in enumerate(indices)],dim=1)
                powers=torch.tensor([13,8,4],dtype=torch.float32,device="cuda").expand(b-a,3)
                y,_=nm.physical_grantfree_uplink_mac(tx,powers,n_rx=2,h_mu_override=(hr[a:b],hi[a:b]),standard_noise_iq=eps[a:b])
                arms=[("plain",c) for c in GRID] if cal else [("plain",coef),("raw_gate",coef)]
                if not cal and a==0 and seed==3001:
                    arms.append(("full",coef))
                for name,c in arms:
                    m.modem.pic_mode,m.modem.pic_scale=name,c
                    demix,_=m.modem.blind_demix_users(y,3,m.latent_channels)
                    for u in ([2] if cal else range(3)):
                        idx=indices[u]
                        xb=targets[idx].cuda()
                        rec=m._decode_from_channel_out(*demix[u])
                        if cal:
                            values=compute_psnr_batch(xb,rec.clamp(0,1))
                            validation[c].extend(values.cpu().tolist())
                            counts["new_validation_outputs"]+=b-a
                            continue
                        sc=metric.evaluate_batch(xb,rec)
                        role=("User1_13dB","User2_8dB","User3_4dB")[u]
                        if name=="full":
                            new=np.asarray([sc[k] for k in PRIMARY]).T
                            old=np.asarray([archived[seed,ids[a+j],role,ids[ii]] for j,ii in enumerate(idx)])
                            error=np.abs(new-old).max(axis=0)
                            assert (error<=[.003,.00005]).all(),(split,role,error)
                            counts["native_qualification_outputs"]+=b-a
                            replay.append(dict(split=split,role=role,max_diff=error.tolist()))
                        else:
                            for j,ii in enumerate(idx):
                                row=dict(split=split,seed=seed,tuple_anchor=ids[a+j],source_id=ids[ii],role=role)
                                rows.append(dict(**row,arm=name,psnr_db=sc["psnr_db"][j],dists=sc["dists"][j]))
                                if name=="plain":
                                    p,d=archived[seed,ids[a+j],role,ids[ii]]
                                    rows.append(dict(**row,arm="full",psnr_db=p,dists=d))
                                    counts["native_reused"]+=1
                            counts["new_test_outputs"]+=b-a
            print(split,seed,"complete",flush=True)
        if cal:
            assert all(len(v)==n and np.isfinite(v).all() for v in map(np.asarray,validation.values()))
            scores={c:float(np.mean(values)) for c,values in validation.items()}
            coef=max(GRID,key=lambda c:(scores[c],-c))
            table(ROOT/"validation_sources.csv",[dict(source_id=ids[(j+2*n//3)%n],damping=c,psnr_db=v)
                  for c in GRID for j,v in enumerate(validation[c])])
            record(ROOT/"frozen_damping.json",{"selected":coef,"validation_mean_psnr":scores,"n":n,"seed":6112312})
        else:
            assert len(rows)==n*27 and all(np.isfinite([r["psnr_db"],r["dists"]]).all() for r in rows)
            table(ROOT/(split+"_frames.csv"),rows)
    record(ROOT/"evaluation.json",{"counts":counts,"native_replay":replay,"frozen_checkpoint":cp["sha256"],
           "validation_selection_sha256":digest(ROOT/"frozen_damping.json")})


if __name__=="__main__":
    if sys.argv[1]=="register": prepare()
    else: run()
