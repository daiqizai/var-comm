"""Frozen two-factor receiver intervention on identical 3-user MAC observations."""
import csv
import sys
import numpy as np
import torch
from common import (PRIMARY, PRIOR, SIC_ARMS, STUDY, SWIN, SWIN_HASH, TEMP, digest,
                    include_sources, load_state, read, record, table, tensor_hash)

include_sources()
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import configure_runtime_env, configure_torch_determinism, seed_everything
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.models import load_swinjscc_backbone
from dir4_rn.noma_semsic_blind import (GrantFreeNOMATransceiver, physical_grantfree_uplink_mac,
                                      sample_multiuser_uplink_channels)

SNR = (13, 8, 4)
CONDITION = "over3x2_13_8_4"


def reference(split):
    source = PRIOR / "evaluation/noma" / (split + "_frame_scores.csv")
    old = {}
    for row in csv.DictReader(source.open()):
        if row["arm"] == "full" and row["condition"] == CONDITION:
            key = (int(row["seed"]), row["tuple_anchor"], row["role"], row["source_id"])
            if key in old:
                raise ValueError(key)
            old[key] = [float(row[m]) for m in PRIMARY]
    return old, digest(source)


def checked_demix(model, y, arm, qualify, ledger):
    modem = model.modem
    modem.cross_subtract = arm in ("both","cancel")
    modem.update_covariance = arm in ("both","covariance")
    original = modem._demix_single_stage
    stages = []
    def observe(*args, **kwargs):
        out = original(*args, **kwargs)
        if qualify:
            stages.append(out)
        ledger["linear_stages_batched"] += 1
        return out
    modem._demix_single_stage = observe
    before = ledger["linear_stages_batched"]
    try:
        outputs, _ = modem.blind_demix_users(y, 3, model.latent_channels)
    finally:
        del modem._demix_single_stage
    if ledger["linear_stages_batched"] - before != 3:
        raise ValueError("Different number of demixing stages")
    if qualify and arm == "neither":
        for first, last in zip(stages[0], stages[-1]):
            for obj1, obj2 in zip(first, last):
                if torch.is_tensor(obj1):
                    assert torch.equal(obj1, obj2)
                else:
                    for k, a in vars(obj1).items():
                        if torch.is_tensor(a):
                            assert torch.equal(a, getattr(obj2,k)), k
    return outputs


@torch.inference_mode()
def main():
    configure_runtime_env(cache_root=TEMP/"cache/noma",num_threads=4)
    seed_everything(20261124)
    configure_torch_determinism()
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    out = STUDY / "noma"
    ident = read(PRIOR/"evaluation/noma/identity.json")
    cp = read(STUDY/"protocol.json")["references"]["frozen_4b"]
    assert digest(cp["path"]) == cp["sha256"] == ident["checkpoints"]["full"]["sha256"]
    assert digest(SWIN) == SWIN_HASH
    backbone = load_swinjscc_backbone(SWIN,model_mode="SwinJSCC_w/_SA",c_bottleneck=96,device="cuda:0")
    model = GrantFreeNOMATransceiver(backbone,method="semsic_noma",n_rx=2,num_rbs=32,hidden_ch=160)
    load_state(model,cp["path"])
    model.cuda().eval()
    metric = OfflineVisualEvaluator(device="cuda:0",include_lpips=False)
    ledger = {"linear_stages_batched":0,"new_quality_pairs":0,"reused_quality_pairs":0,
              "transmission_tuples":0,"replayed_native_quality_pairs":0}
    replay, files = [], {}
    for split, n in (("common500",500),("kodak24",24)):
        ds = load_split_u8(split,allow_holdout=True)
        ids = ds.source_ids
        assert ds.is_holdout and len(set(ids)) == len(ids) == n
        source_path = PRIOR/"evaluation/noma"/(split + "_source_manifest.json")
        assert [{"source_id":k,"chw_u8_sha256":tensor_hash(x)} for k,x in zip(ids,ds.images_u8)] == read(source_path)
        targets = ds.images_u8.float()/255
        stems = torch.cat([model.extract_encoder_stem(x.cuda()).cpu() for x in targets.split(32)])
        old, scores_hash = reference(split)
        assert len(old) == 3*3*n
        files[split] = {"source_manifest_sha256":digest(source_path),"native_scores_sha256":scores_hash}
        rows = []
        for seed in (3001,3002,3003):
            gen = torch.Generator().manual_seed(seed*2089 + 3*97 + 130)
            hr, hi = sample_multiuser_uplink_channels(n,3,2,32,.85,generator=gen)
            noise = torch.randn((n,2,32,384,2),generator=gen,dtype=torch.float32)
            hashes = [tensor_hash(a) for a in (hr,hi,noise)]
            proof = next(z for z in ident["events"][split][CONDITION]["noise"] if z["seed"] == seed)
            assert hashes == proof["tensor_hashes"]
            use_old = False
            for a in range(0,n,25):
                b = min(a+25,n)
                idx = [(np.arange(a,b)+shift)%n for shift in (0,n//3,2*n//3)]
                x = [targets[i].cuda() for i in idx]
                tx = torch.stack([model.encode_single_user_stem(stems[i].cuda(),u)[0] for u,i in enumerate(idx)],dim=1)
                snr = torch.tensor(SNR,device="cuda",dtype=torch.float32).expand(b-a,3)
                y, _ = physical_grantfree_uplink_mac(tx,snr,n_rx=2,h_mu_override=(hr[a:b],hi[a:b]),standard_noise_iq=noise[a:b])
                ledger["transmission_tuples"] += b-a
                for arm in SIC_ARMS:
                    from_file = arm == "both" and a > 0 and use_old
                    if not from_file:
                        demixed = checked_demix(model,y,arm,a==0,ledger)
                    max_error = np.zeros(2)
                    for u,i in enumerate(idx):
                        role = f"User{u+1}_{SNR[u]}dB"
                        keys = [(seed,ids[a+j],role,ids[ii]) for j,ii in enumerate(i)]
                        native = np.asarray([old[k] for k in keys])
                        if from_file:
                            vals = native
                            ledger["reused_quality_pairs"] += b-a
                        else:
                            rec = model._decode_from_channel_out(*demixed[u])
                            scores = metric.evaluate_batch(x[u],rec)
                            vals = np.column_stack([scores[k] for k in PRIMARY])
                            assert np.isfinite(vals).all()
                            ledger["new_quality_pairs"] += b-a
                            if arm == "both" and a == 0:
                                max_error = np.maximum(max_error,np.abs(vals-native).max(axis=0))
                                ledger["replayed_native_quality_pairs"] += b-a
                        for j,k in enumerate(keys):
                            rows.append(dict(split=split,seed=seed,tuple_anchor=k[1],source_id=k[3],role=role,
                                             arm=arm,origin="archive" if from_file else "new",psnr_db=vals[j,0],dists=vals[j,1]))
                    if arm == "both" and a == 0:
                        use_old = bool((max_error <= [.003,.00005]).all())
                        replay.append(dict(split=split,seed=seed,first_batch=b-a,max_diff=max_error.tolist(),reuse=use_old))
            print(split,seed,"all receiver cells complete",flush=True)
        assert len(rows) == n*3*3*4
        table(out/(split+"_frames.csv"),rows)
    record(out/"evaluation.json",{"counts":ledger,"native_replay":replay,"inputs":files,
           "checkpoint_sha256":cp["sha256"],"equal_linear_stages_per_batch_per_arm":3,
           "neither_matches_its_own_stage_one_on_all_qualification_batches":True})


if __name__ == "__main__":
    main()
