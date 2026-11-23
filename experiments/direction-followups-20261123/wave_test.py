"""Cross training and deployment waveforms while keeping the decoder reliability rule fixed."""
import sys
import numpy as np
import torch
from common import (CKPT, PRIMARY, PRIOR, SEED, STEPS, STUDY, SWIN, SWIN_HASH, TEMP, WAVE_ARMS,
                    digest, include_sources, load_state, read, record, table, tensor_hash)

include_sources()
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import configure_runtime_env, configure_torch_determinism
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.models import load_swinjscc_backbone
from dir5_og.otfs_blind import SympOTFSBlindTransceiver, sample_doubly_selective_multipath

TEST_SPEC = {
    "train": list(WAVE_ARMS), "test": list(WAVE_ARMS),
    "reliability": "Frame-uniform rule in all four cells; the common 2-D refiner still sees its local maps.",
    "primary": "Diagonal spread/spread minus no_spread/no_spread, 10 dB at kd=0.10. Both trained on uniform reliability.",
    "secondary": ["diagonal at kd=0.18", "waveform effect within either set of weights",
                  "training-waveform by deployment-waveform interaction"],
    "splits": ["common500", "kodak24"], "seeds": [5001,5002,5003], "kd":[0.10,0.18],
    "sampling": "Original producer channel formula: raw noise first, then multipath from the same CPU generator.",
    "metrics": list(PRIMARY), "source_bootstrap_seed":2026112308, "draws":5000,
}


@torch.inference_mode()
def main():
    assert read(STUDY/"wave_inference_protocol.json") == TEST_SPEC
    configure_runtime_env(cache_root=TEMP/"cache/wave_test",num_threads=4)
    configure_torch_determinism()
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert digest(SWIN) == SWIN_HASH
    bb = load_swinjscc_backbone(SWIN,model_mode="SwinJSCC_w/_SA",c_bottleneck=96,device="cuda:0")
    models, weights, inits = {}, {}, {}
    for tag, method in WAVE_ARMS.items():
        init = read(STUDY/"training"/tag/"initialization.json")
        inits[tag] = init["trainable_sha256"]
        model = SympOTFSBlindTransceiver(bb,method=method)
        path = CKPT/tag/"last_dir5.pt"
        state = load_state(model,path)
        assert state["step"] == STEPS and state["method"] == method
        assert init["parameters"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.modem.uniform_reliability = True
        models[tag] = model.cuda().eval()
        weights[tag] = {"path":str(path),"sha256":digest(path),"parameters":init["parameters"],"step":state["step"]}
    assert len(set(inits.values())) == 1
    metric = OfflineVisualEvaluator(device="cuda:0",include_lpips=False)
    out = STUDY/"wave_test"
    channels, images, nframes = [], {}, 0
    for split,n in (("common500",500),("kodak24",24)):
        ds = load_split_u8(split,allow_holdout=True)
        ids = ds.source_ids
        assert ds.is_holdout and len(ids) == len(set(ids)) == n
        manifest = PRIOR/"evaluation/noma"/(split + "_source_manifest.json")
        assert read(manifest) == [{"source_id":k,"chw_u8_sha256":tensor_hash(x)} for k,x in zip(ids,ds.images_u8)]
        images[split] = digest(manifest)
        target = ds.images_u8.float()/255
        stems = torch.cat([models["spread"].extract_encoder_stem(x.cuda()).cpu() for x in target.split(32)])
        rows = []
        for kd in TEST_SPEC["kd"]:
            for seed in TEST_SPEC["seeds"]:
                gen = torch.Generator().manual_seed(seed*1709 + round(kd*1000)*13 + 10)
                noise = torch.randn((n,48,256,2),generator=gen,dtype=torch.float32)
                g,nu = sample_doubly_selective_multipath(n,kd,generator=gen)
                channels.append({"split":split,"kd":kd,"seed":seed,"noise_taps_doppler_sha256":[tensor_hash(x) for x in (noise,g,nu)]})
                for a in range(0,n,25):
                    b = min(a+25,n)
                    xb,sb = target[a:b].cuda(),stems[a:b].cuda()
                    obs = dict(multipath_override=(g[a:b].cuda(),nu[a:b].cuda()),standard_noise_td_iq=noise[a:b].cuda())
                    for trained,m in models.items():
                        for deploy in WAVE_ARMS:
                            m.modem.use_otfs = deploy == "spread"
                            assert m.modem.uniform_reliability
                            y = m.forward_doubly_selective(sb,10.,kd_max=kd,input_is_stem=True,**obs)["recon"]
                            vals = metric.evaluate_batch(xb,y)
                            for j,id_ in enumerate(ids[a:b]):
                                rows.append(dict(split=split,kd=kd,seed=seed,source_id=id_,trained=trained,
                                    deploy=deploy,psnr_db=vals["psnr_db"][j],dists=vals["dists"][j]))
                            nframes += b-a
                print(split,kd,seed,"four cells complete",flush=True)
        assert len(rows) == n*2*3*4 and all(np.isfinite([r["psnr_db"],r["dists"]]).all() for r in rows)
        table(out/(split+"_frames.csv"),rows)
    record(out/"evaluation.json",{"weights":weights,"train_initial_sha256":inits,"seeds":channels,"sources":images,
           "new_transmissions_and_quality_pairs":nframes,"training_seed":SEED,"bootstrap_unit":"source"})


if __name__ == "__main__":
    if sys.argv[1] == "register":
        if any((STUDY/"wave_test"/k).exists() for k in ("start.json","evaluation.json")):
            raise RuntimeError("Test already started")
        record(STUDY/"wave_inference_protocol.json",TEST_SPEC)
    else:
        main()
