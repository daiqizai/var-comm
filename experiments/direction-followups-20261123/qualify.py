"""Verify the actual cached stem precision, ordering, and identical initial adapter weights."""
import sys
import hashlib
import torch
from common import (SHARED, SWIN, STUDY, TEMP, WAVE_ARMS, SEED, include_sources, record)

include_sources()
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import configure_runtime_env, configure_torch_determinism, seed_everything
from dir2_zpac.models import load_swinjscc_backbone
from dir5_og.otfs_blind import SympOTFSBlindTransceiver


@torch.inference_mode()
def main():
    configure_runtime_env(cache_root=TEMP/"cache/qualification",num_threads=4)
    configure_torch_determinism()
    cache = torch.load(SHARED/"swin_stems_fp16.pt",map_location="cpu",weights_only=True)
    bb = load_swinjscc_backbone(SWIN, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device="cuda:0")
    records, initials = [], {}
    for tag, method in WAVE_ARMS.items():
        seed_everything(SEED)
        m = SympOTFSBlindTransceiver(bb, method=method).cuda().eval()
        h = hashlib.sha256()
        for name,p in m.named_parameters():
            if p.requires_grad:
                h.update(name.encode() + bytes([0]))
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
        initials[tag] = h.hexdigest()
    assert len(set(initials.values())) == 1
    for split, key, starts in (("train20k","train_stems_fp16",(0,9984,19968)),
                               ("val256","val_stems_fp32",(0,192))):
        ds = load_split_u8(split)
        assert not ds.is_holdout and len(set(ds.source_ids)) == len(cache[key]) == len(ds.images_u8)
        for a in starts:
            xb = ds.images_u8[a:a+64].cuda().float()/255
            y = m.extract_encoder_stem(xb).to(torch.float16).float().cpu()
            old = cache[key][a:a+len(xb)].float()
            mx = (y-old).abs().max().item()
            mean = (y-old).abs().mean().item()
            if mx > .016 or mean > .00003:
                raise ValueError((split,a,mx,mean))
            records.append(dict(split=split,start=a,images=len(xb),max_abs=mx,mean_abs=mean,
                                exact_fraction=(y==old).float().mean().item()))
    record(STUDY/"qualify_cache/identity.json", {"cache_comparisons":records,
        "paired_constructor_hashes":initials,"active_training_hashes_checked_separately":True})
    print("Verified stem precision and order", records, flush=True)


if __name__ == "__main__":
    main()
