"""Register both controls and copy only verified source files into this run."""
from common import (
    CKPT, D4, D5, OLD, PRIOR, SEED, SHARED, SNAP, STEPS, STUDY, SWIN, SWIN_HASH,
    TEMP, digest, path_check, read, record, save, single_replace,
)


CACHE_SHA = "ad8c27115bf89e3298a6e7abbb2d864c51aa17adeb928dd046eb84b8e3928b3e"
NOMA_SHA = "b3da369d23df35b95e57f14cb39a4b5521beadae37efcd3cc5952bae026314b3"
WAVE_SHA = "329c902279cfde07a5b45229ff1d6723bd02707c0a60f444afc71653f31f010f"
TRAIN_SHA = "3598196b28ddf270186e706f6da243aced68bf0059112d20acf0815188051171"


def noma_source(text):
    anchor = "            nu_res = (1.0 - omega_all).square().clamp(0.03, 1.0)"
    return single_replace(text, anchor, anchor + """
            if not getattr(self, "cross_subtract", True):
                y_sic_r, y_sic_i = y_s1_r, y_s1_i
            if not getattr(self, "update_covariance", True):
                nu_res = nu_ones""")


def train_source(text):
    text = single_replace(text, "    run_cache = args.shared_cache_dir / args.run_tag",
                          f"    run_cache = Path({str(TEMP/'cache')!r}) / args.run_tag")
    anchor = "            model = SympOTFSBlindTransceiver(backbone, method=args.method).to(device)"
    return single_replace(text, anchor, anchor + '''
            import hashlib
            init_hash = hashlib.sha256()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    init_hash.update(name.encode() + bytes([0]))
                    init_hash.update(param.detach().cpu().contiguous().numpy().tobytes())
            save_json(args.out_dir / "initialization.json", {
                "trainable_sha256": init_hash.hexdigest(),
                "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "method": args.method, "downstream_reliability": "frame_uniform",
            })''')


def main():
    protocol = STUDY / "protocol.json"
    if protocol.exists():
        raise FileExistsError(protocol)
    for p in (TEMP, CKPT):
        path_check(p).mkdir(exist_ok=False, parents=True)
    if digest(SWIN) != SWIN_HASH or digest(SHARED / "swin_stems_fp16.pt") != CACHE_SHA:
        raise ValueError("Encoder weights or training stem cache changed")
    files = {}
    old5 = read(PRIOR / "dir5/published_v2_identity.json")["files"]
    for project, root in [(D4, OLD / "source/experiments" / D4), (D5, SNAP / "experiments" / D5)]:
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root)
            if rel.parts[0] != "src" and path.name != "train_dir5_otfs_genvar.py":
                continue
            txt = path_check(path).read_text()
            ref = digest(path)
            if project == D5 and old5[str(path.relative_to(SNAP))] != ref:
                raise ValueError(path)
            if path.name == "noma_semsic_blind.py":
                assert ref == NOMA_SHA
                txt = noma_source(txt)
            if path.name == "otfs_blind.py":
                assert ref == WAVE_SHA
                txt = single_replace(txt, "        if self.use_otfs:\n",
                                     "        if getattr(self, 'uniform_reliability', True):\n")
            if path.name == "train_dir5_otfs_genvar.py":
                assert ref == TRAIN_SHA
                txt = train_source(txt)
            dst = TEMP / "source" / project / rel
            compile(txt, str(dst), "exec")
            save(dst, txt.encode())
            files[str(dst.relative_to(TEMP))] = {"before": ref, "after": digest(dst)}
    ref = read(PRIOR / "training/noma/full/train_dir4_full.json")["best_checkpoint"]
    if digest(ref["path"]) != ref["sha256"]:
        raise ValueError("SemSIC weight identity")
    record(protocol, {
        "references": {"backbone_sha256": SWIN_HASH, "frozen_4b": ref, "stem_cache_sha256": CACHE_SHA,
                       "producer5_git": "316d82b035b7af8acf3ce2ec95c34c0a0a4df544",
                       "prior_main": "34b790be5d401f82fda57e1e14182798933c7555", "source_files": files},
        "noma": {"train": 0, "arms": ["neither", "cancel", "covariance", "both"],
                 "factors": ["subtract the other estimated waveforms", "shrink other-user residual-interference power"],
                 "constant": "Frozen full weights, common first solve, two complete latent-prior/gate rounds and two new LMMSE solves per arm. Each round receives its own preceding state.",
                 "snr_db": [13,8,4], "rx_antennas": 2, "seeds": [3001,3002,3003],
                 "splits": ["common500","kodak24"], "pairing":"Previously published cyclic tuple construction and image/channel hash checks. Equal airtime. Each user is scored.",
                 "primary":"Weak user PSNR: both minus covariance-only. DISTS, other users, both minus cancel-only, marginal contrasts and factorial interaction reported.",
                 "native":"Reuse native per-event scores only after checking hashes and replaying first batch for each split/seed; max allowed 0.003 dB PSNR and 0.00005 DISTS per replay position; otherwise recompute the native arm and retain diagnostics.",
                 "positive_check":"With neither factor, repeated solves match stage one of this frozen checkpoint."},
        "wave": {"seed":SEED,"steps_per_arm":STEPS,"batch":28,"optimizer":"original AdamW, lr=0.0008, cosine schedule",
                 "train":"Both arms train from identical adapter values on train20k; same recorded per-step draws. Frozen Swin encoder/backbone, 2D estimator, refiner and downstream frame-uniform reliability in both arms. Only payload ISFFT and inverse SFFT differ. Features of the 2D refiner remain local and identical in design.",
                 "arms":["spread","no_spread"],"selection":"Final step 7000 in both arms, independent of validation ranking.",
                 "test":"common500 and Kodak24 separately, 10 dB, kd_max=0.10 and 0.18. Three producer channel draws 5001/5002/5003 per source. Primary comparison spread minus no_spread at 0.10, both PSNR and DISTS, report the full table.",
                 "budget":"12288 complex resource elements, 13056 time samples including cyclic prefix per image. 1/16 per RGB component before cyclic prefix. No extra channel uses or receiver side information."},
        "inference":"5000 percentile bootstrap draws, seed 2026112308 per split; average channel draws before resampling sources (fixed tuple anchors in 4B). Pointwise intervals conditional on these frozen weights, no distribution over training seeds.",
        "gpu":"Shared A800 under existing memory/thermal admission. Additional prelaunch observations of load and reserved memory. No exclusive-device or standardized runtime comparison. At most 90 min per training arm and 30 min per evaluation.",
    })
    print("Registered", digest(protocol), flush=True)


if __name__ == "__main__":
    main()
