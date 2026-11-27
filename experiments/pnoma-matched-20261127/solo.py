"""Hold the nominal weakest user's signal fixed and mute the other three transmitters."""
import os
import numpy as np
from common import (B, HERE, OLD, PRIOR, SNAP, CONDITIONS, OfflineVisualEvaluator,
                    bank, chosen_model, compute_psnr_batch, init, modem_for,
                    physical_grantfree_4x2_mac, sha, source, torch, write_json, write_rows)
from analyze import read, summary
import sys
import time


@torch.inference_mode()
def main():
    start = time.monotonic()
    os.environ["STUDY_SCRATCH"] = str(SNAP.parent)
    os.environ["STUDY_PARENT"] = str(OLD)
    sys.path.insert(0, str(PRIOR))
    import runtime as rt
    from receiver import decode_all, demix
    from dir2_zpac.models import load_swinjscc_backbone
    out = init("solo")
    images, ids, _ = source("common500")
    seed = 1001
    backbone = load_swinjscc_backbone(rt.BACKBONE, model_mode="SwinJSCC_w/_SA",
                                      c_bottleneck=96, device=torch.device("cuda:0")).eval()
    native = rt.weight(backbone, seed)
    pnoma = chosen_model()
    modem = modem_for(seed)
    metric = OfflineVisualEvaluator(torch.device("cuda:0"), include_lpips=False)
    arms = ("pnoma_q2", "semsic_three")
    joined = {(r["condition"], int(r["group"]), r["method"]): r
              for r in read(HERE / "evaluate/group_scores.csv")
              if int(r["draw"]) == seed and r["method"] in arms}
    assert len(joined) == 500
    raw, estimates = [], []
    boot = np.random.default_rng(271128).integers(0, 125, (10000, 125))
    for cond, levels in CONDITIONS.items():
        hr, hi, eps, powers = bank(125, seed, levels, "rotated")
        data = np.full((2, 2, 125, 2), np.nan)
        for st in range(0, 125, 3):
            en = min(st + 3, 125)
            x = images[st * 4:en * 4].cuda().reshape(en - st, 4, 3, 256, 256).float() / 255
            weak = powers[st:en].argmin(1).cuda()
            idx = torch.arange(en - st, device=weak.device)
            active = torch.zeros((en - st, 4), device=weak.device)
            active[idx, weak] = 1
            assert bool((active.sum(1) == 1).all())
            feat = native.extract_encoder_stem(x.flatten(0, 1))
            native_tx = torch.stack([native.encode_single_user_stem(feat[u::4], u)[0]
                                     for u in range(4)], dim=1)
            for k, (method, tx) in enumerate(zip(arms, (pnoma.transmit(x, modem), native_tx))):
                y, _ = physical_grantfree_4x2_mac(tx, powers[st:en].cuda(), active_user_mask=active,
                        n_rx=2, h_mu_override=(hr[st:en], hi[st:en]), standard_noise_iq=eps[st:en])
                if method == "pnoma_q2":
                    rgb = pnoma.receive(y, modem).clamp(0, 1)
                else:
                    symbols, _, _ = demix(native, y, ("late", "power"), False)
                    rgb = decode_all(native, symbols)
                target, recovered = x[idx, weak], rgb[idx, weak]
                p = compute_psnr_batch(target, recovered).tolist()
                d = metric.dists_model(target, recovered, require_grad=False, batch_average=False).reshape(-1).tolist()
                for i, u in enumerate(weak.tolist()):
                    g = st + i
                    old = joined[cond, g, method]
                    assert int(old["weak_slot"]) == u and old["weak_source"] == ids[g * 4 + u]
                    both = [float(old["weak_psnr"]), float(old["weak_dists"])]
                    data[k, :, g, :] = [[p[i], d[i]], both]
                    raw.append(dict(condition=cond, group=g, draw=seed, weak_slot=u, weak_source=old["weak_source"],
                                    method=method, solo_psnr=p[i], solo_dists=d[i],
                                    all_psnr=both[0], all_dists=both[1]))
        assert np.isfinite(data).all()
        for j, name in enumerate(("weak_psnr", "weak_dists")):
            ext, local = data[..., j]
            solo_gap, all_gap = local[0] - ext[0], local[1] - ext[1]
            ext_cost, local_cost = ext[1] - ext[0], local[1] - local[0]
            interaction = local_cost - ext_cost
            assert np.allclose(all_gap, solo_gap + interaction, atol=1e-10, rtol=0)
            for contrast, v in (("pnoma_weak_only", ext[0]), ("semsic_weak_only", local[0]),
                                ("pnoma_all_minus_solo", ext_cost), ("semsic_all_minus_solo", local_cost),
                                ("solo_semsic_minus_pnoma", solo_gap), ("all_semsic_minus_pnoma", all_gap),
                                ("change_in_gap", interaction)):
                estimates.append(dict(condition=cond, draw=seed, metric=name, contrast=contrast,
                                      quartets=125, **summary(v, boot)))
        print("scored solo", cond, flush=True)
    assert len(raw) == 500 and len(estimates) == 28
    write_rows(out / "paired_groups.csv", raw)
    write_rows(out / "paired_summary.csv", estimates)
    write_json(out / "done.json", dict(new_frames=len(raw), reused_all_active=len(raw),
               weights_sha256=sha(B / "liberai-checkpoint/liulu/var_comm/pnoma_matched_20261127_v1/frozen_four.pt"),
               solo_protocol_sha256=sha(HERE / "solo_protocol.json"), wall_s=time.monotonic() - start,
               peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))

if __name__ == "__main__":
    main()
