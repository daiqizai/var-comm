"""Replay the native cancellation gates with RGB interference replicas."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

from evaluate_replicas import BACKBONE, WEIGHTS, ReceiverTrace, digest, record, score_interference
from prepare_snapshot import PACKAGES


class FixedGateRGB(ReceiverTrace):
    def __init__(self, model, reference):
        super().__init__(model, "rgb_reencode", reference)

    def gate(self, step, user, value):
        key = step, user
        assert key not in self.gates
        self.gates[key] = self.reference_gates[key]
        return self.gates[key]


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot", "events", "output", "runtime"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    a = parser.parse_args()
    for pkg in PACKAGES:
        sys.path.insert(0, str(a.snapshot / "experiments" / pkg / "src"))
    from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism
    configure_runtime_env(a.runtime, num_threads=4)
    gpu = admit_gpu(physical_gpu=a.gpu, min_free_mib=25000)
    import torch
    from dir2_zpac.datasets import load_split_u8
    from dir2_zpac.metrics import OfflineVisualEvaluator, compute_psnr_batch
    from dir2_zpac.models import load_swinjscc_backbone
    from dir4_rn.noma_semsic_blind import sample_multiuser_uplink_channels
    from dir6_mb.overloaded_noma_4x2 import OverloadedNOMA4x2Transceiver, build_walsh_16_basis, physical_grantfree_4x2_mac
    configure_torch_determinism(4)
    device = torch.device("cuda:0")
    parent = Path(__file__).with_name("protocol.json")
    reg_file = Path(__file__).with_name("gate_protocol.json")
    cfg = json.loads(parent.read_text())
    reg = json.loads(reg_file.read_text())
    saved = json.loads((a.events / "compute.json").read_text())
    old = json.loads((a.events / "admission.json").read_text())
    assert digest(parent) == reg["parent_protocol_sha256"] == old["protocol_sha256"]
    assert digest(a.snapshot / "snapshot.json") == old["snapshot_sha256"]
    assert digest(BACKBONE) == old["backbone_sha256"]
    assert reg["seeds"] == cfg["seeds"] and reg["snr_db"] == list(cfg["conditions"].values())
    a.output.mkdir(exist_ok=False)
    start = time.time()
    record(a.output / "admission.json", dict(start=start, registration_sha256=digest(reg_file),
           parent_protocol_sha256=digest(parent), script_sha256=digest(Path(__file__)), gpu=gpu))
    with torch.inference_mode():
        backbone = load_swinjscc_backbone(BACKBONE, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device).eval()
        evaluator = OfflineVisualEvaluator(device, include_lpips=False)
        data = load_split_u8("common500", allow_holdout=True)
        images, ids = data.images_u8, data.source_ids
        import hashlib
        assert len(images) == len(set(ids)) == 500
        assert hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest() == saved["image_tensor_hashes"]["common500"]
        stems = []
        for batch in images.split(32):
            x = backbone.encoder.patch_embed(batch.to(device=device, dtype=torch.float32) / 255.)
            for layer in backbone.encoder.layers:
                x = layer(x)
            stems.append(backbone.encoder.norm(x).cpu())
        stems = torch.cat(stems)
        total, max_error = 0, 0.
        fields = ["condition", "seed", "group", "source", "weak_psnr_db", "weak_dists", "interference_nmse_db_r3"]
        for seed in reg["seeds"]:
            model = OverloadedNOMA4x2Transceiver(backbone, method="semsic_3round").to(device).eval()
            ckpt = WEIGHTS / f"seed_{seed}" / "best_indist.pt"
            assert digest(ckpt) == saved["checkpoint_sha256"][str(seed)]
            state = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state"]
            assert set(state) == {k for k in model.state_dict() if not k.startswith("backbone.")}
            model.load_state_dict(state, strict=False)
            for name, snrs in cfg["conditions"].items():
                source_file = a.events / f"common500_{name}_{seed}.csv"
                with source_file.open(newline="") as f:
                    prior = {int(z["group"]):z for z in csv.DictReader(f) if z["arm"]=="latent" and z["slot"]=="3"}
                assert len(prior) == 125
                gen = torch.Generator(device="cpu").manual_seed(730000 + seed * 41 + int(snrs[-1] * 10))
                hr, hi = sample_multiuser_uplink_channels(125, 4, 2, 32, 0.85, generator=gen)
                eps = torch.randn((125, 2, 32, 384, 2), generator=gen)
                snr = torch.tensor(snrs, device=device, dtype=torch.float32)
                dest = a.output / f"common500_{name}_{seed}.csv"
                with dest.with_suffix(".partial").open("x", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    for st in range(0, 125, 12):
                        en = min(st+12, 125)
                        tx = torch.stack([model.encode_single_user_stem(stems[4*st+u:4*en:4].to(device), u)[0]
                                          for u in range(4)], dim=1)
                        observation, truth = physical_grantfree_4x2_mac(tx, snr.expand(en-st,4), n_rx=2,
                            h_mu_override=(hr[st:en],hi[st:en]), standard_noise_iq=eps[st:en])
                        native = ReceiverTrace(model, "latent")
                        model.modem.study_trace = native
                        _, _, aux = model.modem.blind_demix_users(observation, 4, model.latent_channels)
                        arm = FixedGateRGB(model, native.gates)
                        model.modem.study_trace = arm
                        stages, _, new_aux = model.modem.blind_demix_users(observation, 4, model.latent_channels)
                        model.modem.study_trace = None
                        assert torch.equal(aux["active_hat"], new_aux["active_hat"])
                        assert len(arm.gates) == len(native.gates) == 12 and all(torch.equal(v,native.gates[k]) for k,v in arm.gates.items())
                        rec = model._decode_from_channel_out(*stages[3]).clamp(0,1)
                        src = images[4*st+3:4*en:4].to(device=device,dtype=torch.float32) / 255.
                        psnr = compute_psnr_batch(src,rec).tolist()
                        dists = evaluator.dists_model(src,rec,require_grad=False,batch_average=False).reshape(-1).tolist()
                        nerr = score_interference(torch,build_walsh_16_basis,tx,truth,native.subtractions[-1:])[0]
                        rerr = score_interference(torch,build_walsh_16_basis,tx,truth,arm.subtractions[-1:])[0]
                        for j in range(en-st):
                            g = st+j
                            assert prior[g]["source"] == ids[4*g+3]
                            err = abs(nerr[j]-float(prior[g]["interference_nmse_db_r3"]))
                            max_error = max(max_error,err)
                            assert err <= 1.e-5
                            writer.writerow(dict(condition=name, seed=seed, group=g, source=ids[4*g+3],
                                weak_psnr_db=psnr[j], weak_dists=dists[j], interference_nmse_db_r3=rerr[j]))
                            total += 1
                dest.with_suffix(".partial").rename(dest)
                print(name, seed, total, flush=True)
    assert total == 750
    record(a.output / "compute.json",dict(weak_user_scores=total, native_gate_replays=total, actual_new_weak_user_metrics=total*2,
           verified_gate_tensors_per_group=12, max_native_r3_anchor_error_db=max_error, elapsed_seconds=time.time()-start,
           pytorch_reserved_mib=torch.cuda.max_memory_reserved()/2**20, optimizer_steps=0))


if __name__ == "__main__":
    run()
