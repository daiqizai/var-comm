"""Register, calibrate on image-free channels, freeze, and evaluate a delayed CQI budget."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "direction-followups-20261123"))
from common import (D2, OLD, PRIOR, STUDY, SWIN, SWIN_HASH, TEMP_BASE, USER, digest,
                    include_sources, load_state, read, record, table, tensor_hash)
from feedback_codes import decode, delivered_features, encode, lloyd_levels, predict, regression

include_sources()
from dir2_zpac.phy_v2 import OFDMFadingStep, simulate_ofdm_ar1_chain

ROOT = USER / "analysis/var_comm_adaptive_feedback_20261123"
TMP = TEMP_BASE / "adaptive_feedback_exploration_20261123_v1"
SPEC = read(HERE / "protocol.json")
SPLITS = {"common500": 500, "kodak24": 24}
PHY_FIELDS = ("hr_rb", "hi_rb", "hhat_r_rb", "hhat_i_rb", "sigma_n_sq", "sigma_e_sq", "c_r_db")


def register():
    cp = read(STUDY / "feedback/protocol.json")["weights"]["joint"]
    if digest(cp["path"]) != cp["sha256"] or digest(SWIN) != SWIN_HASH:
        raise ValueError("Model identity mismatch")
    src = OLD / "source/experiments" / D2 / "src/dir2_zpac"
    record(ROOT / "registration.json", {
        "protocol": SPEC, "backbone": {"path": str(SWIN), "sha256": SWIN_HASH}, "adapter": cp,
        "split_manifests": {s: digest(PRIOR / "evaluation/noma" / (s + "_source_manifest.json")) for s in SPLITS},
        "sources": {p.name: digest(p) for p in sorted(src.glob("*.py"))},
        "scripts": {p.name: digest(p) for p in sorted(HERE.iterdir()) if p.suffix in (".py", ".json")},
        "prior_common_sha256": digest(HERE.parent / "direction-followups-20261123/common.py"),
    })


def channel_bank(start, per_cell):
    settings = SPEC["calibration"]
    out = []
    for i, snr in enumerate(settings["snr_db"]):
        for j, rho in enumerate(settings["rho_time"]):
            for k in range(per_cell):
                seed = start + i * 10000 + j * 100 + k
                st = simulate_ofdm_ar1_chain(settings["length"], snr, rho, 0, seed, warmup=0)
                out.append(np.array([s.c_r_db for s in st]))
    return out


def fit_mode(mode, step, bank, frozen, slots):
    scheme = SPEC["modes"][mode]
    groups = [[] for _ in range(scheme[1])]
    for obs in bank:
        codes = encode(obs, scheme, frozen["books"], frozen["initial_db"], step)
        values = decode(codes, scheme[0], frozen["books"], frozen["initial_db"], step)
        feats, phases = delivered_features(values, slots, scheme[1], SPEC["conditions"]["delay"])
        for i, group in enumerate(groups):
            mask = phases == i
            group.append(np.column_stack((feats[mask], obs[slots[mask]])))
    by_phase = {}
    for i, group in enumerate(groups):
        data = np.concatenate(group)
        by_phase[str(i)] = regression(data[:, :-1], data[:, -1], SPEC["calibration"]["ridge"])
    return {"step": step, "by_phase": by_phase}


def calibrate():
    cfg, seeds = SPEC["calibration"], SPEC["seeds"]
    fit = channel_bank(seeds["fit_base"], cfg["fit_trajectories_per_cell"])
    val = channel_bank(seeds["validation_base"], cfg["validation_trajectories_per_cell"])
    slots = np.arange(cfg["warmup_blocks"], cfg["length"])
    pool = np.concatenate([v[slots] for v in fit])
    books = {str(b): lloyd_levels(pool, b) for b in (1, 2, 4)}
    books["5"] = (-5. + .8 * np.arange(32)).tolist()
    frozen = {"books": books, "initial_db": float(pool.mean()), "models": {}, "validation_mse_db2": {}}
    for mode, scheme in SPEC["modes"].items():
        candidates = cfg["differential_step_db"] if mode == "differential1" else [None]
        scores, models = [], []
        for step in candidates:
            model = fit_mode(mode, step, fit, frozen, slots) if scheme[0] else {"step": None}
            frozen["models"][mode] = model
            mse = []
            for obs in val:
                codes = encode(obs, scheme, books, frozen["initial_db"], step)
                pred = predict(codes, slots, SPEC, mode, frozen)
                mse.append(np.square(pred - obs[slots]).mean())
            models.append(model)
            scores.append(float(np.mean(mse)))
        frozen["models"][mode] = models[int(np.argmin(scores))]
        frozen["validation_mse_db2"][mode] = [{"step": s, "mse": v} for s, v in zip(candidates, scores)]
    frozen["registration_sha256"] = digest(ROOT / "registration.json")
    record(ROOT / "frozen.json", frozen)
    print("Frozen on", len(fit), "training and", len(val), "validation trajectories", flush=True)


def test_history(split_i, source_i, rep, snr, rho, frozen):
    slot = 40 + (source_i + rep) % 4
    seed = SPEC["seeds"]["test_base"] + split_i * 100000 + rep * 10000 + source_i
    steps = simulate_ofdm_ar1_chain(slot + 1, snr, rho, 0, seed, warmup=0)
    obs = np.array([s.c_r_db for s in steps])
    current = steps[-1]
    result = {k: getattr(current, k) for k in PHY_FIELDS}
    result.update(slot=slot, channel_seed=seed)
    for mode, scheme in SPEC["modes"].items():
        step = frozen["models"][mode]["step"]
        # Exclude the current block and the intervening block from the encoder feed.
        visible = obs[:slot - SPEC["conditions"]["delay"] + 1]
        codes = encode(visible, scheme, frozen["books"], frozen["initial_db"], step)
        pred = float(predict(codes, np.array([slot]), SPEC, mode, frozen)[0])
        result[mode] = pred
        # Mutating every unarrived observation must leave the delivered payload unchanged.
        if source_i == 0:
            corrupted = obs.copy()
            corrupted[len(visible):] = 1000.
            full = encode(corrupted, scheme, frozen["books"], frozen["initial_db"], step)
            assert np.array_equal(full[:len(codes)], codes)
            assert predict(full, np.array([slot]), SPEC, mode, frozen)[0] == pred
    return result


def grid():
    for split_i, (split, n) in enumerate(SPLITS.items()):
        for snr in SPEC["conditions"]["snr_db"]:
            for rho in SPEC["conditions"]["rho_time"]:
                for rep in range(3):
                    tag = f"{split}_snr{snr}_rho{int(rho * 100):02d}_r{rep}"
                    yield split_i, split, n, snr, rho, rep, tag


def prepare():
    frozen = read(ROOT / "frozen.json")
    indices = {}
    for si, split, n, snr, rho, rep, tag in grid():
        vals = [test_history(si, j, rep, snr, rho, frozen) for j in range(n)]
        assert np.array_equal(np.bincount([v["slot"] % 4 for v in vals]), [n // 4] * 4)
        path = TMP / "channels" / (tag + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            np.savez_compressed(output, **{k: np.array([v[k] for v in vals]) for k in vals[0]})
        indices[tag] = digest(path)
    record(ROOT / "channels.json", {"frozen_sha256": digest(ROOT / "frozen.json"), "artifacts": indices,
           "unique_scored_channel_conditions": sum(SPLITS.values()) * 4 * 3,
           "independent_seeds": sum(SPLITS.values()) * 3, "scored_phases": [0,1,2,3]})


def score():
    import torch
    from dir2_zpac.datasets import load_split_u8
    from dir2_zpac.hygiene import configure_runtime_env, configure_torch_determinism, seed_everything
    from dir2_zpac.metrics import OfflineVisualEvaluator
    from dir2_zpac.models import load_swinjscc_backbone
    from dir2_zpac.models_v2 import TransceiverZPACv2
    from dir2_zpac.phy_channel import make_paired_noise

    reg, index = read(ROOT / "registration.json"), read(ROOT / "channels.json")
    assert index["frozen_sha256"] == digest(ROOT / "frozen.json")
    configure_runtime_env(TMP / "cache", num_threads=4)
    configure_torch_determinism(num_threads=4)
    seed_everything(20261123)
    assert digest(SWIN) == reg["backbone"]["sha256"]
    bb = load_swinjscc_backbone(SWIN, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device="cuda:0")
    model = TransceiverZPACv2(bb, method="tr_zpac", hidden_ch=160)
    assert digest(reg["adapter"]["path"]) == reg["adapter"]["sha256"]
    load_state(model, reg["adapter"]["path"])
    model = model.cuda().eval()
    metric = OfflineVisualEvaluator(device="cuda:0", include_lpips=False)
    counts = {}
    with torch.inference_mode():
        for split, n in SPLITS.items():
            ds = load_split_u8(split, allow_holdout=True)
            manifest = PRIOR / "evaluation/noma" / (split + "_source_manifest.json")
            assert ds.is_holdout and len(set(ds.source_ids)) == n
            assert digest(manifest) == reg["split_manifests"][split]
            assert read(manifest) == [{"source_id": k, "chw_u8_sha256": tensor_hash(x)}
                                      for k, x in zip(ds.source_ids, ds.images_u8)]
            targets = ds.images_u8.float() / 255.
            stems = torch.cat([model.extract_encoder_stem(x.cuda()).cpu() for x in targets.split(25)])
            rows = []
            for si, name, _, snr, rho, rep, tag in grid():
                if name != split:
                    continue
                path = TMP / "channels" / (tag + ".npz")
                assert digest(path) == index["artifacts"][tag]
                noise_seed = SPEC["seeds"]["image_noise_base"] + si * 100 + rep
                with np.load(path, allow_pickle=False) as bundle:
                    arr = {k: bundle[k] for k in bundle.files}
                for a in range(0, n, 25):
                    b = min(n, a + 25)
                    xb, stem = targets[a:b].cuda(), stems[a:b].cuda()
                    eps = make_paired_noise((b-a, 256, 96), list(range(a,b)), noise_seed, device="cuda:0")
                    physical = [OFDMFadingStep(c_t_raw_db=0., gamma_avg_db=7., tau_unc=1-.85**4,
                                **{k:arr[k][j] for k in PHY_FIELDS}) for j in range(a,b)]
                    for mode in SPEC["modes"]:
                        states = [replace(st, c_t_raw_db=float(arr[mode][j])) for j, st in enumerate(physical, a)]
                        out = model.forward_channel_v2(stem, 0., 0., 0., channel_mode="ofdm_fading",
                              ofdm_steps=states, standard_noise=eps, equalizer_mode="unbiased_mmse",
                              rate_channels=None, input_is_stem=True)
                        vals = metric.evaluate_batch(xb, out["recon"])
                        for j, st in enumerate(states):
                            i = j + a
                            rows.append(dict(source_id=ds.source_ids[i], source_index=i, split=split,
                                snr_db=snr, rho=rho, repeat=rep, slot=int(arr["slot"][i]), mode=mode,
                                channel_seed=int(arr["channel_seed"][i]), noise_seed=noise_seed,
                                measured_cqi_db=float(st.c_r_db), predicted_cqi_db=st.c_t_raw_db,
                                cqi_sq_error_db2=(st.c_t_raw_db - st.c_r_db)**2,
                                psnr_db=vals["psnr_db"][j], dists=vals["dists"][j]))
                print(tag, "scored", flush=True)
            assert len(rows) == n * 3 * 4 * len(SPEC["modes"])
            table(ROOT / (split + "_frames.csv"), rows)
            counts[split] = len(rows)
    record(ROOT / "score_identity.json", {"counts": counts,
           "frames_sha256": {s:digest(ROOT / (s + "_frames.csv")) for s in SPLITS},
           "torch": str(torch.__version__), "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
           "frozen_sha256": digest(ROOT / "frozen.json"), "channels_sha256": digest(ROOT / "channels.json"),
           "training_steps": 0, "forward_model_calls_at_batch_level": sum((n+24)//25 for n in SPLITS.values())*4*3*6})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "calibrate", "prepare", "score"))
    args = parser.parse_args()
    if args.stage != "register":
        reg = read(ROOT / "registration.json")
        assert reg["protocol"] == SPEC
        assert all(digest(HERE / n) == sha for n, sha in reg["scripts"].items())
        src = OLD / "source/experiments" / D2 / "src/dir2_zpac"
        assert all(digest(src / n) == sha for n, sha in reg["sources"].items())
        assert digest(HERE.parent / "direction-followups-20261123/common.py") == reg["prior_common_sha256"]
    {"register": register, "calibrate": calibrate, "prepare": prepare, "score": score}[args.stage]()
