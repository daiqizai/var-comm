"""Estimate receiver contrasts by resampling disjoint four-image groups."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


MEASURES = ("weak_psnr_db", "weak_dists", "all_psnr_db", "all_dists",
            "interference_nmse_db_r1", "interference_nmse_db_r2", "interference_nmse_db_r3")


def write_rows(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def estimate(groups, bank):
    ci = np.percentile(groups[bank].mean(axis=1), [2.5, 97.5])
    return dict(mean=float(groups.mean()), ci_low=float(ci[0]), ci_high=float(ci[1]))


def read_cell(path, split, cond, seed, n, arms, sources):
    quality = np.full((n, 4, len(arms), 2), np.nan)
    errors = np.full((n, len(arms), 3), np.nan)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            assert (row["split"], row["condition"], int(row["seed"])) == (split, cond, seed)
            g, slot, arm = int(row["group"]), int(row["slot"]), arms.index(row["arm"])
            assert 0 <= g < n and 0 <= slot < 4 and np.isnan(quality[g, slot, arm]).all()
            key = (g, slot)
            assert sources.setdefault(key, row["source"]) == row["source"]
            quality[g, slot, arm] = [float(row["psnr_db"]), float(row["dists"])]
            if slot == 3:
                errors[g, arm] = [float(row[k]) for k in MEASURES[-3:]]
            else:
                assert all(row[k] == "" for k in MEASURES[-3:])
    assert np.isfinite(quality).all() and np.isfinite(errors).all()
    return np.concatenate((quality[:, 3], quality.mean(axis=1), errors), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("protocol", "events", "snapshot", "wait", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    actual_wait = json.loads(args.wait.read_text())["actual_wait"]
    qualification = json.loads((args.events / "qualification.json").read_text())
    compute = json.loads((args.events / "compute.json").read_text())
    admission = json.loads((args.events / "admission.json").read_text())
    assert actual_wait == 0 and qualification["native_equal"] and qualification["gates_verified"] == 12
    assert qualification["observation_immutable"]
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == admission["protocol_sha256"]
    assert not list(args.events.glob("*.partial"))
    args.output.mkdir(exist_ok=False, parents=True)
    seeds, arms = protocol["seeds"], protocol["arms"]
    assert arms == ["latent", "linear_fixed_gate", "rgb_reencode"]
    summary, contrasts, anchors, banks = [], [], [], {}
    observations = 0
    for split, source_count in protocol["splits"].items():
        n = source_count // 4
        bank_seed = {"common500": 2026112401, "kodak24": 2026112402}[split]
        bank = np.random.default_rng(bank_seed).integers(n, size=(10000, n), dtype=np.int32)
        banks[split] = dict(seed=bank_seed, groups=n, draws=len(bank),
                            int32_little_endian_sha256=hashlib.sha256(bank.astype("<i4").tobytes()).hexdigest())
        sources = {}
        upstream = json.loads((args.snapshot / f"anchor_{split}.json").read_text())
        assert upstream["num_disjoint_4tuples"] == n
        for cond in protocol["conditions"]:
            values = np.stack([read_cell(args.events / f"{split}_{cond}_{seed}.csv",
                                         split, cond, seed, n, arms, sources) for seed in seeds])
            assert len(set(sources.values())) == len(sources) == source_count
            observations += n * len(seeds)
            for m, metric in enumerate(MEASURES):
                for a, arm in enumerate(arms):
                    key = dict(split=split, condition=cond, groups=n, seed_realizations=len(seeds),
                               metric=metric)
                    groups = values[:, :, a, m].mean(axis=0)
                    summary.append(dict(**key, arm=arm, **estimate(groups, bank)))
                    if a:
                        delta = (values[:, :, 0, m] - values[:, :, a, m]).mean(axis=0)
                        contrasts.append(dict(**key, comparison=f"latent_minus_{arm}", **estimate(delta, bank)))
                if m < 4:
                    upstream_slot = "weakest_user_slot3" if m < 2 else "all_active_users"
                    upstream_metric = "psnr_db" if m % 2 == 0 else "dists"
                    ref = upstream["conditions"][cond][upstream_slot]["semsic_3round"][upstream_metric]
                    for i, seed in enumerate(seeds):
                        observed = float(values[i, :, 0, m].mean())
                        error = abs(observed - ref["per_seed"][i])
                        anchors.append(dict(split=split, condition=cond, seed=seed, metric=metric,
                                            anchor=ref["per_seed"][i], observed=observed, abs_error=error))
                        assert error <= protocol["published_native_anchor_tolerance"][upstream_metric], anchors[-1]
    assert observations == 786 and compute["score_rows_including_qualification"] == observations * 12 + 12
    assert len(summary) == 84 and len(contrasts) == 56 and len(anchors) == 48
    write_rows(args.output / "summary.csv", summary)
    write_rows(args.output / "paired.csv", contrasts)
    write_rows(args.output / "anchor_check.csv", anchors)
    checks = dict(actual_wait=actual_wait, formal_received_four_user_arrays=observations,
                  formal_paired_user_events=observations * 4, formal_user_arm_rows=observations * 12,
                  development_user_arm_rows=12, native_per_seed_anchor_checks=len(anchors),
                  max_anchor_abs_error={m: max(row["abs_error"] for row in anchors if row["metric"] == m)
                                        for m in MEASURES[:4]}, bootstrap=banks, qualification=qualification,
                  table_rows={"absolute":len(summary), "paired":len(contrasts)},
                  event_files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(args.events.glob("*.csv"))})
    ledger = dict(device=admission["gpu"]["name"], num_devices=compute["gpu_count"],
                  measured_evaluator_wall_s=compute["elapsed_seconds"],
                  evaluator_device_hours=compute["elapsed_seconds"] * compute["gpu_count"] / 3600.,
                  peak_pytorch_reserved_mib=compute["peak_pytorch_reserved_mib"],
                  formal_received_arrays=observations, formal_user_arm_scores=observations * 12,
                  development_received_arrays=1, development_user_arm_scores=12, optimizer_steps=0,
                  additional_rgb_decode_encode_pairs_per_four_user_array=12,
                  additional_formal_rgb_decode_encode_pairs=observations*12, extra_over_air_uses_per_comparison=0,
                  checkpoint_sha256=compute["checkpoint_sha256"], checkpoint_training_runs_reused=3,
                  backbone_sha256=admission["backbone_sha256"], source_pixel_hash=compute["image_tensor_hashes"],
                  protocol_sha256=admission["protocol_sha256"], torch=admission["torch"], cuda=admission["cuda"],
                  resource_note="Wall interval covers the three arms, scoring, and qualification. No isolated arm latency measured.")
    for name, record in (("data_checks", checks), ("resource_ledger", ledger)):
        (args.output / f"{name}.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
