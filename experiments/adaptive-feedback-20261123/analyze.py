"""Direct source-paired bootstrap using the same source draws for all contrasts within each split."""
import csv
import hashlib
from itertools import combinations
import numpy as np
from study import ROOT, SPEC, SPLITS, digest, read, record, table


def ci(x):
    low, high = np.quantile(x, [.025, .975])
    return {"ci_low": float(low), "ci_high": float(high)}


def analyze():
    metrics = SPEC["metrics"]
    modes = list(SPEC["modes"])
    cells = [(s, r) for s in SPEC["conditions"]["snr_db"] for r in SPEC["conditions"]["rho_time"]]
    pairs = list(combinations(range(len(modes)), 2))
    means, deltas, inter, source_rows, banks = [], [], [], [], {}
    score_id = read(ROOT / "score_identity.json")
    for si, (split, n) in enumerate(SPLITS.items()):
        path = ROOT / (split + "_frames.csv")
        assert digest(path) == score_id["frames_sha256"][split]
        data = np.full((n, len(cells), len(modes), 3, len(metrics)), np.nan)
        ids, channels, per_event = {}, {}, {}
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                assert row["split"] == split
                src, rep = int(row["source_index"]), int(row["repeat"])
                cell = cells.index((int(row["snr_db"]), float(row["rho"])))
                arm = modes.index(row["mode"])
                assert ids.setdefault(src, row["source_id"]) == row["source_id"]
                assert channels.setdefault((src, rep), row["channel_seed"]) == row["channel_seed"]
                event = (row["channel_seed"], row["noise_seed"], row["slot"], row["measured_cqi_db"])
                assert per_event.setdefault((src, rep, cell), event) == event
                idx = (src, cell, arm, rep)
                assert np.isnan(data[idx]).all()
                data[idx] = [float(row[m]) for m in metrics]
        assert len(set(ids.values())) == n and len(set(channels.values())) == n * 3
        assert np.isfinite(data).all() and (data[:, :, :, :, 2] >= 0).all()
        av = data.mean(axis=3)
        for i in range(n):
            for c, (s, rho) in enumerate(cells):
                for m, name in enumerate(modes):
                    source_rows.append(dict(split=split, source_index=i, source_id=ids[i], snr_db=s,
                                       rho=rho, mode=name, **dict(zip(metrics, av[i,c,m]))))
        seed = SPEC["seeds"]["bootstrap"] + si
        indices = np.random.default_rng(seed).integers(n, size=(5000, n), dtype=np.int32)
        banks[split] = {"seed": seed, "sha256_index_int32_le": hashlib.sha256(indices.astype("<i4").tobytes()).hexdigest(),
                        "source_order_sha256": hashlib.sha256(("\n".join(ids[i] for i in range(n))+"\n").encode()).hexdigest()}
        weights = np.array([np.bincount(idx, minlength=n) for idx in indices]) / n
        boots = (weights @ av.reshape(n, -1)).reshape((len(weights),) + av.shape[1:])
        mu = av.mean(axis=0)
        for c, (s, rho) in enumerate(cells):
            for k, metric in enumerate(metrics):
                base = dict(split=split, snr_db=s, rho=rho, metric=metric, sources=n, repeats=3)
                for a, name in enumerate(modes):
                    means.append(dict(**base, mode=name, mean=mu[c,a,k], **ci(boots[:,c,a,k])))
                for a, b in pairs:
                    deltas.append(dict(**base, left=modes[a], right=modes[b],
                                     mean_delta=mu[c,a,k]-mu[c,b,k], **ci(boots[:,c,a,k]-boots[:,c,b,k])))
        for s in SPEC["conditions"]["snr_db"]:
            fast, slow = (cells.index((s, r)) for r in SPEC["conditions"]["rho_time"])
            for a, b in pairs:
                for k, metric in enumerate(metrics):
                    vals = (av[:,slow,a,k]-av[:,slow,b,k]) - (av[:,fast,a,k]-av[:,fast,b,k])
                    inter.append(dict(split=split,snr_db=s,metric=metric,left=modes[a],right=modes[b],
                        interaction="slow_minus_fast", mean_delta=float(vals.mean()), **ci(weights @ vals)))
    for name, rows in [("means",means),("paired",deltas),("interactions",inter),("source_means",source_rows)]:
        table(ROOT / (name + ".csv"), rows)
    assert len(means)==144 and len(deltas)==360 and len(inter)==180
    record(ROOT / "statistics.json", {"source_banks":banks, "draws":5000, "means":len(means),
           "paired":len(deltas), "interactions":len(inter), "source_rows":len(source_rows),
           "frames":sum(score_id["counts"].values()), "intervals":"pointwise percentile 95%"})


if __name__ == "__main__":
    analyze()
