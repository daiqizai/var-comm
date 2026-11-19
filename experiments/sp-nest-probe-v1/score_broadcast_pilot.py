"""Assess the registered Pilot50 gate from completed, independent channel receptions."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from verify_pfs_cache import filehash, require


SNR = (4, 13, 19)
METRICS = ("dino", "dists", "dreamsim", "psnr_db")
FAMILIES = ("sp", "single", "oma")


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "xt", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def parse_observations(root, plan):
    catalogue = root / "rx_catalog"
    ref = catalogue / "references.csv"
    require(filehash(ref) == read(catalogue / "completion.json")["reference_sha256"], "quality reference identity")
    quality = {}
    with ref.open() as f:
        for row in csv.DictReader(f):
            slot = int(row["slot"])
            require(row["source_id"] == plan["roster"][slot], "quality reference source")
            key = slot, row["received_token_hash"]
            require(key not in quality, "duplicate quality identity")
            quality[key] = np.array([float(row[m]) for m in METRICS])
    with (root / "sp1_plan" / "physical_cases.csv").open() as f:
        cases = list(csv.DictReader(f))
    scores = np.zeros((len(cases), len(SNR), len(METRICS)), dtype=np.float64)
    counts = np.zeros((len(cases), len(SNR), 4), dtype=np.int8)
    energy = np.zeros((len(cases), 4), dtype=np.float64)
    seen = np.zeros((len(cases), 3), dtype=bool)
    raw_energy, measured, statuses = {}, {}, {}
    for family in FAMILIES:
        folder = root / ("sp1_" + family)
        receipt = read(root / ("sp1_" + family + "_control") / "actual_wait.json")
        done = read(folder / "completion.json")
        require(receipt["returncode"] == 0 and not receipt["timed_out"], (family, "actual wait"))
        require(done["unresolved_source_inputs"] == 0, (family, "received input requires full canonical source decode"))
        raws, starts = [], int(seen.sum())
        measured[family] = {k: done[k] for k in ("waves", "new_receiver_observations", "wall_seconds", "cpu_seconds", "peak_reserved_bytes")}
        family_counts = {s: {} for s in SNR}
        for index in sorted(folder.glob("geometry_*.json")):
            geometry = read(index)
            path = folder / geometry["filename"]
            require(filehash(path) == geometry["sha256"], path)
            rows = 0
            with gzip.open(path, "rt") as f:
                for row in csv.DictReader(f):
                    c, slot, rep = (int(row[k]) for k in ("case", "slot", "repeat"))
                    expected = cases[c]
                    require(0 <= rep < 3 and not seen[c, rep] and int(expected["case"]) == c
                            and int(expected["slot"]) == slot and expected["family"] == family
                            and int(expected["pid"]) == int(row["tx_pid"]), (family, "event identity", c, rep))
                    seen[c, rep] = True
                    e = [float(row["energy_" + k]) for k in ("B", "E", "body", "frame")]
                    require(np.isfinite(e).all() and abs(e[3] - (136.0 + 956.0 * e[2]) / 1024.) < 1.e-6,
                            (c, rep, "energy accounting"))
                    energy[c] += e
                    raws.append(e[3])
                    for i, snr in enumerate(SNR):
                        suffix = "_" + str(snr)
                        status, token = row["status" + suffix], row["token_hash" + suffix]
                        require(not status.startswith("PENDING") and (slot, token) in quality, (c, rep, snr, status, token))
                        family_counts[snr][status] = family_counts[snr].get(status, 0) + 1
                        scores[c, i] += quality[slot, token]
                        gray = token == "GRAY"
                        base_only = status in ("B_EMPTY", "B_ACCEPT") or status.startswith("E_")
                        full = status in ("BOTH_ACCEPT", "S_ACCEPT")
                        require(int(gray) + int(base_only) + int(full) == 1, ("receiver status partition", status))
                        counts[c, i] += np.array([gray, base_only, full, row["sic" + suffix] == "True"], dtype=np.int8)
                    rows += 1
            require(rows == geometry["waves"], ("geometry count", index))
        require(int(seen.sum()) - starts == done["waves"] == plan["unique_cases"][family] * 3
                and done["new_receiver_observations"] == done["waves"] * 3, ("family counts", family))
        raw_energy[family] = dict(mean=float(np.mean(raws)), min=min(raws), max=max(raws))
        statuses[family] = family_counts
    require(seen.all() and int(seen.sum()) == plan["physical_waves"], "every physical case and repeat covered")
    return scores / 3., counts, energy / 3., dict(
        physical_waves=int(seen.sum()), receiver_observations=int(seen.sum() * len(SNR)),
        mean_min_max_actual_frame_Es_over_unique_waves=raw_energy,
        physical_status_counts=statuses, measured=measured)


def select(indices, objective, policies):
    require(len(indices) > 0, "nonempty policy family")
    best = max(objective[indices])
    tied = [int(i) for i in indices if objective[i] == best]
    return min(tied, key=lambda i: policies[i]["id"])


def intervals(samples, bank):
    require(samples.shape == (50, len(METRICS)), "ordered per-source contributions")
    resampled = samples[bank].mean(axis=1)
    lo, hi = np.quantile(resampled, (.025, .975), axis=0)
    return {key: dict(mean=float(mu), lo=float(a), hi=float(b))
            for key, mu, a, b in zip(METRICS, samples.mean(0), lo, hi)}


def frontier(indices, values, policies):
    # All policies remain available in the full candidate table. Choose one lexical ID
    # to label each exactly identical coordinate on this non-randomized frontier.
    coords = {}
    for i in indices:
        x, y = map(float, values[i])
        if (x, y) not in coords or policies[i]["id"] < policies[coords[x, y]]["id"]:
            coords[x, y] = int(i)
    highest = -np.inf
    result = []
    for (x, y), i in sorted(coords.items(), reverse=True):
        if y > highest:
            result.append(i)
            highest = y
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root, out = args.pilot, args.out
    base = root / "sp1_plan"
    plan = read(base / "completion.json")
    for name, checksum in plan["inputs"].items():
        require(filehash(base / name) == checksum, ("original pilot registration", name))
    protocol = read(base / "protocol.json")
    policies = read(base / "policies.json")
    lookup = np.load(base / "policy_to_case.npy", allow_pickle=False)
    bank = np.load(base / "paired_source_bank.npy", allow_pickle=False)
    require(lookup.shape == (50, len(policies)) and bank.shape == (10000, 50)
            and bank.min() == 0 and bank.max() == 49, "source and bootstrap dimensions")
    case_means, counts, energy, audit = parse_observations(root, plan)
    # Keep only policy means here; per-source selected policies are joined on demand.
    means = np.mean(case_means[lookup], axis=0)
    out.mkdir()
    candidate_rows, front_rows, selected_rows, paired_rows, source_rows = [], [], [], [], []
    for idx, policy in enumerate(policies):
        row = dict(policy=policy["id"], family=policy["family"], grid=policy.get("grid", "token"))
        for si, snr in enumerate(SNR):
            row.update({f"{metric}_{snr}": float(means[idx, si, k]) for k, metric in enumerate(METRICS)})
        candidate_rows.append(row)
    write_csv(out / "all_policy_means.csv.gz", candidate_rows)
    sets = {family: [i for i, p in enumerate(policies) if p["family"] == family] for family in FAMILIES}
    qs = [i for i in sets["sp"] if policies[i]["grid"] == "quarter"]
    ws = [i for i in sets["sp"] if policies[i]["grid"] == "whole"]
    gate, operating_points = {}, {}
    for weak, strong in protocol["pairs"]:
        pair = f"{weak}_{strong}"
        s = SNR.index(strong)
        for metric, sign in (("dino", 1), ("dists", -1)):
            v = means[:, [0, s], METRICS.index(metric)]
            for family, ids in (("single", sets["single"]), ("sp", qs), ("sp_whole", ws), ("oma", sets["oma"])):
                for i in frontier(ids, v * sign, policies):
                    front_rows.append(dict(pair=pair, metric=metric, family=family, policy=policies[i]["id"],
                                           weak=float(v[i, 0]), strong=float(v[i, 1])))
        operating_points[pair] = {}
        for weight in protocol["weights"]:
            objective = weight * means[:, 0, 0] + (1 - weight) * means[:, s, 0]
            chosen = {name: select(indices, objective, policies) for name, indices in (
                ("SP", qs), ("SP_WHOLE", ws), ("OMA", sets["oma"]), ("Single_Pareto", sets["single"]))}
            chosen["Single_Weak"] = select(sets["single"], means[:, 0, 0], policies)
            chosen["Single_Strong"] = select(sets["single"], means[:, s, 0], policies)
            selected_sp = policies[chosen["SP"]]
            matched = [i for i in ws if all(policies[i][key] == selected_sp[key]
                       for key in ("variant", "br", "qe", "er", "jcap"))]
            chosen["SP_WHOLE_same_PHY"] = select(matched, objective, policies)
            equal_limit = [i for i in matched if policies[i]["target"] == [selected_sp["target"][0], 0]]
            if equal_limit:
                chosen["SP_WHOLE_same_cap_floor"] = select(equal_limit, objective, policies)
            vals = {}
            for name, idx in chosen.items():
                source_cases = lookup[:, idx]
                all_sources = case_means[source_cases]
                v = weight * all_sources[:, 0] + (1 - weight) * all_sources[:, s]
                vals[name] = v
                summary = intervals(v, bank)
                cal = dict(pair=pair, lambda_weak=weight, family=name, policy=policies[idx]["id"])
                avg_energy = np.mean(energy[source_cases], axis=0)
                selected = dict(**cal, sources=50, repetitions=3, actual_frame_energy=float(avg_energy[3]))
                for j, snr in ((0, weak), (s, strong)):
                    selected.update({f"{k}_{snr}": int(n) for k, n in
                                     zip(("gray", "base_only", "accepted", "sic"), counts[source_cases, j].sum(0))})
                    selected.update({f"{metric}_{'weak' if j == 0 else 'strong'}": float(means[idx, j, k])
                                     for k, metric in enumerate(METRICS)})
                for metric, ci in summary.items():
                    selected.update({metric + "_" + kind: val for kind, val in ci.items()})
                # Stable headings across both possible strong-user SNRs.
                for stem in ("gray", "base_only", "accepted", "sic"):
                    selected[stem + "_weak"] = selected.pop(stem + "_" + str(weak))
                    selected[stem + "_strong"] = selected.pop(stem + "_" + str(strong))
                selected_rows.append(selected)
                if weight == .5:
                    for slot, src in enumerate(plan["roster"]):
                        line = dict(**cal, slot=slot, source_id=src)
                        for k, metric in enumerate(METRICS):
                            line[metric + "_weak"] = float(all_sources[slot, 0, k])
                            line[metric + "_strong"] = float(all_sources[slot, s, k])
                        source_rows.append(line)
            comparisons = {}
            for name in chosen:
                if name == "SP":
                    continue
                stats = intervals(vals["SP"] - vals[name], bank)
                comparisons[name] = stats
                for metric, edge in stats.items():
                    paired_rows.append(dict(pair=pair, lambda_weak=weight, comparison="SP_minus_" + name,
                        metric=metric, sp_policy=selected_sp["id"], other_policy=policies[chosen[name]]["id"], **edge))
            union = select(qs + sets["single"], objective, policies)
            operating_points[pair][str(weight)] = dict(pure_sp=selected_sp["id"],
                best_single=policies[chosen["Single_Pareto"]]["id"], best_oma=policies[chosen["OMA"]]["id"],
                selectable_sp_or_single=policies[union]["id"])
            if weight == .5:
                delta = comparisons["Single_Pareto"]
                oma = comparisons["OMA"]
                drop = -delta["dists"]["mean"]
                relative = drop / float(vals["Single_Pareto"][:, 1].mean())
                margin = float(objective[chosen["SP"]] - objective[chosen["Single_Pareto"]])
                crit = protocol["primary_gate"]
                tests = dict(
                    dists_absolute=drop >= crit["DISTS_drop_min"], dists_relative=relative > crit["DISTS_drop_relative_strict"],
                    dists_paired=delta["dists"]["hi"] < 0, dreamsim_absolute=-delta["dreamsim"]["mean"] >= crit["DreamSim_drop_min"],
                    dreamsim_paired=delta["dreamsim"]["hi"] < 0, oma_dino=oma["dino"]["lo"] > 0,
                    oma_dists=oma["dists"]["hi"] < 0, outside_single_support=margin > 0)
                coordinate = {}
                for family, indices in (("single",sets["single"]), ("oma",sets["oma"])):
                    ref = means[indices][:, [0, s], 0]
                    xy = means[chosen["SP"], [0, s], 0]
                    coordinate[family] = dict(candidates=len(indices),
                        dominates=int(np.sum(np.all(xy >= ref, axis=1) & np.any(xy > ref, axis=1))),
                        dominated_by=int(np.sum(np.all(xy <= ref, axis=1) & np.any(xy < ref, axis=1))))
                gate[pair] = dict(passed=all(tests.values()), tests=tests, dists_relative_drop=relative,
                    dino_support_gain=margin, coordinate_dominance_dino=coordinate,
                    comparisons={k:comparisons[k] for k in ("Single_Pareto","OMA","SP_WHOLE_same_PHY")})
    write_csv(out / "selected_operating_points.csv", selected_rows)
    write_csv(out / "paired_differences.csv", paired_rows)
    write_csv(out / "selected_source_means.csv.gz", source_rows)
    write_csv(out / "observed_frontiers.csv", front_rows)
    save(out / "operating_points.json", operating_points)
    save(out / "actual_compute.json", audit)
    passed = all(row["passed"] for row in gate.values())
    save(out / "decision.json", dict(status="GO" if passed else "NO_GO", pairs=gate,
        selection="Pilot50 DINOv2, equal source weight after three channel repeats",
        interval="10,000 paired source resamples; conditional on pilot-selected policies",
        training_steps=0, formal_common500_frames=0, plan_sha256=filehash(base / "completion.json"),
        bank_sha256=plan["bootstrap_hash"], scorer_sha256=filehash(__file__)))
    print("SP1", "GO" if passed else "NO_GO", audit["physical_waves"], "waveforms fully accounted for", flush=True)


if __name__ == "__main__":
    main()
