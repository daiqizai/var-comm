"""CPU-only tables, measured-point plots and existing-float image comparisons.

No model is imported, no policy is selected and no historical decoder is run.
Each published file must be smaller than ten million bytes. Git is owned by the
calling delivery process, never by this builder.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

LIMIT = 10_000_000
FIXED = (0, 25, 50, 75, 4, 21, 24, 29, 33, 41, 52, 60, 64, 87, 92, 95)
SNRS = (1, 4, 7, 13, 19)
PHYS = ("QPSK", "16QAM")
METHODS = ("whole_policy", "raster_policy", "random_policy", "entropy_policy", "oracle_policy",
           "raster_at_entropy", "random_at_entropy", "oracle_at_entropy")
LABELS = {"whole_policy": "Whole-scale", "raster_policy": "Raster policy",
          "random_policy": "Random policy", "entropy_policy": "Entropy policy",
          "oracle_policy": "Paid mismatch oracle", "raster_at_entropy": "Raster @ entropy K",
          "random_at_entropy": "Random @ entropy K", "oracle_at_entropy": "Paid oracle @ entropy K",
          "P2048": "P2048", "original": "Original"}
METRICS = ("psnr_db", "lpips_alex", "dino_cosine", "dinov2_vitl14_cosine", "clip_image_cosine",
           "dists", "dreamsim", "ms_ssim", "dino_specificity", "resnet50_top1_label",
           "resnet50_top1_source_prediction", "semantic_error", "confidently_wrong")
PLOT_METRICS = ("psnr_db", "lpips_alex", "dinov2_vitl14_cosine", "clip_image_cosine", "dists", "E")
PLOT_LABELS = dict(psnr_db="PSNR (dB, higher)", lpips_alex="LPIPS (lower)",
                  dinov2_vitl14_cosine="DINOv2 ViT-L/14 (higher)", clip_image_cosine="CLIP image cosine (higher)",
                  dists="DISTS (lower)", E="Actual frame energy")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def ident(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rgb_sha(image):
    image = np.asarray(image)
    require(image.shape == (3, 256, 256) and image.dtype == np.float32 and np.isfinite(image).all()
            and image.min() >= 0 and image.max() <= 1, "Invalid saved RGB")
    return hashlib.sha256(b"float32:3,256,256:RGB\0"+np.ascontiguousarray(image).tobytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    require(bool(rows), "Cannot publish an empty table")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite measured value")
    return result


def pretty(value, places=4):
    return "N/A" if value in (None, "") else ("%.*f" % (places, finite(value)))


def markdown_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| "+" | ".join(map(cell, headers))+" |",
                       "| "+" | ".join("---" for _ in headers)+" |"]+
                      ["| "+" | ".join(map(cell, row))+" |" for row in rows])


def summary_index(rows):
    result = {}
    for row in rows:
        key = (int(row["N"]), row["phy_family"], int(row["snr_db"]), row["method"], row["metric"])
        require(key not in result, "Duplicate reported group/metric")
        require(int(row["n_sources"]) == 100 and int(row["n_frames"]) == 300, "Incomplete measured group")
        for field in ("mean", "ci_low", "ci_high"):
            finite(row[field])
        require(finite(row["ci_low"]) <= finite(row["ci_high"]), "Reversed interval")
        result[key] = row
    return result


def policy_index(policy):
    require(policy["sources"] == 1000 and policy["noise_seeds"] == [4101, 4102, 4103]
            and policy["full_grid"] and not policy["development_read"], "Frozen full-grid calibration required")
    result = {(r["phy_family"], int(r["snr_db"]), r["method"]): r for r in policy["cells"]}
    require(len(result) == len(policy["cells"]) == 50, "Incomplete or duplicate policy cells")
    require(set(result) == {(p, s, m) for p in PHYS for s in SNRS for m in ("whole", "raster", "random", "entropy", "oracle")},
            "Policy scope differs")
    return result


def selected_action(policies, phy, snr, method):
    if method.endswith("_at_entropy"):
        action = dict(policies[phy, snr, "entropy"]["action"])
        action["order"] = "whole" if action["q"] == 0 else method.removesuffix("_at_entropy")
        return action
    return dict(policies[phy, snr, method.removesuffix("_policy")]["action"])


def main_rows(measured, resource, policies):
    result = []
    for phy in PHYS:
        for snr in SNRS:
            for method in METHODS:
                action = selected_action(policies, phy, snr, method)
                row = dict(N=2048, phy_family=phy, snr_db=snr, method=method, **{k: action[k] for k in ("m", "q", "order")},
                           partial_active=action["q"] > 0,
                           same_K_degenerate=method.endswith("_at_entropy") and action["q"] == 0,
                           energy_scope="E=4096 exactly" if phy == "QPSK" else "actual fixed-constellation energy",
                           n_sources=100, n_frames=300)
                for metric in METRICS+("E", "header_ok_fraction", "prefix_crc_ok_fraction", "partial_used_fraction", "latent_valid_fraction", "F_sq_error_zero_erasure_proxy", "latent_sq_err_final"):
                    item = measured.get((2048, phy, snr, method, metric))
                    if item is not None:
                        row.update({metric+"_"+field: finite(item[field]) for field in ("mean", "ci_low", "ci_high")})
                require(all(metric+"_mean" in row for metric in METRICS), "Required method metrics missing")
                if phy == "QPSK":
                    require(abs(row["E_mean"]-4096) < 1e-9, "QPSK energy is not 2N")
                result.append(row)
    for snr in SNRS:
        row = dict(N=2048, phy_family="continuous", snr_db=snr, method="P2048", m="", q="", order="",
                   partial_active=False, same_K_degenerate=False, energy_scope="per-frame E=2N; historical floating-point tolerance",
                   n_sources=100, n_frames=300)
        for metric in METRICS+("E",):
            item = resource.get((2048, "continuous", snr, "P2048", metric))
            if item is not None:
                row.update({metric+"_"+field: finite(item[field]) for field in ("mean", "ci_low", "ci_high")})
        require(all(metric+"_mean" in row for metric in ("psnr_db", "lpips_alex", "dinov2_vitl14_cosine")), "Missing final P2048 reference")
        result.append(row)
    return result


def measured_curve(index, phy, snr, family, metric):
    """Fixed resource axis with NaN gaps; no estimated or filled data points."""
    x, y, low, high = [], [], [], []
    for n in (512, 1024, 2048):
        method = "P"+str(n) if family == "P" else family
        p = "continuous" if family == "P" else phy
        row = index.get((n, p, snr, method, metric))
        x.append(n)
        y.append(float("nan") if row is None else finite(row["mean"]))
        low.append(float("nan") if row is None else finite(row["ci_low"]))
        high.append(float("nan") if row is None else finite(row["ci_high"]))
    return np.asarray(x), np.asarray(y), np.asarray(low), np.asarray(high)


def cost_summary(benchmark, checkpoints):
    require(benchmark["sources"] == 2 and benchmark["physical_frames"] == 4140, "Unexpected benchmark workload")
    return dict(benchmark_sources=2, benchmark_physical_frames=4140,
        benchmark_mean_complete_source_grid_seconds=finite(benchmark["mean_source_seconds"]),
        benchmark_qualification_included=False,
        benchmark_per_source=benchmark.get("per_source", []),
        numerical_runtime=benchmark.get("numerical_runtime", {}),
        benchmark_peak_cuda_reserved_bytes=benchmark.get("peak_cuda_reserved_bytes"),
        benchmark_current_cuda_reserved_bytes=benchmark.get("current_cuda_reserved_bytes"),
        development_sources=len(checkpoints), development_method_rows=sum(len(x["rows"]) for x in checkpoints),
        development_saved_source_grid_seconds_sum=sum(finite(x["seconds"]) for x in checkpoints),
        development_saved_source_grid_seconds_mean=float(np.mean([finite(x["seconds"]) for x in checkpoints])),
        development_unique_receiver_states=sum(int(x["unique_receiver_events"]) for x in checkpoints),
        development_unique_quality_images=sum(int(x["unique_quality_images"]) for x in checkpoints),
        interpretation="Cached offline workloads include PHY, decoding, metrics and I/O; source times are not single-frame online latency.",
        online_single_frame_latency="NOT_MEASURED", latency_division_by_method_rows_permitted=False)


def tradeoff_sentences(paired):
    """Describe measured DINO-L contrasts and their separate pixel costs."""
    sentences = []
    for phy in PHYS:
        selected = [row for row in paired if int(row["N"]) == 2048 and row["phy_family"] == phy
                    and row["method_A"] == "entropy_policy" and row["method_B"] == "P2048"]
        dino = {int(row["snr_db"]): row for row in selected if row["metric"] == "dinov2_vitl14_cosine"}
        require(set(dino) == set(SNRS), "Missing primary semantic paired contrasts")
        positive = [s for s, row in sorted(dino.items()) if finite(row["ci_low"]) > 0]
        negative = [s for s, row in sorted(dino.items()) if finite(row["ci_high"]) < 0]
        overlap = len(SNRS)-len(positive)-len(negative)
        costs = {}
        for metric in ("psnr_db", "lpips_alex"):
            values = [finite(row["mean"]) for row in selected if row["metric"] == metric]
            require(len(values) == 5, "Incomplete pixel/perceptual paired contrasts")
            costs[metric] = (min(values), max(values))
        sentences.append("%s：DINOv2-L 的差值区间完全高于 0 的 SNR 为%s；完全低于 0 的为%s；其余 %d 个区间跨越或触及 0。五个 SNR 的 PSNR 均值差范围为 %.4f 至 %.4f dB，LPIPS 均值差为 %.4f 至 %.4f。" %
            (phy, "、".join(str(s)+" dB" for s in positive) or "无",
             "、".join(str(s)+" dB" for s in negative) or "无", overlap,
             *costs["psnr_db"], *costs["lpips_alex"]))
    return sentences


class Builder:
    def __init__(self, root, out, result):
        self.root, self.out, self.result = Path(root).resolve(), Path(out).resolve(), Path(result).resolve()
        require(self.result != self.out and self.out not in self.result.parents, "Published report must be separate from runtime outputs")
        self.result.mkdir(parents=True, exist_ok=True)
        (self.result/"figures").mkdir(exist_ok=True)
        self.inputs = {}
        self.outputs = {}
        self.figure_manifest = []

    def bind(self, path, expected=None):
        path = Path(path)
        digest = sha(path)
        require(expected is None or digest == expected, "Bound input changed: " + str(path))
        self.inputs[str(path)] = digest
        return path

    def output(self, path):
        path = Path(path)
        require(path.stat().st_size < LIMIT, "Publication file exceeds 10 MB: " + str(path))
        self.outputs[str(path.relative_to(self.result))] = dict(sha256=sha(path), bytes=path.stat().st_size)
        return path

    def load(self):
        self.done = read(self.bind(self.out/"score_completion.json"))
        require(self.done["status"] == "M1_N2048_METRICS_COMPLETE" and self.done["sources"] == 100
                and self.done["rows"] == 24000 and not self.done["synthetic"], "Completed real metric receipt required")
        self.dev = read(self.bind(self.out/"development_completion.json"))
        require(self.dev["status"] == "COMPLETE" and self.dev["sources"] == 100 and self.dev["method_rows"] == 24000,
                "Complete real development required")
        self.cal = read(self.bind(self.out/"calibrate_completion.json"))
        require(self.cal["status"] == "COMPLETE" and self.cal["physical_frames"] == 2070000, "Full calibration required")
        self.reg = read(self.bind(self.out/"development_registration.json", self.dev["outputs"][str(self.out/"development_registration.json")]))
        self.run_registration = read(self.bind(self.out/"registration.json", self.dev["registration_sha256"]))
        self.frozen_protocol = None
        if self.run_registration.get("protocol_path"):
            protocol_path = Path(self.run_registration["protocol_path"])
            if not protocol_path.is_absolute():
                protocol_path = self.root/protocol_path
            self.frozen_protocol = read(self.bind(protocol_path, self.run_registration["protocol_sha256"]))
        self.policy = read(self.bind(self.out/"m1_policy.json", self.cal["policy_sha256"]))
        require(self.dev["policy_sha256"] == self.cal["policy_sha256"] == self.reg["policy_sha256"], "Policy differs across completed stages")
        self.policies = policy_index(self.policy)
        self.tables = {}
        for name in ("metrics_summary.csv", "metrics_paired_intervals.csv", "m1_vs_P_paired.csv", "resource_summary.csv"):
            path = self.out/"metrics"/name
            self.tables[name] = read_csv(self.bind(path, self.done["outputs"][str(path)]))
        self.measured = summary_index(self.tables["metrics_summary.csv"])
        self.resource = summary_index(self.tables["resource_summary.csv"])
        self.main = main_rows(self.measured, self.resource, self.policies)
        metricreg = self.out/"metrics/registration.json"
        self.metric_registration = read(self.bind(metricreg, self.done["registration_sha256"]))
        metadata_path = self.out/"metrics/model_metadata.json"
        self.model_metadata = read(self.bind(metadata_path, self.done["outputs"].get(str(metadata_path))))
        self.checkpoints = []
        for index in range(100):
            path = self.out/"development/source_checkpoints"/("%04d.json" % index)
            cp = read(self.bind(path, self.dev["outputs"][str(path)]))
            require(cp["binding"] == ident(self.reg) and cp["payload_sha256"] == ident({k: v for k, v in cp.items() if k != "payload_sha256"}),
                    "Development source seal changed")
            require(cp["source_index"] == index and len(cp["rows"]) == 240, "Development source count changed")
            self.checkpoints.append(cp)
        benchdone = read(self.bind(self.out/"benchmark_completion.json"))
        benchmark = read(self.bind(self.out/"benchmark_report.json", benchdone["outputs"][str(self.out/"benchmark_report.json")]))
        self.cost = cost_summary(benchmark, self.checkpoints)
        return self

    def write_tables(self):
        write_csv(self.result/"MAIN_TABLE.csv", self.main)
        self.output(self.result/"MAIN_TABLE.csv")
        for name in self.tables:
            destination = self.result/name
            shutil.copyfile(self.out/"metrics"/name, destination)
            self.output(destination)
        calpath = self.out/"calibration_summary.csv"
        self.bind(calpath, self.cal["outputs"][str(calpath)])
        shutil.copyfile(calpath, self.result/calpath.name)
        self.output(self.result/calpath.name)
        policy_rows = []
        for (phy, snr, method), value in sorted(self.policies.items()):
            action = value["action"]
            policy_rows.append(dict(N=2048, phy_family=phy, snr_db=snr, method=method,
                m=action["m"], K=action["q"], order=action["order"], partial_active=action["q"] > 0,
                calibration_psnr_db=value["calibration"]["psnr_db"], calibration_lpips=value["calibration"]["lpips_alex"],
                calibration_failure_fraction=value["calibration"]["failure_fraction"]))
        write_csv(self.result/"POLICIES.csv", policy_rows)
        self.output(self.result/"POLICIES.csv")
        for name, value in (("MODEL_METADATA.json", self.model_metadata), ("EVALUATION_REGISTRATION.json", self.metric_registration),
                            ("COSTS.json", self.cost), ("POLICY.json", self.policy),
                            ("RUN_REGISTRATION.json", self.run_registration)):
            write_json(self.result/name, value)
            self.output(self.result/name)
        if self.frozen_protocol is not None:
            write_json(self.result/"PROTOCOL.json", self.frozen_protocol)
            self.output(self.result/"PROTOCOL.json")
        sections = ["# N2048 完整主表", "每行 100 张源图 × 3 个噪声。完整区间在 MAIN_TABLE.csv；N/A 保持缺失。"]
        for phy in PHYS:
            sections += ["## "+phy, "QPSK 每帧 E=4096。" if phy == "QPSK" else "16QAM 使用原固定星座；E 列为实际均值，不能当作同能量比较。"]
            rows = []
            for row in self.main:
                if row["phy_family"] not in (phy, "continuous"):
                    continue
                rows.append([row["snr_db"], LABELS[row["method"]], str(row["m"])+"/"+str(row["q"]) if row["method"] != "P2048" else "—",
                    *[pretty(row.get(m+"_mean"), 2 if m in ("E", "psnr_db") else 4)
                      for m in ("E", "psnr_db", "lpips_alex", "dinov2_vitl14_cosine", "clip_image_cosine", "dists", "resnet50_top1_source_prediction")]])
            sections.append(markdown_table(["SNR", "方法", "m/K", "E", "PSNR↑", "LPIPS↓", "DINOv2-L↑", "CLIP↑", "DISTS↓", "分类一致率↑"], rows))
        path = self.result/"MAIN_TABLE.md"
        path.write_text("\n\n".join(sections)+"\n", encoding="utf-8")
        self.output(path)

    def curves(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
        for phy in PHYS:
            for snr in SNRS:
                figure, axes = plt.subplots(2, 3, figsize=(13.8, 7.4), constrained_layout=True)
                for ax, metric in zip(axes.ravel(), PLOT_METRICS):
                    for family, color, label in (("P", "#4D4D4D", "P (final historical model)"),
                                                 ("whole_policy", "#2878B5", "Whole-scale digital VAR"),
                                                 ("entropy_policy", "#D55E00", "M1 entropy policy")):
                        x, y, lo, hi = measured_curve(self.resource, phy, snr, family, metric)
                        # NaNs leave visible gaps when an original measured point is absent.
                        ax.plot(x, y, marker="o", color=color, label=label)
                        valid = np.isfinite(y)
                        ax.vlines(x[valid], lo[valid], hi[valid], colors=color, alpha=.55)
                    ax.set_xticks([512, 1024, 2048])
                    ax.set_xlabel("N: complex channel uses")
                    ax.set_ylabel(PLOT_LABELS[metric])
                    ax.grid(alpha=.2)
                axes[0, 0].legend(fontsize=8)
                figure.suptitle("%s, %d dB | 100 sources x 3 noise repeats | measured points only" % (phy, snr))
                for suffix in ("png", "pdf"):
                    path = self.result/"figures"/("quality_resource_%s_snr%d.%s" % (phy, snr, suffix))
                    figure.savefig(path, dpi=160 if suffix == "png" else None)
                    self.output(path)
                plt.close(figure)

    def load_fixed_arrays(self):
        runtime = self.root/"outputs/EXTERNAL-COMPARISON-20261004/runtime"
        sys.path.insert(0, str(runtime))
        prepare = importlib.import_module("step0_reference_prepare")
        cache = importlib.import_module("step0_cache_export")
        for module in (prepare, cache):
            self.bind(module.__file__)
        _, receipt, admitted_bindings = prepare.admitted(self.root, "FINAL_P2048_P3060")
        for path, digest in admitted_bindings.items():
            self.bind(path, digest)
        historical = self.root/"outputs/HISTORICAL-METRICS-R2-20261003/FINAL_P2048_P3060"
        histregpath = self.root/"results/historical_metrics_r2_20261003/FINAL_P2048_P3060/registration.json"
        histreg = read(self.bind(histregpath, receipt["bindings"][str(histregpath)]))
        result = {}
        for index in FIXED:
            cp = self.checkpoints[index]
            proof = cp["float_reconstructions"]
            self.bind(proof["path"], self.dev["outputs"][proof["path"]])
            require(proof["sha256"] == self.inputs[proof["path"]], "Float reconstruction proof changed")
            with np.load(proof["path"], allow_pickle=False) as z:
                images, target, ids, slots = z["images"].copy(), z["source_rgb"].copy(), z["row_ids"].tolist(), z["image_slots"].tolist()
            require(ids == [r["replay_row_id"] for r in cp["rows"]] and len(slots) == 240, "Saved float row mapping changed")
            reference = rgb_sha(target)
            rows = {}
            for row, slot in zip(cp["rows"], slots):
                require(row["reference_sha256"] == reference and row["image_sha256"] == rgb_sha(images[slot]), "Saved float image/scientific hash differs")
                if int(row["noise_seed"]) == 2001:
                    rows[row["phy_family"], int(row["snr_db"]), row["method"]] = (images[slot], row)
            hcp_path, hnpz = historical/"source_checkpoints"/("%04d.json" % index), historical/"reconstructions"/("%04d.npz" % index)
            hcp = read(self.bind(hcp_path, receipt["bindings"][str(hcp_path)]))
            self.bind(hnpz, receipt["bindings"][str(hnpz)])
            htarget, himages, hslots = cache.verified_arrays(hcp, histreg, dict(source_index=index, **cp["record"]), "FINAL_P2048_P3060", hnpz)
            require(np.array_equal(htarget, target), "P2048/M1 original float targets differ")
            for row in hcp["rows"]:
                meta = json.loads(row["history_metadata_json"])
                if meta.get("method_id") != "P2048" or int(row["noise_seed"]) != 2001:
                    continue
                require(meta["N"] == 2048 and meta["decoder"] == "Dc" and not meta["label_conditioned"], "Historical P method differs")
                snr = int(float(row["snr_db"]))
                for phy in PHYS:
                    rows[phy, snr, "P2048"] = (himages[hslots[row["history_row_id"]]], row)
            result[index] = dict(target=target, rows=rows, record=cp["record"])
        return result

    def fixed_figures(self):
        from PIL import Image, ImageDraw, ImageFont
        saved = self.load_fixed_arrays()
        font_path = next((p for p in (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("C:/Windows/Fonts/arial.ttf")) if p.exists()), None)
        font = ImageFont.truetype(str(font_path), 18) if font_path else ImageFont.load_default()
        small = ImageFont.truetype(str(font_path), 14) if font_path else ImageFont.load_default()
        panels = {"policies": ("original", "P2048", "whole_policy", "raster_policy", "random_policy", "entropy_policy", "oracle_policy"),
                  "sameK": ("original", "P2048", "entropy_policy", "raster_at_entropy", "random_at_entropy", "oracle_at_entropy")}
        for phy in PHYS:
            for snr in (1, 7, 13):
                for kind, methods in panels.items():
                    for page in range(4):
                        indices = FIXED[page*4:(page+1)*4]
                        canvas = Image.new("RGB", (40+len(methods)*276, 82+4*300), "white")
                        draw = ImageDraw.Draw(canvas)
                        draw.text((20, 12), "N2048 | %s | %d dB | seed 2001 | fixed sources | %s" % (phy, snr, kind), fill="black", font=font)
                        proof = []
                        for column, method in enumerate(methods):
                            draw.text((20+276*column, 48), LABELS[method], fill="black", font=small)
                        for rindex, index in enumerate(indices):
                            source = saved[index]
                            for column, method in enumerate(methods):
                                image = source["target"] if method == "original" else source["rows"][phy, snr, method][0]
                                rgb = np.rint(np.clip(image, 0, 1).transpose(1, 2, 0)*255).astype(np.uint8)
                                x, y = 20+276*column, 78+300*rindex
                                canvas.paste(Image.fromarray(rgb), (x, y))
                                if method == "original":
                                    caption = "source %03d" % index
                                elif method == "P2048":
                                    caption = "P2048; E = 4096"
                                else:
                                    action = selected_action(self.policies, phy, snr, method)
                                    caption = "m%d K%d%s" % (action["m"], action["q"], " | whole-scale" if action["q"] == 0 else "")
                                draw.text((x, y+261), caption, fill="#303030", font=small)
                                proof.append(dict(source_index=index, method=method, image_sha256=rgb_sha(image)))
                        filename = "fixed16_%s_snr%d_%s_page%d.png" % (phy, snr, kind, page+1)
                        path = self.result/"figures"/filename
                        canvas.save(path, optimize=True)
                        self.output(path)
                        self.figure_manifest.append(dict(path=str(path.relative_to(self.result)), N=2048, phy_family=phy,
                            snr_db=snr, noise_seed=2001, source_indices=list(indices), columns=list(methods), float_inputs=proof,
                            display_conversion="round(clamp(float32 RGB,0,1)*255); no image inference or retouching"))
                        canvas.close()
        write_json(self.result/"FIGURE_MANIFEST.json", self.figure_manifest)
        self.output(self.result/"FIGURE_MANIFEST.json")

    def report(self):
        paired = self.tables["m1_vs_P_paired.csv"]
        def matching(phy, snr, metric):
            return next((r for r in paired if int(r["N"]) == 2048 and r["phy_family"] == phy and int(r["snr_db"]) == snr
                         and r["method_A"] == "entropy_policy" and r["method_B"] == "P2048" and r["metric"] == metric), None)
        def interval(row):
            return "N/A" if row is None else "%s [%s, %s]" % tuple(pretty(row[k]) for k in ("mean", "ci_low", "ci_high"))
        comparison = []
        for phy in PHYS:
            for snr in SNRS:
                action = self.policies[phy, snr, "entropy"]["action"]
                comparison.append([phy, snr, "m%d/K%d" % (action["m"], action["q"]),
                    *[interval(matching(phy, snr, metric)) for metric in ("psnr_db", "lpips_alex", "dinov2_vitl14_cosine", "clip_image_cosine", "dists")]])
        order_rows = []
        allpaired = self.tables["metrics_paired_intervals.csv"]
        for phy in PHYS:
            for snr in SNRS:
                k = self.policies[phy, snr, "entropy"]["action"]["q"]
                for comparator in ("raster_at_entropy", "random_at_entropy", "oracle_at_entropy"):
                    selected = {r["metric"]: r for r in allpaired if r["phy_family"] == phy and int(r["snr_db"]) == snr
                                and r["method_A"] == "entropy_policy" and r["method_B"] == comparator}
                    order_rows.append([phy, snr, k, LABELS[comparator],
                                       "K0: no ordering contrast" if k == 0 else "same m/K; oracle pays its mask",
                                       *[interval(selected.get(metric)) for metric in ("psnr_db", "lpips_alex", "dinov2_vitl14_cosine")]])
        samepath = self.result/"SAME_K.md"
        samepath.write_text("# 同 K 排序对照\n\n差值为 entropy − comparator。PSNR、DINO 正值更好；LPIPS 负值更好。\n\n"+
            markdown_table(["PHY", "SNR", "K", "对照", "解释", "ΔPSNR [95%CI]", "ΔLPIPS [95%CI]", "ΔDINOv2-L [95%CI]"], order_rows)+"\n", encoding="utf-8")
        self.output(samepath)
        inactive = [(p, s) for p in PHYS for s in SNRS if self.policies[p, s, "entropy"]["action"]["q"] == 0]
        lines = ["# N2048 完整 M1：同预算质量取舍与尺度内排序", 
            "## 完成范围", 
            "本报告使用完成登记的真实结果：QPSK、16QAM 各 69 个动作，5 个 SNR，原 1000 张校准图 × 3 个噪声，共 2,070,000 次物理链路评测。以原选模规则分别选择 whole/raster/random/entropy/oracle；不读取开发集选模。开发集为原 100 张图 × 3 个噪声，含同 K 消融，共 24,000 条方法记录。本静态报告构建步骤不进行训练或重新推断。",
            "## 相对 P2048 的取舍", 
            *tradeoff_sentences(paired),
            "下表是 **M1 熵序策略 − 最终 P2048**，附源图配对 95% 区间。每张源图先平均 3 个噪声，再进行 10,000 次 bootstrap（seed 20261002）。PSNR、DINO 和 CLIP 正值更好；LPIPS、DISTS 负值更好。区间是复用开发集上的描述性区间，不估计训练 seed 不确定性。",
            markdown_table(["PHY", "SNR", "m/K", "ΔPSNR", "ΔLPIPS", "ΔDINOv2-L", "ΔCLIP", "ΔDISTS"], comparison),
            "QPSK 与 P 每帧均采用 E=2N=4096；16QAM 保持原固定星座，其实际能量单列，不能把它的差值称为严格同能量优势。历史 P 和数字链使用不同噪声命名空间：源图与名义种子配对，实际噪声波形不相同。各指标应分别解释，某一语义指标提升不等于像素或感知全面提升。",
            "完整主表 CSV 另列 F 恢复平方误差与有效 latent 比例。`F_sq_error_zero_erasure_proxy` 在头部擦除时采用零 latent 代理误差，未把灰色输出当作已恢复的 F；原始有效帧误差和擦除代理仍分别保留在逐帧表中。历史 P2048 未保存这项 F 指标，因此不补造该对照差值。",
            "## 部分尺度和排序是否有作用", 
            "熵序策略选为 K=0 的工作点："+("、".join("%s/%ddB" % x for x in inactive) if inactive else "无")+"。这些点实际退化为 whole-scale；raster/random/oracle@K 的相同输出不能作为排序收益证据。",
            "其余点在 [SAME_K.md](SAME_K.md) 中比较相同 m/K 的 raster、random 与 entropy。付费 oracle 按 token 与 VAR 贪心预测是否错配优先选择，并显式付费发送 bitmap；它不是最终图像误差最优选择器，也不是数学性能上界。独立校准的各排序策略见 [完整主表](MAIN_TABLE.md) 与 [策略表](POLICIES.csv)。",
            "## 质量—资源曲线", 
            "仅使用已完成、兼容的 N512、N1024、N2048 点；没有插补缺失结果。每个调制/SNR 单独绘图，QPSK 与 16QAM 不组成未登记的自适应方法。历史 P 的模型与训练 seed/预算保持原记录，曲线不表示只改变 N 的同一训练轨迹；N512 的 40k 预算截断说明继续有效。",
            "![QPSK 7dB](figures/quality_resource_QPSK_snr7.png)",
            "![16QAM 13dB](figures/quality_resource_16QAM_snr13.png)",
            "## 固定样例", 
            "固定 16 张源图与 seed 2001；主图列为原图、P、whole/raster/random/entropy 策略和付费 oracle，另提供同 K 图。所有画面直接取已保存 float32 重建；PNG 仅作显示转换。图像可用于检查内容错误，分类器不一致本身不等价于人工确认的语义幻觉。",
            "![固定样例 QPSK 7dB](figures/fixed16_QPSK_snr7_policies_page1.png)",
            "## 计算与接收代价", 
            "完整校准网格实测的每源平均处理时长为 %.2f 秒（2 源、4,140 次物理帧）。开发集每个完整 240 行方法网格平均 %.2f 秒，共 %d 个不同接收状态。它们是带缓存复用的离线 PHY、生成、指标与写盘总耗时，**不是单帧在线接收时延**；不能用标签行数相除后称为每帧延迟。在线逐帧时延本轮未单独测量，详细成本见 COSTS.json。" % (self.cost["benchmark_mean_complete_source_grid_seconds"], self.cost["development_saved_source_grid_seconds_mean"], self.cost["development_unique_receiver_states"]),
            "## 指标与可追溯性", 
            "保留原 DINOv2 ViT-S/14，并补 DINOv2 ViT-L/14。CLIP、DISTS、DreamSim、MS-SSIM 与独立 ResNet-50 的精确模型版本、权重和预处理见 MODEL_METADATA.json 与 EVALUATION_REGISTRATION.json。分类报告真实标签准确率及与原图分类预测的一致率；本轮 VAR 全部无类别条件。历史缺失指标保持 N/A。KID/FID 留待更大 holdout。",
            "[完整主表](MAIN_TABLE.csv) · [各指标配对区间](metrics_paired_intervals.csv) · [相对 P 的区间](m1_vs_P_paired.csv) · [全部测量资源点](resource_summary.csv) · [完整校准表](calibration_summary.csv)",
            "MANIFEST.json 记录本构建器、完成回执、模型元数据、来源缓存以及每个发布文件的 SHA256。原始权重、张量归档和数据集图片不复制进发布目录；不执行 Git 操作。"]
        path = self.result/"REPORT.md"
        path.write_text("\n\n".join(lines)+"\n", encoding="utf-8")
        self.output(path)

    def build(self):
        self.bind(__file__)
        self.load()
        self.write_tables()
        self.curves()
        self.fixed_figures()
        self.report()
        for path, digest in self.inputs.items():
            require(sha(path) == digest, "Input changed during static report construction")
        manifest = dict(status="STATIC_REPORT_COMPLETE", model_inference=False, training_updates=0,
            metric_recomputation=False, input_bindings=self.inputs, outputs=self.outputs,
            sources=100, method_rows=24000, fixed_source_indices=list(FIXED),
            figures=len(self.figure_manifest), maximum_file_bytes_exclusive=LIMIT,
            score_completion_sha256=sha(self.out/"score_completion.json"),
            policy_sha256=sha(self.out/"m1_policy.json"))
        write_json(self.result/"MANIFEST.json", manifest)
        require((self.result/"MANIFEST.json").stat().st_size < LIMIT, "Manifest exceeds file size limit")
        return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    Builder(args.root, args.out, args.result).build()


if __name__ == "__main__":
    main()
