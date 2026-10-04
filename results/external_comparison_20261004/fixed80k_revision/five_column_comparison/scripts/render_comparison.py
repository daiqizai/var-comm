"""CPU-only export of existing, explicitly identified reconstructions and metrics.

No decoder, learned evaluator, calibration, or channel simulation is imported.
The input manifest is the immutable extraction receipt, not a request to infer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import textwrap
import zipfile

import numpy as np
from PIL import Image

METHODS = ("original", "swin", "hifi", "P", "M1")
LABELS = {"original": "Original", "swin": "SwinJSCC", "hifi": "HiFi-DiffCom", "P": "P", "M1": "M1: entropy order"}
METRICS = ("psnr_db", "lpips_alex", "dino_cosine", "dinov2_vitl14_cosine", "clip_image_cosine", "dists", "dreamsim",
           "ms_ssim", "dino_specificity", "resnet50_top1_label", "resnet50_top1_source_prediction", "semantic_error", "confidently_wrong")
METRIC_LABELS = {"psnr_db": "PSNR ↑", "lpips_alex": "LPIPS ↓", "dino_cosine": "DINOv2-S ↑", "dinov2_vitl14_cosine": "DINOv2-L ↑",
                 "clip_image_cosine": "CLIP ↑", "dists": "DISTS ↓", "dreamsim": "DreamSim ↓", "ms_ssim": "MS-SSIM ↑",
                 "dino_specificity": "DINO specificity ↑", "resnet50_top1_label": "R50 / true label ↑",
                 "resnet50_top1_source_prediction": "R50 / original prediction ↑", "semantic_error": "R50 mismatch ↓",
                 "confidently_wrong": "Confident mismatch ↓"}
FIXED = (0, 25, 50, 75, 4, 21, 24, 29, 33, 41, 52, 60, 64, 87, 92, 95)
BOOTSTRAP_SEED = 20261002


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def asset_key(row):
    return int(row["N"]), int(row["snr_db"]), int(row["source_index"]), row["method"]


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def valid_value(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_bundle(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    base = manifest_path.parent
    manifest = read_json(manifest_path)
    require(manifest["schema"] == "existing_reconstruction_comparison_v1", "Unknown extraction manifest")
    for flag in ("new_inference", "new_reconstruction_export"):
        require(type(manifest.get(flag, False)) is bool, "Export flags must be explicit JSON booleans")
    require(manifest.get("new_metric_evaluation", False) is False and manifest.get("new_training", False) is False
            and manifest.get("new_calibration", False) is False, "This export may not introduce training, calibration, or metric evaluation")
    require(manifest["source_indices"] == list(FIXED), "The original fixed16 display order must be preserved")
    require(int(manifest["noise_seed"]) == 2001, "Only the registered first-tier noise seed is allowed")
    conditions = [(int(row["N"]), int(row["snr_db"])) for row in manifest["conditions"]]
    require(len(conditions) == len(set(conditions)) and conditions, "Duplicate or empty conditions")
    assets = {}
    for row in manifest["assets"]:
        item = asset_key(row)
        require(item not in assets and item[:2] in conditions and item[2] in FIXED and item[3] in METHODS, "Unexpected or duplicate image identity")
        require(row["status"] in ("AVAILABLE", "NOT_RUN", "UNAVAILABLE_CACHE"), "Explicit asset availability required")
        if row["status"] == "AVAILABLE":
            path = resolve(base, row["path"])
            require(path.is_file() and sha(path) == row["sha256"], "Extracted PNG differs from its receipt: " + str(item))
            with Image.open(path) as image:
                require(image.format == "PNG" and image.mode == "RGB" and image.size == (256, 256), "Only original-size RGB PNG assets are admitted")
            row = dict(row, resolved_path=str(path))
        else:
            require(row.get("reason"), "A missing image must have an explicit reason")
        assets[item] = row
    expected = {(n, snr, index, method) for n, snr in conditions for index in FIXED for method in METHODS}
    require(set(assets) == expected, "Every figure cell needs an image or an explicit missing-data record")
    for n, snr in conditions:
        for index in FIXED:
            require(assets[n, snr, index, "original"]["status"] == "AVAILABLE", "Source image is missing")
    for index in FIXED:
        require(len({assets[n, snr, index, "original"]["sha256"] for n, snr in conditions}) == 1, "The source image changes across conditions")
    csv_receipt = manifest["metrics"]
    metrics_path = resolve(base, csv_receipt["path"])
    require(sha(metrics_path) == csv_receipt["sha256"], "Extracted metric CSV differs")
    rows = read_csv(metrics_path)
    indexed = {}
    for row in rows:
        item = asset_key(row)
        require(item in expected and item[3] != "original" and item not in indexed, "Unexpected or duplicate metric row")
        require(int(row["noise_seed"]) == 2001, "Metric noise seed differs from the figure")
        require(assets[item]["status"] != "NOT_RUN", "An unmeasured method cannot have metric rows")
        if row.get("asset_sha256"):
            require(assets[item]["status"] == "AVAILABLE" and row["asset_sha256"] == assets[item]["sha256"], "Metric row points to different PNG bytes")
        for metric in METRICS:
            if row.get(metric, "") not in ("", "NA", "N/A"):
                require(valid_value(row[metric]), "Nonfinite metric value")
        indexed[item] = row
    expected_rows = {item for item, asset in assets.items() if item[3] != "original" and asset["status"] != "NOT_RUN"}
    require(set(indexed) == expected_rows, "Every measured reconstruction must have its existing metric row, even if its image cache is unavailable")
    return manifest, conditions, assets, indexed


def bootstrap_tables(conditions, indexed):
    """Identical source resamples for each complete 16-image group and contrast."""
    order = tuple(sorted(FIXED))
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(order), (10000, len(order)))
    summaries, pairs = [], []

    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        means = values[draws].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return dict(mean=float(values.mean()), ci_low=float(lo), ci_high=float(hi), n_sources=16, n_frames=16,
                    bootstrap_replicates=10000, bootstrap_seed=BOOTSTRAP_SEED, exploratory_fixed_subset=True)

    for n, snr in conditions:
        values = {}
        for method in METHODS[1:]:
            present = [indexed.get((n, snr, index, method)) for index in order]
            for metric in METRICS:
                group = [row[metric] for row in present if row and valid_value(row.get(metric))]
                if len(group) == 16:
                    values[method, metric] = np.asarray(group, dtype=np.float64)
                    summaries.append(dict(N=n, snr_db=snr, method=method, metric=metric, **stats(group)))
        for method_a, method_b in itertools.combinations(METHODS[1:], 2):
            for metric in METRICS:
                if (method_a, metric) not in values or (method_b, metric) not in values:
                    continue
                # Report later column minus earlier column; no common-waveform claim across chains.
                pairs.append(dict(N=n, snr_db=snr, method_A=method_b, method_B=method_a, metric=metric,
                                  comparison_scope="same_fixed_source_N_SNR_and_registered_seed_not_identical_waveform_across_chains",
                                  **stats(values[method_b, metric] - values[method_a, metric])))
    return summaries, pairs


def metric_caption(row):
    if row is None:
        return ""
    value = lambda metric, digits: f"{float(row[metric]):.{digits}f}" if valid_value(row.get(metric)) else "N/A"
    return f"PSNR {value('psnr_db', 2)} | LPIPS {value('lpips_alex', 3)}\nDINOv2-L {value('dinov2_vitl14_cosine', 3)}"


def class_caption(row, source=False):
    if not row:
        return ""
    field = "true_class_name" if source else "resnet50_prediction_name"
    if row.get(field):
        return ("Label: " if source else "R50: ") + row[field]
    return ""


def render_figures(output, manifest, conditions, assets, indexed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    labels = dict(LABELS, **manifest.get("method_labels", {}))
    source_meta = {int(row["source_index"]): row for row in manifest.get("sources", [])}
    images = {}
    for item, asset in assets.items():
        if asset["status"] == "AVAILABLE":
            with Image.open(asset["resolved_path"]) as image:
                images[item] = np.asarray(image).copy()

    def page(n, snr, indices, overview):
        count = len(indices)
        height = 2.82 * count + 2.10
        fig, axes = plt.subplots(count, 5, figsize=(15.6, height), squeeze=False)
        fig.subplots_adjust(left=.026, right=.994, top=1 - .94 / height, bottom=1.11 / height, hspace=.39, wspace=.06)
        for col, method in enumerate(METHODS):
            axes[0, col].set_title(labels[method], fontsize=13, weight="bold", pad=11)
        for position, index in enumerate(indices):
            for col, method in enumerate(METHODS):
                ax = axes[position, col]
                item = n, snr, index, method
                asset = assets[item]
                if asset["status"] == "AVAILABLE":
                    ax.imshow(images[item], interpolation="nearest")
                else:
                    ax.set_facecolor("#f1f3f5")
                    ax.text(.5, .55, "NOT MEASURED" if asset["status"] == "NOT_RUN" else "CACHED IMAGE\nUNAVAILABLE", transform=ax.transAxes,
                            ha="center", va="center", fontsize=13, color="#666666", weight="bold")
                    ax.text(.5, .40, textwrap.fill(asset["reason"], 30), transform=ax.transAxes,
                            ha="center", va="center", fontsize=9, color="#666666")
                    ax.set_xlim(0, 256); ax.set_ylim(256, 0); ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if method == "original":
                    row = source_meta.get(index) or indexed.get((n, snr, index, "swin"))
                    caption = f"Source {index:02d}\n" + class_caption(row, source=True)
                elif asset["status"] == "AVAILABLE":
                    row = indexed[item]
                    caption = metric_caption(row)
                    if not overview and class_caption(row):
                        caption += "\n" + class_caption(row)
                elif asset["status"] == "UNAVAILABLE_CACHE":
                    caption = metric_caption(indexed.get(item))
                else:
                    caption = "No image or metric substituted"
                ax.set_xlabel("\n".join(textwrap.fill(line, 40) for line in caption.splitlines()), fontsize=9, labelpad=6)
        fig.suptitle(f"N = {n} | SNR = {snr} dB | " + ("all 16 fixed examples" if overview else f"fixed examples {FIXED.index(indices[0]) + 1}-{FIXED.index(indices[-1]) + 1}/16"),
                     fontsize=17, weight="bold", y=1 - .17 / height)
        export_label = "Registered models / seeds; cache + authorized re-export" if manifest.get("new_inference", False) else "Existing reconstructions only"
        fig.text(.5, 1 - .46 / height, export_label + " | noise seed 2001 | no new training or calibration", ha="center", fontsize=10)
        fig.text(.5, .16 / height, "Selected illustrative images, not a representative population. Swin: exact 80k, budget-truncated. DINOv2-L: ViT-L/14.\n"
                 "Source identity and resource conditions are matched; distinct communication chains do not share one received waveform.", ha="center", fontsize=9)
        return fig

    for n, snr in conditions:
        stem = f"comparison_N{n}_SNR{snr}"
        overview_png, overview_pdf = output / f"{stem}_all16.png", output / f"{stem}_all16.pdf"
        fig = page(n, snr, FIXED, True)
        fig.savefig(overview_png, dpi=150, facecolor="white")
        fig.savefig(overview_pdf, facecolor="white", metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
        closeups, pdf_path = [], output / f"{stem}_closeups.pdf"
        with PdfPages(pdf_path, metadata={"Title": f"Fixed16 five-column comparison N{n} SNR{snr}", "CreationDate": None, "ModDate": None}) as pdf:
            for part in range(4):
                indices = FIXED[part * 4:part * 4 + 4]
                fig = page(n, snr, indices, False)
                path = output / f"{stem}_page{part + 1}.png"
                fig.savefig(path, dpi=200, facecolor="white"); pdf.savefig(fig, facecolor="white"); plt.close(fig)
                closeups.append(dict(path=str(path), sha256=sha(path), source_indices=list(indices)))
        figures.append(dict(N=n, snr_db=snr, overview_png=str(overview_png), overview_pdf=str(overview_pdf),
                            closeups_pdf=str(pdf_path), png_pages=closeups))
        print(json.dumps(dict(stage="figures", N=n, snr_db=snr, complete=True)), flush=True)
    return figures


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def main_table(output, manifest, conditions, assets, summaries):
    selected = ("psnr_db", "lpips_alex", "dinov2_vitl14_cosine", "clip_image_cosine", "dists")
    summary = {(row["N"], row["snr_db"], row["method"], row["metric"]): row for row in summaries}
    labels = dict(LABELS, **manifest.get("method_labels", {}))
    rows, display = [], []
    for n, snr in conditions:
        for method in METHODS[1:]:
            measured = any(assets[n, snr, index, method]["status"] != "NOT_RUN" for index in FIXED)
            row = dict(N=n, snr_db=snr, method=method, n_sources=16 if measured else 0,
                       status="MEASURED" if measured else "NOT_MEASURED",
                       cached_images=sum(assets[n, snr, index, method]["status"] == "AVAILABLE" for index in FIXED))
            row.update({metric: summary.get((n, snr, method, metric), {}).get("mean", "") for metric in selected})
            rows.append(row)
            values = [f"{row[metric]:.2f}" if metric == "psnr_db" and row[metric] != "" else
                      f"{row[metric]:.3f}" if row[metric] != "" else "未测" if not measured else "N/A" for metric in selected]
            display.append([n, snr, labels[method]] + values)
    write_csv(output / "main_comparison.csv", rows)
    lines = ["# 四种传输方法：固定16图对比", "",
             "各已测行均为相同16张固定样例、名义噪声种子2001的均值。Original仅作图中视觉参照，不列入传输方法排名。该小样本用于展示，不代表完整开发集。", ""]
    lines += table(["N", "SNR/dB", "方法"] + [METRIC_LABELS[metric] for metric in selected], display)
    lines += ["", "↑越大越好，↓越小越好。N2048的M1未正式运行，不能据空缺推断优劣。各链噪声命名空间不同，并非逐元素相同的信道噪声。"
              "Swin使用指定80k模型，训练预算截断，未证明收敛。", "",
              "[完整13指标、95%区间及配对差值](COMPARISON_REPORT.md) · [本表CSV](main_comparison.csv) · [逐帧指标](metrics_per_frame.csv)", ""]
    for n, snr in conditions:
        lines += [f"- N{n} / {snr} dB：[16图总览](figures/comparison_N{n}_SNR{snr}_all16.png) · [总览PDF](figures/comparison_N{n}_SNR{snr}_all16.pdf) · [近景PDF](figures/comparison_N{n}_SNR{snr}_closeups.pdf)"]
    (output / "MAIN_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_text(manifest, conditions, assets, summaries, pairs, figures):
    summary = {(row["N"], row["snr_db"], row["method"], row["metric"]): row for row in summaries}
    paired = {(row["N"], row["snr_db"], row["method_A"], row["method_B"], row["metric"]): row for row in pairs}
    labels = dict(LABELS, **manifest.get("method_labels", {}))
    interval = lambda row: f"{row['mean']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]" if row else "N/A"
    extraction_note = ("本报告整理已完成实验的重建图和已有逐帧指标，并按授权使用原模型、原条件与原种子补导出缺失重建图；没有重新训练、校准或运行指标评测模型。"
                       if manifest.get("new_inference", False) else
                       "本报告仅整理已完成实验的重建图和已有逐帧指标，没有重新训练、推断、校准或运行评测模型。")
    lines = ["# 固定16图：Original / Swin / HiFi / P / M1", "", "[先看一张合并对比表](MAIN_TABLE.md)", "",
             extraction_note + "原图单独作为视觉参照，不把恒等重建的虚构分数放入传输方法排名。", "",
             "范围：原登记的16张固定样例、噪声种子2001。该集合用于展示，不能代表完整100张开发图像或ImageNet总体。Swin使用指定80,000步模型；训练实际暂停于81,551步，保留预算截断与未证明收敛的说明。", "",
             "均值及95%区间以16张源图为单位bootstrap 10,000次，种子20261002。差值使用相同源图的配对重采样；名义噪声种子均为2001，但各链的噪声命名空间不同，信道噪声并非逐元素相同，跨链实际接收波形也不同。", "",
             "DINOv2-L为ViT-L/14；保留DINOv2-S、CLIP ViT-L/14及其余已登记指标。分类器错配只是自动诊断，不等于人工语义错误判决。图像从原缓存导出为256×256 RGB PNG；表中指标沿用原始评测缓存，未在量化PNG上重算。", "",
             "已有指标但图像缓存缺失的项保留指标，并在图中明确标记；未执行的方法不填写指标。只对完整16图且该指标均可用的组计算区间，缺失值不置零、不改用较少样本的均值。P/M1若未保存置信度诊断则标N/A；分类器错配率可由1−原图预测一致率直接得到。", ""]
    if any(n == 2048 for n, _ in conditions):
        lines += ["**N2048 的 M1 未正式运行，不能据此与其他方法比较。** 该列明确保留“NOT MEASURED”，不使用其他带宽或相邻SNR结果替代。", ""]
    lines += manifest.get("report_notes", []) + [""]
    for n, snr in conditions:
        lines += [f"## N{n}，{snr} dB", ""]
        available = []
        for method in METHODS[1:]:
            count = sum(assets[n, snr, index, method]["status"] == "AVAILABLE" for index in FIXED)
            available.append((method, count))
            if count != 16:
                reasons = sorted({assets[n, snr, index, method].get("reason", "") for index in FIXED if assets[n, snr, index, method]["status"] != "AVAILABLE"})
                lines += [f"**{labels[method]}：{count}/16张可用。** " + "; ".join(reasons), ""]
        for selected in (("psnr_db", "lpips_alex", "dinov2_vitl14_cosine", "clip_image_cosine"),
                         ("dists", "dreamsim", "ms_ssim", "dino_cosine", "dino_specificity"),
                         ("resnet50_top1_label", "resnet50_top1_source_prediction", "semantic_error", "confidently_wrong")):
            lines += table(["方法", "已存图像"] + [METRIC_LABELS[metric] for metric in selected],
                           [[labels[method], f"{count}/16"] + [interval(summary.get((n, snr, method, metric))) for metric in selected] for method, count in available]) + [""]
        contrasts = [(b, a) for a, b in itertools.combinations(METHODS[1:], 2)
                     if any((n, snr, b, a, metric) in paired for metric in METRICS)]
        selected = ("psnr_db", "lpips_alex", "dinov2_vitl14_cosine")
        lines += ["同源图配对差值，方向为表中前者减后者。", ""]
        lines += table(["方法差值"] + [METRIC_LABELS[metric] for metric in selected],
                       [[labels[a] + " − " + labels[b]] + [interval(paired.get((n, snr, a, b, metric))) for metric in selected]
                        for a, b in contrasts]) + [""]
        lines += [f"![16图总览](figures/comparison_N{n}_SNR{snr}_all16.png)", "",
                  f"[总览PDF](figures/comparison_N{n}_SNR{snr}_all16.pdf) · [四页近景PDF](figures/comparison_N{n}_SNR{snr}_closeups.pdf)", ""]
        for page in range(1, 5):
            lines += [f"[近景第{page}页](figures/comparison_N{n}_SNR{snr}_page{page}.png)"]
        lines += [""]
    lines += ["## 下载与审计", "", "- [全部逐帧指标](metrics_per_frame.csv)", "- [全部均值与区间](metrics_summary.csv)",
              "- [全部可比方法对的差值与配对区间](metrics_paired_intervals.csv)", "- [图像清单与哈希](asset_inventory.csv)", "",
              "原尺寸PNG与原图独立包作为本地附件交付，不提交Git：`original_size_pngs.zip`含可用重建图和原图；`original_inputs_16.zip`只含16张256×256评测输入图及说明，并非ImageNet原始JPEG。", "",
              "缺测项不补值、不插值，不使用相邻SNR或其他带宽的图像顶替。完整HiFi队列继续保持暂停。", ""]
    return "\n".join(lines)


def export_assets(output, assets):
    directory = output / "images"
    directory.mkdir(parents=True, exist_ok=True)
    inventory, copied = [], {}
    for item, asset in assets.items():
        n, snr, index, method = item
        row = {key: value for key, value in asset.items() if key != "resolved_path"}
        if asset["status"] == "AVAILABLE":
            relative = Path("images") / (f"original/source_{index:02d}.png" if method == "original" else f"N{n}_SNR{snr}/{method}/source_{index:02d}.png")
            destination = output / relative
            if destination in copied:
                require(copied[destination] == asset["sha256"], "Conflicting reused original image")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(asset["resolved_path"], destination)
                copied[destination] = asset["sha256"]
            row["published_path"] = relative.as_posix()
        inventory.append(row)
    write_csv(output / "asset_inventory.csv", inventory)
    with zipfile.ZipFile(output / "original_size_pngs.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(copied):
            archive.write(path, path.relative_to(output).as_posix())
        archive.write(output / "asset_inventory.csv", "asset_inventory.csv")
    with zipfile.ZipFile(output / "original_inputs_16.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        originals = sorted((directory / "original").glob("source_*.png"))
        require(len(originals) == 16, "The source-only archive must contain exactly 16 images")
        for path in originals:
            archive.write(path, path.name)
        archive.writestr("README.txt", "These are the 16 registered evaluation inputs exported as 256 x 256 RGB PNG.\n"
                         "They are the processed model inputs, not the original full-resolution ImageNet JPEG files.\n"
                         "No enhancement or upscaling was applied to these PNG assets.\n"
                         "Source indices retain the original 100-image evaluation identities.\n")
    return len(copied)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tables-only", action="store_true", help="Validate and export tables without figures; never emits a complete release receipt")
    args = parser.parse_args()
    manifest, conditions, assets, indexed = load_bundle(args.manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries, pairs = bootstrap_tables(conditions, indexed)
    main_table(output, manifest, conditions, assets, summaries)
    write_csv(output / "metrics_per_frame.csv", indexed.values())
    write_csv(output / "metrics_summary.csv", summaries)
    write_csv(output / "metrics_paired_intervals.csv", pairs)
    count = export_assets(output, assets)
    figures = [] if args.tables_only else render_figures(output / "figures", manifest, conditions, assets, indexed)
    (output / "COMPARISON_REPORT.md").write_text(report_text(manifest, conditions, assets, summaries, pairs, figures), encoding="utf-8")
    shutil.copyfile(args.manifest, output / "extraction_manifest.json")
    outputs = {str(path.relative_to(output)).replace("\\", "/"): sha(path) for path in output.rglob("*") if path.is_file() and path.name != "export_completion.json"}
    write_json(output / "export_completion.json", dict(status="TABLES_ONLY" if args.tables_only else "EXISTING_RECONSTRUCTION_COMPARISON_COMPLETE",
               source_indices=list(FIXED), noise_seed=2001, conditions=manifest["conditions"], rows=len(indexed), source_pngs=count,
               summaries=len(summaries), paired_intervals=len(pairs), figures=figures, new_inference=manifest.get("new_inference", False),
               new_reconstruction_export=manifest.get("new_reconstruction_export", False), new_metric_evaluation=False,
               exploratory_fixed_subset=True, full_evaluation_resumed=False, manifest_sha256=sha(args.manifest), script_sha256=sha(__file__), outputs=outputs))
    print(json.dumps(dict(status="TABLES_ONLY" if args.tables_only else "COMPLETE", output=str(output), metric_rows=len(indexed))), flush=True)


if __name__ == "__main__":
    main()
