"""Assemble fixed scientific comparisons from verified completed-run PNG exports.

This script performs no model inference, image enhancement, policy selection or
metric recomputation. All displayed scores come from the original float32 run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import textwrap


SOURCES = (0, 25)
BUDGETS = (512, 1024)
SNRS = (1, 7, 19)
METRIC_KEYS = {
    "psnr": ("psnr_db",),
    "lpips": ("lpips_alex",),
    # Never substitute retained dino_cosine, which is the ViT-S/14 metric.
    "dino_l14": ("dinov2_vitl14_cosine", "dinov2_vit_l14_cosine"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value):
    result = float(value)
    if not math.isfinite(result) or result != int(result):
        raise ValueError(f"Expected an integer, got {value!r}")
    return int(result)


def flag(value):
    if value in (True, "True", "true", 1, "1"):
        return True
    if value in (False, "False", "false", 0, "0"):
        return False
    raise ValueError(f"Expected an explicit boolean, got {value!r}")


def metric(row, kind):
    found = [(key, float(row[key])) for key in METRIC_KEYS[kind]
             if key in row and row[key] not in (None, "")]
    if not found:
        raise ValueError(f"Missing {kind}; expected one of {METRIC_KEYS[kind]}")
    if any(not math.isfinite(value) for _, value in found):
        raise ValueError(f"Nonfinite {kind}")
    if any(value != found[0][1] for _, value in found):
        raise ValueError(f"Conflicting explicit {kind} aliases")
    return found[0]


def safe_input(folder, name):
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-folder input: {name}")
    return path


def method_of(frame):
    row = frame["original_row"]
    method = row.get("method") or row.get("control")
    if frame["study"] == "M2_ACTUAL":
        if row.get("control") != "VAR_GUIDED":
            raise ValueError("M2 comparison must use the registered VAR_GUIDED control")
        method = "VAR_GUIDED"
    return method


def validate(manifest, folder):
    if manifest.get("status") != "EXACT_COMPLETED_RGB_EXPORT_PASS":
        raise ValueError("Completed-run RGB identity qualification is required")
    for field in ("scientific_selection_changed", "oracle", "paid_class"):
        if manifest.get(field) is not False:
            raise ValueError(f"Unexpected comparison scope: {field}")
    if (manifest.get("decoder") != "Dc" or manifest.get("noise_seed") != 2001
            or manifest.get("sources_selected_before_rendering") != list(SOURCES)
            or manifest.get("budgets") != list(BUDGETS)
            or manifest.get("snrs_db") != list(SNRS)):
        raise ValueError("The registered fixed comparison grid changed")
    sources = {integer(item["source_index"]): item for item in manifest["sources"]}
    if set(sources) != set(SOURCES) or len(sources) != len(manifest["sources"]):
        raise ValueError("Source population changed or contains duplicates")
    frames = {}
    for frame in manifest["frames"]:
        row, scores = frame["original_row"], frame["completed_metrics"]
        source, budget, snr = (integer(frame["source_index"]), integer(row["N"]),
                               integer(row["snr_db"]))
        method = method_of(frame)
        key = source, budget, snr, method
        if key in frames or frame.get("exact_completed_rgb_match") is not True:
            raise ValueError("Duplicate or unqualified reconstructed image")
        if (scores.get("replay_row_id") != frame["replay_row_id"]
                or scores.get("image_sha256") != frame["float_rgb_sha256"]):
            raise ValueError("Image and measured-row identity differ")
        if scores.get("reference_sha256") != sources[source]["reference_sha256"]:
            raise ValueError("Image and original source identity differ")
        if integer(row["noise_seed"]) != 2001:
            raise ValueError("Nominal noise seed changed")
        if (scores.get("replay_decoder_id") != "Dc"
                or row.get("decoder_id") not in (None, "", "Dc")):
            raise ValueError("Unexpected decoder reference")
        if scores.get("label_conditioned") is not False:
            raise ValueError("Class-conditioned rows cannot enter this comparison")
        if method != f"P{budget}" and row.get("phy_family") != "QPSK":
            raise ValueError("Digital branch must use the QPSK policy")
        if method == "D_U_QPSK" and row.get("class_condition") != "U":
            raise ValueError("The class-unconditional digital branch is required")
        if method == "VAR_GUIDED" and row.get("stage") != "actual_link":
            raise ValueError("An oracle output cannot replace the actual link")
        # P's normalized continuous waveform was produced in float32. Permit
        # its original rounding, while rejecting a different energy budget.
        if row.get("E") not in (None, "") and not math.isclose(
                float(row["E"]), 2 * budget, rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError("The selected frame does not have the stated energy")
        path = safe_input(folder, frame["file"])
        if sha(path) != frame["png_sha256"]:
            raise ValueError("The exported display pixels changed")
        for kind in METRIC_KEYS:
            metric(scores, kind)
        frames[key] = frame
    expected = {(source, budget, snr, method)
                for source in SOURCES for budget in BUDGETS for snr in SNRS
                for method in (f"P{budget}", "D_U_QPSK", "entropy_policy", "VAR_GUIDED")}
    if set(frames) != expected:
        raise ValueError("The fixed 48-frame comparison grid is incomplete")
    for item in sources.values():
        safe_input(folder, item["file"])
    return sources, frames


def detail(frame):
    row = frame["original_row"]
    method = method_of(frame)
    if method.startswith("P"):
        return "连续信道传输", "#4b5563"
    if method == "D_U_QPSK":
        return f"整尺度传输：m={integer(row['action_m'])}", "#4b5563"
    if method == "entropy_policy":
        m, count = integer(row["m"]), integer(row["q"])
        text = f"m={m}，K={count}" + ("（仅整尺度）" if count == 0 else "（熵序）")
        if count and "partial_used" in row and not flag(row["partial_used"]):
            text += "；部分尺度未启用"
            return text, "#a04416"
        return text, "#4b5563"
    projection = str(row["projection"]).replace("g", "", 1).split("_c")
    label = f"{projection[0]}×{projection[0]} / PCA {projection[1]}维"
    if not flag(row["analog_used"]):
        crc_failed = any(not flag(row[field]) for field in
                         ("header_ok", "prefix_crc_ok", "gain_crc_ok") if field in row)
        label += "；CRC失败，未引导" if crc_failed else "；残差未启用"
        return label, "#a04416"
    return label + "；残差引导启用", "#4b5563"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("/home/liulu/projects/VAR_COMM/outputs/TRANSMISSION-COMPARISONS-20261003/comparisons_manifest.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--font", type=Path,
                        help="Optional installed CJK font; defaults to Noto Sans CJK")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    folder = manifest_path.parent
    output = (args.output_dir or folder).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources, frames = validate(manifest, folder)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    from PIL import Image

    font_candidates = ([args.font] if args.font else []) + [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    ]
    font_path = next((path for path in font_candidates if path.is_file()), None)
    if font_path is None:
        raise RuntimeError("A real CJK font is required; provide it with --font")
    font_manager.fontManager.addfont(str(font_path))
    font_property = font_manager.FontProperties(fname=str(font_path))
    font_family = font_property.get_name()
    # Register a second weight if installed, without depending on the host's
    # previously built matplotlib font cache.
    medium = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    if medium.is_file() and medium != font_path:
        font_manager.fontManager.addfont(str(medium))
    plt.rcParams.update({"font.family": font_family, "font.size": 11,
                         "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    output.mkdir(parents=True, exist_ok=True)
    images = {}
    def image(name):
        if name not in images:
            with Image.open(safe_input(folder, name)) as loaded:
                if loaded.mode != "RGB" or loaded.size != (256, 256):
                    raise ValueError("Only the exported 256x256 RGB tiles are accepted")
                images[name] = np.array(loaded)
        return images[name]

    products = []
    for source in SOURCES:
        for budget in BUDGETS:
            fig = plt.figure(figsize=(16, 12.6), dpi=100, facecolor="white")
            fig.text(.5, .972, f"不同信噪比下的传输重建对比  |  N = {budget}",
                     ha="center", va="top", fontsize=23, fontweight="bold", color="#152638")
            label = f"预先固定的样例 {source:03d}  |  {sources[source]['source_id']}"
            fig.text(.5, .935, "\n".join(textwrap.wrap(label, width=130)),
                     ha="center", va="top", fontsize=11, color="#4b5563")
            headers = ("原图", f"P{budget}\n连续传输", "数字 VAR\n无类别条件",
                       "方法一\n部分尺度 + 熵序", "方法二\n模拟残差引导 VAR")
            methods = (None, f"P{budget}", "D_U_QPSK", "entropy_policy", "VAR_GUIDED")
            left, gap, width = .102, .014, .1644
            height = width * 16 / 12.6
            for col, title in enumerate(headers):
                fig.text(left + col * (width + gap) + width / 2, .885, title,
                         ha="center", va="bottom", fontsize=13, fontweight="bold", linespacing=1.4)
            for row_index, snr in enumerate(SNRS):
                top = .860 - row_index * .265
                bottom = top - height
                fig.text(.021, bottom + height / 2, f"{snr} dB", ha="left", va="center",
                         fontsize=21, fontweight="bold", color="#19384d")
                for col, method in enumerate(methods):
                    x = left + col * (width + gap)
                    frame = frames[(source, budget, snr, method)] if method else None
                    name = frame["file"] if frame else sources[source]["file"]
                    ax = fig.add_axes((x, bottom, width, height))
                    ax.imshow(image(name), interpolation="nearest")
                    ax.set_axis_off()
                    if frame:
                        scores = frame["completed_metrics"]
                        _, psnr = metric(scores, "psnr")
                        _, lpips = metric(scores, "lpips")
                        _, dino = metric(scores, "dino_l14")
                        fig.text(x + width / 2, bottom - .009,
                                 f"PSNR {psnr:.2f} dB   |   LPIPS {lpips:.3f}",
                                 ha="center", va="top", fontsize=10.5, color="#18212b")
                        fig.text(x + width / 2, bottom - .027,
                                 f"DINOv2 ViT-L/14  {dino:.3f}",
                                 ha="center", va="top", fontsize=10.5, color="#18212b")
                        text, color = detail(frame)
                        fig.text(x + width / 2, bottom - .045, text,
                                 ha="center", va="top", fontsize=8.8, color=color)
                    else:
                        fig.text(x + width / 2, bottom - .018, "未经过信道的参考图",
                                 ha="center", va="top", fontsize=10.5, color="#69727e")
            caption = (
                f"同一源图、N={budget}、每帧总能量 E={2 * budget}；数字支路使用 QPSK。所有重建均用 Dc 解码，生成时不使用类别标签。\n"
                "名义噪声种子为 2001；各方法的物理噪声序列不同。样例 0 和 25 在制图前已固定，没有按结果好坏挑图。\n"
                "PSNR 越高越好，LPIPS 越低越好，DINOv2-L14 越高越好。数值来自已完成的 float32 评测；PNG 量化仅用于展示。"
            )
            fig.text(.102, .050, caption, ha="left", va="top", fontsize=8.8,
                     color="#53616e", linespacing=1.6)
            stem = f"comparison_source{source:03d}_N{budget}"
            files = []
            for extension in ("png", "pdf"):
                path = output / f"{stem}.{extension}"
                fig.savefig(path, dpi=100 if extension == "png" else 200,
                            metadata={"Title": f"Fixed transmission comparison N{budget}, source {source}"}
                            if extension == "pdf" else None)
                files.append({"file": path.name, "sha256": sha(path)})
            plt.close(fig)
            products.append({"source_index": source, "N": budget, "files": files})
    receipt = {"status": "FIGURES_RENDERED", "manifest_sha256": sha(manifest_path),
               "renderer_sha256": sha(__file__), "products": products,
               "image_enhancement": False, "selection_changed": False,
               "metrics_recomputed": False, "font": font_family,
               "font_path": str(font_path), "font_sha256": sha(font_path),
               "language": "zh-CN",
               "metric_keys": sorted({metric(f["completed_metrics"], k)[0]
                                      for f in frames.values() for k in METRIC_KEYS})}
    (output / "figure_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "figures": len(products), "output": str(output)}))


if __name__ == "__main__":
    main()
