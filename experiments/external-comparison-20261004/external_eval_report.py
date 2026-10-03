"""CPU report and preregistered 16-source figures for the two external methods.

This stage cannot declare the broader external-comparison plan complete. Its
matched P / unconditional digital / M1 comparison remains an explicit gap.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import os
from pathlib import Path
import signal
import textwrap
import time
import uuid
import numpy as np
from external_eval_common import (BUDGETS, METHODS, PHYSICAL_FRAMES, ROWS, SEEDS,
    SNRS, SOURCES, PauseRequested, identity, pixels, read, rgb_sha, seal, sha,
    validate_source, verify, write, write_csv)

FIXED = (0, 25, 50, 75, 4, 21, 24, 29, 33, 41, 52, 60, 64, 87, 92, 95)
LABELS = {METHODS[0]: "SwinJSCC (new shared model)",
          METHODS[1]: "HiFi-DiffCom + Swin (registered adaptation)"}
SHORT = {METHODS[0]: "SwinJSCC", METHODS[1]: "HiFi + Swin"}
METRIC_LABELS = {"psnr_db": "PSNR ↑", "lpips_alex": "LPIPS-Alex ↓",
    "dino_cosine": "DINOv2-S/14 ↑", "dinov2_vitl14_cosine": "DINOv2-L/14 ↑",
    "clip_image_cosine": "CLIP-L/14 ↑", "dists": "DISTS ↓", "dreamsim": "DreamSim ↓",
    "ms_ssim": "MS-SSIM ↑", "dino_specificity": "DINO-S specificity ↑",
    "resnet50_top1_label": "R50 / true label ↑",
    "resnet50_top1_source_prediction": "R50 / source prediction ↑",
    "semantic_error": "Semantic error ↓", "confidently_wrong": "Confidently wrong ↓"}


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    if value in (True, "True", "true", "1"): return True
    if value in (False, "False", "false", "0"): return False
    raise RuntimeError("Explicit boolean required: " + str(value))


def frame_identity(row):
    return (int(row["source_index"]), int(row["N"]), int(row["snr_db"]),
            int(row["noise_seed"]), row["method"])


def validate_inventory(rows):
    indexed = {frame_identity(row): row for row in rows}
    expected = {(i, n, snr, seed, method) for i in range(SOURCES)
                for n in BUDGETS for snr in SNRS for seed in SEEDS for method in METHODS}
    if len(rows) != ROWS or len(indexed) != ROWS or set(indexed) != expected:
        raise RuntimeError("Report requires the exact completed 3600-row inventory")
    for key, row in indexed.items():
        if (boolean(row["synthetic"]) or boolean(row["label_conditioned"])
                or boolean(row["audit_used_to_control_receiver"])):
            raise RuntimeError("Main external report received an invalid scientific row")
        if float(row["E"]) != 2 * key[1] or abs(float(row["actual_energy"]) - 2 * key[1]) > .02:
            raise RuntimeError("Report energy accounting differs")
        if not all(np.isfinite(float(row[name])) for name in METRIC_LABELS):
            raise RuntimeError("Missing or nonfinite completed metric")
        if key[-1] == METHODS[1]:
            other = indexed[key[:-1] + (METHODS[0],)]
            if (row["observed_sha256"] != other["observed_sha256"]
                    or boolean(row["header_accepted"]) != boolean(other["header_accepted"])):
                raise RuntimeError("External pair did not receive the same waveform/header")
            accepted = boolean(row["header_accepted"])
            if (accepted and (not boolean(row["complete_posterior_schedule"])
                    or int(row["NFE"]) != int(row["t_start"]) or int(row["NFE"]) < 2)):
                raise RuntimeError("Incomplete posterior cannot enter the main report")
            if not accepted and (int(row["NFE"]) != 0 or row["fallback"] != "fixed_gray_0.5"):
                raise RuntimeError("Header failure differs from the registered fallback")
    return indexed


def summary_index(rows):
    result = {}
    expected = {(n, snr, method, metric) for n in BUDGETS for snr in SNRS
                for method in METHODS for metric in METRIC_LABELS}
    for row in rows:
        key = (int(row["N"]), int(row["snr_db"]), row["method"], row["metric"])
        if key in result: raise RuntimeError("Duplicate report summary group")
        if int(row["n_sources"]) != SOURCES or int(row["n_frames"]) != 300:
            raise RuntimeError("Summary lacks the registered source/noise population")
        values = [float(row[k]) for k in ("mean", "ci_low", "ci_high")]
        if not np.isfinite(values).all() or values[1] > values[2]:
            raise RuntimeError("Invalid summary interval")
        result[key] = row
    if set(result) != expected: raise RuntimeError("Completed summary metric inventory differs")
    return result


def timing_table(rows):
    groups = defaultdict(list)
    for row in rows: groups[(int(row["N"]), int(row["snr_db"]), row["method"])].append(row)
    answer = []
    for (n, snr, method), group in sorted(groups.items()):
        tx = np.asarray([float(r["TX_seconds"]) for r in group])
        rx = np.asarray([float(r["RX_seconds"]) for r in group])
        accepted = np.asarray([boolean(r["header_accepted"]) for r in group])
        nfe = [int(r["NFE"]) for r in group if boolean(r["header_accepted"])]
        if len(group) != 300 or not np.isfinite(tx).all() or not np.isfinite(rx).all() or min(tx.min(), rx.min()) < 0:
            raise RuntimeError("Incomplete or invalid receiver timing group")
        answer.append(dict(N=n, snr_db=snr, method=method, n_frames=len(group),
            TX_mean_seconds=float(tx.mean()), RX_mean_seconds=float(rx.mean()),
            RX_median_seconds=float(np.median(rx)), RX_p95_seconds=float(np.percentile(rx, 95)),
            RX_mean_accepted_seconds=float(rx[accepted].mean()) if accepted.any() else "",
            accepted_frames=int(accepted.sum()), header_rejected_frames=int((~accepted).sum()),
            header_reject_rate=float((~accepted).mean()),
            offline_false_accepts=sum(boolean(r["offline_false_accept"]) for r in group),
            NFE_min_accepted=min(nfe) if nfe else "", NFE_max_accepted=max(nfe) if nfe else "",
            includes_header_decode=True, includes_model_loading=False,
            standalone_receiver_memory_measured=False,
            quantiles="empirical_per_frame; means include registered CRC-failure outputs"))
    return answer


def fixed_selection(fixed, records):
    if fixed.get("status") != "FROZEN_FIXED_EXAMPLES" or fixed.get("source_indices") != list(FIXED):
        raise RuntimeError("Fixed examples differ from the preregistered 16 sources")
    selected = {r["source_index"]: r for r in fixed["records"]}
    if len(selected) != len(fixed["records"]) or set(selected) != set(FIXED):
        raise RuntimeError("Fixed example records differ")
    for index in FIXED:
        if selected[index] != {k: records[index][k] for k in selected[index]}:
            raise RuntimeError("Fixed example identity differs from the actual evaluation population")
    return FIXED


def plot_figures(config, output, indexed, reconstruction, registration, labels, stopped, progress):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    out = Path(config["output"])
    population = {}
    for index in FIXED:
        checkpoint = validate_source(read(out / "source_checkpoints" / f"{index:04d}.json"), reconstruction["binding"], index)
        with np.load(checkpoint["float_reconstructions"]["path"], allow_pickle=False) as data:
            images = data["images"].copy(); source = pixels(data["source_rgb"]).copy()
            slots = data["image_slots"].tolist()
        pairs = {}
        for record, slot in zip(checkpoint["rows"], slots):
            if record["noise_seed"] == 2001:
                key = frame_identity(record)
                if rgb_sha(images[slot]) != indexed[key]["image_sha256"]:
                    raise RuntimeError("Figure pixels differ from the scored floating reconstruction")
                pairs[key] = images[slot]
        population[index] = (source, pairs)
    def label(value):
        number = int(value)
        return f"{labels[number]} [{number}]" if labels else f"class {number}"
    figures, cells = [], []
    for n in BUDGETS:
        for snr in SNRS:
            pdf_path = output / f"fixed16_N{n}_SNR{snr}.pdf"
            with PdfPages(pdf_path, metadata={"Title": f"Fixed 16 sources; N{n}; SNR{snr}",
                    "Creator": "Verified external comparison", "CreationDate": None, "ModDate": None}) as pdf:
                pages = []
                for page in range(4):
                    if stopped(): raise PauseRequested("Stop requested at a completed figure page")
                    indices = FIXED[page * 4:(page + 1) * 4]
                    fig, axes = plt.subplots(4, 3, figsize=(10.8, 14.6), dpi=120)
                    fig.subplots_adjust(left=.055, right=.985, bottom=.070, top=.900, wspace=.065, hspace=.37)
                    for col, title in enumerate(("Original", "SwinJSCC", "HiFi-DiffCom + Swin")):
                        axes[0, col].set_title(title, fontsize=14, pad=12, weight="bold")
                    for position, index in enumerate(indices):
                        target, reconstructions = population[index]
                        first = indexed[(index, n, snr, 2001, METHODS[0])]
                        images = [target] + [reconstructions[(index, n, snr, 2001, method)] for method in METHODS]
                        for col, image in enumerate(images):
                            axis = axes[position, col]
                            axis.imshow(image.transpose(1, 2, 0), interpolation="nearest", vmin=0, vmax=1)
                            axis.set_xticks([]); axis.set_yticks([])
                            for spine in axis.spines.values(): spine.set_visible(False)
                            if col == 0:
                                caption = (f"Source {index:02d} | true: {label(first['true_class_index'])}\n"
                                           f"R50: {label(first['resnet50_source_prediction'])}")
                            else:
                                row = indexed[(index, n, snr, 2001, METHODS[col - 1])]
                                agree = boolean(row["resnet50_top1_source_prediction"])
                                pred = label(row["resnet50_prediction"])
                                caption = (f"PSNR {float(row['psnr_db']):.2f}  LPIPS {float(row['lpips_alex']):.3f}  "
                                           f"DINO-L {float(row['dinov2_vitl14_cosine']):.3f}\n"
                                           f"R50: {pred} | {'same' if agree else 'different'}\n"
                                           f"p={float(row['resnet50_top1_probability']):.3f} | "
                                           f"{'header accepted' if boolean(row['header_accepted']) else 'header rejected: gray'}")
                                cells.append(dict(N=n, snr_db=snr, page=page + 1, source_index=index,
                                    method=row["method"], replay_row_id=row["replay_row_id"],
                                    image_sha256=row["image_sha256"], reference_sha256=row["reference_sha256"],
                                    observed_sha256=row["observed_sha256"], noise_seed=2001,
                                    psnr_db=float(row["psnr_db"]), lpips_alex=float(row["lpips_alex"]),
                                    dinov2_vitl14_cosine=float(row["dinov2_vitl14_cosine"]),
                                    resnet50_prediction=int(row["resnet50_prediction"])))
                            caption = "\n".join(textwrap.fill(line, 47) for line in caption.splitlines())
                            axis.set_xlabel(caption, fontsize=9, labelpad=6)
                    fig.suptitle(f"N = {n} | SNR = {snr} dB | fixed examples {page * 4 + 1}–{page * 4 + 4} / 16",
                                 fontsize=17, y=.974, weight="bold")
                    fig.text(.5, .944, "Same actual received waveform; paid header; total E = 2N; no class/text side information",
                             ha="center", fontsize=10)
                    fig.text(.5, .031, "Noise seed 2001. Sources were fixed before evaluation; no quality-based selection.\n"
                             "DINO-L = DINOv2 ViT-L/14. R50 agreement compares with the classifier prediction on the original.\n"
                             "HiFi uses a frozen ImageNet ADM prior and the complete registered posterior schedule. "
                             "Matched P / D_U / M1 panels pending.", ha="center", fontsize=9, linespacing=1.35)
                    png_path = output / f"fixed16_N{n}_SNR{snr}_page{page + 1}.png"
                    fig.savefig(png_path, dpi=120, facecolor="white")
                    pdf.savefig(fig, facecolor="white"); plt.close(fig)
                    pages.append(dict(path=str(png_path), sha256=sha(png_path), source_indices=list(indices)))
                    progress("RUNNING", figures_complete=len(figures) * 4 + page + 1, figures_expected=24)
            figures.append(dict(N=n, snr_db=snr, pdf=str(pdf_path), pdf_sha256=sha(pdf_path), png_pages=pages))
    return figures, cells


def interval(row):
    return f"{float(row['mean']):.3f} [{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]"


def markdown_table(headers, values):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + [
        "| " + " | ".join(map(str, row)) + " |" for row in values]


def report_text(summary, timings, paired, figures, selected, qualification):
    lines = ["# External comparison: SwinJSCC and frozen HiFi-DiffCom + Swin", "",
        "This report completes the two external receivers only. The broader comparison plan remains incomplete: "
        "strictly matched P, unconditional digital VAR, and M1 controls must still be merged. "
        "No claim about superiority to this project's methods is made here.", "",
        "## Protocol", "",
        "100 registered development sources × 3 noise seeds (2001/2002/2003) × SNR 1/7/13 dB × N1024/2048: "
        "1,800 actual transmissions and 3,600 reconstructions. Both receivers consume the identical measured full waveform. "
        "The trained model is shared across rates and SNRs. No development-based model or posterior selection; no holdout access.", "",
        "N1024 = 768 data + 256 header complex symbols; N2048 = 1,664 data + 384 header symbols. "
        "The paid QPSK header carries float32 normalization power and the active-channel subset, with CRC16, tail6, convolutional coding and rate matching. "
        "Every frame has E=2N. Rejected or illegal headers produce RGB0.5 in both methods. False-accept audits do not affect receiver decisions.", "",
        "HiFi-DiffCom + Swin is a registered adaptation of the frozen ImageNet ADM posterior to the newly trained SwinJSCC model. "
        "Its schedule uses actual data CBR, the author formula and fixed hyperparameters. All accepted outputs finish the complete schedule; "
        "no shortened qualification samples enter these tables. No generative training is performed.", "",
        f"Selected Swin checkpoint: step {selected['step']}; SHA256 `{selected['checkpoint_sha256']}`. "
        f"Native qualification runtime: PyTorch {qualification['torch']}, CUDA {qualification['cuda']}, {qualification['device']}.", "",
        "The original DINO column is DINOv2 ViT-S/14; the additional DINO-L column is DINOv2 ViT-L/14. "
        "CLIP uses ViT-L/14. Classification uses the independent ImageNet ResNet-50. "
        "No DINO or CLIP backbone is shared with these two transmission pipelines.", "",
        "Intervals are the original paired source bootstrap: average three noise repetitions within each source, "
        "then resample 100 sources 10,000 times (seed 20261002). KID/FID are deferred; external RGB methods have no comparable F recovery error.", ""]
    for title, names in (("Image quality", ("psnr_db", "lpips_alex", "dinov2_vitl14_cosine", "clip_image_cosine")),
        ("Perception and original DINO", ("dists", "dreamsim", "ms_ssim", "dino_cosine", "dino_specificity")),
        ("Classification and semantic error", ("resnet50_top1_label", "resnet50_top1_source_prediction", "semantic_error", "confidently_wrong"))):
        lines += ["## " + title, "", "Mean [95% source-bootstrap interval].", ""]
        table = [[n, snr, SHORT[method]] + [interval(summary[n, snr, method, metric]) for metric in names]
                 for n in BUDGETS for snr in SNRS for method in METHODS]
        lines += markdown_table(["N", "SNR dB", "Method"] + [METRIC_LABELS[m] for m in names], table) + [""]
    lines += ["Semantic error is disagreement with ResNet-50's prediction on the original, not a direct human judgment. "
        "Confidently wrong additionally requires reconstruction top-1 probability ≥ 0.5. "
        "Ground-truth label accuracy is reported separately; neither receiver is given the label.", "",
        "## Paired HiFi minus Swin", "", "Both members of each pair use the same actual received waveform. "
        "Positive deltas favor HiFi for higher-is-better metrics; negative deltas favor HiFi for lower-is-better metrics.", ""]
    names = ("psnr_db", "lpips_alex", "dinov2_vitl14_cosine", "semantic_error", "confidently_wrong")
    paired_by_key = {(int(r["N_A"]), int(r["snr_A"]), r["metric"]): r for r in paired}
    if len(paired_by_key) != 6 * len(METRIC_LABELS): raise RuntimeError("Paired comparison inventory differs")
    lines += markdown_table(["N", "SNR dB"] + [METRIC_LABELS[m] for m in names],
        [[n, snr] + [interval(paired_by_key[n, snr, m]) for m in names] for n in BUDGETS for snr in SNRS]) + [""]
    lines += ["All metrics and intervals: [summary](metrics_summary.csv), [paired intervals](metrics_paired_intervals.csv), "
        "[per-source values](metrics_per_source.csv), [per-frame values](metrics_per_frame.csv).", "",
        "## Receiver cost and header failures", "",
        "Seconds per frame on the same GPU and runtime. Receiver timing includes actual header decoding and reconstruction, "
        "with CUDA synchronization; model loading, metric scoring and file writing are excluded. "
        "Means include rejected-header frames; accepted-only means expose actual generation cost. P95 is an empirical frame quantile. "
        "Standalone receiver memory was not measured: both receivers share one process and the ADM model remains resident.", ""]
    table = []
    for r in timings:
        accepted = "—" if r["RX_mean_accepted_seconds"] == "" else f"{r['RX_mean_accepted_seconds']:.3f}"
        table.append([r["N"], r["snr_db"], SHORT[r["method"]], f"{r['RX_mean_seconds']:.3f}",
            f"{r['RX_median_seconds']:.3f}", f"{r['RX_p95_seconds']:.3f}", accepted,
            f"{r['header_rejected_frames']}/300", r["offline_false_accepts"],
            f"{r['NFE_min_accepted']}–{r['NFE_max_accepted']}"])
    lines += markdown_table(["N", "SNR", "Method", "RX mean", "RX median", "RX P95", "RX accepted mean", "Rejected", "False accept audit", "NFE accepted"], table)
    lines += ["", "[Complete timing and header audit table](receiver_cost.csv).", "",
        "## Fixed examples", "", "The 16 preregistered sources are 0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95. "
        "All use noise seed 2001. Four pages per N/SNR preserve readable 256×256 image details. "
        "SNR7 is the planned main view; SNR1 and SNR13 are supplements. No sample was selected using reconstruction quality.", ""]
    for figure in sorted(figures, key=lambda f: (f["N"], {7: 0, 1: 1, 13: 2}[f["snr_db"]])):
        lines += [f"### N{figure['N']}, SNR {figure['snr_db']} dB", "",
            f"[Four-page PDF](figures/{Path(figure['pdf']).name}) · " + " · ".join(
                f"[PNG {i + 1}](figures/{Path(page['path']).name})" for i, page in enumerate(figure["png_pages"])), "",
            f"![Fixed examples page 1](figures/{Path(figure['png_pages'][0]['path']).name})", ""]
    lines += ["## Required matched project controls remain pending", "",
        "N1024 P, D_U and M1 have historical unified-metric outputs, but they require strict source/metric/protocol identity checks "
        "before entering this report. Existing N2048 digital results transmit a paid true label; they cannot be relabeled unconditional D_U. "
        "A registered unconditional N2048 policy needs separate calibration and evaluation. "
        "Until those controls are added, these figures and tables answer only the Swin-versus-HiFi comparison.", "",
        "The two external methods have their own RGB encoder and decoder. The same-Dc restriction applies within the project's "
        "mechanism comparison; it is not asserted for these external architectures. Other historic external methods are not silently merged.", ""]
    return "\n".join(lines)


def generate(config_path, fixed_path, class_names, progress, stopped):
    config = read(config_path)
    from external_eval import validate_config
    validate_config(config)
    out, result = Path(config["output"]), Path(config["result"])
    completion_path = out / "completion.json"
    completion = read(completion_path)
    if (completion.get("status") != "EXTERNAL_EVALUATION_COMPLETE" or completion.get("synthetic") is not False
            or completion.get("rows") != ROWS or completion.get("physical_frames") != PHYSICAL_FRAMES
            or completion.get("sources") != SOURCES or completion.get("sampler_step_limit") is not None
            or completion.get("selection_uses_development") is not False or completion.get("holdout_access") is not False):
        raise RuntimeError("Complete scientific evaluation is required before report generation")
    verify(completion["bindings"]); verify(completion["outputs"])
    reconstruction_path = out / "reconstruction_completion.json"
    if sha(reconstruction_path) != completion["reconstruction_completion_sha256"]:
        raise RuntimeError("Scoring and reconstruction populations differ")
    reconstruction = read(reconstruction_path)
    verify(reconstruction["bindings"]); verify(reconstruction["outputs"])
    registration = read(out / "reconstruction_registration.json")
    fixed_selection(read(fixed_path), registration["source_identity"])
    labels = read(class_names) if class_names else None
    if labels is not None and (not isinstance(labels, list) or len(labels) != 1000 or not all(isinstance(x, str) for x in labels)):
        raise RuntimeError("Class names must be an ordered 1000-string ImageNet list")
    own = {str(Path(__file__).resolve()): sha(__file__),
           str(Path(__file__).with_name("external_eval_common.py").resolve()): sha(Path(__file__).with_name("external_eval_common.py"))}
    inputs = {**own, str(config_path): sha(config_path), str(fixed_path): sha(fixed_path),
              str(completion_path): sha(completion_path), str(reconstruction_path): sha(reconstruction_path)}
    if class_names: inputs[str(class_names)] = sha(class_names)
    report_registration = dict(status="EXTERNAL_TWO_METHOD_REPORT_REGISTERED", input_bindings=inputs,
        source_indices=list(FIXED), budgets=list(BUDGETS), snrs=list(SNRS), noise_seed=2001,
        methods=list(METHODS), figures_pages_per_group=4, full_plan_complete=False,
        missing_matched_controls=["P", "D_U", "M1"], sample_selection_uses_quality=False, synthetic=False)
    seal(out / "report_registration.json", report_registration)
    report_completion_path = out / "report_completion.json"
    if report_completion_path.exists():
        done = read(report_completion_path)
        if done["report_registration_sha256"] != sha(out / "report_registration.json"):
            raise RuntimeError("Existing report registration differs")
        verify(done["outputs"]); progress("COMPLETE", full_plan_complete=False); return done
    if stopped(): raise PauseRequested("Stop requested before report generation")
    indexed = validate_inventory(csv_rows(result / "metrics_per_frame.csv"))
    summary = summary_index(csv_rows(result / "metrics_summary.csv"))
    timings = timing_table(list(indexed.values()))
    paired = csv_rows(result / "metrics_paired_intervals.csv")
    for row in paired:
        if row["method_A"] != METHODS[1] or row["method_B"] != METHODS[0] or row["comparison_scope"] != "same_source_noise_seed_and_identical_measured_full_waveform":
            raise RuntimeError("Report paired contrast is not the registered same-observation comparison")
    figures_dir = result / "figures"; figures_dir.mkdir(parents=True, exist_ok=True)
    figures, cells = plot_figures(config, figures_dir, indexed, reconstruction, registration, labels, stopped, progress)
    write_csv(result / "receiver_cost.csv", timings)
    selected = read(Path(config["training_output"]) / "selected_swin.json")
    qualification = read(config["hifi_qualification_path"])
    text = report_text(summary, timings, paired, figures, selected, qualification)
    path = result / "EXTERNAL_REPORT.md"
    temporary = path.with_suffix(".md.tmp"); temporary.write_text(text, encoding="utf-8"); os.replace(temporary, path)
    figures_manifest = dict(status="VERIFIED_FIXED16_EXTERNAL_FIGURES", figures=figures, cells=cells,
        source_indices=list(FIXED), noise_seed=2001, image_source="measured_float32_reconstructions",
        metric_source="completed_unified_metrics_per_frame", full_plan_complete=False, input_bindings=inputs)
    seal(result / "figures_manifest.json", figures_manifest)
    outputs = {str(p): sha(p) for p in (path, result / "receiver_cost.csv", result / "figures_manifest.json", out / "report_registration.json")}
    for figure in figures:
        outputs[figure["pdf"]] = figure["pdf_sha256"]
        outputs.update({page["path"]: page["sha256"] for page in figure["png_pages"]})
    verify(inputs)
    done = dict(status="EXTERNAL_TWO_METHOD_REPORT_COMPLETE", synthetic=False, full_plan_complete=False,
        missing_matched_controls=["P", "D_U", "M1"], sources=SOURCES, physical_frames=PHYSICAL_FRAMES,
        rows=ROWS, fixed_sources=16, png_figures=24, pdf_figures=6, figure_cells=len(cells),
        input_bindings=inputs, outputs=outputs, report_registration_sha256=sha(out / "report_registration.json"),
        evaluation_completion_sha256=sha(completion_path), sample_selection_uses_quality=False,
        holdout_access=False)
    seal(report_completion_path, done); progress("COMPLETE", full_plan_complete=False)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixed-examples", type=Path, required=True)
    parser.add_argument("--class-names", type=Path)
    parser.add_argument("--launch-id", default=None)
    args = parser.parse_args()
    config = read(args.config)
    out = Path(config["output"]); out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / "evaluation.lock").open("a+"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    token = args.launch_id or str(uuid.uuid4())
    def progress(status, **values):
        write(out / "report_status.json", dict(status=status, stage="report", pid=os.getpid(), launch_id=token,
            safe_pause_handler_installed=True, updated=time.time(), **values))
    progress("STARTING")
    failure = out / "report_failure.json"
    try:
        if failure.exists(): raise RuntimeError("Previous report failure needs review")
        generate(args.config.resolve(), args.fixed_examples.resolve(),
            args.class_names.resolve() if args.class_names else None, progress, lambda: stop[0])
    except PauseRequested as exc:
        progress("PAUSED", reason=str(exc)); raise SystemExit(75)
    except BaseException as exc:
        if isinstance(exc, SystemExit): raise
        if not failure.exists():
            write(failure, dict(status="FAILED_REQUIRES_REVIEW", stage="report", error=repr(exc),
                launch_id=token, automatic_restart_allowed=False))
        progress("FAILED_REQUIRES_REVIEW", error=repr(exc)); raise


if __name__ == "__main__":
    main()
