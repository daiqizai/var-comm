#!/usr/bin/env python3
"""Measure only the new whole-frame entropy path at the existing CPU endpoints."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from benchmark_frozen_systems import assert_gpu_available, digest, gpu_state
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.online_timing import SOURCE_INDICES, compare_arrays, timed_call
from var_comm.study import artifact_hashes, create_output, seeded_noise, sha256, snapshot, verify_snapshot, write_csv, write_json
from var_comm.whole_entropy import decode_phy, decode_source, encode_prefixes, transmit


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        print("PLAN ONLY: three arithmetic modes on old32 sources; no existing-system benchmark rerun")
        return
    receipt = json.loads((arguments.matrix_run / "completion.json").read_text())
    if receipt["population"] != "development" or receipt["status"] != "FIXED_MODE_MATRIX_COMPLETE":
        raise RuntimeError("the new entropy timing requires completed original-development waveforms")
    output = create_output(arguments.output_dir)
    started = time.perf_counter()
    rows = []
    try:
        assert_gpu_available()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0")
        config_path = ROOT / "configs/communication_decision_study.json"
        config = json.loads(config_path.read_text())
        paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
        vae, var = load_models(paths, device)
        before = {"vae": state_sha256(vae), "var": state_sha256(var)}
        if any(value != receipt["frozen_before"][name] for name, value in before.items()):
            raise RuntimeError("timing used a different visual/source probability model")
        source_hashes = snapshot(output, [Path(__file__), config_path, ROOT / "src/var_comm/whole_entropy.py",
            ROOT / "src/var_comm/entropy.py", ROOT / "src/var_comm/next_scale_prior.py", ROOT / "src/var_comm/scale_channel.py"])
        population = json.loads((arguments.matrix_run / "population.json").read_text())
        write_json(output / "metadata.json", {"started_local": datetime.now().astimezone().isoformat(), "GPU": gpu_state(),
            "matrix_receipt_sha256": sha256(arguments.matrix_run / "completion.json"), "source_hashes": source_hashes,
            "scope": "same_CPU_endpoints_different_session_VAE_VAR_only_resident", "no_existing_control_rerun": True})

        def tx(pixels, label, mode):
            image = torch.from_numpy(pixels[None]).float().div(127.5).sub(1).to(device)
            encoded = vae.img_to_idxBl(image)
            source = [value[0].cpu().numpy() for value in encoded[:mode]]
            payload = encode_prefixes(vae, var, source, label, device, modes=(mode,))[mode]
            return transmit(payload, label, mode)[0]

        def rx(observed, snr):
            return decode_source(decode_phy(observed, snr), vae, var, device)["image"]

        warmed = set()
        for position, index in enumerate(SOURCE_INDICES):
            assert_gpu_available()
            target = population[index]
            with np.load(ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-001" / f"images/{index:03d}/reconstructions.npz", allow_pickle=False) as original:
                pixels = np.rint(original["source"] * 255).astype(np.uint8)
            directory = arguments.matrix_run / "images" / f"{index:04d}"
            with (directory / "per_frame.csv").open(newline="", encoding="utf-8") as handle:
                selected = {(float(row["snr_db"]), int(row["mode"])): row for row in csv.DictReader(handle)
                            if row["family"] == "arithmetic" and int(row["seed"]) == 2001}
            with np.load(directory / "transmissions.npz", allow_pickle=False) as transmissions, np.load(directory / "reconstructions.npz", allow_pickle=False) as images:
                for snr_index, snr in enumerate(config["snrs_db"]):
                    noise = seeded_noise(target["image_id"], 2001, (3060, 2)) / np.sqrt(10 ** (snr / 10))
                    for mode in (7, 8, 9):
                        if (snr, mode) not in warmed:
                            signal = transmissions[f"signal_arithmetic_m{mode}"].astype(np.float64)
                            for warmup in range(2):
                                tx(pixels, int(target["class_index"]), mode)
                                rx(signal + noise, snr)
                            warmed.add((snr, mode))
                    for repeat in range(3):
                        assert_gpu_available()
                        shift = (position * len(config["snrs_db"]) * 3 + snr_index * 3 + repeat) % 3
                        order = (7, 8, 9)[shift:] + (7, 8, 9)[:shift]
                        for mode in order:
                            expected_signal = transmissions[f"signal_arithmetic_m{mode}"].astype(np.float64)
                            observed = expected_signal + noise
                            row = selected[snr, mode]
                            expected_image = images["images"][int(row["image_index_in_archive"])]
                            if digest(observed) != row["received_sha256"]:
                                raise RuntimeError("entropy timing changed the recorded observation")
                            signal, tx_seconds = timed_call(lambda: tx(pixels, int(target["class_index"]), mode), torch.cuda.synchronize)
                            image, rx_seconds = timed_call(lambda: rx(observed, snr), torch.cuda.synchronize)
                            signal_error, exact_signal = compare_arrays(signal, expected_signal, 0., "actual arithmetic waveform")
                            image_error, exact_image = compare_arrays(image, expected_image, 1e-6, "actual arithmetic output")
                            rows.append({"image_index": index, "image_id": target["image_id"], "snr_db": snr, "noise_seed": 2001,
                                "mode": mode, "arm": f"arithmetic_m{mode}", "repeat": repeat, "TX_seconds": tx_seconds,
                                "RX_seconds": rx_seconds, "processing_sum_seconds": tx_seconds + rx_seconds,
                                "total_complex_uses": 3060, "total_energy": float(np.square(signal).sum()),
                                "signal_max_error": signal_error, "image_max_error": image_error, "signal_exact": exact_signal,
                                "image_exact": exact_image, "received_sha256": digest(observed), "image_sha256": digest(image),
                                "timing_scope": "CPU_to_CPU_contiguous_TX_and_RX"})
                        assert_gpu_available()
            write_csv(output / "per_call.csv", rows)
            write_json(output / "status.json", {"status": "ENTROPY_ONLY_TIMING", "pid": __import__("os").getpid(),
                "completed_sources": position + 1, "source_count": 32, "rows": len(rows), "updated_local": datetime.now().astimezone().isoformat()})
            print(f"entropy-only timing {position+1}/32 rows={len(rows)}", flush=True)
        after = {"vae": state_sha256(vae), "var": state_sha256(var)}
        if after != before or len(rows) != 2016:
            raise RuntimeError("entropy timing is incomplete or frozen weights changed")
        verify_snapshot(source_hashes)
        summary = []
        for snrs in ([snr] for snr in config["snrs_db"]):
            for mode in (7, 8, 9):
                selected_rows = [row for row in rows if row["snr_db"] in snrs and row["mode"] == mode]
                summary.append({"snrs_db": "+".join(map(str, snrs)), "mode": mode,
                    **{metric.replace("seconds", "ms"): float(np.mean([row[metric] for row in selected_rows]) * 1000)
                       for metric in ("TX_seconds", "RX_seconds", "processing_sum_seconds")}})
        write_csv(output / "summary.csv", summary)
        write_json(output / "completion.json", {"status": "WHOLE_ENTROPY_CPU_ENDPOINT_TIMING_COMPLETE", "completed_local": datetime.now().astimezone().isoformat(),
            "source_images": 32, "rows": len(rows), "new_existing_system_timing": False, "frozen_before": before, "frozen_after": after,
            "max_signal_error": max(row["signal_max_error"] for row in rows), "max_image_error": max(row["image_max_error"] for row in rows),
            "source_hashes": source_hashes, "elapsed_seconds": time.perf_counter() - started,
            "research_goal_complete": False, "output_hashes": artifact_hashes(output)})
    except BaseException as error:
        if rows:
            write_csv(output / "partial_calls.csv", rows)
        write_json(output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
