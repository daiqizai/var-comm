#!/usr/bin/env python3
"""Frozen-model entropy roundtrip and independent renderer/probability equivalence checks."""

import argparse
import copy
from datetime import datetime
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from benchmark_frozen_systems import assert_gpu_available
from var_comm.next_scale_prior import load_models, next_scale_log_probs, state_sha256
from var_comm.prefix_training_data import read_image_population
from var_comm.progressive import complete_image
from var_comm.study import artifact_hashes, create_output, sha256, snapshot, write_json
from var_comm.whole_entropy import VarScaleStream, decode_phy, decode_source, encode_prefixes, transmit


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915/ENTROPY_GPU_CHECK_001")
    arguments = parser.parse_args()
    if not arguments.execute:
        print("PLAN ONLY: two calibration sources, codec correctness only; no image quality selection")
        return
    output = create_output(arguments.output_dir)
    try:
        assert_gpu_available()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0")
        paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
        for name in ("vae_checkpoint", "var_checkpoint"):
            if sha256(paths[name]) != paths[name + "_sha256"]:
                raise RuntimeError("frozen visual model changed")
        vae, var = load_models(paths, device)
        before = {"vae": state_sha256(vae), "var": state_sha256(var)}
        pixels, labels, identifiers, bindings = read_image_population("calibration")
        rows, archives = [], {}
        for index in (0, 999):
            assert_gpu_available()
            image = pixels[index:index + 1].float().div(127.5).sub(1).to(device)
            source = [values[0].cpu().numpy() for values in vae.img_to_idxBl(image)]
            label = int(labels[index])
            original = {}
            for mode in (7, 8, 9):
                prefix = [torch.as_tensor(tokens, device=device)[None] for tokens in source[:mode - 1]]
                original[mode] = next_scale_log_probs(var, vae, prefix, torch.tensor([label], device=device))[0].cpu().numpy()
            errors = []
            with VarScaleStream(vae, var, label, device) as stream:
                for scale in range(9):
                    actual = stream.log_probs()
                    if scale + 1 in original:
                        error = float(np.max(np.abs(actual - original[scale + 1])))
                        if error > .0002:
                            raise RuntimeError("streaming probabilities differ from the frozen prior implementation")
                        errors.append(error)
                    stream.advance(source[scale])
            payloads = encode_prefixes(vae, var, source, label, device)
            for mode in (7, 8, 9):
                signal, ledger = transmit(payloads[mode], label, mode)
                phy = decode_phy(signal, 19.)
                candidate = decode_source(phy, vae, var, device)
                if not phy["header"]["accepted"] or not phy["body_crc_accepted"] or not candidate["source_complete"]:
                    raise RuntimeError("noiseless whole-frame protocol did not accept a complete source")
                if not all(np.array_equal(first, second) for first, second in zip(candidate["prefix"], source[:mode])):
                    raise RuntimeError("actual arithmetic entropy decoding did not recover the sent tokens")
                expected = complete_image(vae, var, source[:mode], label, device)
                image_error = float(np.max(np.abs(candidate["image"] - expected)))
                if image_error > 1e-6:
                    raise RuntimeError("entropy decoder's carried state changed final RGB")
                corrupted = copy.deepcopy(phy)
                corrupted["payload"][len(corrupted["payload"]) // 2] ^= 1
                corrupted["body_crc_accepted"] = False
                damaged = decode_source(corrupted, vae, var, device)
                canonical = complete_image(vae, var, damaged["prefix"], corrupted["label"], device)
                candidate_error = float(np.max(np.abs(damaged["image"] - canonical)))
                if candidate_error > 1e-6 or not damaged["prefix"]:
                    raise RuntimeError("failed candidate was dropped or rendered with a different context")
                rows.append({"source_index": index, "image_id": identifiers[index], "mode": mode,
                    "prior_max_error": max(errors), "clean_image_max_error": image_error,
                    "failed_candidate_image_max_error": candidate_error, "failed_candidate_scales": len(damaged["prefix"]), **ledger})
                archives[f"source{index}_m{mode}_payload"] = payloads[mode]["payload"]
                archives[f"source{index}_m{mode}_signal"] = signal
            print(f"entropy GPU roundtrip {index}: lengths=" + str({mode: len(payloads[mode]['payload']) for mode in (7,8,9)}), flush=True)
        after = {"vae": state_sha256(vae), "var": state_sha256(var)}
        if after != before:
            raise RuntimeError("entropy checking modified a frozen model")
        source_hashes = snapshot(output, [Path(__file__), ROOT / "src/var_comm/whole_entropy.py", ROOT / "src/var_comm/entropy.py",
            ROOT / "src/var_comm/next_scale_prior.py", ROOT / "src/var_comm/progressive.py", ROOT / "src/var_comm/scale_channel.py",
            ROOT / "configs/communication_decision_study.json", ROOT / "reports/whole_entropy_and_mode_policy_protocol_20260915.md"])
        np.savez(output / "roundtrips.npz", **archives)
        write_json(output / "checks.json", rows)
        write_json(output / "completion.json", {"status": "WHOLE_ENTROPY_CODEC_GPU_CHECK_PASS", "completed_local": datetime.now().astimezone().isoformat(),
            "source_role": "two_existing_calibration_images", "cases": len(rows), "no_quality_based_selection": True,
            "synthetic_failed_payload_tests_not_channel_performance": True, "frozen_before": before, "frozen_after": after,
            "source_hashes": source_hashes, "output_hashes": artifact_hashes(output)})
    except BaseException as error:
        write_json(output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
