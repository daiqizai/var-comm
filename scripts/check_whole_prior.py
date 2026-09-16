#!/usr/bin/env python3
"""Check independent hypothesis caches and full prefix probabilities against official VAR."""

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from var_comm.next_scale_prior import PATCH_NUMS, load_models, next_scale_log_probs, state_sha256
from var_comm.progressive import split_prefix
from var_comm.scale_channel import bits_to_indices, indices_to_bits
from var_comm.study import artifact_hashes, create_output, snapshot, write_json
from var_comm.whole_frame import cache_empty, source_tables


@torch.no_grad()
def teacher_scores(vae, var, prefix_bits, label, device, future_value=0):
    prefixes = [split_prefix(bits_to_indices(bits), 8) for bits in prefix_bits]
    batch = len(prefixes)
    tensors = [torch.tensor(np.stack([prefix[index] for prefix in prefixes]), device=device, dtype=torch.long) for index in range(8)]
    tensors += [torch.full((batch, size ** 2), future_value, device=device, dtype=torch.long) for size in PATCH_NUMS[8:]]
    labels = torch.full((batch,), label, device=device, dtype=torch.long)
    probabilities = torch.log_softmax(var(labels, vae.quantize.idxBl_to_var_input(tensors)).float(), dim=-1)
    offset, scores = 0, []
    for index, tensor in enumerate(tensors[:8]):
        length = PATCH_NUMS[index] ** 2
        scores.append(probabilities[:, offset:offset + length].gather(-1, tensor[..., None]).squeeze(-1).sum(dim=1, dtype=torch.float64))
        offset += length
    return probabilities[:, offset:offset + 169].cpu().numpy(), torch.stack(scores, dim=1).cpu().numpy()


def main():
    output = create_output(ROOT / "outputs/VAR-WHOLE-PRIOR-SELFCHECK-001")
    try:
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        config = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())
        vae, var = load_models(config["paths"], torch.device("cuda:0"))
        before = {"vae": state_sha256(vae), "var": state_sha256(var)}
        target = json.loads((ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-001/populations.json").read_text())["resource_calibration"][0]
        with np.load(ROOT / "outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001/source_tokens.npz", allow_pickle=False) as cache:
            index = cache["image_ids"].tolist().index(target["image_id"])
            source = cache["tokens"][index][:255]
        hypotheses = np.stack([indices_to_bits(source)] * 4)
        for row, location in ((1, 3059), (2, 1092), (3, 100)):
            hypotheses[row, location] ^= 1
        device = torch.device("cuda:0")
        tables, prefix_scores = source_tables(vae, var, hypotheses, target["class_index"], device)
        received = [split_prefix(bits_to_indices(bits), 8) for bits in hypotheses]
        tensors = [torch.tensor(np.stack([prefix[index] for prefix in received]), device=device, dtype=torch.long) for index in range(8)]
        old = next_scale_log_probs(var, vae, tensors, torch.full((4,), target["class_index"], device=device, dtype=torch.long)).cpu().numpy()
        official, official_prefix = teacher_scores(vae, var, hypotheses, target["class_index"], device)
        changed_future, changed_prefix = teacher_scores(vae, var, hypotheses, target["class_index"], device, 4095)
        old_error = float(np.max(np.abs(tables - old)))
        teacher_error = float(np.max(np.abs(tables - official)))
        prefix_error = float(np.max(np.abs(prefix_scores - official_prefix)))
        assert old_error == 0 and teacher_error < 2e-4 and prefix_error < 0.005
        assert np.array_equal(official, changed_future) and np.array_equal(official_prefix, changed_prefix)
        assert cache_empty(var) and before == {"vae": state_sha256(vae), "var": state_sha256(var)}
        np.savez(output / "hypotheses.npz", bits=hypotheses, prefix_scores=prefix_scores, label=np.array(target["class_index"]))
        sources = snapshot(output, [Path(__file__), ROOT / "src/var_comm/whole_frame.py"])
        result = {"status": "WHOLE_PRIOR_SELFCHECK_PASS", "disjoint_engineering_image": target["image_id"],
                  "hypotheses": 4, "old_q9_max_error": old_error, "teacher_q9_max_error": teacher_error,
                  "teacher_prefix_logprob_max_error": prefix_error, "future_token_invariance": True,
                  "frozen_models": before, "source_hashes": sources, "output_hashes": artifact_hashes(output)}
        write_json(output / "selfcheck.json", result)
        print(result, flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
