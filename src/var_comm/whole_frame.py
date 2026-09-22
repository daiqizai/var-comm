"""Receiver-only whole-frame hypotheses and complete conditional source scores."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from torch.nn import functional as functional

from .next_scale_prior import PATCH_NUMS
from .progressive import prefix_key, split_prefix
from .scale_channel import bits_to_indices, crc16
from .whole_list import ordered_paths


def cache_empty(var):
    return all(not block.attn.caching and block.attn.cached_k is None and block.attn.cached_v is None for block in var.blocks)


@torch.no_grad()
def source_tables(vae, var, prefix_bits, label, device):
    if not cache_empty(var) or var.training or vae.training or var.cond_drop_rate != 0:
        raise RuntimeError("model state is not an isolated frozen receiver")
    received = [split_prefix(bits_to_indices(bits), 8) for bits in prefix_bits]
    batch = len(received)
    labels = torch.full((batch,), int(label), dtype=torch.long, device=device)
    tensors = [torch.tensor(np.stack([prefix[index] for prefix in received]), dtype=torch.long, device=device) for index in range(8)]
    condition = var.class_emb(labels)
    positions = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    next_input = condition[:, None].expand(-1, var.first_l, -1) + var.pos_start.expand(batch, var.first_l, -1) + positions[:, :var.first_l]
    latent = condition.new_zeros(batch, var.Cvae, PATCH_NUMS[-1], PATCH_NUMS[-1])
    prefix_scores, offset = [], 0
    for block in var.blocks:
        block.attn.kv_caching(True)
    try:
        for scale in range(9):
            size = PATCH_NUMS[scale]
            offset += size ** 2
            hidden = next_input
            conditioned = var.shared_ada_lin(condition)
            for block in var.blocks:
                hidden = block(x=hidden, cond_BD=conditioned, attn_bias=None)
            log_probs = functional.log_softmax(var.get_logits(hidden, condition).float(), dim=-1)
            if scale == 8:
                probabilities = log_probs.cpu().numpy()
                scores = torch.stack(prefix_scores, dim=1).cpu().numpy()
                if not np.isfinite(probabilities).all() or not np.isfinite(scores).all():
                    raise FloatingPointError("nonfinite complete source probability")
                return probabilities, scores
            prefix_scores.append(log_probs.gather(-1, tensors[scale][..., None]).squeeze(-1).sum(dim=1, dtype=torch.float64))
            embedded = vae.quantize.embedding(tensors[scale]).transpose(1, 2).reshape(batch, var.Cvae, size, size)
            latent, next_input = vae.quantize.get_next_autoregressive_input(scale, len(PATCH_NUMS), latent, embedded)
            next_input = var.word_embed(next_input.reshape(batch, var.Cvae, -1).transpose(1, 2)) + positions[:, offset:offset + PATCH_NUMS[scale + 1] ** 2]
    finally:
        for block in var.blocks:
            block.attn.kv_caching(False)
    raise RuntimeError("missing last-scale table")


def conditional_branches(evidence, prefixes, config, tables=None, prefix_log_probs=None):
    choices, branches = [], []
    uniform = np.zeros((169, 4096), dtype=np.float64)
    for index, bits in enumerate(prefixes["bits"]):
        state = int(bits_to_indices(bits[-6:], 6)[0])
        prior = uniform if tables is None else tables[index]
        tick = time.perf_counter()
        listed = ordered_paths(evidence[6120:], config["candidates_per_prefix"], prior=prior, initial_state=state,
                               payload_bits=2028, crc_initial=crc16(bits), stop_on_crc=True,
                               queue_limit=config["maximum_heap_nodes"])
        elapsed = time.perf_counter() - tick
        prefix_nll = 0.0 if prefix_log_probs is None else -float(prefix_log_probs[index].sum())
        joint_scores = listed["local_scores"] + prefixes["local_scores"][index] + prefix_nll
        accepted = np.flatnonzero(listed["accepted"])
        if len(accepted):
            position = int(accepted[0])
            choices.append((float(joint_scores[position]), index, position))
        branches.append({"prefix_rank": index, "initial_state": state, "prefix_crc_state": crc16(bits),
                         "prefix_channel_cost": float(prefixes["local_scores"][index]), "prefix_source_nll": prefix_nll,
                         "joint_scores": joint_scores, "list": listed, "seconds": elapsed})
    selected = min(choices) if choices else None
    information = None if selected is None else np.concatenate((prefixes["bits"][selected[1]], branches[selected[1]]["list"]["bits"][selected[2]]))
    return {"information": information, "selected": None if selected is None else list(selected), "branches": branches}


def assisted_receivers(evidence, label, vae, var, device, config):
    tick = time.perf_counter()
    prefixes = ordered_paths(evidence, config["distinct_prefixes"], stop_bits=3060, queue_limit=config["maximum_heap_nodes"])
    prefix_seconds = time.perf_counter() - tick
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    tables, prefix_scores = source_tables(vae, var, prefixes["bits"], label, device)
    torch.cuda.synchronize(device)
    prior_seconds = time.perf_counter() - tick
    peak_gpu = torch.cuda.max_memory_allocated(device)
    tick = time.perf_counter()
    informed = conditional_branches(evidence, prefixes, config, tables, prefix_scores)
    informed_seconds = prefix_seconds + prior_seconds + time.perf_counter() - tick
    tick = time.perf_counter()
    uniform = conditional_branches(evidence, prefixes, config)
    uniform_seconds = prefix_seconds + time.perf_counter() - tick
    if not cache_empty(var):
        raise RuntimeError("source-hypothesis branch polluted the next receiver")
    keys = [prefix_key(split_prefix(bits_to_indices(bits), 8), label) for bits in prefixes["bits"]]
    return {"prefixes": prefixes, "prefix_seconds": prefix_seconds, "prior_seconds": prior_seconds,
            "prior_batch_calls": 1, "prior_prefix_count": len(keys), "prefix_keys": keys,
            "q9_sha256": [hashlib.sha256(table.tobytes()).hexdigest() for table in tables],
            "prefix_log_probs": prefix_scores, "peak_gpu_allocated_bytes": peak_gpu,
            "VAR": informed, "uniform": uniform, "VAR_seconds": informed_seconds, "uniform_seconds": uniform_seconds}


def public_native_record(listed, arrays, key):
    arrays[key] = np.packbits(listed["bits"], axis=1)
    return {"array_key": key, "information_bits": listed["bits"].shape[1], "candidate_count": len(listed["bits"]),
            "rank_scores": listed["rank_scores"].tolist(), "local_scores": listed["local_scores"].tolist(),
            "accepted": listed["accepted"].tolist(), "stats": listed["stats"]}


def public_assisted_record(result, arrays, key):
    output = {name: value for name, value in result.items() if name not in ("prefixes", "VAR", "uniform", "prefix_log_probs")}
    output["prefix_log_probs"] = result["prefix_log_probs"].tolist()
    output["prefixes"] = public_native_record(result["prefixes"], arrays, key + "_prefix")
    for family in ("VAR", "uniform"):
        record = result[family]
        output[family] = {"selected": record["selected"], "branches": []}
        for branch in record["branches"]:
            entry = {name: value for name, value in branch.items() if name not in ("list", "joint_scores")}
            entry["joint_scores"] = branch["joint_scores"].tolist()
            entry["list"] = public_native_record(branch["list"], arrays, key + "_" + family + "_" + str(branch["prefix_rank"]))
            output[family]["branches"].append(entry)
    return output
