"""Calibration-only Q_VAR/Q_direct/Q_gray from frozen official VAR operators.

The original receiver uses class1000 and greedy argmax, without CFG or random
sampling. This module preserves that deterministic rule. Its ideal-state table
is not a noisy-channel evaluation, and CRC acceptance is not treated as proof
of correctness. Actual-link receivers may reuse an image only after matching
the actual accepted token payload and receiver identity.
"""
from __future__ import annotations
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
GENERATION = "unconditional_class1000_greedy_argmax_no_CFG_no_sampling"
PRIMARY_METRIC = "dinov2_vitl14_cosine"
DOMAIN = b"float32:3,256,256:RGB\0"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    return value


def identity(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def seal(path, value):
    if Path(path).exists():
        require(read(path) == jsonable(value), "Frozen artifact changed: "+str(path))
    else:
        write(path, value)


def rgb_sha(value):
    value = np.asarray(value)
    require(value.dtype == np.float32 and value.shape == (3, 256, 256) and np.isfinite(value).all()
            and value.min() >= 0 and value.max() <= 1, "Expected finite original float32 CHW RGB")
    return hashlib.sha256(DOMAIN+np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_state(m, K):
    require(type(m) is int and type(K) is int and 4 <= m <= 10 and K >= 0, "Registered integer m/K required")
    if m == 10:
        require(K == 0, "No eleventh scale exists")
        return m, K
    length = SIZES[m]**2
    require(K <= length, "Partial K exceeds the next scale")
    return (m+1, 0) if K == length else (m, K)


def state_id(m, K):
    m, K = canonical_state(m, K)
    return "m%d_K%d" % (m, K)


def validate_states(values):
    require(isinstance(values, list) and values, "Frozen resource-derived state list required")
    result, seen = [], set()
    for value in values:
        m, k = canonical_state(value["m"], value["K"])
        name = state_id(m, k)
        require(value["state_id"] == name, "State must use its normalized m/K identifier")
        require(name not in seen, "Duplicate normalized source state")
        seen.add(name)
        result.append(dict(state_id=name, m=m, K=k))
    return sorted(result, key=lambda x: (x["m"], x["K"]))


def split_tokens(tokens):
    tokens = np.asarray(tokens)
    require(tokens.shape == (sum(s*s for s in SIZES),) and np.issubdtype(tokens.dtype, np.integer)
            and np.all((tokens >= 0) & (tokens < 4096)), "Original ten-scale VQ tokens required")
    return [x.astype(np.int64, copy=True) for x in np.split(tokens, np.cumsum([s*s for s in SIZES])[:-1])]


def entropy_order(old_rx, logits):
    """Same coarser-prefix entropy, rounding and index tie-break as original M1."""
    logits = np.asarray(logits)
    values = old_rx.entropy_values(logits)
    rounded = np.rint(values*10**old_rx.ENTROPY_DECIMALS).astype(np.int64)
    return np.lexsort((np.arange(len(values), dtype=np.int64), -rounded))


def accepted_payload_key(prefix, positions, values, receiver_identity):
    """Pure actual accepted information key; never accepts image IDs or truth."""
    require(len(prefix) <= 10 and len(positions) == len(values), "Accepted state shape differs")
    h = hashlib.sha256((identity(receiver_identity)+"|accepted-prefix-v1|").encode())
    h.update(len(prefix).to_bytes(2, "little"))
    for index, tokens in enumerate(prefix):
        tokens = np.asarray(tokens)
        require(tokens.shape == (SIZES[index]**2,) and np.issubdtype(tokens.dtype, np.integer)
                and np.all((tokens >= 0) & (tokens < 4096)), "Invalid accepted complete scale")
        h.update(np.asarray(tokens, dtype="<u2").tobytes())
    positions, values = np.asarray(positions), np.asarray(values)
    if len(positions):
        require(len(prefix) < 10 and np.issubdtype(positions.dtype, np.integer) and np.issubdtype(values.dtype, np.integer)
                and len(set(positions.tolist())) == len(positions) and np.all((positions >= 0) & (positions < SIZES[len(prefix)]**2))
                and np.all((values >= 0) & (values < 4096)), "Invalid accepted partial payload")
    h.update(np.asarray(positions, dtype="<u2").tobytes())
    h.update(np.asarray(values, dtype="<u2").tobytes())
    return h.hexdigest()


class SourceRenderer:
    """One calibration source, frozen prefix snapshots, arbitrary frozen states.

    Models are never batched with another source here. Every state remains the
    original B=1 numeric path. Snapshot acceleration needs real qualification;
    ``use_snapshots=False`` provides a fresh original forward for that check.
    """
    def __init__(self, loaded, old_rx, tokens, states, *, use_snapshots=False):
        self.loaded, self.rx, self.torch = loaded, old_rx, old_rx.torch
        self.vae, self.var = loaded["vae"], loaded["var"]
        self.device = loaded["device"]
        self.states = validate_states(states)
        self.scales = split_tokens(tokens)
        self.use_snapshots = bool(use_snapshots)
        self.snapshots, self.orders, self.outputs = {}, {}, {}
        self.prior_forward_calls = 0
        require(tuple(self.vae.quantize.v_patch_nums) == SIZES, "Frozen official quantizer differs")
        require(not self.var.training and not self.vae.training and not loaded["decoder"].training,
                "Frozen evaluation-mode visual models required")
        self.receiver_identity = dict(models=loaded["identity"]["models"], generation=GENERATION,
            order="original_M1_entropy_rounded_%d_digits" % old_rx.ENTROPY_DECIMALS,
            direct_missing="zero_embedding; absent scales skipped", code_sha256=sha(__file__))
        if self.use_snapshots:
            with self.torch.no_grad():
                self._prepare()

    def _logits(self, prior, scale):
        self.prior_forward_calls += 1
        return prior.logits(scale)

    def _snapshot(self, prior, logits):
        kv = []
        for block in self.var.blocks:
            attention = block.attn
            require(all(hasattr(attention, key) for key in ("caching", "cached_k", "cached_v")) and attention.caching,
                    "Official VAR KV cache interface is unavailable")
            kv.append((None if attention.cached_k is None else attention.cached_k.detach().clone(),
                       None if attention.cached_v is None else attention.cached_v.detach().clone()))
        return dict(prior=copy.copy(prior), x=prior.x.detach().clone(), fhat=prior.fhat.detach().clone(),
                    offset=prior.offset, logits=None if logits is None else logits.detach().clone(), kv=kv)

    def _prepare(self):
        needed = {x["m"] for x in self.states}
        maximum = max(needed)
        prior = self.rx._Prior(self.vae, self.var, self.device)
        try:
            for scale in range(min(maximum+1, 10)):
                logits = self._logits(prior, scale)
                if scale in needed:
                    self.snapshots[scale] = self._snapshot(prior, logits)
                if scale < maximum:
                    prior.advance(self.scales[scale], scale)
            if 10 in needed:
                self.snapshots[10] = self._snapshot(prior, None)
        finally:
            prior.close()

    def _restore(self, m):
        snapshot = self.snapshots[m]
        prior = copy.copy(snapshot["prior"])
        prior.x, prior.fhat, prior.offset = snapshot["x"].clone(), snapshot["fhat"].clone(), snapshot["offset"]
        for block, (key, value) in zip(self.var.blocks, snapshot["kv"]):
            block.attn.kv_caching(True)
            block.attn.cached_k = None if key is None else key.clone()
            block.attn.cached_v = None if value is None else value.clone()
        return prior, snapshot["logits"]

    def _fresh(self, m):
        prior = self.rx._Prior(self.vae, self.var, self.device)
        try:
            for scale in range(m):
                self._logits(prior, scale)
                prior.advance(self.scales[scale], scale)
            logits = None if m == 10 else self._logits(prior, m)
            return prior, logits
        except BaseException:
            prior.close()
            raise

    def _open(self, m):
        return self._restore(m) if self.use_snapshots else self._fresh(m)

    def direct_partial(self, latent, m, positions, values):
        """Official one-scale contribution with genuinely zero unknown vectors."""
        if not len(positions):
            return latent.clone()
        torch, q, side = self.torch, self.vae.quantize, SIZES[m]
        embedded = q.embedding.weight.new_zeros(1, q.embedding.embedding_dim, side*side)
        accepted = q.embedding(torch.as_tensor(np.asarray(values)[None], dtype=torch.long, device=self.device))
        embedded[:, :, torch.as_tensor(positions, dtype=torch.long, device=self.device)] = accepted.transpose(1, 2)
        result, _ = q.get_next_autoregressive_input(m, 10, latent.clone(), embedded.reshape(1, q.embedding.embedding_dim, side, side))
        return result

    def render(self, state, receiver):
        require(receiver in ("VAR", "direct"), "Only registered unconditional receivers are supported")
        m, k = canonical_state(state["m"], state["K"])
        require(state_id(m, k) in {x["state_id"] for x in self.states}, "State is outside frozen resource endpoints")
        cache_key = (m, k, receiver)
        if cache_key in self.outputs:
            return self.outputs[cache_key]
        torch = self.torch
        with torch.no_grad():
            prior, logits = self._open(m)
            tokens = [x.copy() for x in self.scales[:m]]
            positions = np.empty(0, dtype=np.int64)
            values = np.empty(0, dtype=np.int64)
            try:
                if k:
                    if m not in self.orders:
                        self.orders[m] = entropy_order(self.rx, logits[0].float().cpu().numpy())
                    positions = self.orders[m][:k].copy()
                    values = self.scales[m][positions].copy()
                if receiver == "VAR":
                    for scale in range(m, 10):
                        current = logits if scale == m else self._logits(prior, scale)
                        indices = current[0].argmax(-1).cpu().numpy().astype(np.int64)
                        if scale == m and k:
                            indices[positions] = values
                        tokens.append(indices.copy())
                        prior.advance(indices, scale)
                elif k:
                    prior.fhat = self.direct_partial(prior.fhat, m, positions, values)
                latent = prior.fhat.float().cpu().clone()
            finally:
                prior.close()
            image = self.loaded["decoder"](latent.to(self.device))[0].float().cpu().numpy().copy()
            rgb_sha(image)
            actual_key = accepted_payload_key(self.scales[:m], positions, values,
                                             dict(self.receiver_identity, receiver=receiver))
            result = dict(image=image, fhat=latent, tokens=tokens, positions=positions, values=values,
                          accepted_payload_key=actual_key, receiver=receiver,
                          known_tokens_fixed=bool(receiver == "direct" or not k or np.array_equal(tokens[m][positions], values)),
                          generation=GENERATION if receiver == "VAR" else "official_partial_zero_embedding_direct",
                          label_conditioned=False)
            self.outputs[cache_key] = result
            return result

    def close(self):
        for block in self.var.blocks:
            block.attn.kv_caching(False)
        self.snapshots.clear()
        self.outputs.clear()


class ReceivedRenderer:
    """Actual receiver: only CRC-accepted payload values, never source truth.

    The link dispatcher handles first-group/header rejection as gray before
    entering this class. A second-group rejection passes its actual accepted
    first-group prefix with K=0. Undetected corruptions remain actual inputs.
    """
    def __init__(self, loaded, old_rx):
        self.loaded, self.rx, self.torch = loaded, old_rx, old_rx.torch
        self.vae, self.var, self.device = loaded["vae"], loaded["var"], loaded["device"]
        require(tuple(self.vae.quantize.v_patch_nums) == SIZES, "Frozen official quantizer differs")
        require(not self.var.training and not self.vae.training and not loaded["decoder"].training,
                "Frozen evaluation-mode visual models required")
        self.receiver_identity = dict(models=loaded["identity"]["models"], generation=GENERATION,
            order="original_M1_entropy_rounded_%d_digits" % old_rx.ENTROPY_DECIMALS,
            direct_missing="zero_embedding; absent scales skipped", code_sha256=sha(__file__))

    direct_partial = SourceRenderer.direct_partial

    def render_received(self, prefix_tokens, partial_values, m, K, receiver="VAR", cache=None):
        require(canonical_state(m, K) == (m, K), "Received profile must already use normalized m/K")
        require(receiver in ("VAR", "direct") and len(prefix_tokens) == m, "Actual accepted prefix length/receiver differs")
        prefix = []
        for scale, token in enumerate(prefix_tokens):
            value = np.asarray(token)
            require(value.shape == (SIZES[scale]**2,) and np.issubdtype(value.dtype, np.integer)
                    and np.all((value >= 0) & (value < 4096)), "Invalid actual accepted prefix")
            prefix.append(value.astype(np.int64, copy=True))
        values = np.asarray(partial_values)
        require(values.shape == (K,) and (not K or np.issubdtype(values.dtype, np.integer))
                and np.all((values >= 0) & (values < 4096)), "Invalid actual accepted partial payload")
        values = values.astype(np.int64, copy=True)
        torch = self.torch
        with torch.no_grad():
            prior = self.rx._Prior(self.vae, self.var, self.device)
            tokens = [x.copy() for x in prefix]
            try:
                for scale in range(m):
                    prior.logits(scale)
                    prior.advance(prefix[scale], scale)
                logits = None if m == 10 else prior.logits(m)
                positions = entropy_order(self.rx, logits[0].float().cpu().numpy())[:K].copy() if K else np.empty(0, dtype=np.int64)
                key = accepted_payload_key(prefix, positions, values, dict(self.receiver_identity, receiver=receiver))
                if cache is not None and key in cache:
                    result = cache[key]
                    require(result["accepted_payload_key"] == key, "Received cache key differs")
                    return result
                if receiver == "VAR":
                    for scale in range(m, 10):
                        current = logits if scale == m else prior.logits(scale)
                        indices = current[0].argmax(-1).cpu().numpy().astype(np.int64)
                        if scale == m and K:
                            indices[positions] = values
                        tokens.append(indices.copy())
                        prior.advance(indices, scale)
                elif K:
                    prior.fhat = self.direct_partial(prior.fhat, m, positions, values)
                latent = prior.fhat.float().cpu().clone()
            finally:
                prior.close()
            image = self.loaded["decoder"](latent.to(self.device))[0].float().cpu().numpy().copy()
            rgb_sha(image)
            result = dict(image=image, fhat=latent, tokens=tokens, positions=positions, values=values,
                accepted_payload_key=key, receiver=receiver,
                known_tokens_fixed=bool(receiver == "direct" or not K or np.array_equal(tokens[m][positions], values)),
                generation=GENERATION if receiver == "VAR" else "official_partial_zero_embedding_direct", label_conditioned=False)
            if cache is not None:
                cache[key] = result
            return result


def qualify(native, states, indices=(0, 1)):
    """Real calibration parity: fresh vs snapshot; no policy or metric fitting."""
    states = validate_states(states)
    data = native.data("calibration")
    torch, checks = native.receiver.torch, []
    # Cover each registered scale, its smallest and largest partial state, and K0.
    chosen = []
    for m in sorted({s["m"] for s in states}):
        group = [s for s in states if s["m"] == m]
        chosen.extend([group[0], group[-1]])
    # The generic received-payload implementation extends the historical m<=7
    # partial helper. Exercise its m8/m9 boundaries even if a given resource
    # enumeration cannot admit these engineering-only states.
    engineering_ids = set()
    for m, k in ((8, 0), (8, 1), (8, 84), (9, 0), (9, 1), (9, 128)):
        name = state_id(m, k)
        if name not in {s["state_id"] for s in states}:
            engineering_ids.add(name)
        chosen.append(dict(state_id=name, m=m, K=k))
    chosen = list({s["state_id"]: s for s in chosen}.values())
    for index in indices:
        tokens = data["T"][index].cpu().numpy()
        fresh = SourceRenderer(native.loaded, native.receiver, tokens, chosen, use_snapshots=False)
        fast = SourceRenderer(native.loaded, native.receiver, tokens, chosen, use_snapshots=True)
        try:
            for state in chosen:
                for receiver in ("VAR", "direct"):
                    a, b = fresh.render(state, receiver), fast.render(state, receiver)
                    require(torch.equal(a["fhat"], b["fhat"]) and np.array_equal(a["image"], b["image"]),
                            "Real snapshot path is not exactly equal to fresh original arithmetic")
                    require(len(a["tokens"]) == len(b["tokens"]) and all(np.array_equal(x, y) for x, y in zip(a["tokens"], b["tokens"])),
                            "Snapshot path changes generated tokens")
                    require(a["accepted_payload_key"] == b["accepted_payload_key"] and b["known_tokens_fixed"], "Snapshot changed accepted information")
                    actual = ReceivedRenderer(native.loaded, native.receiver).render_received(
                        fresh.scales[:state["m"]], a["values"], state["m"], state["K"], receiver)
                    require(actual["accepted_payload_key"] == a["accepted_payload_key"]
                            and torch.equal(actual["fhat"], a["fhat"]) and np.array_equal(actual["image"], a["image"]),
                            "Source Q differs from the actual accepted-payload receiver")
                    if state["K"] == 0:
                        prefix = fresh.scales[:state["m"]]
                        if receiver == "direct":
                            old = native.common.legacy.prefix_latent(native.loaded["vae"], prefix).cpu()
                        else:
                            old = native.common.legacy.complete_latent(native.loaded["vae"], native.loaded["var"], prefix, 1000, native.loaded["device"]).float().cpu()
                        require(torch.equal(old, b["fhat"]), "K0 does not match the original whole-prefix receiver")
                    if state["K"] and state["m"] <= 7:
                        m, k = state["m"], state["K"]
                        original_logits = native.receiver.prefix_logits(native.loaded["vae"], native.loaded["var"], fresh.scales[:m], native.loaded["device"])
                        old_positions = native.receiver.order_positions(fresh.scales[:m], "entropy", k, original_logits)
                        require(np.array_equal(old_positions, b["positions"]), "Entropy order differs from original M1")
                    checks.append(dict(source_index=index, state_id=state["state_id"], receiver=receiver,
                                       image_sha256=rgb_sha(b["image"]), exact_fresh_snapshot=True,
                                       actual_accepted_payload_exact=True,
                                       engineering_boundary_only=state["state_id"] in engineering_ids))
            partial = next(s for s in chosen if s["K"])
            original = fresh.render(partial, "VAR")
            wrong = original["values"].copy(); wrong[0] = (wrong[0]+1) % 4096
            wrong_result = ReceivedRenderer(native.loaded, native.receiver).render_received(
                fresh.scales[:partial["m"]], wrong, partial["m"], partial["K"])
            require(wrong_result["known_tokens_fixed"] and np.array_equal(wrong_result["tokens"][partial["m"]][wrong_result["positions"]], wrong)
                    and wrong_result["accepted_payload_key"] != original["accepted_payload_key"],
                    "Receiver substituted clean values for accepted corrupt payload")
            checks.append(dict(source_index=index, state_id=partial["state_id"], receiver="VAR",
                engineering_corruption_only=True, actual_wrong_accepted_values_clamped=True,
                measured_CRC_false_acceptance=False))
            with torch.no_grad():
                for m in sorted({s["m"] for s in chosen if s["m"] < 10}):
                    prefix_latent = native.common.legacy.prefix_latent(native.loaded["vae"], fresh.scales[:m])
                    empty = fast.direct_partial(prefix_latent, m, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
                    full = fast.direct_partial(prefix_latent, m, np.arange(SIZES[m]**2), fresh.scales[m])
                    reference = native.common.legacy.prefix_latent(native.loaded["vae"], fresh.scales[:m+1])
                    require(torch.equal(empty, prefix_latent) and torch.equal(full, reference),
                            "Direct zero/full partial boundary differs from whole-scale accumulation")
                    checks.append(dict(source_index=index, m=m, receiver="direct", engineering_boundary_only=True,
                                       K0_exact=True, full_scale_exact=True))
        finally:
            fresh.close()
            fast.close()
    native.frozen()
    return dict(status="REAL_SOURCE_QUALITY_PARITY_PASS", source_role="calibration", synthetic=False,
                source_indices=list(indices), generation=GENERATION, random_sampling=False,
                checks=checks, frozen_identity=native.loaded["identity"], source_sha256=sha(__file__))


def validate_reuse_registration(registration, renderer_identity, source_ids):
    require(registration["population"] == "calibration" and registration["label_conditioned"] is False,
            "Development or class-conditioned source Q cannot be reused")
    require(registration["renderer_identity"] == renderer_identity and registration["source_ids"] == source_ids,
            "Reuse changed visual/numeric/generation/source identities")
    require(registration["metric"] == PRIMARY_METRIC, "A different selection metric was registered")


def _read_checkpoint(path, binding):
    value = read(path)
    require(value["binding"] == binding and value["payload_sha256"] == identity({k: v for k, v in value.items() if k != "payload_sha256"}),
            "Source Q checkpoint seal changed")
    proof = value["float_reconstructions"]
    require(sha(proof["path"]) == proof["sha256"], "Source Q float archive changed")
    with np.load(proof["path"], allow_pickle=False) as archive:
        require(set(archive.files) == {"images", "source_rgb", "row_ids", "image_slots"}, "Unexpected source Q float schema")
        images, source_rgb = archive["images"].copy(), archive["source_rgb"].copy()
        slots, row_ids = archive["image_slots"].tolist(), archive["row_ids"].tolist()
    require(row_ids == [r["row_id"] for r in value["rows"]] and len(slots) == len(row_ids), "Source Q row mapping changed")
    require(len(set(row_ids)) == len(row_ids) and len({(r['state_id'],r['receiver']) for r in value['rows']}) == len(row_ids),
            "Duplicate source-Q state/receiver identity")
    for row, slot in zip(value["rows"], slots):
        require(type(slot) is int and 0 <= slot < len(images), "Invalid source-Q image slot")
        require(row["image_sha256"] == rgb_sha(images[slot]) and row["reference_sha256"] == rgb_sha(source_rgb), "Source Q image fingerprint changed")
    return value, source_rgb, {(r["state_id"], r["receiver"]): (r, images[slot]) for r, slot in zip(value["rows"], slots)}


def _csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_quality(native, states_path, out, score_images, score_identity, *, source_limit=300,
                reuse_from=None, qualification=None, stop_requested=lambda: False):
    """Build a sealed screen300 or final1000 Q bundle; no development is loaded.

    ``states_path`` is frozen by resource enumeration before this function.
    ``score_images(record, images)`` returns one metric dict per unique image;
    selection uses only the named DINOv2-L field. Independent ConvNeXt is loaded
    on development only. Label/classification data never enter SourceRenderer.
    """
    require(source_limit in (300, 1000), "Only registered screen300/final1000 populations")
    out, states_path = Path(out).resolve(), Path(states_path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    states_registration = read(states_path)
    require(states_registration["registered_before_quality"] is True and states_registration.get("development_read", False) is False,
            "Resource states must be fixed before reading quality or development")
    states = validate_states(states_registration["states"])
    data = native.data("calibration")
    require(len(data["records"]) == len(data["T"]) == 1000, "Original complete calibration population required")
    sources = [r["image_id"] for r in data["records"]]
    require(len(set(sources)) == 1000, "Repeated calibration sources")
    renderer_identity = dict(frozen_identity=native.loaded["identity"], numerical_runtime=native.flags,
        generation=GENERATION, random_sampling=False, class_embedding=1000,
        source_sha256=sha(__file__), direct_missing="zero_embedding; absent scales skipped")
    if qualification is None:
        qualification = qualify(native, states)
    require(qualification["status"] == "REAL_SOURCE_QUALITY_PARITY_PASS" and qualification["source_role"] == "calibration"
            and qualification["frozen_identity"] == native.loaded["identity"] and qualification["source_sha256"] == sha(__file__),
            "Matching real source-Q parity qualification required")
    seal(out/"qualification.json", qualification)
    registration = dict(population="calibration", stage="screen300" if source_limit == 300 else "final1000",
        source_count=source_limit, source_ids=sources, preprocessing_ids=[r["preprocessing_id"] for r in data["records"]],
        data_bindings=data.get("bindings", {}), renderer_identity=renderer_identity,
        score_identity=jsonable(score_identity), metric=PRIMARY_METRIC,
        states=states, states_registration_path=str(states_path), states_registration_sha256=sha(states_path),
        source_sha256=sha(__file__), label_conditioned=False, development_read=False, training_updates=0,
        qualification_sha256=sha(out/"qualification.json"),
        independent_convnext_role="development only; excluded from selection and candidate refinement")
    seal(out/"registration.json", registration)
    binding = identity(registration)
    reuse_reg = reuse_done = None
    if reuse_from is not None:
        reuse_from = Path(reuse_from).resolve()
        require(reuse_from != out, "Use normal resume for the same output; reuse requires a different completed bundle")
        reuse_reg = read(reuse_from/"registration.json")
        reuse_done = read(reuse_from/"completion.json")
        require(reuse_done["status"] == "SOURCE_QUALITY_COMPLETE"
                and reuse_done["registration_sha256"] == sha(reuse_from/"registration.json"), "Reuse bundle is incomplete")
        validate_reuse_registration(reuse_reg, renderer_identity, sources)
    allrows, outputs = [], {}
    started = time.monotonic()
    requested = [(s["state_id"], receiver) for s in states for receiver in ("VAR", "direct")]+[("gray", "gray")]
    for index in range(source_limit):
        if stop_requested() or (out/"STOP").exists():
            write(out/"status.json", dict(status="PAUSED_AT_SOURCE_BOUNDARY", sources=index, total_sources=source_limit))
            native.frozen()
            return None
        record = dict(data["records"][index], source_index=index)
        source_rgb = record["pixels"].astype(np.float32)/np.float32(255)
        path = out/"source_checkpoints"/("%04d.json" % index)
        if path.exists():
            checkpoint, resumed_source, _ = _read_checkpoint(path, binding)
            require(checkpoint['source_index'] == index and checkpoint['source_id'] == record['image_id']
                    and checkpoint['preprocessing_id'] == record['preprocessing_id'] and np.array_equal(resumed_source,source_rgb),
                    "Resumed source-Q checkpoint belongs to a different source")
        else:
            reference_sha = rgb_sha(source_rgb)
            reused, reuse_proof = {}, None
            if reuse_from is not None:
                previous_path = reuse_from/"source_checkpoints"/("%04d.json" % index)
                if str(previous_path) in reuse_done["outputs"]:
                    require(sha(previous_path) == reuse_done["outputs"][str(previous_path)], "Reusable source Q checkpoint is unbound")
                    previous, previous_source, reused = _read_checkpoint(previous_path, identity(reuse_reg))
                    require(previous["source_id"] == record["image_id"] and previous["preprocessing_id"] == record["preprocessing_id"]
                            and np.array_equal(source_rgb, previous_source), "Reuse has a different source reference")
                    reuse_proof = dict(checkpoint=str(previous_path), sha256=sha(previous_path),
                                       registration_sha256=sha(reuse_from/"registration.json"))
            missing_states = [s for s in states if any((s["state_id"], receiver) not in reused for receiver in ("VAR", "direct"))]
            began = time.monotonic()
            renderer = SourceRenderer(native.loaded, native.receiver, data["T"][index].cpu().numpy(), missing_states,
                                      use_snapshots=True) if missing_states else None
            rows, images, image_slots, slots, reusable_metrics = [], [], [], {}, {}
            try:
                for sid, receiver in requested:
                    old, image = reused.get((sid, receiver), (None, None))
                    if old is not None:
                        details = dict(accepted_payload_key=old["accepted_payload_key"], generation=old["generation"], label_conditioned=False)
                    elif receiver == "gray":
                        image = np.full((3, 256, 256), .5, dtype=np.float32)
                        details = dict(accepted_payload_key="gray", generation="constant_RGB_0.5", label_conditioned=False)
                    else:
                        state = next(s for s in states if s["state_id"] == sid)
                        rendered = renderer.render(state, receiver)
                        image = rendered["image"]
                        details = {k: rendered[k] for k in ("accepted_payload_key", "generation", "label_conditioned")}
                    image_hash = rgb_sha(image)
                    if image_hash not in slots:
                        slots[image_hash] = len(images)
                        images.append(image)
                    state = next((s for s in states if s["state_id"] == sid), dict(m=0, K=0))
                    row = dict(source_index=index, source_id=record["image_id"], preprocessing_id=record["preprocessing_id"],
                        state_id=sid, m=state["m"], K=state["K"], receiver=receiver, population="calibration",
                        image_sha256=image_hash, reference_sha256=reference_sha,
                        row_id=identity(dict(source_id=record["image_id"], state_id=sid, receiver=receiver)),
                        **details)
                    rows.append(row)
                    image_slots.append(slots[image_hash])
                    if old is not None and reuse_reg["score_identity"] == jsonable(score_identity):
                        # Previous source-Q rows separate identity fields from metric fields.
                        metrics = {key: value for key, value in old.items() if key not in row}
                        require(PRIMARY_METRIC in metrics, "Reusable metric row lacks its registered objective")
                        reusable_metrics[image_hash] = metrics
                # Image caching never shares metrics between different source references.
                if hasattr(score_images, "prepare_source"):
                    score_images.prepare_source(record, source_rgb, index)
                pending_slots = [slot for slot, image in enumerate(images) if rgb_sha(image) not in reusable_metrics]
                pending_scores = score_images(record, [images[slot] for slot in pending_slots]) if pending_slots else []
                require(len(pending_scores) == len(pending_slots), "Metric callback returned an incomplete image batch")
                scored = [None]*len(images)
                for slot, metrics in zip(pending_slots, pending_scores):
                    scored[slot] = metrics
                for slot, image in enumerate(images):
                    if scored[slot] is None:
                        scored[slot] = reusable_metrics[rgb_sha(image)]
                protected = set(rows[0])
                for metrics in scored:
                    require(isinstance(metrics, dict) and not (protected & set(metrics)), "Metric callback overrides source/receiver identity")
                    require(PRIMARY_METRIC in metrics and math.isfinite(float(metrics[PRIMARY_METRIC])), "Primary DINOv2-L metric missing/nonfinite")
                for row, slot in zip(rows, image_slots):
                    row.update(jsonable(scored[slot]))
            finally:
                if renderer is not None:
                    renderer.close()
            archive_path = out/"reconstructions"/("%04d.npz" % index)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            temp = archive_path.with_name(archive_path.name+".tmp")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, images=np.stack(images), source_rgb=source_rgb,
                                    row_ids=np.asarray([r["row_id"] for r in rows]), image_slots=np.asarray(image_slots, dtype=np.int64))
            os.replace(temp, archive_path)
            checkpoint = dict(binding=binding, source_index=index, source_id=record["image_id"],
                preprocessing_id=record["preprocessing_id"], rows=rows, source_seconds=time.monotonic()-began,
                reused_image_rows=sum((sid, receiver) in reused for sid, receiver in requested), reuse_proof=reuse_proof,
                unique_images=len(images), float_reconstructions=dict(path=str(archive_path), sha256=sha(archive_path)),
                development_read=False, training_updates=0)
            checkpoint["payload_sha256"] = identity(checkpoint)
            seal(path, checkpoint)
        require([(r["state_id"], r["receiver"]) for r in checkpoint["rows"]] == requested, "Resumed quality state coverage changed")
        require(all(r['source_index'] == index and r['source_id'] == record['image_id']
                    and r['preprocessing_id'] == record['preprocessing_id'] and r['population'] == 'calibration'
                    and r['label_conditioned'] is False for r in checkpoint['rows']), "Source-Q row identity differs")
        allrows.extend(checkpoint["rows"])
        outputs[str(path)] = sha(path)
        proof = checkpoint["float_reconstructions"]
        outputs[proof["path"]] = proof["sha256"]
        write(out/"status.json", dict(status="RUNNING", population="calibration", sources=index+1,
            total_sources=source_limit, state_count=len(states), rows=len(allrows), elapsed_seconds=time.monotonic()-started))
    table = out/"quality_per_source.csv"
    _csv(table, allrows)
    summary = []
    for sid, receiver in requested:
        part = [row for row in allrows if row["state_id"] == sid and row["receiver"] == receiver]
        require(len(part) == source_limit, "Incomplete source-Q state population")
        numeric = [key for key in part[0] if key not in ("source_index", "m", "K", "label_conditioned")
                   and isinstance(part[0][key], (int, float))]
        for metric in numeric:
            values = [float(row[metric]) for row in part if row.get(metric) is not None]
            if len(values) == source_limit and all(math.isfinite(x) for x in values):
                summary.append(dict(state_id=sid, receiver=receiver, metric=metric, mean=float(np.mean(values)),
                                    std=float(np.std(values, ddof=1)), sources=source_limit, population="calibration"))
    summary_path = out/"quality_summary.csv"
    _csv(summary_path, summary)
    outputs.update({str(table): sha(table), str(summary_path): sha(summary_path), str(out/"qualification.json"): sha(out/"qualification.json")})
    native.frozen()
    complete = dict(status="SOURCE_QUALITY_COMPLETE", population="calibration", stage=registration["stage"],
        sources=source_limit, source_ids=sources[:source_limit], metric=PRIMARY_METRIC,
        rows=len(allrows), states=states, outputs=outputs, registration_sha256=sha(out/"registration.json"),
        independent_validation_excluded_from_selection=True, development_read=False, training_updates=0)
    seal(out/"completion.json", complete)
    write(out/"status.json", dict(status="COMPLETE", sources=source_limit, rows=len(allrows)))
    return complete
