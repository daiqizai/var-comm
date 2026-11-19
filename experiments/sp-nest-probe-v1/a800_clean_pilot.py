"""Build one internally consistent SP0 pilot on A800, from native received bits."""
import argparse
import copy
import hashlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from profiles import OFF, QUARTER, count
from qualify_native_cuda import snapshot
from split_prefixes import BASES, FAMILY, SplitPrefixes
from verify_pfs_cache import digest, filehash, require, verified_bytes


def write_json(path, value, *, update=False):
    tmp = path.with_suffix(".tmp")
    with tmp.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    if not update and path.exists():
        raise FileExistsError(path)
    tmp.replace(path)


def array_hash(x):
    a = np.ascontiguousarray(x)
    return digest(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def bind_native(root, routes, counts, dev):
    import torch

    sys.path.insert(0, str(root / "experiments/layered-entropy-feasibility-20261014"))
    sys.path.insert(0, str(root / "experiments/wcl-evidence-closure-20261009/scripts"))
    from native import definitions
    from t1_codec_runtime import SourceCodec
    from t1_entropy_core import SIZES

    for f in routes["files"] + routes["author_files"] + [routes["dist"]]:
        require(filehash(f["path"]) == f["sha256"], f["path"])
    paths = {"var_source": routes["author_root"]}
    for record in routes["files"]:
        path = record["path"]
        key = {"vae_ch160v4096z32.pth": "vae_checkpoint", "var_d16.pth": "var_checkpoint"}.get(
            Path(path).name, "decoder_checkpoint")
        paths[key] = path
    ns = dict(torch=torch, Path=Path, sys=sys, np=np, hashlib=hashlib, PATCH_NUMS=SIZES, VOCAB_SIZE=4096)
    definitions(root / "src/var_comm/next_scale_prior.py", {"load_models", "state_sha256"}, ns)
    vae, var = ns["load_models"](paths, dev)
    dn = dict(torch=torch, nn=torch.nn, copy=copy)
    definitions(root / "experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py",
                {"ContinuousDecoder"}, dn)
    decoder = dn["ContinuousDecoder"](vae).to(dev)
    state = torch.load(paths["decoder_checkpoint"], map_location="cpu", weights_only=True)
    decoder.load_state_dict(state["decoder"], strict=True)
    del state
    decoder.eval().requires_grad_(False)
    states = {k: ns["state_sha256"](m) for k, m in dict(vae=vae, var=var, decoder=decoder).items()}
    require(states == routes["models"], "all three frozen state identities")
    pn = dict(torch=torch, np=np, SIZES=SIZES)
    definitions(root / "experiments/scale-causal-partial-residual-20261002/partial_receiver.py", {"_Prior"}, pn)
    rn = dict(np=np, time=time, SIZES=SIZES, require=require, token_count=lambda m, K=0: OFF[m] + K)
    definitions(root / "experiments/content-real-64qam-20261006/h_source_driver.py", {"render_received"}, rn)
    native = SimpleNamespace(torch=torch, receiver=SimpleNamespace(_Prior=pn["_Prior"]),
        loaded=dict(vae=vae, var=var, decoder=decoder, device=dev, identity=dict(models=states)),
        flags=dict(deterministic=True, matmul_tf32=False, cudnn_tf32=False, precision="highest"))
    codec = SourceCodec(root, native=native)

    def render(received, endpoint):
        counts["native_renders"] += 1
        with torch.no_grad():
            return rn["render_received"](native, received, *endpoint)

    return codec, render, states


def record_tables(codec, counts, role):
    tables = {}

    def cdf(family, provider, scale):
        require(family == FAMILY and scale == provider.scale, "probability context order")
        table = codec._cdf(family, provider, scale)
        counts[role] += 1
        if scale in tables:
            require(np.array_equal(table, tables[scale]), "replayed coarse RX probability table")
        else:
            tables[scale] = np.ascontiguousarray(table)
        return table

    adapter = SimpleNamespace(Encoder=codec.Encoder, Decoder=codec.Decoder, _inference=codec._inference,
                              _context=codec._context, _cdf=cdf)
    return SplitPrefixes(adapter), tables


class RecoveredContext:
    """Only an actually decoded coarse prefix can address a cached receiver CDF."""
    def __init__(self, tables):
        self.tables = tables
        self.prefix = bytearray()
        self.scale = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def advance(self, values):
        self.prefix.extend(np.asarray(values, dtype="<i8").tobytes())
        self.scale += 1

    def cdf(self, scale):
        require(self.scale == scale, "RX replay scale")
        return self.tables[(scale, digest(self.prefix))]


def registered_single_streams(codec, tables, tokens):
    single = codec.Encoder()
    streams = []
    for s in range(10):
        table = tables[s]
        for pos in range(OFF[s], OFF[s + 1]):
            idx = pos - OFF[s]
            single.encode(tokens[pos:pos + 1], table[idx:idx + 1])
            if pos + 1 >= OFF[4]:
                streams.append(single.finish())
    require(len(streams) == 651, "single NeST token grid")
    return streams


def one_source(slot, row, codec, render, previous, out, counts):
    with np.load(io.BytesIO(verified_bytes(row["archive"])), allow_pickle=False) as z:
        truth, pixels = z["tokens"], z["pixels"]
    require(digest(pixels.tobytes()) == row["preprocessing_id"] and
            digest(b"int64:680\0" + truth.astype("<i8", copy=False).tobytes()) == row["tokens_sha256"], "source identity")
    folder = out / f"slot_{slot:02d}"
    folder.mkdir()
    if slot == 0:
        old = json.loads((previous / "outcome.json").read_bytes())
        require(old["source_id"] == row["source_id"] and old["gpu_uuid"] == os.environ["CUDA_VISIBLE_DEVICES"]
                and filehash(previous / "candidate_streams.npz") == old["bytes_archive_sha256"], "existing A800 slot 0")
        with np.load(previous / "candidate_streams.npz", allow_pickle=False) as z:
            streams = {k: z[k] for k in z.files}
        require(len(streams) == 101, "saved A800 TX grid")
        tx_hashes = [r["observed"] for r in old["cdf"]]
        counts["reused_a800_tx_sources"] += 1
    else:
        coder, tx_tables = record_tables(codec, counts, "tx_prior_scales")
        single, base, enhancement = coder.encode(truth)
        streams = {f"single_{m}_{k}": a for (m, k), a in single.items()}
        streams.update({f"base_{j}": a for j, a in base.items()})
        streams.update({f"enhancement_{j}_{m}_{k}": a for (j, m, k), a in enhancement.items()})
        tx_hashes = [array_hash(tx_tables[s]) for s in range(10)]
        counts["new_tx_sources"] += 1

    # Receiver bootstraps from two terminated streams, through independent original VAR contexts.
    rx, rx_tables = record_tables(codec, counts, "rx_prior_scales")
    base = rx.decode_base(4, streams["base_4"])
    extra = rx.decode_enhancement(4, (10, 0), base, streams["enhancement_4_10_0"])
    recovered = np.concatenate((base, extra))
    require([array_hash(rx_tables[s]) for s in range(10)] == tx_hashes, "all independent RX/TX integer CDFs")
    require(np.array_equal(recovered, truth), "audit after independent decode")
    # Slot 0 already had its own TX passes and archived hashes for all ten tables. Reuse
    # those exact table contents for the extra 651-length TX grid, without a second model traversal.
    token_bits = registered_single_streams(codec, rx_tables if slot == 0 else tx_tables, truth)
    for end in QUARTER:
        require(np.array_equal(token_bits[count(end) - OFF[4]], streams[f"single_{end[0]}_{end[1]}"]),
                "independent arithmetic termination on quarter grid")
    offsets = np.cumsum([0] + [len(x) for x in token_bits], dtype=np.int64)
    np.savez_compressed(folder / "single_tokens.npz", offsets=offsets, bits=np.concatenate(token_bits))
    np.savez_compressed(folder / "split.npz", **streams)

    table_keys = [(s, digest(recovered[:OFF[s]].astype("<i8", copy=False).tobytes())) for s in range(10)]
    cache = {key: rx_tables[key[0]] for key in table_keys}
    np.savez_compressed(folder / "rx_cdfs.npz", **{f"s{s}": rx_tables[s] for s in range(10)})
    cached_codec = SimpleNamespace(Encoder=codec.Encoder, Decoder=codec.Decoder, _inference=codec._inference,
        _context=lambda family: RecoveredContext(cache), _cdf=lambda family, provider, s: provider.cdf(s))
    cached = SplitPrefixes(cached_codec)
    prefixes, rows = {}, []

    def remember(end, values):
        require(np.array_equal(values, recovered[:count(end)]), "independent RX paths disagree")
        if end in prefixes:
            require(np.array_equal(prefixes[end], values), "overlapping RX endpoints")
        prefixes[end] = values

    for j in BASES:
        b = streams[f"base_{j}"]
        decoded_base = base if j == 4 else cached.decode_base(j, b)
        remember((j, 0), decoded_base)
        for m, k in QUARTER:
            if count((m, k)) <= OFF[j]:
                continue
            e = streams[f"enhancement_{j}_{m}_{k}"]
            extra = recovered[OFF[4]:] if (j, m, k) == (4, 10, 0) else cached.decode_enhancement(
                j, (m, k), decoded_base, e)
            values = np.concatenate((decoded_base, extra))
            remember((m, k), values)
            rows.append(dict(j=j, m=m, K=k, base_bits=len(b), enhancement_bits=len(e),
                             base_bits_sha256=array_hash(b), enhancement_bits_sha256=array_hash(e),
                             received_tokens_sha256=array_hash(values)))
    require(len(rows) == 72 and len(prefixes) == 25, "actual split and received endpoint grid")
    counts["canonical_base_streams"] += 4
    counts["canonical_enhancement_streams"] += len(rows)
    if slot == 0:
        write_json(out / "independent_rx_pass.json", dict(pilot_slot=0, source_id=row["source_id"],
                   received_tokens=len(recovered), integer_cdfs_equal=10, canonical_bases=4,
                   canonical_enhancements=len(rows), previous_a800_tx_verified=True))
    images = {f"e_{m}_{k}": render(prefixes[(m, k)], (m, k)) for m, k in QUARTER}
    np.savez(folder / "native_images.npz", **images)
    np.savez_compressed(folder / "accepted_rx_tokens.npz", values=recovered)
    require(len(images) == 25, "clean renders")
    write_json(folder / "done.json", dict(slot=slot, source_id=row["source_id"],
               source_pixels_sha256=row["preprocessing_id"], source_archive_sha256=row["archive"]["sha256"],
               tx_cdf_sha256=tx_hashes, independent_rx_cdf_keys=table_keys, lengths=rows,
               token_grid_count=len(token_bits), source_truth_used_at_rx=False,
               native_images={k: array_hash(v) for k, v in images.items()},
               files={p.name: filehash(p) for p in sorted(folder.glob("*.npz"))}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("repo", "pilot", "audit", "routes", "previous", "out"):
        ap.add_argument("--" + n, type=Path, required=True)
    a = ap.parse_args()
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "deterministic CUDA workspace")
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    require(uuid.startswith("GPU-") and "," not in uuid, "one physical GPU")
    audit = json.loads(a.audit.read_bytes())
    require(filehash(a.routes) == audit["route_manifest_sha256"] and filehash(a.pilot) == audit["pilot_manifest_sha256"],
            "registered routes and roster")
    sources = json.loads(a.pilot.read_bytes())["records"]
    routes = json.loads(a.routes.read_bytes())
    require(len(sources) == 50 and [s["pilot_slot"] for s in sources] == list(range(50)), "fixed Pilot50")
    a.out.mkdir()
    counts = dict(new_tx_sources=0, reused_a800_tx_sources=0, tx_prior_scales=0, rx_prior_scales=0,
                  native_renders=0, canonical_base_streams=0, canonical_enhancement_streams=0)
    begin, cpu = time.time(), time.process_time()
    completed = 0
    try:
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
        os.nice(10)
        gpu = next(r for r in snapshot() if r["uuid"] == uuid)
        require("A800" in gpu["name"] and gpu["free_MiB"] > 10240, "shared GPU headroom")
        write_json(a.out / "registration.json", dict(gpu=gpu, cap_GiB=6, pilot_sha256=filehash(a.pilot),
                   route_sha256=filehash(a.routes), script_sha256=filehash(__file__), planned={"sources":50,"split_edges":3600,
                   "single_token_lengths":32550,"renders":1250},
                   planned_order="first certify the independently received stream of fixed source 0, then all 50 in roster order",
                   training_steps=0, wireless_frames=0, old_H800_streams_used=0))
        import torch
        torch.set_num_threads(6)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        dev = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(dev)
        require(torch.cuda.device_count() == 1 and str(props.uuid) == uuid[4:], "GPU identity")
        torch.cuda.set_per_process_memory_fraction((6 << 30) / props.total_memory, dev)
        runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                       cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), threads=6, interop_threads=2)
        require(runtime == routes["native_qualification"]["numeric_settings"]["runtime"], "frozen numeric runtime")
        codec, render, states = bind_native(a.repo, routes, counts, dev)
        write_json(a.out / "identity.json", dict(gpu_uuid=uuid, physical_name=props.name, runtime=runtime,
                   model_states=states, tf32=False, precision="FP32", source_and_receiver_device_same=True))
        for slot, source in enumerate(sources):
            require(next(r for r in snapshot() if r["uuid"] == uuid)["free_MiB"] > 4096, "shared GPU free memory")
            one_source(slot, source, codec, render, a.previous, a.out, counts)
            completed += 1
            write_json(a.out / "progress.json", dict(completed_sources=completed, total_sources=50,
                       counts=counts, elapsed_seconds=time.time()-begin), update=True)
            print(f"Pilot50 {completed}/50; {counts['native_renders']} native RX outputs", flush=True)
        require(counts["new_tx_sources"] == 49 and counts["reused_a800_tx_sources"] == 1 and
                counts["canonical_enhancement_streams"] == 3600 and counts["native_renders"] == 1250,
                "complete pilot")
        write_json(a.out / "completion.json", dict(status="A800_PILOT_SOURCE_AND_RX_COMPLETE", counts=counts,
                   sources=50, source_lengths=3600, single_lengths=32550,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(dev), training_steps=0, wireless_frames=0,
                   wall_seconds=time.time()-begin, cpu_seconds=time.process_time()-cpu))
    except BaseException as error:
        write_json(a.out / "failure.json", dict(error=repr(error), completed_sources=completed,
                   counts=counts, wall_seconds=time.time()-begin))
        raise


if __name__ == "__main__":
    main()
