from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import torch

from latent_enhancement.runtime import CONFIG, OUTPUT, digest, settings, verify_snapshot, write_json


PHASE = Path(__file__).resolve().parents[2]
CONFIG_B = PHASE / "config.json"
OUT_B = OUTPUT / "stage_B_v1"
CACHE = OUTPUT / "cache_v1"
NAMES = ("enhancement512", "enhancement1024", "receiver_only_refiner")


def config_b():
    return json.loads(CONFIG_B.read_text())


def stable_digest(value):
    """Digest structured cache identity without depending on dict ordering."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def sample_key_digest(image_ids, snrs_db, noise_seeds):
    """Digest every source/SNR/noise key, including the full image identity."""
    keys = [(str(image_id), float(snr), int(seed))
            for image_id in image_ids for snr in snrs_db for seed in noise_seeds]
    return stable_digest(keys)


def validate_sample_grid(record, *, image_ids, snrs_db, noise_seeds, arrays=()):
    """Reject caches whose declared grid is incomplete or reordered."""
    actual_ids = [str(value) for value in record.get("image_ids", [])]
    if actual_ids != [str(value) for value in image_ids] or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError("cache image identities changed or are duplicated")
    if [float(value) for value in record.get("snrs_db", [])] != [float(value) for value in snrs_db]:
        raise RuntimeError("cache SNR grid changed")
    if [int(value) for value in record.get("noise_seeds", [])] != [int(value) for value in noise_seeds]:
        raise RuntimeError("cache noise seed grid changed")
    expected = (len(actual_ids), len(snrs_db), len(noise_seeds))
    for name in arrays:
        if name not in record:
            raise RuntimeError(f"cache is missing {name}")
        shape = tuple(record[name].shape[:3])
        if shape != expected:
            raise RuntimeError(f"cache {name} grid {shape} != {expected}")


def bind_files(files):
    return {str(Path(path).resolve()): digest(path) for path in files}


def stage_b_sources():
    return bind_files([CONFIG, CONFIG_B, PHASE / "README.md", *sorted((PHASE / "src").rglob("*.py"))])


def validate_gate(completion, result, selection):
    if completion.get("status") != "STAGE_A_COMPLETE_NOT_FULL_EXPERIMENT":
        raise RuntimeError("stage A is not complete")
    if not completion.get("frozen_vae_state_unchanged"):
        raise RuntimeError("original VAE preservation not established")
    if not completion.get("stage_B_eligible") or not result.get("stage_B_eligible"):
        return False
    if completion["selection"] != selection or result["selection"] != selection:
        raise RuntimeError("stage-A selected checkpoint receipts disagree")
    if selection.get("step", 0) <= 0:
        raise RuntimeError("untrained decoder cannot qualify")
    return True


def freeze_selected_decoder():
    directory = OUTPUT / "stage_A_v1"
    completion = json.loads((directory / "completion.json").read_text())
    result = json.loads((directory / "continuous_reference_result.json").read_text())
    selection = json.loads((directory / "selected.json").read_text())
    if not validate_gate(completion, result, selection):
        return None
    registration = json.loads((directory / "registration.json").read_text())
    verify_snapshot(registration["source_snapshot"])
    checkpoint = Path(selection["checkpoint"]).resolve()
    if not checkpoint.is_relative_to((directory / "checkpoints").resolve()):
        raise RuntimeError("selected decoder checkpoint escaped original run")
    if digest(checkpoint) != selection["checkpoint_sha256"]:
        raise RuntimeError("selected decoder checkpoint SHA mismatch")
    cache_complete = json.loads((CACHE / "completion.json").read_text())
    if digest(CACHE / "training_statistics.json") != cache_complete["statistics_sha256"]:
        raise RuntimeError("training-only normalization statistics changed")
    gate = {"status": "STAGE_B_DECODER_FROZEN_FROM_ACTUAL_STAGE_A_SELECTION", "selection": selection,
            "stage_A_bindings": bind_files([directory / "completion.json", directory / "continuous_reference_result.json",
                                            directory / "selected.json", directory / "registration.json"]),
            "frozen_stage_A_sources": registration["source_snapshot"],
            "statistics_sha256": cache_complete["statistics_sha256"], "base_config_sha256": digest(CONFIG),
            "stage_B_config_sha256": digest(CONFIG_B)}
    destination = OUT_B / "decoder_gate.json"
    if destination.exists():
        if json.loads(destination.read_text()) != gate:
            raise RuntimeError("frozen decoder gate changed; do not reselect")
    else:
        write_json(destination, gate)
    return gate


def decoder_gate_path():
    """Return an explicit gate context without changing the historical default."""
    override = os.environ.get("VAR_COMM_DECODER_GATE")
    return Path(override).expanduser().resolve() if override else OUT_B / "decoder_gate.json"


def load_gate():
    gate = json.loads(decoder_gate_path().read_text())
    verify_snapshot(gate["stage_A_bindings"])
    verify_snapshot(gate["frozen_stage_A_sources"])
    if digest(CONFIG) != gate["base_config_sha256"] or digest(CONFIG_B) != gate["stage_B_config_sha256"]:
        raise RuntimeError("frozen stage-B recipe changed")
    if digest(CACHE / "training_statistics.json") != gate["statistics_sha256"]:
        raise RuntimeError("training-only statistics changed")
    return gate


def validate_gpu_qualification():
    gate_path = decoder_gate_path()
    qualification = gate_path.parent / "qualification/completion.json"
    record = json.loads(qualification.read_text())
    if record["status"] != "STAGE_B_REAL_PHY_AND_GRADIENT_QUALIFICATION_PASS":
        raise RuntimeError("real-model GPU qualification did not pass")
    verify_snapshot(record["source_bindings"])
    if record["decoder_gate_sha256"] != digest(gate_path):
        raise RuntimeError("qualified decoder selection changed")


def scale_statistics(device="cpu"):
    record = json.loads((CACHE / "training_statistics.json").read_text())
    if record["source_role"] != "training_only":
        raise RuntimeError("normalization statistics are not training-only")
    return torch.tensor(record["s_F"], dtype=torch.float32, device=device).clamp_min(settings()["s_F_floor"])


def load_decoder(vae, device):
    from latent_enhancement.latent import ContinuousDecoder

    gate = load_gate()
    selected = gate["selection"]
    if digest(selected["checkpoint"]) != selected["checkpoint_sha256"]:
        raise RuntimeError("selected Dc weights changed")
    checkpoint = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)
    decoder = ContinuousDecoder(vae).to(device)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    return decoder.eval().requires_grad_(False)


def cache_shard(population, index):
    path = CACHE / population / f"shard_{index:04d}.pt"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if digest(path) != receipt["sha256"]:
        raise RuntimeError(f"source latent cache changed: {path}")
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("scope") != receipt.get("scope"):
        raise RuntimeError(f"source latent cache scope receipt changed: {path}")
    # Latent shards are keyed only by image identity; their cache.py receipt
    # stores a dedicated image-only digest instead of the PHY grid digest.
    image_ids = [str(value) for value in record.get("image_ids", [])]
    image_hash = stable_digest(image_ids)
    if record.get("sample_key_sha256") != image_hash:
        raise RuntimeError(f"source latent cache sample identities changed: {path}")
    return record, path, receipt
