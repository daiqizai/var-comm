"""Durable full development evaluation; native generation and metrics are separate.

CLI: python external_eval.py --config registered.json --stage reconstruct|score
There is deliberately no source limit, SNR override, or sampler step-limit flag.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid
import numpy as np
from external_eval_common import *

HERE = Path(__file__).resolve().parent
OWN = ("external_eval.py", "external_eval_common.py", "external_eval_score.py",
       "hifi_swin_operator.py", "hifi_swin_sampler.py", "hifi_swin_schedule.py",
       "hifi_swin_tests.py", "hifi_swin_schedule_tests.py", "swin_model.py",
       "swin_protocol.py", "swin_replay.py", "swin_train.py", "swin_data.py", "swin_state.py",
       "step0_reference_prepare.py", "step0_reference_protocol.py", "step0_cache_export.py",
       "step0_reference_metrics.py")


def code_bindings():
    return {str((HERE / name).resolve()): sha(HERE / name) for name in OWN}


def validate_config(config):
    if any(key in config for key in ("step_limit", "limit_sources", "limit_frames", "sampler_steps")):
        raise RuntimeError("Diagnostic truncation options are forbidden in main evaluation")
    root = Path(config["root"]).resolve()
    expected = {"output": root / "outputs/EXTERNAL-COMPARISON-20261004/evaluation",
                "result": root / "results/external_comparison_20261004/evaluation",
                "training_output": root / "outputs/EXTERNAL-COMPARISON-20261004/swin_training",
                "hifi_qualification_path": root / "outputs/EXTERNAL-COMPARISON-20261004/hifi_qualification/qualification.json"}
    for key, path in expected.items():
        if Path(config[key]).resolve() != path:
            raise RuntimeError("Registered external path differs: " + key)
    if (config.get("version") != "EXTERNAL-EVALUATION-20261004-R1"
            or config.get("total_N") != list(BUDGETS) or config.get("snrs") != list(SNRS)
            or config.get("noise_seeds") != list(SEEDS) or config.get("source_count") != SOURCES
            or config.get("sampling_seed") != 23 or config.get("schedule_mode") != "actual_data_cbr"
            or config.get("holdout_access") is not False):
        raise RuntimeError("External evaluation scope differs from its registration")
    return root


def selected_gate(config):
    import torch
    train = Path(config["training_output"])
    completion = read(train / "completion.json")
    if completion.get("status") != "SWIN_TRAINING_COMPLETE" or completion.get("synthetic") is not False:
        raise RuntimeError("Only a completed calibration-selected Swin may enter development")
    verify(completion["outputs"])
    selected = read(train / "selected_swin.json")
    registration = read(train / "registration.json")
    if (selected["registration_sha256"] != sha(train / "registration.json")
            or completion["registration_sha256"] != selected["registration_sha256"]
            or sha(selected["checkpoint"]) != selected["checkpoint_sha256"]
            or selected.get("development_read") is not False or selected.get("holdout_read") is not False):
        raise RuntimeError("Selected Swin provenance differs")
    qualification_path = Path(config["hifi_qualification_path"])
    proof = read(qualification_path)
    if (proof.get("status") != "HIFI_SWIN_QUALIFICATION_PASS"
            or any(proof.get(key) is not True for key in ("full_input_gradient_pass", "native_forward_parity_pass", "model_parameters_unchanged"))
            or proof.get("selected_checkpoint_sha256") != selected["checkpoint_sha256"]):
        raise RuntimeError("Selected model lacks the real HiFi gradient/native qualification")
    if proof.get("torch") != torch.__version__ or proof.get("cuda") != torch.version.cuda:
        raise RuntimeError("Main posterior runtime differs from its real qualification")
    from hifi_swin_sampler import ADM_SHA256
    if proof.get("adm_checkpoint_sha256") != ADM_SHA256 or sha(config["adm_checkpoint"]) != ADM_SHA256:
        raise RuntimeError("HiFi qualification has a different frozen ImageNet prior")
    own = code_bindings()
    for name in ("hifi_swin_operator.py", "hifi_swin_sampler.py", "hifi_swin_schedule.py",
                 "hifi_swin_tests.py", "hifi_swin_schedule_tests.py"):
        path = str((HERE / name).resolve())
        if proof.get("source_bindings", {}).get(path) != own[path]:
            raise RuntimeError("HiFi qualification source differs: " + name)
    for name in ("swin_model.py", "swin_protocol.py", "swin_replay.py"):
        path = str((HERE / name).resolve())
        if registration["bindings"].get(path) != own[path]:
            raise RuntimeError("Evaluation Swin adapter differs from trained protocol: " + name)
    verify(proof["source_bindings"])
    verify(proof["input_bindings"])
    bindings = {str(train / name): sha(train / name) for name in
                ("completion.json", "selected_swin.json", "registration.json", "qualification.json")}
    bindings.update({str(Path(selected["checkpoint"]).resolve()): selected["checkpoint_sha256"],
                     str(qualification_path.resolve()): sha(qualification_path),
                     str(Path(config["adm_checkpoint"]).resolve()): ADM_SHA256})
    return selected, bindings


def base_row(spec, method, source, target_sha):
    data_n, header_n = ((768, 256) if spec["N"] == 1024 else (1664, 384))
    return dict(experiment="EXTERNAL_SWIN_HIFI_20261004", scope="external_matched_total_budget",
        **spec, source_id=source["image_id"], preprocessing_id=source["preprocessing_id"],
        true_class_index=int(source["class_index"]), method=method, projection="", control="",
        phy_family="continuous_AWGN_with_paid_QPSK_header", output_role="main",
        decoder_id="SwinJSCC_decoder" if method == METHODS[0] else "ImageNet_ADM_with_SwinJSCC_likelihood",
        label_conditioned=False, is_main_conclusion=True, classification_interpretation="independent_classifier",
        class_side_information_bits=0, text_side_information_bits=0,
        E=2 * spec["N"], data_N=data_n, header_N=header_n,
        noise_namespace="SWIN-EXTERNAL-20261004", nominal_noise_pairing="same_source_and_seed_other_methods_have_distinct_physical_noise_namespaces",
        reference_sha256=target_sha, replay_row_id=row_id(spec, method), synthetic=False,
        dino_model_id="DINOv2_ViT-S/14", lpips_model_id="LPIPS_Alex",
        dinov2_vitl14_model_id="DINOv2_ViT-L/14", F_recovery_error_status="NOT_APPLICABLE_external_RGB_interface",
        selection_uses_development=False, standalone_receiver_memory_measured=False)


def reconstruct_frame(codec, receiver, source, spec, directory, binding, selected):
    """Full paired physical frame; audit TX metadata only after both RX calls."""
    import torch
    from swin_protocol import layout, transmit_frame, standard_noise
    from swin_replay import receive
    target = pixels(source["rgb"])
    u8 = np.rint(target * 255).astype(np.uint8)
    if __import__("hashlib").sha256(u8.tobytes()).hexdigest() != source["preprocessing_id"]:
        raise RuntimeError("TX uint8 preprocessing identity differs")
    image = torch.from_numpy(u8.copy()).to("cuda:0").float().div(255)[None]
    n, snr, seed = spec["N"], spec["snr_db"], spec["noise_seed"]
    torch.cuda.synchronize(); started = time.perf_counter()
    with torch.no_grad():
        data, power, indices = codec.encode_data(image, snr, layout(n)["channels"])
        tx_power, tx_indices = float(power[0]), tuple(indices[0].cpu().tolist())
        transmitted = transmit_frame(data[0].cpu().numpy(), tx_power, tx_indices, n)
    torch.cuda.synchronize(); tx_seconds = time.perf_counter() - started
    observed = transmitted + standard_noise(source["image_id"], seed, n, snr) / 10 ** (snr / 20)
    observed_before = array_sha(observed)
    # Receiver arguments contain the full actual observation and public N/SNR.
    torch.cuda.synchronize(); started = time.perf_counter()
    baseline, rx = receive(codec, observed, n, snr)
    torch.cuda.synchronize(); base_seconds = time.perf_counter() - started
    torch.cuda.synchronize(); started = time.perf_counter()
    generated, hifi_receipt = receiver.receive_frame(observed, n, snr, sampling_seed=23)
    torch.cuda.synchronize(); hifi_seconds = time.perf_counter() - started
    validate_full_sampler(hifi_receipt, rx.accepted)
    if array_sha(observed) != observed_before:
        raise RuntimeError("A receiver modified the shared measured waveform")
    if not rx.accepted and not (np.all(baseline == .5) and np.all(generated == .5)):
        raise RuntimeError("Paired CRC failure did not produce identical fixed gray")
    audit = audit_received_context(tx_power, tx_indices, rx)
    images = np.stack([pixels(baseline), pixels(generated)])
    rows = []
    for i, method in enumerate(METHODS):
        row = base_row(spec, method, source, rgb_sha(target))
        row.update(image_sha256=rgb_sha(images[i]), observed_sha256=observed_before,
            transmitted_sha256=array_sha(transmitted), actual_energy=float(np.square(transmitted).sum()),
            TX_seconds=tx_seconds, RX_seconds=base_seconds if i == 0 else hifi_seconds,
            header_accepted=bool(rx.accepted), header_crc_accepted=bool(rx.crc_accepted),
            header_fields_legal=bool(rx.fields_legal), fallback="" if rx.accepted else "fixed_gray_0.5",
            selected_checkpoint_sha256=selected["checkpoint_sha256"], selected_step=selected["step"],
            NFE=0 if i == 0 else hifi_receipt["NFE"],
            t_start="" if i == 0 else hifi_receipt.get("t_start", ""),
            complete_posterior_schedule=bool(i == 1 and rx.accepted),
            replay_parity_passed=True, replay_parity_basis="new_actual_waveform_cache_integrity_not_historical_metric_replay",
            **audit)
        rows.append(row)
    archive = directory / "reconstructions.npz"
    digest = atomic_npz(archive, images=images, observed=observed, transmitted=transmitted)
    value = dict(binding=binding, frame=spec, source_id=source["image_id"],
        reference_sha256=rgb_sha(target), tx_input_preprocessing="same_uint8_source_div255",
        tx_input_float_sha256=rgb_sha(image[0].detach().cpu().numpy()),
        rows=rows, hifi_receipt=hifi_receipt, archive=str(archive), archive_sha256=digest,
        observed_sha256=observed_before, transmitted_sha256=array_sha(transmitted),
        receiver_arguments="observed_full_Nx2,N,SNR,public_sampling_seed_only", synthetic=False)
    value["payload_sha256"] = identity(value)
    validate_frame_receipt(value, binding, spec, source)
    seal(directory / "frame.json", value)
    return value


def reconstruct(config, config_path, progress, stopped):
    import torch
    from swin_train import configure_runtime
    from swin_protocol import configure_phy
    from swin_replay import load_selected
    from hifi_swin_sampler import FrozenHiFiSwinReceiver
    from step0_reference_prepare import targets_from_completed
    root, out = Path(config["root"]), Path(config["output"])
    own = code_bindings()
    selected, selection_bindings = selected_gate(config)
    records, target_bindings = targets_from_completed(root)
    if len(records) != SOURCES or [r["source_index"] for r in records] != list(range(SOURCES)):
        raise RuntimeError("Exact registered 100-source population required")
    input_bindings = {**own, **selection_bindings, **target_bindings, str(config_path): sha(config_path)}
    registration = dict(status="EXTERNAL_RECONSTRUCTION_REGISTERED", config=config,
        source_identity=[{k: v for k, v in r.items() if k != "rgb"} for r in records],
        source_float_sha256=[rgb_sha(r["rgb"]) for r in records], bindings=input_bindings,
        sources=SOURCES, physical_frames=PHYSICAL_FRAMES, rows=ROWS, training_updates=0,
        policy_selection_updates=0, holdout_access=False, synthetic=False,
        schedule_mode="actual_data_cbr", sampler_step_limit=None,
        waveform_pairing="Swin_and_HiFi_identical_full_observation", posterior_sampling_seed=23)
    seal(out / "reconstruction_registration.json", registration)
    binding = identity(registration)
    if (out / "reconstruction_completion.json").exists():
        done = read(out / "reconstruction_completion.json")
        if done["binding"] != binding: raise RuntimeError("Completed reconstruction binding differs")
        verify(done["outputs"]); progress("COMPLETE", sources_complete=100, rows=ROWS); return
    if stopped(): raise PauseRequested("Stop requested before model initialization")
    gpu_available(); configure_runtime(); configure_phy(root)
    codec, checked = load_selected(config["training_output"], config["swin_vendor"])
    if checked != selected: raise RuntimeError("Selection changed during model loading")
    receiver = FrozenHiFiSwinReceiver(codec.native, config["diffcom_vendor"], config["adm_checkpoint"])
    progress("RUNNING", sources_complete=0, frames_complete=0, rows=0)
    started, completed_frames, rows_out = time.time(), 0, []
    output_bindings = {str(out / "reconstruction_registration.json"): sha(out / "reconstruction_registration.json")}
    for index, source in enumerate(records):
        verify(own)
        if stopped(): raise PauseRequested("Stop requested at a committed frame boundary")
        gpu_available()
        checkpoint = out / "source_checkpoints" / f"{index:04d}.json"
        if checkpoint.exists():
            done = validate_source(read(checkpoint), binding, index)
            if done["source_id"] != source["image_id"]:
                raise RuntimeError("Committed source identity differs from admitted population")
        else:
            frames = []
            for spec in frame_specs(index):
                if stopped(): raise PauseRequested("Stop requested at a committed frame boundary")
                gpu_available()
                directory = out / "frames" / frame_key(spec)
                path = directory / "frame.json"
                if path.exists():
                    frame = validate_frame_receipt(read(path), binding, spec, source)
                else:
                    frame = reconstruct_frame(codec, receiver, source, spec, directory, binding, selected)
                frames.append(frame)
                progress("RUNNING", sources_complete=index, frames_complete=completed_frames + len(frames),
                    rows=2 * (completed_frames + len(frames)), last_frame=frame_key(spec),
                    elapsed_seconds=time.time() - started, last_RX_seconds=frame["rows"][1]["RX_seconds"])
            images, slots, lookup, rows = [], [], {}, []
            frame_bindings = {}
            for frame in frames:
                path = Path(frame["archive"])
                frame_bindings[str(path)] = frame["archive_sha256"]
                frame_bindings[str(path.parent / "frame.json")] = sha(path.parent / "frame.json")
                with np.load(path, allow_pickle=False) as archive:
                    for row, image in zip(frame["rows"], archive["images"]):
                        digest = rgb_sha(image)
                        if digest not in lookup:
                            lookup[digest] = len(images); images.append(image.copy())
                        slots.append(lookup[digest]); rows.append(row)
            cache = out / "reconstructions" / f"{index:04d}.npz"
            digest = atomic_npz(cache, images=np.stack(images), source_rgb=pixels(source["rgb"]),
                row_ids=np.asarray(expected_ids(index)), image_slots=np.asarray(slots, dtype=np.int64))
            done = dict(binding=binding, source_index=index, source_id=source["image_id"], rows=rows,
                frame_bindings=frame_bindings, float_reconstructions=dict(path=str(cache), sha256=digest,
                    dtype="float32", layout="CHW", rows=36, unique_images=len(images),
                    row_ids=expected_ids(index), image_slots=slots, reference_sha256=rgb_sha(source["rgb"])))
            done["payload_sha256"] = identity(done)
            validate_source(done, binding, index); seal(checkpoint, done)
        output_bindings[str(checkpoint)] = sha(checkpoint)
        output_bindings[done["float_reconstructions"]["path"]] = done["float_reconstructions"]["sha256"]
        output_bindings.update(done["frame_bindings"])
        rows_out.extend(done["rows"]); completed_frames += 18
        progress("RUNNING", sources_complete=index + 1, frames_complete=completed_frames, rows=len(rows_out))
    if len(rows_out) != ROWS: raise RuntimeError("Incomplete full development reconstruction coverage")
    verify(own)
    write_csv(out / "reconstruction_per_frame.csv", rows_out)
    output_bindings[str(out / "reconstruction_per_frame.csv")] = sha(out / "reconstruction_per_frame.csv")
    seal(out / "reconstruction_completion.json", dict(status="EXTERNAL_RECONSTRUCTIONS_COMPLETE",
        binding=binding, sources=SOURCES, physical_frames=PHYSICAL_FRAMES, rows=ROWS,
        outputs=output_bindings, bindings=input_bindings, synthetic=False, sampler_step_limit=None,
        selection_uses_development=False, elapsed_seconds=time.time() - started))
    progress("COMPLETE", sources_complete=SOURCES, frames_complete=PHYSICAL_FRAMES, rows=ROWS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("reconstruct", "score"), required=True)
    parser.add_argument("--launch-id", default=None)
    args = parser.parse_args()
    config_path = args.config.resolve(); config = read(config_path); validate_config(config)
    out = Path(config["output"]); out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / "evaluation.lock").open("a+"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    launch_id = args.launch_id or str(uuid.uuid4())
    def progress(status, **values):
        write(out / f"{args.stage}_status.json", dict(status=status, stage=args.stage,
            pid=os.getpid(), launch_id=launch_id, safe_pause_handler_installed=True, updated=time.time(), **values))
    progress("STARTING")
    failure = out / f"{args.stage}_failure.json"
    try:
        if failure.exists(): raise RuntimeError("Previous stage failure needs review before a restart")
        if args.stage == "reconstruct":
            reconstruct(config, config_path, progress, lambda: stop[0])
        else:
            from external_eval_score import score
            score(config, config_path, progress, lambda: stop[0])
    except PauseRequested as exc:
        progress("PAUSED", reason=str(exc)); raise SystemExit(75)
    except BaseException as exc:
        if isinstance(exc, SystemExit): raise
        if not failure.exists():
            write(failure, dict(status="FAILED_REQUIRES_REVIEW", stage=args.stage, error=repr(exc),
                launch_id=launch_id, automatic_restart_allowed=False))
        progress("FAILED_REQUIRES_REVIEW", error=repr(exc)); raise


if __name__ == "__main__":
    main()
