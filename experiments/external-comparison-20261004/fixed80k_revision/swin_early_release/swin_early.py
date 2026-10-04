"""Exact80k Swin-only physical evaluation, independently published before HiFi."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import sys
import time
import uuid
import numpy as np
from swin_early_common import *


def register(root):
    p=locations(root);parent=read(p['parent_config']);request=HERE/'user_request.json'
    require(request.is_file(),'Swin-first publication request record is required')
    requested=read(request);pause=read(HERE/'pause_verified.json')
    require(requested.get('selected_checkpoint_step')==80000 and requested.get('training_resume_allowed') is False
        and requested.get('paired_cache_must_be_preserved') is True,'Swin-first request must preserve exact80k/paused training/paired cache')
    require(pause.get('delivery/status.json',{}).get('status')=='PAUSED'
        and pause.get('evaluation/reconstruct_status.json',{}).get('status')=='PAUSED'
        and len(pause.get('processes',{}))==2 and all(v.get('exists') is False for v in pause['processes'].values()),
        'Original paired generation and controller must be safely paused and exited')
    import external_eval
    external_eval.validate_config(parent)
    parent_registration=p['parent_output']/'reconstruction_registration.json'
    require(parent_registration.is_file(),'Paused original paired reconstruction registration is required')
    old=read(parent_registration);verify(old['bindings'])
    require(old['rows']==3600 and old['sources']==100 and old['physical_frames']==1800,
        'Original paired source registration differs')
    snapshot=[];bindings=code_bindings()
    bindings.update({str(request):sha(request),str(p['parent_config']):sha(p['parent_config']),
        str(parent_registration):sha(parent_registration),str(HERE/'pause_verified.json'):sha(HERE/'pause_verified.json')})
    for path in sorted((p['parent_output']/'frames').glob('*/frame.json')):
        value=read(path)
        require(value.get('binding')==identity(old),'Paired cache belongs to another registration')
        archive=Path(value['archive']);require(sha(archive)==value['archive_sha256'],'Paired cache archive changed')
        require(value['rows'][0]['selected_checkpoint_sha256']==CHECKPOINT_SHA and value['rows'][0]['selected_step']==80000,
            'Paired cache is not the exact80k checkpoint')
        snapshot.append(dict(frame=value['frame'],receipt=str(path),receipt_sha256=sha(path),archive=str(archive),archive_sha256=sha(archive)))
        bindings[str(path)]=sha(path);bindings[str(archive)]=sha(archive)
    config=dict(parent,version=VERSION,output=str(p['output']),result=str(p['result']),
        methods=list(METHODS),rows=1800,full_comparison_complete=False,
        early_source_bindings=bindings,paired_cache_snapshot=snapshot,user_request_path=str(request))
    seal(p['config'],config);validate_config(config)
    return config


def selected_for_early(config):
    import external_eval
    return external_eval.selected_gate(read(locations(config['root'])['parent_config']))


def reconstruct_frame(codec,receiver,source,spec,directory,binding,selected):
    import torch
    from swin_protocol import layout,transmit_frame,standard_noise
    from swin_replay import receive
    from external_eval import base_row
    config=read(HERE/'config.json');parent_registration=read(locations(config['root'])['parent_output']/'reconstruction_registration.json')
    matching=[v for v in config['paired_cache_snapshot'] if v['frame']==spec]
    require(len(matching)<=1,'Duplicate inherited paired cache')
    target=pixels(source['rgb']);u8=np.rint(target*255).astype(np.uint8)
    require(__import__('hashlib').sha256(u8.tobytes()).hexdigest()==source['preprocessing_id'],'TX uint8 preprocessing differs')
    image=torch.from_numpy(u8.copy()).to('cuda:0').float().div(255)[None]
    if matching:
        inherited=matching[0];require(sha(inherited['receipt'])==inherited['receipt_sha256'],'Inherited paired receipt changed')
        value=paired.validate_frame_receipt(read(inherited['receipt']),identity(parent_registration),spec,source)
        with np.load(value['archive'],allow_pickle=False) as archive:
            baseline=archive['images'][0].copy();observed=archive['observed'].copy();transmitted=archive['transmitted'].copy()
        row=dict(value['rows'][0]);tx_input_sha=value['tx_input_float_sha256']
        # Real native replay of every inherited Swin frame proves this separate
        # fast path reproduces both the paid waveform and the existing pixels.
        n,snr,seed=spec['N'],spec['snr_db'],spec['noise_seed']
        with torch.no_grad():
            data,power,indices=codec.encode_data(image,snr,layout(n)['channels'])
            replayed=transmit_frame(data[0].cpu().numpy(),float(power[0]),tuple(indices[0].cpu().tolist()),n)
        replay_observed=replayed+standard_noise(source['image_id'],seed,n,snr)/10**(snr/20)
        replay_rgb,replay_context=receive(codec,replay_observed,n,snr)
        require(np.array_equal(replayed,transmitted) and np.array_equal(replay_observed,observed)
            and rgb_sha(pixels(replay_rgb))==row['image_sha256'] and bool(replay_context.accepted)==row['header_accepted'],
            'Swin-only native replay differs from frozen paired waveform/pixels; stop instead of weakening parity')
        origin=dict(kind='exact_completed_paired_Swin_pixels',receipt=inherited['receipt'],receipt_sha256=inherited['receipt_sha256'],
            archive=inherited['archive'],archive_sha256=inherited['archive_sha256'],fresh_native_waveform_and_RGB_bitexact=True)
    else:
        n,snr,seed=spec['N'],spec['snr_db'],spec['noise_seed']
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.no_grad():
            data,power,indices=codec.encode_data(image,snr,layout(n)['channels'])
            tx_power,tx_indices=float(power[0]),tuple(indices[0].cpu().tolist())
            transmitted=transmit_frame(data[0].cpu().numpy(),tx_power,tx_indices,n)
        torch.cuda.synchronize();tx_seconds=time.perf_counter()-started
        observed=transmitted+standard_noise(source['image_id'],seed,n,snr)/10**(snr/20)
        before=array_sha(observed)
        torch.cuda.synchronize();started=time.perf_counter()
        baseline,rx=receive(codec,observed,n,snr)
        torch.cuda.synchronize();rx_seconds=time.perf_counter()-started
        require(array_sha(observed)==before,'Receiver changed observed waveform')
        audit=audit_received_context(tx_power,tx_indices,rx)
        row=base_row(spec,METHODS[0],source,rgb_sha(target))
        row.update(image_sha256=rgb_sha(pixels(baseline)),observed_sha256=before,transmitted_sha256=array_sha(transmitted),
            actual_energy=float(np.square(transmitted).sum()),TX_seconds=tx_seconds,RX_seconds=rx_seconds,
            header_accepted=bool(rx.accepted),header_crc_accepted=bool(rx.crc_accepted),header_fields_legal=bool(rx.fields_legal),
            fallback='' if rx.accepted else 'fixed_gray_0.5',selected_checkpoint_sha256=selected['checkpoint_sha256'],selected_step=selected['step'],
            NFE=0,t_start='',complete_posterior_schedule=False,replay_parity_passed=True,
            replay_parity_basis='new_actual_waveform_cache_integrity_not_historical_metric_replay',**audit)
        tx_input_sha=rgb_sha(image[0].detach().cpu().numpy())
        origin=dict(kind='new_Swin_only_same_registered_physical_algorithm',ADM_loaded=False,posterior_steps=0)
    expected=transmitted+standard_noise(source['image_id'],spec['noise_seed'],spec['N'],spec['snr_db'])/10**(spec['snr_db']/20)
    require(np.array_equal(observed,expected),'Early release noise differs from original physical namespace')
    require(tx_input_sha==rgb_sha(image[0].detach().cpu().numpy()),'Inherited input float differs')
    archive=directory/'reconstructions.npz'
    digest=atomic_npz(archive,images=np.stack([pixels(baseline)]),observed=observed,transmitted=transmitted)
    value=dict(binding=binding,frame=spec,source_id=source['image_id'],reference_sha256=rgb_sha(target),
        tx_input_preprocessing='same_uint8_source_div255',tx_input_float_sha256=tx_input_sha,rows=[row],
        archive=str(archive),archive_sha256=digest,observed_sha256=array_sha(observed),transmitted_sha256=array_sha(transmitted),
        receiver_arguments='observed_full_Nx2,N,SNR_only',synthetic=False,hifi_evaluated=False,cache_origin=origin)
    value['payload_sha256']=identity(value);validate_frame_receipt(value,binding,spec,source)
    seal(directory/'frame.json',value);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--stage',choices=('register','reconstruct','score'),required=True);parser.add_argument('--launch-id')
    args=parser.parse_args()
    from fixed80k_adapter import install
    install(args.root)
    if args.stage=='register':
        config=register(args.root);print('SWIN_EARLY_REGISTERED',len(config['paired_cache_snapshot']));return
    p=locations(args.root);config=read(p['config']);validate_config(config);out=p['output'];out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'evaluation.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
    token=args.launch_id or str(uuid.uuid4())
    def progress(status,**fields):
        write(out/(args.stage+'_status.json'),dict(status=status,stage=args.stage,pid=os.getpid(),launch_id=token,
            safe_pause_handler_installed=True,updated=time.time(),methods=list(METHODS),**fields))
    progress('STARTING');failure=out/(args.stage+'_failure.json')
    try:
        require(not failure.exists(),'Prior early-stage failure requires review')
        if args.stage=='reconstruct':reconstruct(config,p['config'],progress,lambda:stopped[0])
        else:
            from swin_early_score import score
            score(config,p['config'],progress,lambda:stopped[0])
    except PauseRequested as error:progress('PAUSED',reason=str(error));raise SystemExit(75)
    except BaseException as error:
        if isinstance(error,SystemExit):raise
        seal(failure,dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False))
        progress('FAILED_REQUIRES_REVIEW',error=repr(error));raise


def reconstruct(config, config_path, progress, stopped):
    import torch
    from swin_train import configure_runtime
    from swin_protocol import configure_phy
    from swin_replay import load_selected
    from step0_reference_prepare import targets_from_completed
    root, out = Path(config["root"]), Path(config["output"])
    own = code_bindings()
    own.update(config["early_source_bindings"])
    selected, selection_bindings = selected_for_early(config)
    records, target_bindings = targets_from_completed(root)
    if len(records) != SOURCES or [r["source_index"] for r in records] != list(range(SOURCES)):
        raise RuntimeError("Exact registered 100-source population required")
    input_bindings = {**own, **selection_bindings, **target_bindings, str(config_path): sha(config_path)}
    registration = dict(status="SWIN_EARLY_RECONSTRUCTION_REGISTERED", config=config,
        source_identity=[{k: v for k, v in r.items() if k != "rgb"} for r in records],
        source_float_sha256=[rgb_sha(r["rgb"]) for r in records], bindings=input_bindings,
        sources=SOURCES, physical_frames=PHYSICAL_FRAMES, rows=ROWS, training_updates=0,
        policy_selection_updates=0, holdout_access=False, synthetic=False,
        schedule_mode="actual_data_cbr", sampler_step_limit=None,
        waveform_pairing="Swin_only_same_registered_physical_waveform; HiFi_pending", hifi_evaluated=False, posterior_sampling_seed=23)
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
    receiver = None  # No ADM loaded or sampled in the one-method release.
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
                    rows=completed_frames + len(frames), last_frame=frame_key(spec),
                    elapsed_seconds=time.time() - started, last_RX_seconds=frame["rows"][0]["RX_seconds"])
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
                    dtype="float32", layout="CHW", rows=18, unique_images=len(images),
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
    seal(out / "reconstruction_completion.json", dict(status="SWIN_EARLY_RECONSTRUCTIONS_COMPLETE",
        binding=binding, sources=SOURCES, physical_frames=PHYSICAL_FRAMES, rows=ROWS,
        outputs=output_bindings, bindings=input_bindings, synthetic=False, sampler_step_limit=None,
        selection_uses_development=False, full_comparison_complete=False, hifi_evaluated=False,
        cached_swin_native_parity_count=len(config['paired_cache_snapshot']), elapsed_seconds=time.time() - started))
    progress("COMPLETE", sources_complete=SOURCES, frames_complete=PHYSICAL_FRAMES, rows=ROWS)



if __name__=='__main__':main()
