"""Only fixed16 x one noise x two budgets x two SNRs; full posterior per frame."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import sys
import time
import uuid
import numpy as np
from hifi16_common import *


def register(root):
    p=locations(root);parent=read(p['parent_config']);request=read(HERE/'user_request.json')
    require(request.get('source_indices')==list(SOURCE_INDICES) and request.get('budgets')==list(BUDGETS)
        and request.get('snrs_db')==list(SNRS) and request.get('noise_seeds')==list(SEEDS)
        and request.get('physical_frames')==64 and request.get('paired_rows')==128
        and request.get('full_queue_resume_allowed') is False and request.get('training_resume_allowed') is False,
        'Only the explicitly requested fixed16 tier is authorized')
    pause=read(HERE/'pause_verified.json')
    require(bool(pause.get('processes')) and all(v.get('exists') is False for v in pause['processes'].values()),
        'Original hierarchy must exit before fixed16 registration')
    import external_eval
    external_eval.validate_config(parent)
    parent_reg=p['parent_output']/'reconstruction_registration.json';old=read(parent_reg);verify(old['bindings'])
    require(old.get('rows')==3600 and old.get('sources')==100 and old.get('synthetic') is False,'Original full registration differs')
    binds=code_bindings();binds.update({str(f):sha(f) for f in (p['parent_config'],parent_reg,HERE/'user_request.json',HERE/'pause_verified.json',
        p['runtime'].parent/'fixed_examples.json',p['root']/'experiments/unified-metrics-20261002/analysis.py')})
    wanted={frame_key(s) for i in SOURCE_INDICES for s in frame_specs(i)};snapshot=[]
    for path in sorted((p['parent_output']/'frames').glob('*/frame.json')):
        value=read(path)
        if frame_key(value['frame']) not in wanted:continue
        require(value['binding']==identity(old) and value['rows'][0]['selected_checkpoint_sha256']==CHECKPOINT_SHA,
            'Cached pair has a different physical registration or checkpoint')
        archive=Path(value['archive']);require(sha(archive)==value['archive_sha256'],'Cached paired archive changed')
        snapshot.append(dict(frame=value['frame'],receipt=str(path),receipt_sha256=sha(path),archive=str(archive),archive_sha256=sha(archive)))
        binds[str(path)]=sha(path);binds[str(archive)]=sha(archive)
    # All matching completed Swin-only observations must be identical. These
    # receipts are provenance checks, never inputs to the HiFi receiver.
    early=p['output'].parents[1]/'swin_early_release/evaluation'
    for index in SOURCE_INDICES:
        for spec in frame_specs(index):
            file=early/'frames'/frame_key(spec)/'frame.json';receipt=read(file)
            require(receipt['rows'][0]['selected_checkpoint_sha256']==CHECKPOINT_SHA
                and receipt['rows'][0]['selected_step']==80000,'Swin reference checkpoint differs')
            binds[str(file)]=sha(file);binds[receipt['archive']]=receipt['archive_sha256']
    config=dict(parent,version=VERSION,output=str(p['output']),result=str(p['result']),source_count=16,
        source_indices=list(SOURCE_INDICES),methods=list(METHODS),rows=128,physical_frames=64,snrs=list(SNRS),noise_seeds=list(SEEDS),
        full_comparison_complete=False,full_queue_resume_allowed=False,fixed16_source_bindings=binds,
        paired_cache_snapshot=snapshot,user_request_path=str(HERE/'user_request.json'))
    seal(p['config'],config);validate_config(config)
    return config


def selected_gate(config):
    import external_eval
    return external_eval.selected_gate(read(locations(config['root'])['parent_config']))


def reconstruct_frame(codec,receiver,source,spec,directory,binding,selected):
    import external_eval
    config=read(HERE/'config.json');p=locations(config['root'])
    matches=[v for v in config['paired_cache_snapshot'] if v['frame']==spec]
    require(len(matches)<=1,'Duplicate cached fixed16 physical frame')
    if matches:
        inherited=matches[0];require(sha(inherited['receipt'])==inherited['receipt_sha256'],'Cached paired receipt changed')
        original=read(inherited['receipt']);old=read(p['parent_output']/'reconstruction_registration.json')
        paired.validate_frame_receipt(original,identity(old),spec,source)
        value=dict(original,binding=binding,cache_origin=dict(receipt=inherited['receipt'],receipt_sha256=inherited['receipt_sha256'],
            reused_complete_author_schedule=True))
        value.pop('payload_sha256');value['payload_sha256']=identity(value)
        paired.validate_frame_receipt(value,binding,spec,source);seal(directory/'frame.json',value)
    else:
        value=external_eval.reconstruct_frame(codec,receiver,source,spec,directory,binding,selected)
    early=p['output'].parents[1]/'swin_early_release/evaluation/frames'/frame_key(spec)/'frame.json'
    expected=read(early)
    require(expected['payload_sha256']==identity({k:v for k,v in expected.items() if k!='payload_sha256'})
        and sha(expected['archive'])==expected['archive_sha256'],'Completed Swin reference changed')
    a,b=value['rows'][0],expected['rows'][0]
    require(all(a[k]==b[k] for k in ('image_sha256','observed_sha256','transmitted_sha256','selected_checkpoint_sha256','selected_step','header_accepted')),
        'HiFi-paired Swin waveform/pixels differ from the already completed Swin release')
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--stage',choices=('register','reconstruct','score'),required=True);parser.add_argument('--launch-id')
    args=parser.parse_args()
    from fixed80k_adapter import install
    install(args.root)
    if args.stage=='register':
        config=register(args.root);print('HIFI_FIXED16_REGISTERED',len(config['paired_cache_snapshot']));return
    p=locations(args.root);config=read(p['config']);validate_config(config);out=p['output'];out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'evaluation.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    token=args.launch_id or str(uuid.uuid4())
    def progress(status,**fields):
        write(out/(args.stage+'_status.json'),dict(status=status,stage=args.stage,pid=os.getpid(),launch_id=token,
            safe_pause_handler_installed=True,updated=time.time(),fixed16_exploratory=True,**fields))
    progress('STARTING');failure=out/(args.stage+'_failure.json')
    try:
        require(not failure.exists(),'Previous fixed16 failure requires review')
        if args.stage=='reconstruct':reconstruct(config,p['config'],progress,lambda:stop[0])
        else:
            from hifi16_score import score
            score(config,p['config'],progress,lambda:stop[0])
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
    from hifi_swin_sampler import FrozenHiFiSwinReceiver
    from step0_reference_prepare import targets_from_completed
    root, out = Path(config["root"]), Path(config["output"])
    own = code_bindings()
    own.update(config["fixed16_source_bindings"])
    selected, selection_bindings = selected_gate(config)
    records, target_bindings = targets_from_completed(root)
    if len(records)!=100 or [r["source_index"] for r in records]!=list(range(100)):
        raise RuntimeError("Original 100-source identity changed")
    records=[records[i] for i in SOURCE_INDICES]
    input_bindings = {**own, **selection_bindings, **target_bindings, str(config_path): sha(config_path)}
    registration = dict(status="HIFI_FIXED16_RECONSTRUCTION_REGISTERED", config=config,
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
        verify(done["outputs"]); progress("COMPLETE", sources_complete=16, rows=ROWS); return
    if stopped(): raise PauseRequested("Stop requested before model initialization")
    gpu_available(); configure_runtime(); configure_phy(root)
    codec, checked = load_selected(config["training_output"], config["swin_vendor"])
    if checked != selected: raise RuntimeError("Selection changed during model loading")
    receiver = FrozenHiFiSwinReceiver(codec.native, config["diffcom_vendor"], config["adm_checkpoint"])
    progress("RUNNING", sources_complete=0, frames_complete=0, rows=0)
    started, completed_frames, rows_out = time.time(), 0, []
    output_bindings = {str(out / "reconstruction_registration.json"): sha(out / "reconstruction_registration.json")}
    for position, source in enumerate(records):
        index=source["source_index"]
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
                progress("RUNNING", sources_complete=position, frames_complete=completed_frames + len(frames),
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
                    dtype="float32", layout="CHW", rows=8, unique_images=len(images),
                    row_ids=expected_ids(index), image_slots=slots, reference_sha256=rgb_sha(source["rgb"])))
            done["payload_sha256"] = identity(done)
            validate_source(done, binding, index); seal(checkpoint, done)
        output_bindings[str(checkpoint)] = sha(checkpoint)
        output_bindings[done["float_reconstructions"]["path"]] = done["float_reconstructions"]["sha256"]
        output_bindings.update(done["frame_bindings"])
        rows_out.extend(done["rows"]); completed_frames += 4
        progress("RUNNING", sources_complete=position + 1, frames_complete=completed_frames, rows=len(rows_out))
    if len(rows_out) != ROWS: raise RuntimeError("Incomplete full development reconstruction coverage")
    verify(own)
    write_csv(out / "reconstruction_per_frame.csv", rows_out)
    output_bindings[str(out / "reconstruction_per_frame.csv")] = sha(out / "reconstruction_per_frame.csv")
    seal(out / "reconstruction_completion.json", dict(status="HIFI_FIXED16_RECONSTRUCTIONS_COMPLETE",
        binding=binding, sources=SOURCES, physical_frames=PHYSICAL_FRAMES, rows=ROWS,
        outputs=output_bindings, bindings=input_bindings, synthetic=False, sampler_step_limit=None,
        selection_uses_development=False, full_comparison_complete=False, full_queue_resume_allowed=False, fixed16_exploratory=True,
        source_indices=list(SOURCE_INDICES), paired_with_completed_Swin_release=True, elapsed_seconds=time.time() - started))
    progress("COMPLETE", sources_complete=SOURCES, frames_complete=PHYSICAL_FRAMES, rows=ROWS)



if __name__=='__main__':main()
