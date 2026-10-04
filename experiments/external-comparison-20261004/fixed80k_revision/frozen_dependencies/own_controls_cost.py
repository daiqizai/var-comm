"""Fixed cal4 receiver benchmark: own (VAR env), external (Swin env), report (CPU)."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import time
import uuid
import numpy as np
import external_eval_common as c
import own_controls_cost_common as k


def name(spec):return f"{spec['source_index']:04d}_N{spec['N']}_SNR{spec['snr_db']}_{spec['method']}"

def row_base(spec,source,runtime):
    return dict(**spec,source_id=source['image_id'],preprocessing_id=source['preprocessing_id'],
        true_class_index=source['class_index'],reference_sha256=source['rgb_sha256'],runtime=runtime,
        source_population='original_calibration_indices_0_1_2_3',batch_size=1,timed_repeats=1,
        warmup_per_method_N_per_invocation=1,E=2*spec['N'],synthetic=False,NFE=0,
        complete_posterior_schedule=False,t_start=0)

def save_frame(directory,binding,spec,source,runtime,row,image,wave,observed,**extra):
    archive=directory/'pixels_and_waveform.npz'
    digest=c.atomic_npz(archive,image=c.pixels(image),transmitted=wave,observed=observed)
    row.update(image_sha256=c.rgb_sha(image),transmitted_sha256=c.array_sha(wave),observed_sha256=c.array_sha(observed),
        actual_energy=float(np.square(np.asarray(wave,dtype=np.float64)).sum()))
    value=dict(binding=binding,spec=spec,runtime=runtime,row=row,archive=str(archive),archive_sha256=digest,**extra)
    value['payload_sha256']=c.identity(value);k.validate_frame(value,binding,spec,source,runtime)
    path=directory/'frame.json';c.seal(path,value);return path,value

def measured_own(native,spec,source,runtime,directory,binding):
    wave,observed,policy=native.prepare(spec) # TX, clean source and policy selection remain outside timing.
    original=c.array_sha(observed);native.torch.cuda.synchronize();started=time.perf_counter()
    image,event=native.receive(observed,spec['N'],spec['snr_db'],spec['method'])
    native.torch.cuda.synchronize();seconds=time.perf_counter()-started
    k.require(c.array_sha(observed)==original,'Receiver changed the actual observation')
    row=row_base(spec,source,runtime);row.update(event,selected_policy=policy,RX_seconds=seconds)
    return save_frame(directory,binding,spec,source,runtime,row,image,wave,observed)

def collect_outputs(paths):
    outputs={}
    for path in paths:
        frame=c.read(path);outputs[str(path)]=c.sha(path);outputs[frame['archive']]=frame['archive_sha256']
    return outputs

def registration(out,stage,bindings,runtime,population):
    value=dict(status='RECEIVER_COST_REGISTERED',stage=stage,protocol=k.PROTOCOL,bindings=bindings,
        runtime=runtime,sources=population['sources'],expected_specs=k.specs(stage),training_updates=0,
        source_population='original_calibration_indices_0_1_2_3',development_images_read=False,holdout_access=False)
    path=out/(stage+'_registration.json');c.seal(path,value);return c.identity(value),path

def finish(out,stage,reg_path,binding,bindings,frames,warmups,progress):
    k.require(len(frames)==len(k.specs(stage)),'Incomplete fixed receiver benchmark')
    outputs=collect_outputs([*frames,*warmups]);outputs[str(reg_path)]=c.sha(reg_path)
    outputs.update({str(Path(p).parent/'warmup.json'):c.sha(Path(p).parent/'warmup.json') for p in frames})
    c.verify(bindings);c.verify(outputs)
    c.seal(out/(stage+'_completion.json'),dict(status=k.STATUS[stage],binding=binding,registration_sha256=c.sha(reg_path),
        rows=len(frames),frames=[str(p) for p in frames],warmup_frames=[str(p) for p in warmups],
        protocol=k.PROTOCOL,same_population_all5=True,synthetic=False,bindings=bindings,outputs=outputs))
    k.load_stage(out,stage);progress('COMPLETE',rows=len(frames))

def run_own(root,config_path,out,token,progress,guard,handler):
    c.gpu_available()
    from own_controls_cost_native import OwnReceivers
    native=OwnReceivers(root,handler);guard();runtime=k.gpu_identity(native.torch)
    sources=native.sources;archive=out/'calibration_originals.npz'
    digest=c.atomic_npz(archive,source_rgb=np.stack([r['rgb'] for r in sources]))
    population=dict(protocol=k.PROTOCOL,source_indices=[0,1,2,3],sources=[{kk:vv for kk,vv in r.items() if kk!='rgb'} for r in sources],
        archive=str(archive),archive_sha256=digest,original_population_proof=native.population_proof,
        bindings=native.bindings,development_images_read=False,holdout_access=False)
    pp=out/'population.json';c.seal(pp,population)
    bindings={**k.code_bindings(),**native.bindings,str(config_path):c.sha(config_path),str(pp):c.sha(pp),str(archive):digest}
    binding,rp=registration(out,'own',bindings,runtime,population)
    warms={};frames=[];allwarm=[]
    for spec in k.specs('own'):
        guard();c.gpu_available();native.native.common.check()
        source=sources[spec['source_index']];path=out/'own_frames'/name(spec)/'frame.json'
        if path.exists():
            value=k.validate_frame(c.read(path),binding,spec,source,runtime)
            link=c.read(path.parent/'warmup.json');warm=Path(link['path'])
            k.require(c.sha(warm)==link['sha256'],'Original warmup receipt changed')
            allwarm.append(warm)
        else:
            group=(spec['N'],spec['method'])
            if group not in warms:
                warm_spec=dict(spec,source_index=0,snr_db=13)
                warm_path,_=measured_own(native,warm_spec,sources[0],runtime,out/'warmups'/token/name(warm_spec),binding)
                warms[group]=warm_path;allwarm.append(warm_path);guard()
            link=path.parent/'warmup.json';c.seal(link,dict(path=str(warms[group]),sha256=c.sha(warms[group])))
            path,value=measured_own(native,spec,source,runtime,path.parent,binding)
        frames.append(path);progress('RUNNING',rows=len(frames),rows_expected=72)
    native.check()
    finish(out,'own',rp,binding,bindings,frames,sorted(set(allwarm)),progress)

def run_external(root,config_path,config,out,token,progress,guard,handler):
    import torch
    from external_eval import selected_gate,reconstruct_frame,code_bindings
    from swin_train import configure_runtime
    from swin_protocol import configure_phy
    from swin_replay import load_selected
    from hifi_swin_sampler import FrozenHiFiSwinReceiver
    _,own_reg,own_bindings=k.load_stage(out,'own')
    pp=out/'population.json';population=c.read(pp);sources=k.validate_population(population,population['archive'])
    selected,selection=selected_gate(config);c.gpu_available();configure_runtime();configure_phy(root)
    runtime=k.gpu_identity(torch)
    k.require(all(runtime[x]==own_reg['runtime'][x] for x in ('uuid','name','driver')),'Benchmark must use the same GPU/driver as own receivers')
    bindings={**k.code_bindings(),**code_bindings(),**selection,**own_bindings,str(config_path):c.sha(config_path),
              str(pp):c.sha(pp),population['archive']:population['archive_sha256']}
    binding,rp=registration(out,'external',bindings,runtime,population);guard()
    codec,checked=load_selected(config['training_output'],config['swin_vendor'])
    k.require(checked==selected,'Selected Swin changed while loading')
    receiver=FrozenHiFiSwinReceiver(codec.native,config['diffcom_vendor'],config['adm_checkpoint'])
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    warms={};frames=[];allwarm=[];native_outputs={}
    def pair(spec,directory,source):
        base={kk:vv for kk,vv in spec.items() if kk!='method'}
        native_path=directory/'native'/'frame.json'
        if native_path.exists():raw=c.validate_frame_receipt(c.read(native_path),binding,base,source)
        else:raw=reconstruct_frame(codec,receiver,source,base,directory/'native',binding,selected)
        native_outputs[str(native_path)]=c.sha(native_path);native_outputs[raw['archive']]=raw['archive_sha256']
        with np.load(raw['archive'],allow_pickle=False) as data:
            images=data['images'].copy();wave=data['transmitted'].copy();observed=data['observed'].copy()
        result={}
        for j,method in enumerate(c.METHODS):
            actual=dict(base,method=method);row=row_base(actual,source,runtime)
            row.update({kk:raw['rows'][j][kk] for kk in ('RX_seconds','header_accepted','NFE','t_start','complete_posterior_schedule')})
            row['t_start']=row['t_start'] or 0
            path,value=save_frame(directory/method,binding,actual,source,runtime,row,images[j],wave,observed,
                hifi_receipt=raw['hifi_receipt'],native_receipt=str(native_path),native_receipt_sha256=c.sha(native_path))
            result[method]=path
        return result
    for i in range(4):
        for n in (1024,2048):
            for s in (1,7,13):
                guard();c.gpu_available();source=sources[i];spec=dict(source_index=i,N=n,snr_db=s,noise_seed=4101)
                directory=out/'external_frames'/f'{i:04d}_N{n}_SNR{s}'
                if all((directory/m/'frame.json').exists() for m in c.METHODS):
                    for m in c.METHODS:
                        path=directory/m/'frame.json';value=k.validate_frame(c.read(path),binding,dict(spec,method=m),source,runtime)
                        link=c.read(path.parent/'warmup.json');k.require(c.sha(link['path'])==link['sha256'],'Warmup changed');allwarm.append(Path(link['path']))
                    paths={m:directory/m/'frame.json' for m in c.METHODS}
                else:
                    if n not in warms:
                        warms[n]=pair(dict(source_index=0,N=n,snr_db=13,noise_seed=4101),out/'warmups'/token/f'external_N{n}',sources[0])
                        allwarm+=list(warms[n].values());guard()
                    for m in c.METHODS:c.seal(directory/m/'warmup.json',dict(path=str(warms[n][m]),sha256=c.sha(warms[n][m])))
                    paths=pair(spec,directory,source)
                frames+=list(paths.values());progress('RUNNING',rows=len(frames),rows_expected=48)
    # Native receipts include complete-schedule and shared-waveform verification.
    for path in [*frames,*allwarm]:
        frame=c.read(path);npth=Path(frame['native_receipt']);raw=c.read(npth)
        native_outputs[str(npth)]=c.sha(npth);native_outputs[raw['archive']]=raw['archive_sha256']
    bindings.update(native_outputs)
    finish(out,'external',rp,binding,bindings,frames,sorted(set(allwarm)),progress)

def report(out,progress):
    rows=[];bindings=k.code_bindings();runtime=[]
    for stage in k.STATUS:
        rr,reg,bb=k.load_stage(out,stage);rows+=rr;bindings.update(bb);runtime.append(reg['runtime'])
    k.require(all(all(r[x]==runtime[0][x] for x in ('uuid','name','driver')) for r in runtime),'Receiver benchmark GPUs differ')
    summary=k.summarize(rows);outputs={}
    for filename,rr in (('per_frame.csv',rows),('summary.csv',summary)):
        p=out/filename;c.write_csv(p,rr);outputs[str(p)]=c.sha(p)
    done=dict(status='OWN_CONTROLS_RECEIVER_COST_COMPLETE',rows=120,groups=30,same_population_all5=True,
        protocol=k.PROTOCOL,summary=summary,bindings=bindings,outputs=outputs,**{kk:vv for kk,vv in k.PROTOCOL.items() if kk not in ('version','same_population_all5')})
    c.seal(out/'completion.json',done);k.verify_report(out/'completion.json');progress('COMPLETE',rows=120,groups=30)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('own','external','report'),required=True);parser.add_argument('--launch-id',default=None)
    args=parser.parse_args();root=args.root.resolve();config_path=args.config.resolve();config=c.read(config_path)
    from external_eval import validate_config
    k.require(validate_config(config)==root,'Explicit root/config mismatch')
    out=k.folder(root);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'benchmark.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=[False]
    def handler(*_):stop[0]=True
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    token=args.launch_id or str(uuid.uuid4())
    def guard():
        if stop[0]:raise c.PauseRequested('Requested pause at a complete receiver frame boundary')
    def progress(status,**fields):c.write(out/(args.stage+'_status.json'),dict(status=status,pid=os.getpid(),launch_id=token,
        safe_pause_handler_installed=True,updated=time.time(),**fields))
    progress('STARTING');failure=out/(args.stage+'_failure.json')
    try:
        k.require(not failure.exists(),'Previous benchmark failure requires review')
        if (out/(args.stage+'_completion.json')).exists() and args.stage!='report':
            k.load_stage(out,args.stage);progress('COMPLETE');return
        guard()
        if args.stage=='own':run_own(root,config_path,out,token,progress,guard,handler)
        elif args.stage=='external':run_external(root,config_path,config,out,token,progress,guard,handler)
        else:report(out,progress)
    except c.PauseRequested as exc:progress('PAUSED',reason=str(exc));raise SystemExit(75)
    except BaseException as exc:
        if isinstance(exc,SystemExit):raise
        if not failure.exists():c.write(failure,dict(status='FAILED_REQUIRES_REVIEW',error=repr(exc),automatic_restart_allowed=False))
        progress('FAILED_REQUIRES_REVIEW',error=repr(exc));raise

if __name__=='__main__':main()
