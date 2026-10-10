#!/usr/bin/env python3
"""Register and fill only fixed16 missing Swin/BPG display frames; never score."""
from __future__ import annotations
import argparse
import base64
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

FIXED=(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
SNRS=(1,10,19)
METHODS=('SWIN80K','ADAPTIVE_BPG')
SCHEMA='WCL_T5_MISSING_FIXED16_FRAMES_V1'
CHECKPOINT='8857b8c5ed91a5c316084e8c252a8e2168e424a767378389436d96268e65ba21'
STOP=False
def stop(*_):
    global STOP;STOP=True
def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(d):
    require(sha(d['path'])==d['sha256'],'Changed pinned input: '+d['path']);return read(d['path'])
def verify(bindings):
    for p,h in bindings.items():require(sha(p)==h,'Changed binding: '+p)
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(canonical(v)+'\n');f.flush();os.fsync(f.fileno())
def digest_array(a):
    import numpy as np
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+b':'+str(a.shape).encode()+b'\0'+a.tobytes()).hexdigest()
def source_pixels(record):
    import numpy as np
    require(sha(record['archive'])==record['archive_sha256'],'Changed source archive')
    with np.load(record['archive'],allow_pickle=False) as z:pixels=z['pixels'].copy()
    require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256) and
        hashlib.sha256(pixels.tobytes()).hexdigest()==record['preprocessing_id'],'Original fixed16 source pixels differ')
    return pixels
def key(method,index,snr):return f'{method}_source{index:04d}_N1024_snr{snr}_seed2001'
@contextlib.contextmanager
def lock(path):
    import fcntl
    with Path(path).open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def original_adaptive(path):
    a=read(path);require(a['schema']=='A5_ADAPTIVE_FIXED16_RECEIVE_REQUEST_V1' and a['source_indices']==list(FIXED)
        and a['SNRs']==[13] and a['noise_seeds']==[2001],'Exact completed A5 fixed16 request required')
    q=checked(a['original_adaptive_request']);freeze=checked(a['original_adaptive_freeze']);qual=checked(a['original_adaptive_qualification'])
    for ref,child in (('original_adaptive_freeze_completion','original_adaptive_freeze'),
                      ('original_adaptive_qualification_completion','original_adaptive_qualification')):
        done=checked(a[ref]);require(done['status']=='ADAPTIVE_BPG_STAGE_COMPLETE_V1' and done['actual_children_waited']
            and done['worker_exit_codes']==[0] and done['request_sha256']==a['original_adaptive_request']['sha256']
            and done['outputs'][a[child]['path']]==a[child]['sha256'],'Original actual-wait qualification/freeze differs')
    require(freeze['request_sha256']==qual['request_sha256']==a['original_adaptive_request']['sha256']
        and freeze['calibration_source_count']==100 and freeze['noise_count']==3 and not freeze['holdout_used_for_selection'],
        'Frozen calibration-only adaptive policy required')
    require({s:freeze['selected_profile_ids'][str(s)] for s in SNRS}=={1:3000,10:3007,19:3013},'Original adaptive winners changed')
    require(len(q['catalogue'])==15,'Retain all15 original header profiles and source capacities')
    verify(q['source_bindings']);verify(q['phy_dependency_bindings'])
    for d in (q['python'],q['bpgenc'],q['bpgdec']):require(sha(d['path'])==d['sha256'],'Original executable changed')
    rule=checked(q['source_rule']);require(rule['resolutions']==[256,128,64,32]
        and rule['coarse_qps']==[0,8,16,24,32,40,48,51],'Frozen adaptive source rule differs')
    done=read(Path(a['out'])/'completion.json')
    require(done['status']=='A5_ADAPTIVE_FIXED16_ACTUAL_CHILDREN_WAIT_ZERO' and done['actual_children_waited']
        and done['worker_exit_codes']==[0,0] and done['source_count']==16 and done['frame_count']==16
        and done['SNRs']==[13] and done['noise_seeds']==[2001] and done['request_sha256']==sha(path)
        and done['independent_ledger']['unresolved']==0,'Completed old actual A5 cache required')
    return a,q,freeze,qual,done

def inspect_swin(path,index,snr,record):
    p=Path(path)
    if not p.exists():
        require(not (p.parent/'reconstructions.npz').exists(),'Unreceipted existing Swin archive; manual review required')
        return None
    f=read(p);spec=f['frame'];row=f['rows'][0]
    require(spec==dict(source_index=index,N=1024,snr_db=snr,noise_seed=2001)
        and f['source_id']==record['source_id'] and len(f['rows'])==1
        and row['selected_checkpoint_sha256']==CHECKPOINT and row['selected_step']==80000,'Existing Swin frame identity differs')
    require(sha(f['archive'])==f['archive_sha256'] and f['synthetic'] is False,'Existing Swin archive differs')
    return dict(receipt=pin(p),archive=dict(path=f['archive'],sha256=f['archive_sha256']),array_key='images',array_slot=0)

def register(args):
    root=Path(args.root).resolve();out=Path(args.out).resolve()
    require(out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009') and not out.exists(),'Fresh independent T5 output required')
    require(0<args.max_seconds<=3600 and time.time()<args.deadline_unix<=time.time()+14400,'Finite near-term display replay window')
    a,q,freeze,qual,old_done=original_adaptive(args.a5_request)
    a3=checked(a['a3_request']);require(a3['records']==a['records'] and not a3['holdout_used'],'Frozen development sources differ')
    baseline=read(args.baseline_request)
    require(baseline['Swin_identity']['selected_checkpoint_sha256']==CHECKPOINT and baseline['Swin_identity']['selected_step']==80000,'Exact original Swin80k identity')
    bindings={str(Path(__file__).resolve()):sha(__file__),str(Path(__file__).with_name('t2_ledger.py').resolve()):sha(Path(__file__).with_name('t2_ledger.py'))}
    inputs={str(Path(args.a5_request).resolve()):sha(args.a5_request),str(Path(args.baseline_request).resolve()):sha(args.baseline_request)}
    inputs.update({d['path']:d['sha256'] for k,d in a.items() if isinstance(d,dict) and set(d)=={'path','sha256'}})
    inputs[str(Path(a['out'])/'completion.json')]=sha(Path(a['out'])/'completion.json')
    display_manifest=root/'outputs/MAIN-RAW64-20261007/development_fixed16_cached_display_v2/examples_manifest.json'
    inputs[str(display_manifest)]=sha(display_manifest)
    native=baseline['swin_native_build']
    for d in native.values():require(sha(d['path'])==d['sha256'],'Existing Swin native build missing/changed');inputs[d['path']]=d['sha256']
    runtime=Path(baseline['swin_train']).parent
    paths=[runtime/(n+'.py') for n in ('swin_train','swin_protocol','swin_model','swin_data','swin_state','swin_replay')]
    paths += [Path(baseline['swin_fixed_adapter']),root/'src/var_comm/scale_channel.py',root/'src/var_comm/__init__.py']
    old_bindings={}
    for name in ('bindings','source_bindings','frozen_source_bindings','input_bindings'):
        if isinstance(baseline.get(name),dict):old_bindings.update(baseline[name])
    for p in paths:
        require(old_bindings.get(str(p))==sha(p),'Original Swin dependency is not checksum-bound: '+str(p));bindings[str(p)]=sha(p)
    inputs[baseline['swin_python']]=sha(baseline['swin_python'])
    view=Path(baseline['swin_training_view']);reg=read(view/'registration.json');selected=read(view/'selected_swin.json')
    require(selected['step']==80000 and selected['checkpoint_sha256']==CHECKPOINT and sha(selected['checkpoint'])==CHECKPOINT,'Frozen checkpoint bytes changed')
    inputs[selected['checkpoint']]=CHECKPOINT
    for name in ('registration.json','selected_swin.json','qualification.json','completion.json'):inputs[str(view/name)]=sha(view/name)
    for p,h in reg['bindings'].items():
        if p.startswith(baseline['swin_vendor']+'/'):require(sha(p)==h,'Original Swin vendor changed');bindings[p]=h
    frames=[];caches=[];reuse=[]
    for ordinal,record in enumerate(a['records']):
        index=record['source_index'];require(index==FIXED[ordinal],'Fixed16 ordering differs');source_pixels(record)
        inputs[record['archive']]=record['archive_sha256']
        cp=Path(a['out'])/'codec/development'/f'{ordinal:04d}'/'completion.json'
        require(old_done['outputs'].get(str(cp))==sha(cp),'Existing complete source-codec cache missing/changed: '+str(cp))
        coded=read(cp)
        require(coded['status']=='ADAPTIVE_BPG_SOURCE_CODEC_COMPLETE_V1' and coded['source_index']==ordinal
            and coded['source_id']==record['source_id'] and coded['preprocessing_id']==record['preprocessing_id']
            and coded['request_sha256']==sha(args.a5_request) and len(coded['fits'])==15,'Existing adaptive source-codec identity differs')
        cache=dict(ordinal=ordinal,source_index=index,completion=pin(cp));inputs[str(cp)]=sha(cp);caches.append(cache)
        for snr in SNRS:
            p=root/'outputs/EXTERNAL-COMPARISON-20261004/fixed80k_revision/swin_early_release/evaluation/frames'/f'source{index:04d}_N1024_snr{snr}_seed2001'/'frame.json'
            existing=inspect_swin(p,index,snr,record)
            require(snr!=1 or existing is not None,'Expected existing Swin1 missing; do not silently extend32-frame budget')
            frame=dict(method='SWIN80K',source_index=index,ordinal=ordinal,snr_db=snr,noise_seed=2001,key=key('SWIN80K',index,snr),
                action='REUSE' if existing else 'GENERATE',old_expected_receipt=str(p),existing=existing)
            frames.append(frame)
            if existing:
                for field in ('receipt','archive'):inputs[existing[field]['path']]=existing[field]['sha256']
            profile_id=freeze['selected_profile_ids'][str(snr)];profile=next(p for p in q['catalogue'] if p['profile_id']==profile_id)
            fit=coded['fits'][str(profile_id)];require(fit['capacity_bytes']==profile['capacity_bytes'],'Original capacity-fit mismatch')
            selected=fit['selected']
            if selected:
                require(coded['outputs'].get(selected['stream'])==selected['stream_sha256']==sha(selected['stream'])
                    and selected['complete_BPG_bytes']==Path(selected['stream']).stat().st_size<=profile['capacity_bytes'], 'Cached selected complete BPG bytes differ')
                inputs[selected['stream']]=selected['stream_sha256']
            frames.append(dict(method='ADAPTIVE_BPG',source_index=index,ordinal=ordinal,snr_db=snr,noise_seed=2001,
                key=key('ADAPTIVE_BPG',index,snr),action='GENERATE',profile_id=profile_id,counter=ordinal,source_fit=fit,source_codec=pin(cp)))
        # All existing continuous/raw/direct display sources are pinned, never inferred.
        entries=[('P1024',1,root/'outputs/MAIN-RAW64-20261007/p1_replay_r2/images'/f'{index:04d}.npz','P1024_snr1_noise2001',None),
            ('P1024',10,root/'outputs/PRIOR-AWARE-UEP-20261004-V1/p1024_scores/reconstructions'/f'{index:04d}.npz','images',6),
            ('P1024',19,root/'outputs/CONTENT-REAL-64QAM-20261006/H/p600_replay_v1/images'/f'{index:04d}.npz','P1024_snr19_noise2001',None),
            ('RAW64_WHOLE_PARTIAL_DIRECT',None,root/'outputs/MAIN-RAW64-20261007/development_same_rx_images_r2/images'/f'{index:04d}.npz',None,None)]
        for method,snr,p,k,slot in entries:
            require(p.is_file(),'Existing display source missing (no automatic replay): '+str(p));inputs[str(p)]=sha(p)
            reuse.append(dict(method=method,snr_db=snr,source_index=index,archive=pin(p),array_key=k,array_slot=slot,
                noise_seed=6201 if method.startswith('RAW') else 2001))
    count={m:sum(f['method']==m and f['action']=='GENERATE' for f in frames) for m in METHODS}
    require(count['SWIN80K']<=32 and count['ADAPTIVE_BPG']==48,'Minimal missing-work budget exceeded')
    r=dict(schema=SCHEMA,root=str(root),out=str(out),source_indices=list(FIXED),snrs=list(SNRS),noise_seeds=[2001],N=1024,
        population='development_fixed16',holdout_used=False,records=a['records'],frames=frames,reuse=reuse,source_caches=caches,
        original_a5_request=pin(args.a5_request),original_baseline_request=pin(args.baseline_request),source_bindings=bindings,input_bindings=inputs,
        frame_caps=count,packet_caps={'SWIN80K':count['SWIN80K'],'ADAPTIVE_BPG':96},total_frame_cap=sum(count.values()),
        packet_cap=count['SWIN80K']+96,new_source_encodings=0,new_metric_calls=0,new_bootstrap_calls=0,training_updates=0,policy_selection=False,
        max_seconds=args.max_seconds,deadline_unix=args.deadline_unix,stop_files=[str(root/'STOP'),str(out/'STOP')],
        Swin19_note="19 dB is outside Swin's training and calibration range.",new_entropy_column=False,
        bpg_noise_namespace='A3_DEVELOPMENT_TIMING',bpg_counter='original fixed16 ordinal at each SNR',
        method_noise_pairing='External seed2001; preserve each original method noise namespace. Raw mechanism uses original6201; no same-waveform claim.',
        visual_lock=str(root/'outputs/CONTENT-REAL-64QAM-20261006/shared_visual.lock'),auto_resume=False)
    out.mkdir(parents=True);save(out/'request.json',r)
    result=dict(status='REGISTERED_NOT_RUN',request=pin(out/'request.json'),frame_caps=count,packet_caps=r['packet_caps'],new_source_encodings=0,
        commands={m:[baseline['swin_python'] if m=='SWIN80K' else q['python']['path'],'-B',str(Path(__file__).resolve()),'run','--request',str(out/'request.json'),'--method',m] for m in METHODS})
    save(out/'registration.json',result);return result

def verify_request(path):
    r=read(path);require(r['schema']==SCHEMA and r['source_indices']==list(FIXED) and r['snrs']==list(SNRS)
        and r['noise_seeds']==[2001] and r['N']==1024 and r['total_frame_cap']<=80 and r['packet_cap']<=128
        and not r['new_source_encodings'] and not r['new_metric_calls'] and not r['policy_selection'],'Frozen display request differs')
    verify(r['source_bindings']);verify(r['input_bindings']);return r
def guard(r,started):
    require(not STOP and time.time()<r['deadline_unix'] and time.monotonic()-started<r['max_seconds']
        and not any(Path(p).exists() for p in r['stop_files']),'STOP/deadline reached; no automatic retry')
def archive_frame(path,image,pixels,tx=None,rx=None):
    import numpy as np
    from PIL import Image
    image=np.asarray(image,dtype=np.float32)
    require(image.shape==(3,256,256) and np.isfinite(image).all() and ((image>=0)&(image<=1)).all(),'Finite RGB image required')
    path.mkdir(parents=True,exist_ok=True);archive=path/'reconstruction.npz'
    require(not archive.exists(),'Unreceipted output retained; never overwrite')
    arrays=dict(rgb=image,source_rgb=pixels.astype(np.float32)/np.float32(255))
    if tx is not None:arrays.update(transmitted=tx,observed=rx)
    np.savez_compressed(archive,**arrays)
    image_path=path/'reconstruction.png';Image.fromarray(np.rint(image.transpose(1,2,0)*255).astype(np.uint8)).save(image_path)
    return {str(p):sha(p) for p in (archive,image_path)}

class BpgLedger:
    def __init__(self,ledger):self.ledger=ledger
    def decode(self,phase,event_id,kind,request,callback):
        return self.ledger.call(dict(event_id=event_id,phase=phase,kind=kind),request,callback)

def run_bpg(r,path,frames,out,ledger,started):
    import numpy as np
    a,q,freeze,qual,_=original_adaptive(r['original_a5_request']['path'])
    require(os.path.abspath(sys.executable)==q['python']['path'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Original CPU-only adaptive environment')
    os.setpriority(os.PRIO_PROCESS,0,15)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):require(os.environ.get(name)=='2','Two CPU threads must be set before launch')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    dirs=[str(Path(p).parent) for p in q['source_bindings'] if p.endswith(('/adaptive_codec.py','/bpg_codec.py','/bpg_phy.py'))]
    sys.path[:0]=list(dict.fromkeys(dirs))
    from adaptive_codec import ReceiverCodec
    from bpg_phy import Link
    cfg=dict(q,out=str(out),deadline_unix=r['deadline_unix'],stop_files=r['stop_files'],records={'development':r['records']},
        original_native_codec_completion={'development':a['empty_native_cache_index']},original_native_root=a['disabled_native_cache_root'],
        _path=str(Path(path).resolve()),_sha256=sha(path),_worker=0)
    link=Link(cfg);require(canonical(link.identity)==canonical(qual['backend_identity']) and canonical(link.plans)==canonical(qual['layouts']),
        'Real constructor must match already-qualified backend and all15 actual layouts')
    receiver=ReceiverCodec(cfg,'development');rows=[];charged=BpgLedger(ledger)
    for f in frames:
        guard(r,started);record=r['records'][f['ordinal']];pixels=source_pixels(record);directory=out/'frames'/f['key']
        save(directory/'reservation.json',dict(frame=f,request_sha256=sha(path),reserved_before_transmit=True))
        profile=next(p for p in q['catalogue'] if p['profile_id']==f['profile_id']);selected=f['source_fit']['selected']
        image=np.full((3,256,256),.5,dtype=np.float32);tx=rx=None;outcome=None;status='SOURCE_UNFIT';receipt=None
        if selected:
            payload=Path(selected['stream']).read_bytes();require(hashlib.sha256(payload).hexdigest()==selected['stream_sha256'],'Selected cached stream changed')
            tx=link.transmit(payload,profile,f['counter']);rx=link.noisy(tx,r['bpg_noise_namespace'],f['snr_db'],f['source_index'],2001)
            outcome=link.receive(rx,f['snr_db'],f['counter'],charged,'development',f['key']);status=outcome['status']
            if outcome['body'] and outcome['body']['parser_accepted']:
                received=base64.b64decode(outcome['body']['payload_b64']);decoded,status,receipt=receiver.decode(received)
                if decoded is not None:image=decoded
                outcome['body']=dict(outcome['body'],received_bytes_sha256=hashlib.sha256(received).hexdigest())
                outcome['body'].pop('payload_b64',None)
        outputs=archive_frame(directory,image,pixels,tx,rx)
        if receipt:outputs[receipt]=sha(receipt);outputs.update(read(receipt)['outputs'])
        row=dict(frame=f,status=status,source_id=record['source_id'],preprocessing_id=record['preprocessing_id'],received=outcome,
            source_codec=f['source_codec'],selected_source_stream=selected,new_source_encodings=0,receiver_arguments='actual parsed complete BPG bytes only',
            transmitted_sha256=None if tx is None else digest_array(tx),observed_sha256=None if rx is None else digest_array(rx),
            actual_energy=None if tx is None else float(np.square(tx.astype(np.float64)).sum()),outputs=outputs,
            request_sha256=sha(path),synthetic=False)
        save(directory/'completion.json',row);rows.append(row)
    return rows,dict(torch=torch.__version__,device='CPU',threads=2,interop_threads=1,backend_identity=link.identity,new_source_encodings=0)

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def run_swin(r,path,frames,out,ledger,started):
    import numpy as np
    b=checked(r['original_baseline_request'])
    require(os.path.abspath(sys.executable)==b['swin_python'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Original isolated deterministic single-GPU environment required')
    sys.path[:0]=b['swin_pythonpath']
    import torch
    import swin_train as train
    import swin_protocol as protocol
    import swin_replay as replay
    for module in (train,protocol,replay):require(r['source_bindings'].get(str(Path(module.__file__).resolve()))==sha(module.__file__),'Original module import differs')
    train.available_gpu();train.configure_runtime();protocol.configure_phy(r['root'])
    library=protocol._phy().load_native();require(str(library._name)==b['swin_native_build']['library']['path']
        and sha(library._name)==b['swin_native_build']['library']['sha256'],'Use original existing native header library, no rebuild')
    adapter=load_module(b['swin_fixed_adapter'],'_t5_original_fixed80_adapter');model,selected=adapter.load_selected(b['swin_training_view'],b['swin_vendor'],'cuda:0')
    require(selected['step']==80000 and selected['checkpoint_sha256']==CHECKPOINT,'Exact paused80k checkpoint');initial=train.model_hash(model)
    original=replay.decode_header;active={};actual_context=[];rows=[]
    def decode(observed,snr,N):
        require(not actual_context,'Exactly one header decoder call per registered frame')
        def call():
            result=original(observed,snr,N);actual_context.append(result);return dataclasses.asdict(result)
        ledger.call(dict(event_id=active['key']+':header',kind='Swin_header',phase='development'),
            dict(observed_sha256=digest_array(observed),snr_db=snr,N=N),call)
        require(len(actual_context)==1,'Actual header decoder must execute once; no unresolved/cached restart')
        return actual_context[0]
    replay.decode_header=decode
    try:
        with torch.no_grad():
            for f in frames:
                guard(r,started);train.available_gpu();record=r['records'][f['ordinal']];pixels=source_pixels(record)
                directory=out/'frames'/f['key'];save(directory/'reservation.json',dict(frame=f,request_sha256=sha(path),reserved_before_neural_encoder=True))
                active.clear();active.update(f);actual_context.clear()
                image=torch.from_numpy(pixels.copy()).to('cuda:0').float().div(255)[None]
                data,power,indices=model.encode_data(image,f['snr_db'],protocol.layout(1024)['channels'])
                tx=protocol.transmit_frame(data[0].cpu().numpy(),float(power[0]),tuple(indices[0].cpu().tolist()),1024)
                rx=tx+protocol.standard_noise(record['source_id'],2001,1024,f['snr_db'])/10**(f['snr_db']/20)
                before=digest_array(rx);rgb,context=replay.receive(model,rx,1024,f['snr_db'])
                require(before==digest_array(rx) and len(actual_context)==1,'Receiver mutated observations or omitted actual header')
                outputs=archive_frame(directory,rgb,pixels,tx,rx)
                row=dict(frame=f,source_id=record['source_id'],preprocessing_id=record['preprocessing_id'],status='DECODED' if context.accepted else 'HEADER_REJECT_GRAY',
                    received=dataclasses.asdict(context),receiver_arguments='observed full N1024 IQ, N, SNR only',selected_step=80000,checkpoint_sha256=CHECKPOINT,
                    transmitted_sha256=digest_array(tx),observed_sha256=before,actual_energy=float(np.square(tx).sum()),outputs=outputs,
                    out_of_training_and_calibration_range=f['snr_db']==19,request_sha256=sha(path),synthetic=False)
                save(directory/'completion.json',row);rows.append(row)
        require(train.model_hash(model)==initial,'Frozen Swin model changed')
    finally:replay.decode_header=original
    return rows,dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),model_sha256=initial,precision='FP32',batch_size=1)

def run(path,method):
    from t2_ledger import Ledger
    r=verify_request(path);require(method in METHODS,'Known method required');out=Path(r['out'])/method;out.mkdir(exist_ok=True)
    frames=[f for f in r['frames'] if f['method']==method and f['action']=='GENERATE'];started=time.monotonic()
    require(len(frames)==r['frame_caps'][method],'Registered actual missing-frame coverage changed')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    with lock(out/'owner.lock'):
        if (out/'completion.json').exists():
            done=read(out/'completion.json');require(done['request_sha256']==sha(path),'Completed request differs');verify(done['outputs']);return done
        require(not (out/'attempt.json').exists(),'Existing attempt requires review; never automatically restart')
        # A previously missing old Swin artifact appearing meanwhile blocks replay.
        if method=='SWIN80K':
            for f in frames:require(not Path(f['old_expected_receipt']).exists(),'Existing old output appeared after registration; re-register a smaller job')
        guard(r,started);save(out/'attempt.json',dict(status='STARTED_NOT_COMPLETE',request_sha256=sha(path),method=method,pid=os.getpid(),started_unix=time.time()))
        ledger=Ledger(out/'new_packet_budget.sqlite',sha(path),r['packet_caps'][method]) if frames else None
        try:
            if not frames:rows,runtime=[],{'model_loaded':False}
            elif method=='SWIN80K':
                with lock(r['visual_lock']):rows,runtime=run_swin(r,path,frames,out,ledger,started)
            else:rows,runtime=run_bpg(r,path,frames,out,ledger,started)
            snapshot=ledger.snapshot() if ledger else dict(total=0,unresolved=0,cap=0)
            require(snapshot['unresolved']==0 and len(rows)==len(frames),'Unresolved/missing actual frames')
            if method=='SWIN80K':require(snapshot['total']==len(frames),'One actual header per Swin frame')
            else:require(snapshot['total']==sum(row['received']['packet_calls'] if row['received'] else 0 for row in rows),'Actual BPG packet count differs from frame receipts')
            outputs={str(out/'frames'/row['frame']['key']/'completion.json'):sha(out/'frames'/row['frame']['key']/'completion.json') for row in rows}
            for row in rows:outputs.update(row['outputs'])
            guard(r,started);verify(r['source_bindings']);verify(r['input_bindings'])
            done=dict(status='T5_MISSING_FIXED16_BRANCH_COMPLETE',method=method,request_sha256=sha(path),frame_count=len(rows),
                frame_keys=[f['key'] for f in frames],packet_budget=snapshot,outputs=outputs,runtime=runtime,elapsed_seconds=time.monotonic()-started,
                population='development_fixed16',holdout_used=False,new_source_encodings=0,new_metrics=0,new_bootstrap=0,training_updates=0,policy_selection=False,
                requires_external_actual_wait_exit_receipt=True)
            save(out/'completion.json',done);return done
        except BaseException as error:
            save(out/'failure.json',dict(status='FAILED_PRESERVED_NO_RETRY',error=repr(error),request_sha256=sha(path),
                packet_budget=None if ledger is None else ledger.snapshot()));raise
        finally:
            if ledger is not None:ledger.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('register')
    for n in ('root','out','a5-request','baseline-request'):a.add_argument('--'+n,required=True)
    a.add_argument('--deadline-unix',type=float,required=True);a.add_argument('--max-seconds',type=int,default=3600)
    a=sub.add_parser('run');a.add_argument('--request',required=True);a.add_argument('--method',choices=METHODS,required=True)
    args=p.parse_args();print(canonical(register(args) if args.command=='register' else run(args.request,args.method)),flush=True)
if __name__=='__main__':main()
