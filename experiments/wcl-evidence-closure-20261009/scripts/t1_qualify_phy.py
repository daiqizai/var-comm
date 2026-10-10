#!/usr/bin/env python3
"""Bound real-LDPC qualification: 96 body + 145 paid-header decodes, cap241.

prepare constructs actual Sionna layouts only. run executes metered decoders.
selfcheck is explicitly non-PHY framing/dispatch engineering, not qualification.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import t1_phy as phy
from t2_ledger import Ledger

SCHEMA='WCL_T1_REAL_PHY_QUALIFICATION_20261009_V1'
CAP=241
SEED=20261009041

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=phy.canonical(obj)+'\n'
    if path.exists():phy.require(path.read_text(encoding='utf-8')==text,'Existing immutable record differs: '+str(path));return
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('x',encoding='utf-8') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
def save_status(path,obj):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8');os.replace(temporary,path)
def csvout(path,rows):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
@contextlib.contextmanager
def lock(path):
    import fcntl
    with Path(path).open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def cpu_environment():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
def guard(r,started):
    phy.require(time.monotonic()-started<r['max_seconds'],'Qualification time cap reached')
    phy.require(not any(Path(p).exists() for p in r['stop_files']),'Explicit project/qualification STOP file')

def prepare(args):
    root=Path(args.root).resolve();out=Path(args.out).resolve();request=Path(args.request or out/'request.json').resolve()
    phy.require(out.is_relative_to(root/'outputs') and 'WCL' in str(out).upper(),'Independent outputs/WCL namespace required')
    phy.require(request.is_relative_to(out),'Qualification request belongs in its new output directory')
    out.mkdir(parents=True,exist_ok=True)
    pins=phy.collect_source_bindings(root)
    rt=phy.Runtime(root);catalogue=rt.build_catalogue()
    phy.require(not rt.torch.cuda.is_initialized(),'Qualification must remain CPU-only')
    reference=Path(rt.reference_qualification)
    config=dict(protocol=phy.PROTOCOL,N=1024,header_symbols=68,body_symbols=956,
      modulation_bits=[2,4,6],nominal_rates=list(phy.RATES),actual_m=[4,5,6,7,8,9],families=list(phy.FAMILIES),
      source_body='uint13 L | L arithmetic bits | known zero information padding | CRC16',
      body_information_k='floor(956*q*nominal_rate)',source_capacity='k-29',pure_arithmetic=True,
      length_fallback='By actual length only: target m7/8/9, then m-1 down to m4; source codec owns selection',
      failure_rule='CRC/length/known-padding rejection gives no source payload; entropy parser/canonical check is separate',
      receiver_profile='actual paid RX header family/m/q/r, even when different from TX',
      power='Fixed constellation averageEs2; measured actual frame energy, no per-frame normalization')
    req=dict(schema=SCHEMA,root=str(root),out=str(out),source_bindings=pins,
      input_bindings={str(reference):phy.sha(reference)},config=config,catalogue=catalogue,
      backend_identity=rt.backend.identity,original_backend_identity=rt.backend.old.identity,
      environment=dict(python=platform.python_version(),python_executable=sys.executable,torch=rt.torch.__version__,
        sionna=rt.backend.old.identity['version'],cpu_threads=rt.torch.get_num_threads(),
        interop_threads=rt.torch.get_num_interop_threads(),cuda_initialized=rt.torch.cuda.is_initialized(),
        deterministic=rt.torch.are_deterministic_algorithms_enabled(),TF32=False),
      execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
      payload_seed=SEED,body_modes=['4_noiseless','4_AWGN60'],body_decoder_calls=96,
      header_ids=list(range(144))+[4095],header_decoder_calls=145,packet_cap=CAP,
      neural_model_calls=0,source_images_read=0,max_seconds=args.max_seconds,
      stop_files=[str(root/'STOP'),str(out/'STOP')],original_ledger_opened=False)
    phy.verify_bindings(pins)
    write(out/'constructor_catalogue.json',catalogue);write(request,req)
    if catalogue['rejected_layouts']:
        write(out/'constructor_rejections.json',dict(status='UNSUPPORTED_LAYOUTS_STOP',rejected=catalogue['rejected_layouts']))
        raise RuntimeError('Actual Sionna rejected a planned layout; receipt preserved, do not silently shrink grid')
    phy.require(len(catalogue['layouts'])==12 and len(catalogue['profiles'])==144,'Required12 layouts and144 profiles')
    return dict(status='PREPARED_ACTUAL_CONSTRUCTORS_NOT_PACKET_QUALIFIED',request=str(request),
                request_sha256=phy.sha(request),layouts=12,profiles=144,planned_packet_decodes=241,
                actual_packet_decodes=0,neural_model_calls=0)

def qualify(args):
    request=Path(args.request).resolve();r=phy.read(request);rh=phy.sha(request);out=Path(r['out'])
    phy.require(r['schema']==SCHEMA and r['payload_seed']==SEED and r['packet_cap']==241,'Frozen qualification scope')
    phy.require(r['body_decoder_calls']==96 and r['header_decoder_calls']==145 and r['header_ids']==list(range(144))+[4095],'Fixed241 decode plan')
    phy.require(r['source_bindings'].get(str(Path(__file__).resolve()))==phy.sha(__file__),'Bound qualifier source differs')
    phy.verify_bindings(r['source_bindings']);phy.verify_bindings(r['input_bindings'])
    with lock(out/'qualify.lock'):
        completion=out/'completion.json'
        if completion.exists():
            c=phy.read(completion);phy.require(c['request_sha256']==rh and c['status']=='PASS','Existing qualification differs')
            phy.verify_bindings(c['outputs']);return dict(status='REUSE_COMPLETE_REAL_PHY_QUALIFICATION',packet_decodes=0,original_packet_decodes=c['packet_decode_count'])
        phy.require(not (out/'failure.json').exists(),'Previous failure preserved; diagnose before a new version')
        started=time.monotonic();rt=phy.Runtime(r['root']);rt.use_catalogue(r['catalogue'])
        phy.require(not rt.torch.cuda.is_initialized(),'Qualification must remain CPU-only')
        phy.require(rt.backend.identity==r['backend_identity'],'Current real backend identity differs')
        ledger=Ledger(out/'qualification_ledger.sqlite',rh,CAP);bodyrows=[];headerrows=[]
        try:
            for layout_index,layout in enumerate(r['catalogue']['layouts']):
                profile=next(p for p in r['catalogue']['profiles'] if p['layout_id']==layout['layout_id'])
                capacity=profile['source_capacity_bits']
                for index in range(8):
                    guard(r,started)
                    rng=np.random.default_rng(np.random.SeedSequence([SEED,layout_index,index]))
                    mode=index%4;length=[2,capacity,max(2,capacity//2),capacity-1][mode]
                    payload=np.zeros(length,dtype=np.uint8) if mode==0 else np.ones(length,dtype=np.uint8) if mode==1 else rng.integers(0,2,length,dtype=np.uint8)
                    counter=layout_index*8+index;body,tx=phy.transmit_body(rt,payload,profile,counter,'T1_PHY_QUALIFICATION')
                    noise=np.zeros_like(body) if index<4 else rng.standard_normal(body.shape).astype(np.float32)*np.float32(1e-3)
                    observed=(body+noise).astype(np.float32)
                    event_id=f'{SCHEMA}/layout{layout_index:02d}/body{index}'
                    rx=phy.receive_body(rt,observed,profile,60.,counter,ledger,event_id,'qualification','T1_PHY_QUALIFICATION')
                    expected=phy.pack_body(payload,profile)
                    correct=bool(rx['crc_accepted'] and rx['parser_accepted'] and rx['payload']==payload.tolist() and np.array_equal(rx['decoded_bits'],expected))
                    header_E=float(np.square(np.asarray(rt.header.transmit(profile['profile_id']),dtype=np.float64)).sum());frame_E=tx['body_energy']+header_E
                    row=dict(event_id=event_id,layout_index=layout_index,layout_id=layout['layout_id'],profile_id=profile['profile_id'],
                      q=profile['q'],nominal_rate=profile['nominal_rate'],k=profile['k'],n=profile['n'],index=index,
                      noiseless=index<4,snr_db=60,payload_bits=length,payload_sha256=phy.array_sha(payload),
                      noise_sha256=phy.array_sha(noise),received_sha256=phy.array_sha(observed),
                      exact_information_equal=bool(np.array_equal(rx['decoded_bits'],expected)),correct=correct,
                      crc_accepted=rx['crc_accepted'],parser_accepted=rx['parser_accepted'],
                      header_symbols=68,body_symbols=956,frame_padding_symbols=0,
                      body_energy=tx['body_energy'],header_energy=header_E,E_frame=frame_E,rho=frame_E/(2*1024),
                      energy_population='qualification random bits; not image scientific energy distribution')
                    write(out/'body_cases'/f'{layout_index:02d}_{index}.json',dict(row=row,tx=tx,rx=rx))
                    bodyrows.append(row);phy.require(correct,'Actual real-LDPC roundtrip failed: '+event_id)
                save_status(out/'status.json',dict(status='RUNNING_REAL_QUALIFICATION',complete_body=len(bodyrows),complete_header=0,ledger=ledger.snapshot()))
                print(phy.canonical(dict(status='LAYOUT_COMPLETE',layout=layout_index,q=profile['q'],rate=profile['nominal_rate'],actual_body_decodes=len(bodyrows))),flush=True)
            for pid in r['header_ids']:
                guard(r,started);wave=np.asarray(rt.header.transmit(pid),dtype=np.float64)
                event_id=f'{SCHEMA}/header{pid:04d}'
                rx=phy.receive_header(rt,wave,60.,ledger,event_id,'qualification')
                known=pid<144
                correct=(rx['header_crc_ok'] is True and rx['header_fields_legal'] is known and rx['header_ok'] is known and rx['profile_id']==(pid if known else None))
                p=phy.select_received_profile(rt,rx)
                row=dict(rx,event_id=event_id,known=known,correct=correct,
                    selected_profile_id=None if p is None else p['profile_id'],header_energy=float(np.square(wave).sum()),
                    received_sha256=phy.array_sha(wave))
                # Preserve the sent test ID separately when the rejected RX ID is None.
                row['test_profile_id']=pid
                write(out/'header_cases'/f'{pid:04d}.json',row);headerrows.append(row)
                phy.require(correct and (p is None or p['profile_id']==pid),'Actual paid-header roundtrip failed: '+event_id)
            snap=ledger.snapshot();phy.require(snap==dict(total=241,unresolved=0,cap=241),'Actual independent qualification count differs')
            phy.require(len(bodyrows)==96 and len(headerrows)==145,'Incomplete241 actual qualification')
            csvout(out/'body_qualification.csv',bodyrows);csvout(out/'header_qualification.csv',headerrows)
            phy.verify_bindings(r['source_bindings']);phy.verify_bindings(r['input_bindings'])
            outputs={str(p):phy.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and (p.parent.name in ('body_cases','header_cases') or p.name in ('body_qualification.csv','header_qualification.csv','constructor_catalogue.json'))}
            c=dict(status='PASS',schema=SCHEMA,request_path=str(request),request_sha256=rh,source_bindings=r['source_bindings'],
              input_bindings=r['input_bindings'],backend_identity=rt.backend.identity,
              catalogue_sha256=r['catalogue']['catalogue_sha256'],profile_count=144,actual_layout_count=12,
              actual_constructor_validation=True,actual_body_decodes=96,actual_header_decodes=145,packet_decode_count=241,
              ledger=snap,ledger_path=str(out/'qualification_ledger.sqlite'),neural_model_calls=0,source_images_read=0,
              original_ledger_opened=False,new_bootstrap=0,elapsed_seconds=time.monotonic()-started,
              implementation_scope='Real LDPC and paid convolutional header; generated random qualification bits only, no image quality conclusion',
              outputs=outputs)
            write(completion,c);save_status(out/'status.json',dict(status='COMPLETE_REAL_PHY_QUALIFICATION',ledger=snap))
            return {k:c[k] for k in ('status','actual_layout_count','actual_body_decodes','actual_header_decodes','packet_decode_count','neural_model_calls','elapsed_seconds')}
        except BaseException as exc:
            save_status(out/'failure.json',dict(status='FAILED_PRESERVED',exception_type=type(exc).__name__,error=str(exc),
                complete_body=len(bodyrows),complete_header=len(headerrows),ledger=ledger.snapshot(),request_sha256=rh))
            raise
        finally:ledger.close()

def selfcheck(args):
    rows=phy.profile_templates();checks=0
    phy.require(len(rows)==144 and {p['family'] for p in rows}==set(phy.FAMILIES),'Paid profile grid')
    phy.require(len({(p['k'],p['n'],p['q']) for p in rows})==12,'12 integer resource templates')
    for p in rows:
        for length in (2,p['source_capacity_bits']//2,p['source_capacity_bits']):
            bits=np.random.default_rng(SEED+p['profile_id']+length).integers(0,2,length,dtype=np.uint8)
            info=phy.pack_body(bits,p);parsed=phy.parse_body(info,p)
            phy.require(parsed['payload']==bits.tolist() and parsed['status']=='PAYLOAD_PARSED','Format roundtrip')
            corrupt=info.copy();corrupt[-1]^=1
            phy.require(phy.parse_body(corrupt,p)['status']=='CRC_REJECT','CRC failure cannot provide payload')
            checks+=2
        payload=np.array([0,1],dtype=np.uint8);info=phy.pack_body(payload,p)
        info[15]=1;info=phy.append_crc(info[:-16])
        phy.require(phy.parse_body(info,p)['invalid_reason']=='NONZERO_KNOWN_PADDING','Known padding validation')
        checks+=1
    parameters=inspect.signature(phy.receive_frame).parameters
    phy.require(not any(k in parameters for k in ('tx_profile','payload','tokens','target','profile_id')),'Receive-frame API must not accept TX truth')
    checks+=1
    phy.require({p['m'] for p in rows}==set(range(4,10)),'Actual fallback m4..9')
    phy.require(all(p['token_count']==30 for p in rows if p['m']==4),'m4 prefix is 30 tokens')
    minimum_capacity=min(p['source_capacity_bits'] for p in rows)
    phy.require(minimum_capacity==927 and 30*24+2<minimum_capacity,'m4 arithmetic worst-case reserve')
    checks+=3
    noise=phy.standard_noise('format_only_source',1)
    phy.require(noise.shape==(1024,2) and noise.dtype==np.float64,'Full-frame standard noise')
    phy.require(np.array_equal(noise,phy.standard_noise('format_only_source',1)),'Noise repeatability')
    phy.require(not np.array_equal(noise,phy.standard_noise('format_only_source',2)),'Independent registered noise seed')
    checks+=3
    result=dict(status='PASS_ENGINEERING_ONLY_NOT_REAL_PHY_QUALIFICATION',checks=checks,profile_templates=144,
        layout_templates=12,actual_encoder_constructors=0,actual_packet_decodes=0,neural_model_calls=0,
        note='No fake backend or decoder. Actual Sionna constructors and241 real decodes are still required remotely.')
    if args.output:write(args.output,result)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--request');p.add_argument('--max-seconds',type=int,default=3600)
    p=sub.add_parser('run');p.add_argument('--request',required=True)
    p=sub.add_parser('selfcheck');p.add_argument('--output')
    args=ap.parse_args()
    if args.command!='selfcheck':cpu_environment()
    result={'prepare':prepare,'run':qualify,'selfcheck':selfcheck}[args.command](args)
    print(phy.canonical(result),flush=True)

if __name__=='__main__':main()
