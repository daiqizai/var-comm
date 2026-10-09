"""One finite development extension of the already-frozen adaptive BPG baseline.

Only N1024 / 13 dB / seed2001 / fixed16. Original A1 runtime and ledgers are
read-only. Two CPU workers share a new charge-before-decode ledger capped at32.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

FIXED=[0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]
METHOD='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(d):
    require(sha(d['path'])==d['sha256'],'Pinned input changed: '+d['path']);return read(d['path'])
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if p.exists():require(p.read_bytes()==data,'Immutable output differs: '+str(p));return
    tmp=p.with_suffix(p.suffix+'.pending')
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def verify(outputs):
    for p,h in outputs.items():require(sha(p)==h,'Completed output changed: '+p)
def guard(r):
    require(time.time()<r['deadline_unix'],'A5 adaptive deadline reached')
    require(not any(Path(p).exists() for p in r['stop_files']),'A5 adaptive STOP requested')


def configure(path):
    r=read(path);rh=sha(path)
    require(r['schema']=='A5_ADAPTIVE_FIXED16_RECEIVE_REQUEST_V1' and r['worker_sha256']==sha(__file__),'Frozen A5 script/request mismatch')
    require(r['source_indices']==FIXED and r['SNRs']==[13] and r['noise_seeds']==[2001]
        and r['max_seconds']==1800 and r['phase_caps']=={'development':32}
        and r['resources']==dict(workers=2,threads=2,nice=15,affinities=[[10,11],[12,13]])
        and not r['training_updates'] and not r['policy_selection'] and r['new_qualification_calls']==0,'Scope/budget changed')
    q=pin(r['original_adaptive_request']);freeze=pin(r['original_adaptive_freeze']);fn=pin(r['original_adaptive_freeze_completion'])
    qualification=pin(r['original_adaptive_qualification']);qn=pin(r['original_adaptive_qualification_completion']);a3=pin(r['a3_request'])
    require(fn['status']==qn['status']=='ADAPTIVE_BPG_STAGE_COMPLETE_V1'
        and fn['actual_children_waited'] and qn['actual_children_waited']
        and fn['worker_exit_codes']==qn['worker_exit_codes']==[0]
        and fn['outputs'].get(r['original_adaptive_freeze']['path'])==r['original_adaptive_freeze']['sha256']
        and qn['outputs'].get(r['original_adaptive_qualification']['path'])==r['original_adaptive_qualification']['sha256'], 'Actual frozen/qualified parent receipts required')
    require(freeze['request_sha256']==fn['request_sha256']==qualification['request_sha256']==qn['request_sha256']==r['original_adaptive_request']['sha256']
        and freeze['calibration_source_count']==100 and freeze['noise_count']==3
        and not freeze['holdout_used_for_selection'] and freeze['selected_profile_ids']['13']==3009,'Actual 100cal frozen13dB profile required')
    require(a3['source_indices']==FIXED and r['records']==a3['records'] and not a3['holdout_used'],'Original fixed16 development records required')
    for p,h in q['source_bindings'].items():require(sha(p)==h,'Original adaptive runtime changed: '+p)
    for d in [q['bpgenc'],q['bpgdec'],q['python']]:require(sha(d['path'])==d['sha256'],'Original executable changed')
    rule=pin(q['source_rule'])
    require(rule['status']=='SOURCE_RULE_FROZEN_MCS_NOT_SELECTED' and rule['resolutions']==[256,128,64,32]
        and rule['coarse_qps']==[0,8,16,24,32,40,48,51], 'Actual frozen source rule changed')
    empty=pin(r['empty_native_cache_index'])
    require(empty==dict(status='NO_DEVELOPMENT_NATIVE_STREAM_CACHE_AVAILABLE',outputs={},scientific_result=False)
        and not Path(r['disabled_native_cache_root']).exists(),'Native cache must be explicitly absent, never borrowed from another source population')
    dirs=[str(Path(p).parent) for p in q['source_bindings'] if p.endswith(('/adaptive_codec.py','/bpg_codec.py','/bpg_phy.py'))]
    sys.path[:0]=list(dict.fromkeys(dirs))
    cfg=dict(q,out=r['out'],deadline_unix=r['deadline_unix'],stop_files=r['stop_files'],
        records={'development':r['records']},original_native_codec_completion={'development':r['empty_native_cache_index']},
        original_native_root=r['disabled_native_cache_root'],_path=str(Path(path).resolve()),_sha256=rh)
    # Keep ALL original capacities in the source search and ALL qualified header
    # profiles. Trimming either would alter the frozen selection/receiver rule.
    require(cfg['catalogue']==q['catalogue'] and len(cfg['catalogue'])==15,'Frozen source/receiver catalogue changed')
    profile=next(p for p in cfg['catalogue'] if p['profile_id']==3009)
    require(profile==r['profile'],'Frozen selected MCS differs')
    return r,cfg,qualification,profile


def worker(path,index):
    r,cfg,qualification,profile=configure(path)
    require(index in (0,1),'Only two workers')
    os.sched_setaffinity(0,r['resources']['affinities'][index]);os.setpriority(os.PRIO_PROCESS,0,15)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','This is CPU-only')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from adaptive_codec import source_rows,ReceiverCodec
    from codec_stage import source_pixels
    from bpg_phy import Link,rgb_sha
    from bpg_codec import digest
    from supplement_ledger import Ledger
    cfg['_worker']=index;guard(r);link=Link(cfg)
    require(digest(link.identity)==digest(qualification['backend_identity'])
        and digest(link.plans)==digest(qualification['layouts']),'Already-qualified implementation/layout changed')
    ledger=Ledger(r['independent_ledger'],r['phase_caps'],sha(path));receiver=ReceiverCodec(cfg,'development')
    out=Path(r['out']);outputs={}
    for ordinal in range(index,16,2):
        guard(r);record=r['records'][ordinal];i=record['source_index'];require(i==FIXED[ordinal],'Source index order changed')
        cp=out/'sources'/f'{i:04d}.json'
        if cp.exists():
            done=read(cp);require(done['request_sha256']==sha(path) and done['source_id']==record['source_id'],'Existing A5 source differs')
            verify(done['outputs']);outputs[str(cp)]=sha(cp);outputs.update(done['outputs']);continue
        pixels=source_pixels(record);target=pixels.astype(np.float32)/np.float32(255)
        coded=source_rows(cfg,'development',ordinal,pixels)
        require(len(coded['rows'])<=208 and coded['source_id']==record['source_id'],'Frozen source search exceeded cap or changed identity')
        codec_cp=out/'codec/development'/f'{ordinal:04d}'/'completion.json'
        selected=coded['fits']['3009']['selected'];image=np.full((3,256,256),.5,dtype=np.float32)
        status='SOURCE_UNFIT';outcome=None;calls=0;different=None;energy=None;tx_sha=None;rx_sha=None
        local={str(codec_cp):sha(codec_cp)};frame=f'development:{i:04d}:SNR13:noise2001:MCS3009'
        # Counter is a public deterministic frame counter, never a receiver hint.
        counter=ordinal
        if selected:
            payload=Path(selected['stream']).read_bytes();require(hashlib.sha256(payload).hexdigest()==selected['stream_sha256'],'Frozen selected stream changed')
            tx=link.transmit(payload,profile,counter);rx=link.noisy(tx,r['noise_namespace'],13,i,2001)
            tx_sha=link.hp.array_sha(tx);rx_sha=link.hp.array_sha(rx);energy=float(np.square(tx.astype(np.float64)).sum())
            outcome=link.receive(rx,13,counter,ledger,'development',frame);calls=outcome['packet_calls'];status=outcome['status']
            if outcome['body'] and outcome['body']['parser_accepted']:
                received=base64.b64decode(outcome['body']['payload_b64']);different=received!=payload
                decoded,status,receipt=receiver.decode(received);local[receipt]=sha(receipt)
                if decoded is not None:image=decoded
        body=dict(outcome['body']) if outcome and outcome['body'] else None
        if body and body.get('payload_b64') is not None:body['received_BPG_sha256']=hashlib.sha256(base64.b64decode(body.pop('payload_b64'))).hexdigest()
        archive=out/'reconstructions'/f'{i:04d}.npz';archive.parent.mkdir(parents=True,exist_ok=True)
        require(not archive.exists(),'Unreceipted reconstruction archive; do not repeat link automatically')
        with archive.open('xb') as f:np.savez_compressed(f,rgb=image,source_rgb=target);f.flush();os.fsync(f.fileno())
        local[str(archive)]=sha(archive);mse=float(np.square(image.astype(np.float64)-target.astype(np.float64)).mean())
        require(mse>0,'Unexpected perfect frame; explicit PSNR handling required')
        row=dict(status='A5_ADAPTIVE_FIXED_SOURCE_COMPLETE_V1',request_sha256=sha(path),method=METHOD,
            source_index=i,codec_source_ordinal=ordinal,source_id=record['source_id'],preprocessing_id=record['preprocessing_id'],
            evaluation_class_index=record['evaluation_class_index'],population='development',N=1024,snr_db=13,noise_seed=2001,
            noise_namespace=r['noise_namespace'],frame_id=frame,counter=counter,profile_id=3009,q=profile['q'],k=profile['k'],n=profile['n'],
            header_symbols=68,body_symbols=956,padding_symbols=0,capacity_bytes=profile['capacity_bytes'],
            selected_resolution=selected['resolution'] if selected else None,selected_qp=selected['qp'] if selected else None,
            complete_BPG_bytes=selected['complete_BPG_bytes'] if selected else None,source_encoding_fit=selected is not None,
            source_unfit=selected is None,actual_link_executed=selected is not None,link_status=status,
            gray_substitution=status!='BPG_DECODED',display_role='source_unfit_gray_no_received_frame' if selected is None else 'actual_received_output',
            packet_decoder_calls=calls,header=outcome['header'] if outcome else None,body=body,
            transmitted_sha256=tx_sha,observation_sha256=rx_sha,actual_frame_energy=energy,
            undetected_payload_difference=different,psnr_db=-10*math.log10(mse),mse=mse,
            image_sha256=rgb_sha(image),reference_sha256=rgb_sha(target),
            float_reconstruction=dict(path=str(archive),sha256=sha(archive),image_key='rgb',reference_key='source_rgb',dtype='float32',layout='CHW'),
            original_frozen_source_rule=cfg['source_rule'],original_frozen_policy=r['original_adaptive_freeze'],
            source_truth_used_by_receiver=False,policy_selection=False,new_qualification_calls=0,training_updates=0,outputs=local)
        save(cp,row);outputs[str(cp)]=sha(cp);outputs.update(local)
        print(json.dumps(dict(worker=index,source_index=i,link_status=status,selected_resolution=row['selected_resolution'],packet_calls=calls)),flush=True)
    save(out/f'worker_{index}.json',dict(status='A5_ADAPTIVE_FIXED16_SHARD_COMPLETE',request_sha256=sha(path),worker=index,
        source_indices=FIXED[index::2],outputs=outputs))


def run(path):
    import fcntl
    r,cfg,_,_=configure(path);out=Path(r['out']);out.mkdir(parents=True,exist_ok=True)
    from supplement_ledger import Ledger,root_snapshot
    with (out/'owner.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completed=out/'completion.json'
        if completed.exists():
            done=read(completed);require(done['request_sha256']==sha(path),'Completed request changed');verify(done['outputs']);print('Already complete; no new work');return
        guard(r);before=root_snapshot(cfg['original_root_ledger_readonly'])
        ledger=Ledger(r['independent_ledger'],r['phase_caps'],sha(path));ledger.assert_resume()
        attempt=out/('attempt_'+str(time.time_ns()));attempt.mkdir();children=[];handles=[];started=time.monotonic()
        def interrupt(signum,_):raise KeyboardInterrupt('A5 owner signal '+str(signum))
        for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,interrupt)
        try:
            for index in (0,1):
                argv=[cfg['python']['path'],'-B',str(Path(__file__).resolve()),'--request',str(Path(path).resolve()),'--worker',str(index)]
                log=attempt/f'worker_{index}.log';handle=log.open('xb');handles.append(handle)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,env=env);children.append(child)
                save(attempt/f'launch_{index}.json',dict(pid=child.pid,argv=argv,worker=index,log=str(log)))
            while any(p.poll() is None for p in children):
                guard(r);require(time.monotonic()-started<1800,'Owner wall-time cap reached')
                require(not any(p.poll() not in (None,0) for p in children),'Child failed; stop sibling without successor')
                time.sleep(.5)
            codes=[p.wait() for p in children];require(codes==[0,0],'Both workers must exit zero')
            for f in handles:f.flush();os.fsync(f.fileno());f.close()
            outputs={};rows=[]
            for index in (0,1):
                cp=out/f'worker_{index}.json';d=read(cp);require(d['request_sha256']==sha(path),'Worker request mismatch');verify(d['outputs'])
                outputs[str(cp)]=sha(cp);outputs.update(d['outputs'])
            for index in FIXED:rows.append(read(out/'sources'/f'{index:04d}.json'))
            require(len(rows)==16 and [x['source_id'] for x in rows]==[x['source_id'] for x in r['records']],'Fixed16 source count/order mismatch')
            snapshot=ledger.assert_resume();require(snapshot['total']==sum(x['packet_decoder_calls'] for x in rows)<=32,'Finite packet count mismatch')
            after=root_snapshot(cfg['original_root_ledger_readonly']);require(after==before,'Original root ledger changed')
            for p in attempt.iterdir():outputs[str(p)]=sha(p)
            save(completed,dict(status='A5_ADAPTIVE_FIXED16_ACTUAL_CHILDREN_WAIT_ZERO',request_sha256=sha(path),worker_sha256=sha(__file__),
                actual_children_waited=True,worker_exit_codes=codes,source_count=16,source_indices=FIXED,frame_count=16,
                SNRs=[13],noise_seeds=[2001],noise_namespace=r['noise_namespace'],profile_id=3009,
                source_unfit=sum(x['source_unfit'] for x in rows),decoded=sum(x['link_status']=='BPG_DECODED' for x in rows),
                independent_ledger=snapshot,new_PHY_call_cap=32,new_qualification_calls=0,new_metric_calls=0,
                training_updates=0,policy_selection=False,actual_population='development',holdout_used=False,
                root_budget_before=before,root_budget_after=after,original_adaptive_request=r['original_adaptive_request'],
                original_frozen_policy=r['original_adaptive_freeze'],outputs=outputs,elapsed_seconds=time.monotonic()-started))
            print(json.dumps(dict(status='COMPLETE',completion=str(completed),PHY_calls=snapshot['total'])),flush=True)
        except BaseException:
            for p in children:
                if p.poll() is None:p.terminate()
            for p in children:
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            for f in handles:
                if not f.closed:f.close()
            save(attempt/'failure.json',dict(status='FAILED_NO_AUTOMATIC_SUCCESSOR',traceback=traceback.format_exc(),
                actual_children_waited=True,exit_codes=[p.returncode for p in children]));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--worker',type=int);a=p.parse_args()
    if a.worker is None:run(a.request)
    else:
        def stopped(signum,_):raise KeyboardInterrupt('A5 worker signal '+str(signum))
        for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,stopped)
        worker(a.request,a.worker)
