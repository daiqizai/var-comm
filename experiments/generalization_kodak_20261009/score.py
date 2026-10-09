"""Label-free four-metric scoring of completed frozen Kodak24 reconstructions.

No transmission, reconstruction model, true-class label, training or bootstrap.
Only the existing hash-bound metric backend is loaded, at batch one/FP32.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

METHODS=('VAR_UNCONDITIONAL','P1024','BPG_ADAPTIVE','SWIN80K')
SNRS=(4,10,19)
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction')


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pinned(d):
    require(sha(d['path'])==d['sha256'],'Pinned file changed: '+d['path'])
    return read(d['path'])
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if p.exists():require(p.read_bytes()==data,'Existing output differs: '+str(p));return
    temporary=p.with_suffix(p.suffix+'.pending')
    with temporary.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temporary,p)
def csv_write(path,rows):
    import io
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    data=buf.getvalue().encode();p=Path(path)
    if p.exists():require(p.read_bytes()==data,'Existing CSV differs');return
    p.write_bytes(data)
def load(d,name):
    require(sha(d['path'])==d['sha256'],'Metric backend source changed')
    spec=importlib.util.spec_from_file_location(name,d['path']);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def rgb(np,value):
    x=np.asarray(value)
    require(x.dtype==np.float32 and x.shape==(3,256,256)and np.isfinite(x).all()and x.min()>=0 and x.max()<=1,
            'Expected untouched CHWfloat32 RGB01')
    return np.ascontiguousarray(x)
def rgb_sha(np,value):
    x=rgb(np,value);return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()


def inputs(r):
    require(r['schema']=='KODAK24_FROZEN_FOUR_METRIC_REQUEST_V1'and r['N']==1024
        and r['snr_db']==list(SNRS)and r['metrics']==list(METRICS),'Kodak frozen four-metric scope')
    records=r['records'];require(len(records)==24 and [x['source_index']for x in records]==list(range(24))
        and [x['source_id']for x in records]==[f'kodak/kodim{i+1:02d}'for i in range(24)]
        and all(x['preprocessing_id']=='kodak_rgb_center_crop_256_v1'for x in records),'Fixed ordered24 source records')
    methods=r['method_inputs'];require(len(methods)==4 and {x['method']for x in methods}==set(METHODS),'Exactly four named methods')
    frames=[]
    for entry in methods:
        completion=pinned(entry['completion']);table=pinned(entry['frames'])
        require(type(table)is list and len(table)==216,'Actual216-frame method table')
        expected_status='KODAK24_ADAPTIVE_BPG_ACTUAL_CHILDREN_WAIT_ZERO'if entry['method']=='BPG_ADAPTIVE'else'COMPLETE'
        require(completion['status']==expected_status and completion['source_count']==24
            and completion.get('frame_count',completion.get('frames'))==216,'Actual completed method required')
        require(completion.get('actual_children_waited',True)is True,'Known unfinished reconstruction owner')
        require(completion['outputs'].get(entry['frames']['path'])==entry['frames']['sha256'],'Method completion does not bind frames')
        seeds=entry['noise_seeds'];require(seeds==[2001,2002,2003],'Exactly three frozen noises per method')
        expected={(i,s,n)for i in range(24)for s in SNRS for n in seeds};seen=set()
        for row in table:
            i,s,n=row['source_index'],row['snr_db'],row['noise_seed'];key=(i,s,n)
            require(row['method']==entry['method']and key in expected and key not in seen
                and row['source_id']==records[i]['source_id']and row.get('N',1024)==1024,'Exact source/method/noise scope')
            require(isinstance(row['status'],str)and row['status'],'Failure/success status must be preserved')
            require(row['reconstruction']['key']=='rgb','Frozen reconstruction NPZ key is rgb')
            seen.add(key);frames.append(row)
        require(seen==expected,'Missing reconstruction frames; no success-only filtering')
    require(len(frames)==864,'All four216-frame methods required')
    return records,frames


def run(path):
    import numpy as np
    from PIL import Image
    r=read(path);request_hash=sha(path);records,frames=inputs(r);out=Path(r['out'])
    require(r.get('new_metric_call_cap',864)==864 and r.get('reference_preparation_cap',24)==24,'Bounded Kodak-only metrics')
    out.mkdir(parents=True,exist_ok=True)
    cpdone=out/'completion.json'
    if cpdone.exists():
        d=read(cpdone);require(d['status']=='KODAK24_FOUR_METRICS_COMPLETE_V1'and d['request_sha256']==request_hash,'Different completed score request')
        require(all(sha(p)==h for p,h in d['outputs'].items()),'Completed score output changed')
        print(json.dumps(dict(status='COMPLETE_REUSED',new_metric_calls=0)));return
    backend_request=pinned(r['backend_request'])
    require(r['backend_module']['sha256']==backend_request['worker_sha256'],'Use the actually executed frozen backend loader')
    require(sys.platform.startswith('linux')and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Existing FP32 evaluation GPU environment')
    import fcntl
    lock=Path(backend_request['visual_lock']).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    own=(out/'score.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();backend=None;torch=None;rows=[];outputs={};new_scores=0;new_references=0
    def guard():
        require(time.monotonic()-started<r.get('max_seconds',3600)and time.time()<r.get('deadline_unix',float('inf'))
            and not any(Path(p).exists()for p in r.get('stop_files',[])),'Stopped/deadline; preserve completed sources')
    for index,record in enumerate(records):
        guard();cp=out/'sources'/f'{index:04d}.json';reservation=cp.with_suffix('.reserved.json')
        if cp.exists():
            d=read(cp);require(d['status']=='KODAK24_SOURCE_METRICS_COMPLETE_V1'and d['request_sha256']==request_hash
                and d['source_id']==record['source_id']and len(d['rows'])==144,'Existing source checkpoint differs')
            rows.extend(d['rows']);outputs[str(cp)]=sha(cp);continue
        require(not reservation.exists(),'Unresolved source scoring attempt; no automatic repeat')
        require(sha(record['png']['path'])==record['png']['sha256'],'Frozen center-crop PNG changed')
        with Image.open(record['png']['path'])as image:
            require(image.mode=='RGB'and image.size==(256,256),'Use frozen256x256RGB center crop without resizing')
            pixels=np.ascontiguousarray(np.asarray(image,dtype=np.uint8).transpose(2,0,1))
        require(hashlib.sha256(pixels.tobytes()).hexdigest()==record['pixels_chw_uint8_sha256'],'Original Kodak pixel hash changed')
        target=rgb(np,pixels.astype(np.float32)/np.float32(255));reference_hash=rgb_sha(np,target)
        if backend is None:
            old=load(r['backend_module'],'_kodak_existing_backend_loader')
            backend=old.build_backend(backend_request,pinned(backend_request['original_FINAL_FREEZE']))
            import torch as torch_module
            torch=torch_module
        write(reservation,dict(request_sha256=request_hash,source_index=index,source_id=record['source_id'],
            reference_sha256=reference_hash,metric_call_cap=36,reference_preparation_cap=1))
        backend.check();device=backend.evaluator.device
        x=torch.from_numpy(target[None]).to(device)
        with torch.inference_mode(),backend.metric.no_network(),torch.autocast(device_type=device.type,enabled=False):
            source_feature=backend.metric.dinov2_vitl14_features(backend.evaluator.models['dinov2_vitl14'],x)
            source_prediction=backend.classifier.predict(target)
        new_references+=1;cache={};source_rows=[]
        for frame in [f for f in frames if f['source_index']==index]:
            guard();archive=frame['reconstruction'];require(sha(archive['path'])==archive['sha256'],'Actual received reconstruction changed')
            with np.load(archive['path'],allow_pickle=False)as z:image=rgb(np,z['rgb'].copy())
            image_hash=rgb_sha(np,image)
            if image_hash not in cache:
                backend.check();y=torch.from_numpy(image[None]).to(device)
                with torch.inference_mode(),backend.metric.no_network(),torch.autocast(device_type=device.type,enabled=False):
                    # Exactly the existing LPIPS argument order and RGB[-1,1] transform.
                    lpips=float(backend.lpips(y*2-1,x*2-1).reshape(-1)[0].item())
                    feature=backend.metric.dinov2_vitl14_features(backend.evaluator.models['dinov2_vitl14'],y)
                    cosine=float(torch.nn.functional.cosine_similarity(source_feature,feature,dim=-1)[0].item())
                    reconstruction_prediction=backend.classifier.predict(image)
                mse=float(np.square(image.astype(np.float64)-target.astype(np.float64)).mean())
                # Preserve an exact match as inf, never a fabricated PSNR cap.
                psnr=-10*math.log10(mse)if mse>0 else'inf'
                values=dict(psnr_db=psnr,lpips_alex=lpips,dinov2_vitl14_cosine=cosine,
                    convnext_top1_source_prediction=int(reconstruction_prediction==source_prediction),
                    reconstruction_prediction=int(reconstruction_prediction))
                require(all(math.isfinite(float(values[m]))for m in METRICS if m!='psnr_db'),'Finite frozen metric output')
                cache[image_hash]=values;new_scores+=1;backend.check()
            common={k:frame[k]for k in ['method','source_index','source_id','snr_db','noise_seed','status']}
            common.update(N=1024,population='Kodak24',reference_sha256=reference_hash,reconstruction_sha256=image_hash,
                reconstruction_archive_sha256=archive['sha256'],preprocessing_id=record['preprocessing_id'],
                metric_backend_identity=backend.identity,true_class_label_available=False,
                source_prediction=source_prediction,reconstruction_prediction=cache[image_hash]['reconstruction_prediction'],
                frame_level_noise_pairing_claimed=False)
            source_rows.extend(dict(common,metric=m,value=cache[image_hash][m])for m in METRICS)
        require(len(source_rows)==144 and len(cache)<=36,'Complete source36-frame metric grid')
        write(cp,dict(status='KODAK24_SOURCE_METRICS_COMPLETE_V1',request_sha256=request_hash,source_id=record['source_id'],
            source_index=index,reference_sha256=reference_hash,source_prediction=source_prediction,
            unique_reconstruction_scores=len(cache),reference_preparations=1,rows=source_rows,
            metric_backend_identity=backend.identity,actual_population='Kodak24',true_class_labels_used=False))
        outputs[str(cp)]=sha(cp);rows.extend(source_rows)
        print(json.dumps(dict(completed_sources=index+1,new_unique_scores=new_scores,new_reference_preparations=new_references)),flush=True)
    require(len(rows)==3456 and sha(path)==request_hash,'Complete864framesx4metrics with unchanged request')
    csv_write(out/'rows.csv',rows);outputs[str(out/'rows.csv')]=sha(out/'rows.csv')
    checkpoints=[read(out/'sources'/f'{i:04d}.json')for i in range(24)]
    total_scores=sum(d['unique_reconstruction_scores']for d in checkpoints)
    require(total_scores<=864 and sum(d['reference_preparations']for d in checkpoints)==24,'Finite Kodak metric cost')
    identity=checkpoints[0]['metric_backend_identity'];require(all(d['metric_backend_identity']==identity for d in checkpoints),'Same metric backend for all sources')
    write(cpdone,dict(status='KODAK24_FOUR_METRICS_COMPLETE_V1',request_sha256=request_hash,script_sha256=sha(__file__),
        source_count=24,noise_count=3,frame_count=864,frame_count_per_method=216,metric_rows=3456,metrics=list(METRICS),
        methods=list(METHODS),snr_db=list(SNRS),N=1024,source_ids=[x['source_id']for x in records],
        original_metric_backend_identity=identity,actual_population='Kodak24',true_class_labels_used=False,
        classifier_metric='agreement with source-image prediction;not accuracy',metric_batch_size=1,
        unique_reconstruction_scores=total_scores,reference_preparations=24,new_scores_this_run=new_scores,
        new_references_this_run=new_references,LPIPS_rule='existing backend.lpips(reconstruction*2-1,source*2-1)',
        DINO_rule='existing dinov2_vitl14_features and cosine_similarity',PSNR_rule='per-image float64 RGB01 MSE before averaging',
        old_statistics_modified=False,new_PHY_calls=0,new_channel_simulations=0,new_reconstructions=0,new_bootstrap=0,
        training_updates=0,policy_selection=False,outputs=outputs,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(status='COMPLETE',frame_count=864,metric_rows=3456,new_unique_scores=new_scores)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True)
    run(parser.parse_args().request)
