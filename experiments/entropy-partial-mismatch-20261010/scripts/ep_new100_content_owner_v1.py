"""Finite2CPU original preprocessing and complete new100 dedup gate, before Encoder.

Consumes only the actual fixed selection and its closed exact100 transport.
There is no source replacement, source-count reduction, model or PHY entry point.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import importlib.util
import io as byteio
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import ep_new100_content_io_v1 as io
import ep_new100_selection_owner_v1 as selection_owner
import ep_new100_download_owner_v1 as download
import ep_source_population as population

SCHEMA='EP_NEW100_CONTENT_OWNER_V1'
PASS='EP_NEW100_CONTENT_ASSETS_READY_NO_ENCODER'
PREPROCESS_SHA='568217086c94ee7607abc256415d705a11702ffcaaed9c79ca04d5e9a62da562'
NATIVE_SHA='eb641b1f3139960b67b6342417eba09fb0aaed26784c93f99c1d85c9437a4868'
require=io.require


def tools():
    names=selection_owner.TOOLS+('ep_new100_download_owner_v1.py','ep_new100_content_owner_v1.py')
    return {n:io.sha(Path(__file__).with_name(n)) for n in names}


def mirrored(desc,mapping):
    require(desc['path'] in mapping,'Exact closed download mirror missing')
    actual=mapping[desc['path']]
    require(actual['sha256']==desc['sha256'] and actual.get('bytes')==desc.get('bytes'),'Transport mirror byte pin differs')
    return io.read(actual)


def selection_closure(r):
    done=io.read(r['selection_completion']);outer=io.read(r['selection_owner_actual_wait'])
    require(done['status']==selection_owner.PASS and done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and
        done['actual_child_wait']['actual_wait'] is True and done['actual_child_wait']['returncode']==0 and
        done['actual_child_wait']['interrupted_or_timeout'] is False and outer['actual_wait'] is True and outer['returncode']==0,
        'Actual selection owner wait0 required')
    request=selection_owner.registration(done['request']);worker=io.read(done['worker_completion'])
    require(worker['status']==selection_owner.PASS and worker['request_sha256']==done['request']['sha256'] and worker['results']==done['results'],
        'Closed selection worker differs')
    reg=selection_owner.closed_registry(request);selected=io.read(done['results']['selection']);policy=io.read(done['results']['policy_bundle'])
    plan=download.validate_plan(io.read(done['results']['download_plan']))
    require(all(plan[k]==done['results'][k] for k in ('selection','policy_bundle','registry')) and selected['registry']==plan['registry'] and
        plan['registry']==request['metadata']['registry'] and policy['schema']=='EP_NEW100_FOUR_FROZEN_POLICY_PROJECTION_V1' and
        policy['strategy_selection_uses_new100'] is False,'Selection policies/registry must remain fixed')
    require([x['source_id'] for x in plan['records']]==selected['source_ids'],'Exact selected order required')
    return done,selected,plan,reg


def transport_closure(r,plan,selection_done):
    done=mirrored(r['download_completion'],r['download_mirrors']);worker=mirrored(done['worker_completion'],r['download_mirrors'])
    require(done['schema']==worker['schema']==download.SCHEMA and done['status']==worker['status']==download.PASS and
        done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and done['actual_child_wait']['actual_wait'] is True and
        done['actual_child_wait']['returncode']==0 and done['actual_child_wait']['interrupted_or_timeout'] is False and
        worker['request_sha256']==done['request']['sha256'] and worker['results']==done['results'],'Closed download actual wait0 required')
    transport=io.read(r['transport_manifest'])
    require(r['transport_manifest']['sha256']==done['results']['transport_manifest']['sha256'] and
        transport['schema']=='EP_NEW100_OFFICIAL_JPEG_TRANSPORT_V1' and transport['source_count']==len(transport['records'])==100 and
        transport['selection_completion']==r['selection_completion'] and transport['download_plan']==selection_done['results']['download_plan'] and
        all(transport[k]==plan[k] for k in ('selection','policy_bundle','registry','official')) and
        transport['payload_requests']==100 and transport['payload_bytes']==plan['maximum_payload_bytes'] and
        transport['replacements']==0 and transport['automatic_retry'] is False and transport['historical_byte_SHA_equality_claimed'] is False and
        transport['image_decodes']==transport['model_calls']==0,'Exact fixed100 transport provenance differs')
    archive=r['transport_archive'];require(archive['sha256']==transport['archive']['sha256']==done['results']['transport_archive']['sha256'] and
        archive['bytes']==transport['archive']['bytes']==done['results']['transport_archive']['bytes'] and
        archive['bytes']<=plan['maximum_payload_bytes']+200*512+10240,'Bounded exact tar identity differs')
    for i,(actual,expected) in enumerate(zip(transport['records'],plan['records'])):
        require(all(actual[k]==v for k,v in expected.items()) and actual['transport_member']=='%04d.JPEG'%i and
            actual['historical_JPEG_sha256'] is None and actual['historical_byte_SHA_equality_claimed'] is False and
            population.SHA.fullmatch(actual['newly_observed_official_JPEG_sha256']),'Payload changed or retrospective SHA invented')
    return transport


def registration(desc):
    r=io.read(desc)
    require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['tool_bindings']==tools() and
        r['worker_python']==str(io.FROZEN) and r['claim_name']=='canonical_content_once.json' and
        r['caps']==dict(canonical_preprocess=100,horizontal_flip_hashes=100,distinct_original_JPEG_files=100,
            model_load=0,encoder=0,source_tx=0,source_rx=0,packet_decodes=0),'Finite original-content registration differs')
    require(r['original_preprocess']['sha256']==PREPROCESS_SHA and r['native_import_binding']['sha256']==NATIVE_SHA,
        'Original preprocessing/native import qualification required')
    io.bytes_checked(r['original_preprocess']);io.bytes_checked(r['native_import_binding']);io.bytes_checked(r['controlled_environment'])
    expected_root=io.RT/'assets/frozen_namespace_v1/external/home/liulu/.local/lib/python3.10/site-packages'
    require(r['frozen_import_paths']=={'torch':str(expected_root/'torch/__init__.py'),'numpy':str(expected_root/'numpy/__init__.py'),
        'torchvision':str(expected_root/'torchvision/__init__.py'),'PIL.Image':str(expected_root/'PIL/Image.py')},'Frozen CPU imports required')
    done,selected,plan,reg=selection_closure(r);transport=transport_closure(r,plan,done)
    # Payloads are not opened during prepare/registration; worker verifies them.
    archive=io.inside(r['transport_archive']['path']);require(archive.is_file() and archive.stat().st_size==r['transport_archive']['bytes'],
        'Exact-size incoming transport required')
    return r,done,selected,plan,reg,transport


def verify_tar(archive,transport,out,guard):
    """Verify exact safe100 file list before any canonical image decoding."""
    require(io.sha(archive)==transport['archive']['sha256'],'Transport SHA mismatch')
    records=transport['records'];result=[]
    with tarfile.open(archive,'r:') as tar:
        for i,row in enumerate(records):
            guard();member=tar.next();require(member is not None and member.isfile() and not member.issym() and not member.islnk() and
                member.name==row['transport_member']=='%04d.JPEG'%i and member.size==row['bytes'] and not member.pax_headers,
                'Only exact ordered100 regular flat JPEG members allowed')
            f=tar.extractfile(member);raw=f.read(member.size+1);f.close()
            require(len(raw)==member.size and hashlib.sha256(raw).hexdigest()==row['newly_observed_official_JPEG_sha256'],
                'Original JPEG payload SHA mismatch')
            path=out/member.name
            with path.open('xb') as dest:dest.write(raw);dest.flush();os.fsync(dest.fileno())
            result.append(io.pin(path))
        require(tar.next() is None,'Unregistered extra tar member; no access to additional candidates')
    return result


def load_preprocessor(r):
    import torch
    import numpy as np
    import PIL
    from PIL import Image
    import torchvision
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
    observed={'torch':torch.__file__,'numpy':np.__file__,'torchvision':torchvision.__file__,'PIL.Image':Image.__file__}
    require(observed==r['frozen_import_paths'],'Unchanged frozen CPU package paths required')
    path=Path(r['original_preprocess']['path']);spec=importlib.util.spec_from_file_location('ep_new100_original_preprocess',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(io.sha(path)==PREPROCESS_SHA,'Original preprocessing changed during import')
    return module,torch,np,dict(imports=observed,python=sys.version,torch=torch.__version__,numpy=np.__version__,
        PIL=PIL.__version__,torchvision=torchvision.__version__,cuda_initialized=False)


def worker(desc,owner_pid):
    r,done,selected,plan,reg,transport=registration(desc);boundary=lambda:io.guard(r,owner_pid);boundary()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in io.THREADS) and
        set(os.sched_getaffinity(0))==set(r['cpu_slots']) and Path(sys.executable).resolve()==io.FROZEN.resolve(),
        'Admitted hidden-CUDA frozen2CPU interpreter required')
    out=io.inside(Path(r['out'])/'run');images=out/'original_JPEG';images.mkdir();pixels_dir=out/'pixels';pixels_dir.mkdir()
    originals=verify_tar(r['transport_archive']['path'],transport,images,boundary)
    module,torch,np,env=load_preprocessor(r);io.save(out/'loaded_cpu_environment.json',env)
    reserved=[];completed=[];actual=[];assets=[]
    with (out/'canonical_events.jsonl').open('x',encoding='utf8') as events:
        def event(value):events.write(json.dumps(value,sort_keys=True)+'\n');events.flush();os.fsync(events.fileno())
        def one(i):
            boundary();raw=io.bytes_checked(originals[i],maximum=16<<20)
            normalized,pixel_sha=module.preprocess(byteio.BytesIO(raw))
            require(normalized.shape==(3,256,256) and normalized.device.type=='cpu','Original canonical output shape/device differs')
            pixels=normalized.add(1).mul(127.5).round().to(torch.uint8).contiguous();array=pixels.numpy()
            require(hashlib.sha256(array.tobytes()).hexdigest()==pixel_sha,'Canonical uint8 hash differs')
            flip=hashlib.sha256(pixels.flip(-1).contiguous().numpy().tobytes()).hexdigest()
            path=pixels_dir/('%04d.npz'%i)
            with path.open('xb') as f:np.savez_compressed(f,pixels=array)
            with np.load(path,allow_pickle=False) as check:
                require(check.files==['pixels'] and check['pixels'].dtype==np.uint8 and check['pixels'].shape==(3,256,256) and
                    hashlib.sha256(check['pixels'].tobytes()).hexdigest()==pixel_sha,'Saved pixels-only content archive differs')
            require(not torch.cuda.is_initialized(),'CPU preprocessing initialized CUDA')
            return pixel_sha,flip,io.pin(path)
        # At most two submitted tasks, no unbounded queue; all submitted tasks
        # are actually joined, including a failing batch. No retries occur.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for start in range(0,100,2):
                boundary();futures=[]
                for i in (start,start+1):
                    event(dict(kind='reserve',source_index=i));reserved.append(i);futures.append((i,pool.submit(one,i)))
                errors=[]
                for i,future in futures:
                    try:
                        pre,flip,archive=future.result();row=plan['records'][i];completed.append(i)
                        event(dict(kind='complete',source_index=i,pixel_sha256=pre,horizontal_flip_pixel_sha256=flip))
                        actual.append(dict(source_index=i,source_id=row['source_id'],original_file_sha256=originals[i]['sha256'],
                            pixel_sha256=pre,horizontal_flip_pixel_sha256=flip))
                        assets.append(dict(source_index=i,source_id=row['source_id'],canonical_source_id=row['canonical_source_id'],
                            evaluation_class_index=row['evaluation_class_index'],original_JPEG=originals[i],pixels_archive=archive,
                            preprocessing_id=pre,canonical_pixel_sha256=pre,horizontal_flip_pixel_sha256=flip,
                            original_provenance='Official original val archive; newly observed SHA, no retrospective local-JPEG SHA claim'))
                    except BaseException as error:event(dict(kind='failure',source_index=i,error=repr(error)));errors.append(error)
                if errors:raise errors[0]
    require(reserved==completed==list(range(100)),'Exactly100 resolved canonical preprocessing calls required')
    receipt=dict(schema='EP_CANDIDATE100_CONTENT_RECEIPT_V1',selection=done['results']['selection'],pixel_hash_domain=population.PIXEL_DOMAIN,
        Encoder_calls_before_check=0,VAR_calls_before_check=0,metric_calls_before_check=0,records=actual,
        source_count=100,original_preprocess=r['original_preprocess'],transport_manifest=r['transport_manifest'])
    cp=io.save(out/'content_receipt.json',receipt)
    gate=population.check_content(reg,done['results']['registry'],selected,done['results']['selection'],receipt,cp)
    gp=io.save(out/'content_gate.json',gate)
    require(gate['status']=='EP_NEW100_COMPLETE_CONTENT_DEDUP_PASS','Duplicate found; fixed100 preserved, no replacements or Encoder')
    manifest=dict(schema='EP_NEW100_SOURCE_CONTENT_ASSETS_V1',source_count=100,records=assets,
        selection=done['results']['selection'],policy_bundle=done['results']['policy_bundle'],registry=done['results']['registry'],
        content_receipt=cp,content_gate=gp,original_preprocess=r['original_preprocess'],model_calls=0)
    mp=io.save(out/'source_manifest.json',manifest);boundary()
    results={k:done['results'][k] for k in ('selection','policy_bundle','registry')}
    results.update(content_receipt=cp,content_gate=gp,source_manifest=mp)
    io.save(out/'worker_completion.json',dict(schema=SCHEMA,status=PASS,request_sha256=desc['sha256'],results=results,
        canonical_reserved=100,canonical_completed=100,unresolved=0,model_calls=0,packet_calls=0,CUDA_initialized=False))


def prepare(a):
    spec=io.read(dict(path=a.spec,sha256=a.spec_sha256));out=io.inside(a.out)
    expected={'selection_completion','selection_owner_actual_wait','download_completion','download_mirrors','transport_manifest','transport_archive',
        'original_preprocess','native_import_binding','controlled_environment','frozen_import_paths'}
    require(set(spec)==expected and not out.exists(),'Fresh exact CPU content specification required')
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',**spec,out=str(out),worker_python=str(io.FROZEN),tool_bindings=tools(),
        claim_name='canonical_content_once.json',caps=dict(canonical_preprocess=100,horizontal_flip_hashes=100,distinct_original_JPEG_files=100,
            model_load=0,encoder=0,source_tx=0,source_rx=0,packet_decodes=0),cpu_slots=a.cpu_slots,
        max_seconds=a.max_seconds,deadline_unix=a.deadline_unix,automatic_retry=False,automatic_successor=False)
    io.validate_execution(r);out.mkdir(parents=True);rp=io.save(out/'request.json',r);registration(rp);return rp


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--spec',required=True);a.add_argument('--spec-sha256',required=True);a.add_argument('--out',required=True)
    a.add_argument('--deadline-unix',type=float,required=True);a.add_argument('--max-seconds',type=int,default=600)
    a.add_argument('--cpu-slots',type=lambda x:[int(v) for v in x.split(',')],required=True)
    for name in ('run','_worker'):
        a=sub.add_parser(name);a.add_argument('--request',required=True);a.add_argument('--request-sha256',required=True)
        if name=='_worker':a.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(a)));return
    desc=dict(path=a.request,sha256=a.request_sha256)
    if a.command=='_worker':worker(desc,a.owner_pid)
    else:print(json.dumps(io.run_owner(desc,Path(__file__).absolute(),lambda d:registration(d)[0],PASS)))


if __name__=='__main__':main()
