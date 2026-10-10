"""Explicit metadata-only owner: closed full1000 policy, then one fixed new100 selection.

prepare binds metadata only. Only an explicit run can rank the pool. No image or
network access, replacement, quality call, or automatic successor is available.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
import ep_new100_content_io_v1 as io
import ep_new100_confirmation_core_v1 as core
import ep_source_population as population

SCHEMA='EP_NEW100_FIXED_SELECTION_OWNER_V1'
PASS='EP_NEW100_SELECTION_FIXED_OFFICIAL_PAYLOADS_BOUND'
POOL_SHA='f911d508d95febb4ab77224da3d99c99d1e8dc876a838f724991f0a949a91d8e'
HEADER_SHA='59ab8fc066354867c794aa2bed74dfc27b79205a4502b512d329fde840d048cb'
CLASS_SHA='576ffc8db4d9ced560af66e55c185fcd2d1a078b9ffa9315b372ea15f2532959'
REGISTRY_SHA='7b8320d259aabc24e77d86f4da51414bd8a15e796f370200cc0d28e88c45ee8b'
SCOPE_SHA='062c258e78bda428d5055673d0b0e60012378bbf7b469f65485e49cbf61bdfbe'
CORE_SHA='0e688098e8ecf41daccd5cdc93f91a08bbeac1d4f40b273a42d6b89d05284bc8'
FULL_OWNER_SHA='0ff1e074c8881e4f15eff7c90943431eed092c578e0ac6ae712360396dfa1ebb'
URL='https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar'
ETAG='"192076000-4c2722fbfc580"'
TOTAL=6744924160
TOOLS=('ep_new100_selection_owner_v1.py','ep_new100_content_io_v1.py',
       'ep_new100_confirmation_core_v1.py','ep_source_population.py','ep_plan.py')
require=io.require


def tools():return {n:io.sha(Path(__file__).with_name(n)) for n in TOOLS}


def validate_fixed_metadata(r):
    expected=dict(pool=POOL_SHA,header_index=HEADER_SHA,class_map=CLASS_SHA,registry=REGISTRY_SHA,scope_completion=SCOPE_SHA)
    require(set(r['metadata'])==set(expected),'Exact five frozen metadata inputs required')
    for k,digest in expected.items():
        require(r['metadata'][k]['sha256']==digest,'Wrong fixed metadata: '+k)
        io.bytes_checked(r['metadata'][k])
    require(r['full_calibration']['request']['path']==str(io.RT/'qualification/h800_ep_full_metrics_v2_attempt1/registered/request.json') and
        r['full_calibration']['completion']['path']==str(io.RT/'qualification/h800_ep_full_metrics_v2_attempt1/registered/run/completion.json') and
        r['full_calibration']['owner_actual_wait']['path']==str(io.RT/'qualification/h800_ep_full_metrics_v2_attempt1/run_owner_actual_wait.json'),
        'Exact actually closed full1000 attempt required')
    require(set(r['full_calibration'])=={'request','completion','owner_actual_wait'},'Full closure pins required')
    for desc in r['full_calibration'].values():core.pin(desc);io.bytes_checked(desc,512<<20)
    require(set(r['original_policies'])=={'raw_plan','raw_freeze','whole_policy'},'Exact original policy inputs required')
    for desc in r['original_policies'].values():io.bytes_checked(desc)


def registration(desc):
    r=io.read(desc)
    require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['tool_bindings']==tools() and
        r['worker_python']==str(io.ENGINEERING) and r['claim_name']=='metadata_selection_once.json' and
        r['selection_seed']==population.SELECTION_SEED and r['science_caps']==core.scientific_caps() and
        r['maximum_selected_sources']==100 and r['pixel_reads']==r['model_calls']==r['packet_calls']==0,
        'Finite metadata-only registration changed')
    require(io.sha(core.__file__)==CORE_SHA,'Frozen confirmation contract changed')
    validate_fixed_metadata(r);return r


def closed_registry(r):
    reg=io.read(r['metadata']['registry']);scope=io.read(r['metadata']['scope_completion'])
    require(scope['status']=='RECORDED_STUDY_SCOPE_AND_CANONICAL_HASH_COVERAGE_CLOSED' and scope['canonical_hash_gaps']==0 and
        scope['registry']['sha256']==r['metadata']['registry']['sha256'] and scope['source_count']==24959 and
        reg['schema']=='EP_STUDY_SOURCE_EXCLUSION_REGISTRY_V1' and reg['study_scope_closed'] is True and
        reg['full_canonical_content_dedup_ready'] is True and reg['source_count']==24959 and
        len(reg['records'])==len(set(reg['canonical_source_ids']))==24959 and reg['pixel_hash_missing_count']==0 and
        all(row['pixel_sha256'] for row in reg['records']),'Complete formal recorded-study canonical coverage required')
    return reg


def pool_and_headers(pool,lines,classes):
    """Validation only: no selection keys or source ranking are computed here."""
    rows=pool['records'];require(len(rows)==50000,'Exact original val50000 metadata required')
    ids={population.canonical_id(r['source_id']) for r in rows}
    require(ids=={'imagenet-val:%08d'%i for i in range(1,50001)},'Full original val identity coverage required')
    mapping={}
    for row in classes['entries']:
        synset,index=row['synset'],row['class_index']
        require(type(index) is int and 0<=index<1000 and mapping.setdefault(synset,index)==index,'Original class mapping inconsistent')
    require(len(mapping)==1000 and set(mapping.values())==set(range(1000)),'Original1000 class mapping required')
    headers={}
    for row in lines:
        m=re.fullmatch(r'ILSVRC2012_val_(\d{8})\.JPEG',row['name']);require(m,'Only official original JPEG entries')
        index=int(m[1]);require(index not in headers and row['source_index']==index,'Duplicate/mismatched official header')
        require(type(row['bytes']) is int and 0<row['bytes']<=16<<20 and row['offset']%512==0 and
            row['data_offset']==row['offset']+512 and row['next_offset']==row['data_offset']+((row['bytes']+511)//512)*512 and
            0<=row['offset']<row['data_offset']<row['next_offset']<=TOTAL and core.SHA.fullmatch(row['header_sha256']),
            'Invalid bounded original tar header')
        headers[index]=row
    require(set(headers)==set(range(1,50001)),'Exactly50000 official headers required')
    for row in rows:
        require(row['source_id'].split('/')[0] in mapping and type(row['original_bytes']) is int and row['original_bytes']>0,
            'Pool class or original byte count missing')
    return headers,mapping


def download_plan(selection,headers,mapping,selection_pin,policy_pin,registry_pin,metadata):
    require(selection['source_count']==len(selection['records'])==100,'Exactly the fixed100, no replacements')
    records=[]
    for i,row in enumerate(selection['records']):
        require(row['source_index']==i,'Selection order changed')
        h=headers[int(row['canonical_source_id'].split(':')[1])]
        # This check is deliberately AFTER fixed selection was saved by worker().
        require(row['original_bytes']==h['bytes'],'SELECTED_ORIGINAL_OFFICIAL_LENGTH_MISMATCH_STOP_NO_REPLACEMENT: '+row['source_id'])
        records.append(dict(source_index=i,source_id=row['source_id'],canonical_source_id=row['canonical_source_id'],
            evaluation_class_index=mapping[row['source_id'].split('/')[0]],original_path=row['path'],original_bytes=row['original_bytes'],
            official_member=h['name'],data_offset=h['data_offset'],bytes=h['bytes'],header_sha256=h['header_sha256']))
    require(sum(r['bytes'] for r in records)<=512<<20,'Exact selected100 transfer exceeds fixed byte budget')
    return dict(schema='EP_NEW100_OFFICIAL_EXACT_DOWNLOAD_PLAN_V1',selection=selection_pin,policy_bundle=policy_pin,registry=registry_pin,
        source_count=100,records=records,official=dict(url=URL,etag=ETAG,archive_bytes=TOTAL),metadata=metadata,
        maximum_payload_requests=100,maximum_payload_bytes=sum(r['bytes'] for r in records),automatic_retry=False,
        replacements_allowed=False,historical_byte_SHA_equality_claimed=False,model_calls=0)


def full_gate(r):
    p=Path(__file__).with_name('h800_ep_full_metric_owner_v2.py')
    require(io.sha(p)==FULL_OWNER_SHA,'Unchanged fullmetric provider required')
    owner=importlib.import_module('h800_ep_full_metric_owner_v2')
    # Frozen original validator checks the complete request, prior actual owners,
    # runtime/model pins and scientific caps. No model constructor is called.
    runner=owner.engine()
    def validate(desc):return owner.registration(desc['path'],desc['sha256'],runner)
    read=lambda d:io.read(d,maximum=512<<20)
    closure,winners=core.verify_full_calibration(r['full_calibration'],read,validate,owner.evaluation.winners)
    return core.policies(r['original_policies'],winners,closure,read)


def worker(desc,owner_pid):
    r=registration(desc);io.guard(r,owner_pid)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))==set(r['cpu_slots']),
        'Hidden CUDA and coordinated2CPU required')
    out=io.inside(Path(r['out'])/'run');reg=closed_registry(r)
    policy=full_gate(r);pp=io.save(out/'four_frozen_policies.json',policy)
    io.guard(r,owner_pid)
    pool=io.read(r['metadata']['pool']);headers,mapping=pool_and_headers(pool,
        [json.loads(x) for x in io.bytes_checked(r['metadata']['header_index']).decode().splitlines()],io.read(r['metadata']['class_map']))
    # No call above computes rankings or reads any candidate image bytes.
    selected=population.select_population(reg,r['metadata']['registry'],pool,r['metadata']['pool'],r['selection_seed'])
    sp=io.save(out/'selection.json',selected)
    plan=download_plan(selected,headers,mapping,sp,pp,r['metadata']['registry'],r['metadata'])
    dp=io.save(out/'download_plan.json',plan);io.guard(r,owner_pid)
    result=dict(selection=sp,policy_bundle=pp,registry=r['metadata']['registry'],download_plan=dp)
    io.save(out/'worker_completion.json',dict(schema=SCHEMA,status=PASS,request_sha256=desc['sha256'],results=result,
        selected_sources=100,pixel_reads=0,model_calls=0,packet_calls=0,selection_replacements=0))


def prepare(a):
    spec=io.read(dict(path=a.spec,sha256=a.spec_sha256));out=io.inside(a.out)
    require(not out.exists() and set(spec)=={'metadata','full_calibration','original_policies'},'Fresh exact input specification required')
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',**spec,out=str(out),tool_bindings=tools(),worker_python=str(io.ENGINEERING),
        claim_name='metadata_selection_once.json',selection_seed=population.SELECTION_SEED,science_caps=core.scientific_caps(),
        maximum_selected_sources=100,pixel_reads=0,model_calls=0,packet_calls=0,cpu_slots=a.cpu_slots,
        max_seconds=a.max_seconds,deadline_unix=a.deadline_unix,automatic_retry=False,automatic_successor=False)
    io.validate_execution(r);validate_fixed_metadata(r);out.mkdir(parents=True);return io.save(out/'request.json',r)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--spec',required=True);a.add_argument('--spec-sha256',required=True);a.add_argument('--out',required=True)
    a.add_argument('--deadline-unix',type=float,required=True);a.add_argument('--max-seconds',type=int,default=3600)
    a.add_argument('--cpu-slots',type=lambda x:[int(v) for v in x.split(',')],required=True)
    for name in ('run','_worker'):
        a=sub.add_parser(name);a.add_argument('--request',required=True);a.add_argument('--request-sha256',required=True)
        if name=='_worker':a.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(a)));return
    desc=dict(path=a.request,sha256=a.request_sha256)
    if a.command=='_worker':worker(desc,a.owner_pid)
    else:print(json.dumps(io.run_owner(desc,Path(__file__).absolute(),registration,PASS)))


if __name__=='__main__':main()
