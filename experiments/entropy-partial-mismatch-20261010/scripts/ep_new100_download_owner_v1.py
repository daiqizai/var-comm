"""Finite PC owner for the fixed100 official original JPEG payloads; no image imports.

Input is an exact mirror of the actual selection owner artifacts. One Range GET
per selected payload, no retry/redirect/replacement and no archive-header scan.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
import ep_new100_content_io_v1 as io
import ep_new100_selection_owner_v1 as selection_owner

SCHEMA='EP_NEW100_OFFICIAL_DOWNLOAD_OWNER_V1'
PASS='EP_NEW100_EXACT100_OFFICIAL_JPEG_TRANSPORT_CLOSED'
LOCAL_BASE=Path(__file__).resolve().parents[3]/'.research'
require=io.require


def local(path):return io.inside(path,LOCAL_BASE)
def load(desc):return io.read(desc,base=LOCAL_BASE)


def mirrored(desc,mapping):
    require(desc['path'] in mapping,'Exact selection mirror entry missing')
    actual=mapping[desc['path']]
    require(actual['sha256']==desc['sha256'] and actual.get('bytes')==desc.get('bytes'),'Mirror must retain original bytes')
    return load(actual)


def validate_plan(plan):
    require(plan['schema']=='EP_NEW100_OFFICIAL_EXACT_DOWNLOAD_PLAN_V1' and plan['source_count']==len(plan['records'])==100 and
        plan['official']==dict(url=selection_owner.URL,etag=selection_owner.ETAG,archive_bytes=selection_owner.TOTAL) and
        plan['maximum_payload_requests']==100 and plan['automatic_retry'] is False and plan['replacements_allowed'] is False and
        plan['historical_byte_SHA_equality_claimed'] is False,'Exact fixed official download plan required')
    seen=set()
    for i,row in enumerate(plan['records']):
        require(row['source_index']==i and row['canonical_source_id'] not in seen and 0<row['bytes']<=16<<20 and
            row['bytes']==row['original_bytes'] and row['data_offset']%512==0 and
            512<=row['data_offset']<row['data_offset']+row['bytes']<=selection_owner.TOTAL,'Fixed selected record invalid')
        seen.add(row['canonical_source_id'])
    require(plan['maximum_payload_bytes']==sum(r['bytes'] for r in plan['records'])<=512<<20,'Exact bounded bytes required')
    return plan


def registration(desc):
    r=load(desc);require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['automatic_retry'] is False and
        r['automatic_successor'] is False and r['max_seconds']==900 and r['tool_bindings']==tools(),'Download registration changed')
    local(r['out']);require(type(r['deadline_unix']) in (float,int),'Finite deadline required')
    done=mirrored(r['selection_completion'],r['mirrors']);outer=mirrored(r['selection_owner_actual_wait'],r['mirrors'])
    require(done['status']==selection_owner.PASS and done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and
        done['actual_child_wait']['actual_wait'] is True and done['actual_child_wait']['returncode']==0 and
        done['actual_child_wait']['interrupted_or_timeout'] is False and outer['actual_wait'] is True and outer['returncode']==0,
        'Actual successful selection owner required')
    worker=mirrored(done['worker_completion'],r['mirrors'])
    require(worker['status']==selection_owner.PASS and worker['request_sha256']==done['request']['sha256'] and worker['results']==done['results'],
        'Actual selection worker result binding differs')
    plan=validate_plan(mirrored(done['results']['download_plan'],r['mirrors']))
    require(all(plan[k]==done['results'][k] for k in ('selection','policy_bundle','registry')),'Selected policy/registry links differ')
    selected=mirrored(plan['selection'],r['mirrors']);policy=mirrored(plan['policy_bundle'],r['mirrors'])
    require(selected['status']=='EP_SOURCE_IDS_FIXED_NO_PIXEL_ACCESS' and selected['registry']==plan['registry'] and
        policy['schema']=='EP_NEW100_FOUR_FROZEN_POLICY_PROJECTION_V1' and policy['strategy_selection_uses_new100'] is False,
        'Actual fixed selection and frozen policy projection required')
    require([x['source_id'] for x in plan['records']]==selected['source_ids'] and
        all(a['original_bytes']==b['bytes'] for a,b in zip(selected['records'],plan['records'])),'Plan differs from fixed100')
    return r,plan,done


def tools():
    names=('ep_new100_download_owner_v1.py','ep_new100_content_io_v1.py','ep_new100_selection_owner_v1.py',
        'ep_new100_confirmation_core_v1.py','ep_source_population.py','ep_plan.py')
    return {n:io.sha(Path(__file__).with_name(n)) for n in names}


def guard(r,owner=None):
    require(time.time()<r['deadline_unix'] and not (LOCAL_BASE/'STOP').exists() and not (Path(r['out'])/'STOP').exists(),
        'Download stopped or expired')
    if owner is not None:require(os.getppid()==owner,'Download owner lost')


def fetch_one(row,path,deadline,connection_factory=http.client.HTTPSConnection):
    start=row['data_offset'];end=start+row['bytes']-1
    connection=connection_factory('image-net.org',timeout=max(.1,min(30,deadline-time.time())))
    try:
        connection.request('GET','/data/ILSVRC/2012/ILSVRC2012_img_val.tar',headers={
            'Range':f'bytes={start}-{end}','If-Match':selection_owner.ETAG,'Accept-Encoding':'identity','Connection':'close'})
        response=connection.getresponse()
        require(response.status==206 and response.getheader('ETag')==selection_owner.ETAG and
            response.getheader('Content-Range')==f'bytes {start}-{end}/{selection_owner.TOTAL}' and
            response.getheader('Content-Length')==str(row['bytes']) and response.getheader('Content-Encoding') in (None,'identity'),
            'Exact206 bounded official payload required; do not read full archive or redirect')
        digest=hashlib.sha256();count=0
        with Path(path).open('xb') as f:
            while count<row['bytes']:
                require(time.time()<deadline,'Finite payload download deadline exceeded')
                block=response.read(min(1<<16,row['bytes']-count));require(block,'Incomplete payload; no retry')
                f.write(block);digest.update(block);count+=len(block)
            f.flush();os.fsync(f.fileno())
        require(count==row['bytes'],'Exact payload bytes required')
        return dict(bytes=count,sha256=digest.hexdigest())
    finally:connection.close()


def worker(desc,owner):
    r,plan,done=registration(desc);guard(r,owner);out=local(Path(r['out'])/'run')
    images=out/'images';images.mkdir();records=[]
    with (out/'payload_events.jsonl').open('x',encoding='utf8') as events:
        def event(row):events.write(json.dumps(row,sort_keys=True)+'\n');events.flush();os.fsync(events.fileno())
        for row in plan['records']:
            guard(r,owner);name='%04d.JPEG'%row['source_index'];event(dict(kind='reserve',source_index=row['source_index']))
            value=fetch_one(row,images/name,r['deadline_unix'])
            records.append(dict(row,transport_member=name,newly_observed_official_JPEG_sha256=value['sha256'],
                historical_JPEG_sha256=None,historical_byte_SHA_equality_claimed=False))
            event(dict(kind='complete',source_index=row['source_index'],**value))
    guard(r,owner);archive=out/'exact100_original_JPEG.tar'
    with tarfile.open(archive,'x',format=tarfile.USTAR_FORMAT) as tar:
        for row in records:
            path=images/row['transport_member'];info=tarfile.TarInfo(row['transport_member']);info.size=row['bytes'];info.mode=0o600
            with path.open('rb') as f:tar.addfile(info,f)
    transport=dict(schema='EP_NEW100_OFFICIAL_JPEG_TRANSPORT_V1',source_count=100,records=records,archive=io.pin(archive),
        selection=plan['selection'],policy_bundle=plan['policy_bundle'],registry=plan['registry'],
        download_plan=done['results']['download_plan'],selection_completion=r['selection_completion'],
        official=plan['official'],payload_requests=100,payload_bytes=sum(x['bytes'] for x in records),
        replacements=0,automatic_retry=False,historical_byte_SHA_equality_claimed=False,image_decodes=0,model_calls=0)
    tp=io.save(out/'transport_manifest.json',transport);guard(r,owner)
    io.save(out/'worker_completion.json',dict(schema=SCHEMA,status=PASS,request_sha256=desc['sha256'],
        results=dict(transport_manifest=tp,transport_archive=io.pin(archive)),payload_requests=100,image_decodes=0,model_calls=0))


def run(desc):
    r,_,_=registration(desc);guard(r);require(0<r['deadline_unix']-time.time()<=900,'At most900seconds fresh download window')
    out=local(Path(r['out'])/'run');require(not out.exists(),'Fresh download attempt only');out.mkdir()
    io.save(LOCAL_BASE/'ep_new100_official_v1/download_once_claim.json',dict(request=desc,pid=os.getpid(),automatic_retry=False))
    child=None;wait=None
    try:
        with (out/'child.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'_worker','--request',desc['path'],
                '--request-sha256',desc['sha256'],'--owner-pid',str(os.getpid())],stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            started=time.monotonic()
            try:code=child.wait(timeout=min(900,r['deadline_unix']-time.time()));interrupted=False
            except BaseException:child.kill();code=child.wait();interrupted=True
        wait=dict(actual_wait=True,returncode=code,interrupted_or_timeout=interrupted,elapsed_seconds=time.monotonic()-started,
            log_sha256=io.sha(out/'child.log'));io.save(out/'actual_child_wait.json',wait)
        require(code==0 and not interrupted,'Download failed; preserve selected100 and partial payloads, no replacement/retry')
        done=load(io.pin(out/'worker_completion.json'));require(done['status']==PASS and done['request_sha256']==desc['sha256'],'Wrong worker closure')
        return io.save(out/'completion.json',dict(schema=SCHEMA,status=PASS,request=desc,actual_child_wait=wait,
            actual_children_waited=True,worker_exit_codes=[0],worker_completion=io.pin(out/'worker_completion.json'),results=done['results']))
    except BaseException as error:
        if child is not None and child.poll() is None:child.kill();child.wait()
        io.save(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),actual_child_wait=wait,
            source_reselection_allowed=False,automatic_retry=False));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--spec',required=True);a.add_argument('--spec-sha256',required=True);a.add_argument('--out',required=True)
    a.add_argument('--deadline-unix',type=float,required=True)
    for name in ('run','_worker'):
        a=sub.add_parser(name);a.add_argument('--request',required=True);a.add_argument('--request-sha256',required=True)
        if name=='_worker':a.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args()
    if a.command=='prepare':
        spec=load(dict(path=a.spec,sha256=a.spec_sha256));out=local(a.out)
        require(not out.exists() and set(spec)=={'selection_completion','selection_owner_actual_wait','mirrors'},'Fresh exact mirror spec required')
        r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',**spec,out=str(out),tool_bindings=tools(),max_seconds=900,
            deadline_unix=a.deadline_unix,automatic_retry=False,automatic_successor=False)
        require(0<a.deadline_unix-time.time()<=900,'Fresh finite download deadline required')
        out.mkdir(parents=True);print(json.dumps(io.save(out/'request.json',r)));return
    desc=dict(path=a.request,sha256=a.request_sha256)
    if a.command=='_worker':worker(desc,a.owner_pid)
    else:print(json.dumps(run(desc)))


if __name__=='__main__':main()
