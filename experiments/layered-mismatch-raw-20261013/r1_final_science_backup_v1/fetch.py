"""Explicit later fetch, byte verification, ordinary alias restore and light sync.

Import is inert. No SSH occurs until fetch() is explicitly requested. Prior
dependency receipts are inspected; their large archives are not rehashed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
from build import PACKAGE, SCHEMA, FLAGS, DEPENDENCIES, safe_parts, payload_name, require, sha

def extended_local_path(value):
    path=Path(value)
    if os.name!='nt':return path
    require(path.is_absolute() and '..' not in path.parts,'Absolute canonical Windows path required')
    text=str(path)
    return path if text.startswith('\\\\?\\') else Path('\\\\?\\'+text)

W=extended_local_path('C:/Users/11946/Documents/ChatGPT/comm')
RECOVERY=extended_local_path('D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011')
D=RECOVERY/PACKAGE
HOST='DevMachineModel-H100-4C-P1660-01'
REMOTE='/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/'+PACKAGE+'_delivery'
ARCHIVE=PACKAGE+'.tar.gz'


def no_links(path):
    for p in (path,*path.parents):
        if p.exists() or p.is_symlink():
            info=p.lstat()
            require(not p.is_symlink() and not (getattr(info,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024)),
                    'Local symlink/junction forbidden')


def destination(root,name):
    parts=safe_parts(name);p=root.joinpath(*parts);no_links(p)
    require(p.resolve().is_relative_to(root.resolve()),'Extraction escaped destination')
    return p


def save(path,value):
    content=json.dumps(value,sort_keys=True,indent=2)+'\n'
    if path.exists():require(path.read_text(encoding='utf8')==content,'Existing receipt differs; preserve it')
    else:
        with path.open('x',encoding='utf8',newline='\n') as f:f.write(content)


def scope(value):
    require(value['schema']==SCHEMA and all(value.get(k)==v and type(value.get(k))==type(v) for k,v in FLAGS.items()),'Wrong completed-R1 scope')


def dependency_status():
    results=[]
    for expected in DEPENDENCIES:
        folder=RECOVERY/expected['package'];no_links(folder)
        receipt=json.loads((folder/'local_verification.json').read_text(encoding='utf8'))
        require(receipt['sha256']==expected['archive_sha256'] and receipt['server_receipt_sha256']==expected['server_receipt_sha256']
                and receipt['archive_SHA_PASS'] is True and receipt['all_payload_SHA_PASS'] is True,'Old dependency verification receipt is missing/different')
        require((folder/(expected['package']+'.tar.gz')).is_file(),'Old dependency archive no longer present')
        results.append(dict(**expected,local_receipt=str(folder/'local_verification.json'),receipt_sha256=sha(folder/'local_verification.json'),
                            old_payload_rehashed=False,status='EXISTING_VERIFIED_RECEIPT_READ'))
    return results


def validate_manifest(manifest,receipt,names):
    scope(manifest);require(manifest['git_commit']==receipt['git_commit'],'Git receipt differs')
    files=manifest['files'];aliases=manifest['aliases']
    require(len(files)==receipt['payload_files'] and len(aliases)==receipt['alias_files']
            and len(files)+len(aliases)==receipt['restored_files'],'Manifest counts differ')
    require(sum(r['bytes'] for r in files)==receipt['payload_bytes'] and sum(r['bytes'] for r in aliases)==receipt['alias_bytes'],'Manifest sizes differ')
    all_names=[r['path'] for r in files+aliases]
    require(len(set(n.casefold() for n in all_names))==len(all_names),'Duplicate/case-colliding restored path')
    for name in all_names:payload_name(name)
    expected={r['path']:r for r in files}
    require(set(names)==set(expected)|{'manifest.json'},'Unexpected or missing archive member')
    for alias in aliases:
        require(alias['reason']=='SAME_INODE' and alias['target'] in expected,'Alias target is not a canonical payload')
        target=expected[alias['target']]
        require(alias['bytes']==target['bytes'] and alias['sha256']==target['sha256'],'Alias bytes differ')
    directories=manifest['empty_directories']
    require(len(set(x.casefold() for x in directories))==len(directories),'Duplicate empty directories')
    for name in directories:
        payload_name(name+'/.empty_directory_metadata.json')
        require(name.startswith('runtime/') and name.casefold() not in {x.casefold() for x in all_names},'Invalid empty directory scope')
        require(not any(x.startswith(name+'/') for x in all_names),'Claimed empty directory contains payload')
    return expected


def materialize_aliases(root,aliases):
    """Regular verified copies, preserving every fixed original path."""
    for alias in aliases:
        source=destination(root,alias['target']);dest=destination(root,alias['path'])
        require(source.is_file() and source.stat().st_size==alias['bytes'] and sha(source)==alias['sha256'],'Alias source differs')
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);no_links(dest.parent)
            temp=dest.with_name(dest.name+'.alias_'+str(os.getpid())+'.partial')
            with source.open('rb') as a,temp.open('xb') as b:shutil.copyfileobj(a,b,8<<20)
            require(sha(temp)==alias['sha256'],'Alias restore SHA differs');temp.rename(dest)
        require(dest.is_file() and dest.stat().st_size==alias['bytes'] and sha(dest)==alias['sha256'],'Restored alias differs')


def extract(archive,receipt):
    # No tar extraction API: every regular file has a validated fixed destination.
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        require(len(set(n.casefold() for n in names))==len(names),'Duplicate archive members')
        require(all(m.isfile() and not m.issym() and not m.islnk() for m in members),'Link/nonregular archive member')
        for name in names:payload_name(name)
        require('manifest.json' in names,'Manifest missing')
        mm=tar.getmember('manifest.json');require(mm.size<256<<20,'Manifest exceeds fixed safety ceiling')
        raw=tar.extractfile(mm).read();require(hashlib.sha256(raw).hexdigest()==receipt['manifest_sha256'],'Manifest SHA differs')
        manifest=json.loads(raw);expected=validate_manifest(manifest,receipt,names)
        expected['manifest.json']=dict(bytes=len(raw),sha256=receipt['manifest_sha256'])
        for member in members:
            wanted=expected[member.name];require(member.size==wanted['bytes'],'Archive size differs')
            target=destination(D,member.name)
            if target.exists():
                require(target.is_file() and target.stat().st_size==wanted['bytes'] and sha(target)==wanted['sha256'],'Existing payload differs; preserve it')
                continue
            target.parent.mkdir(parents=True,exist_ok=True);no_links(target.parent)
            temporary=target.with_name(target.name+'.extract_'+str(os.getpid())+'.partial')
            with temporary.open('xb') as out,tar.extractfile(member) as incoming:shutil.copyfileobj(incoming,out,8<<20)
            require(temporary.stat().st_size==wanted['bytes'] and sha(temporary)==wanted['sha256'],'Extracted SHA differs')
            temporary.rename(target)
    materialize_aliases(D,manifest['aliases'])
    for name in manifest['empty_directories']:
        path=destination(D,name);path.mkdir(parents=True,exist_ok=True);no_links(path)
        require(path.is_dir() and not any(path.iterdir()),'Empty failure-evidence directory differs')
    for row in manifest['files']+manifest['aliases']:
        p=destination(D,row['path']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Final restored SHA differs')
    return manifest


def sync_light(manifest):
    synced=0
    for row in manifest['files']+manifest['aliases']:
        if not row['path'].startswith('repo/'):continue
        require(row['bytes']<=10_000_000,'Large result remains in recovery; unchanged Git filter')
        source=destination(D,row['path']);target=destination(W,row['path'][5:])
        target.parent.mkdir(parents=True,exist_ok=True);no_links(target.parent)
        if not target.exists() or sha(target)!=row['sha256']:
            temp=target.with_name(target.name+'.R1sync_'+str(os.getpid())+'.partial')
            with source.open('rb') as a,temp.open('xb') as b:shutil.copyfileobj(a,b,8<<20)
            require(sha(temp)==row['sha256'],'Light copy changed');os.replace(temp,target)
        require(sha(target)==row['sha256'],'Light sync failed');synced+=1
    return synced


def fetch(receipt_sha,sync=False):
    require(os.name=='nt' and len(receipt_sha)==64,'Explicit server receipt on Windows required')
    no_links(D);require(RECOVERY.is_dir(),'Lu_Data recovery root is unavailable')
    dependencies=dependency_status()
    response=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'cat',REMOTE+'/receipt.json'],
                            capture_output=True,timeout=30,check=True)
    require(hashlib.sha256(response.stdout).hexdigest()==receipt_sha,'Server receipt SHA differs')
    receipt=json.loads(response.stdout);scope(receipt)
    require(receipt['archive']==REMOTE+'/'+ARCHIVE and receipt['archive_payload_SHA_PASS'] is True,'Unverified archive')
    D.mkdir(exist_ok=True);archive=destination(D,ARCHIVE)
    require(shutil.disk_usage(D).free>=receipt['payload_bytes']+receipt['alias_bytes']+receipt['bytes']+(512<<20),'Insufficient disk margin')
    if not archive.exists():
        temporary=D/(ARCHIVE+'.download_'+str(os.getpid())+'.partial')
        require(not temporary.exists(),'Existing partial transfer requires explicit inspection')
        subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST+':'+receipt['archive'],str(temporary)],check=True,timeout=7200)
        require(temporary.stat().st_size==receipt['bytes'] and sha(temporary)==receipt['sha256'],'Downloaded SHA differs');temporary.rename(archive)
    require(archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256'],'Archive SHA differs')
    manifest=extract(archive,receipt);synced=sync_light(manifest) if sync else 0
    result=dict(receipt,server_receipt_sha256=receipt_sha,local_archive=str(archive),local_directory=str(D),archive_SHA_PASS=True,
                all_payload_SHA_PASS=True,all_alias_SHA_PASS=True,safe_extraction_PASS=True,synced_light_files=synced,
                dependency_receipt_checks=dependencies,old_dependency_payloads_reverified=False)
    save(D/'server_receipt.json',receipt);save(D/'local_verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--receipt-sha256',required=True);parser.add_argument('--sync-light',action='store_true')
    args=parser.parse_args();print(json.dumps(fetch(args.receipt_sha256,args.sync_light)),flush=True)
