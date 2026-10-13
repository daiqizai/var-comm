"""Download and stream-verify a closed EMAIN increment onto Lu_Data.

No extraction of large duplicate assets, no deletion, no scientific calls.
Existing destination files are reused only when their SHA matches the receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time

ALIAS='DevMachineModel-H100-4C-P1660-01'
REMOTE='/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/entropy_main_common500_delivery_20261013_v1'
DEST=Path('D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011/entropy_main_common500_20261013_v1')
WORK=Path('C:/Users/11946/Documents/ChatGPT/comm')
ARCHIVE='entropy_main_common500_20261013_v1.tar.gz'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def safe_write(path,data):
    path=Path(path)
    if path.exists():require(path.read_bytes()==data,'Existing file differs; preserve it: '+str(path));return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data)


def download(name,digest=None):
    target=DEST/name
    if target.exists():
        require(digest is not None and sha(target)==digest,'Existing destination requires matching expected SHA: '+str(target))
        return target
    partial=DEST/(name+'.download')
    require(not partial.exists(),'Interrupted download is preserved: '+str(partial))
    subprocess.run(['scp','-q',ALIAS+':'+REMOTE+'/'+name,str(partial)],check=True)
    if digest is not None:require(sha(partial)==digest,'Downloaded SHA differs: '+name)
    require(not target.exists(),'Destination appeared during transfer')
    partial.rename(target)
    return target


def verify(sync_light=False):
    receipt=json.loads((DEST/'backup_archive_receipt.json').read_text())
    archive=DEST/ARCHIVE
    require(receipt['status']=='REMOTE_E1_E2_INCREMENT_CREATED_LOCAL_VERIFICATION_PENDING','Unexpected archive receipt')
    require(archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256'],'Archive length/SHA failed')
    checked=set();synced=0
    with tarfile.open(archive,'r:gz') as tar:
        manifest_file=tar.extractfile('backup_manifest.json')
        require(manifest_file is not None,'Missing manifest')
        manifest_bytes=manifest_file.read()
        require(hashlib.sha256(manifest_bytes).hexdigest()==receipt['manifest_sha256'],'Manifest SHA differs')
        manifest=json.loads(manifest_bytes)
        require(manifest['schema']=='EMAIN_E1_E2_INCREMENTAL_BACKUP_V1' and manifest['scope']=='E1_E2','Wrong increment identity')
        expected={r['path']:r for r in manifest['files']}
        require(len(expected)==manifest['file_count']==receipt['payload_files'],'Manifest names duplicate/count differs')
        seen=set()
        for member in tar:
            p=PurePosixPath(member.name)
            require(member.isfile() and not member.issym() and not member.islnk() and not p.is_absolute()
                    and '..' not in p.parts and ':' not in member.name and member.name not in seen,'Unsafe/duplicate archive member')
            seen.add(member.name)
            stream=tar.extractfile(member);h=hashlib.sha256();size=0;parts=[]
            light=sync_light and member.name.startswith('repo/results/entropy_main_common500_20261013/')
            keep=member.name in ('backup_manifest.json','RESTORE.md') or light
            for block in iter(lambda:stream.read(8<<20),b''):
                size+=len(block);h.update(block)
                if keep:parts.append(block)
            if member.name in expected:
                row=expected[member.name]
                require(size==row['bytes'] and h.hexdigest()==row['sha256'],'Payload SHA failed: '+member.name)
                checked.add(member.name)
            else:
                require(member.name in ('backup_manifest.json','RESTORE.md'),'Undeclared payload')
                field='manifest_sha256' if member.name=='backup_manifest.json' else 'restore_sha256'
                require(h.hexdigest()==receipt[field],'Metadata SHA failed')
                safe_write(DEST/member.name,b''.join(parts))
            if light:
                target=WORK.joinpath(*p.parts[1:]).resolve()
                require(target.is_relative_to(WORK.resolve()),'Light sync escaped workspace')
                safe_write(target,b''.join(parts));synced+=1
        require(checked==set(expected),'Payload members missing')
        require(seen==set(expected)|{'backup_manifest.json','RESTORE.md'},'Archive member set differs')
    result=dict(status='PASS_ARCHIVE_AND_EVERY_PAYLOAD_SHA',verified_unix=time.time(),
                archive_sha256=receipt['sha256'],archive_bytes=receipt['bytes'],payload_files_verified=len(checked),
                verified_by_streamed_extraction=True,full_extracted_duplicate_kept=False,
                light_result_files_synced_to_workspace=synced,independently_complete_restore=False,
                dependencies=manifest['dependencies'],old_dependencies_reverified=False,
                E3_timing_included=False,deleted_files=0,new_scientific_calls=0)
    destination=DEST/'local_verification.json'
    if destination.exists():
        previous=json.loads(destination.read_text());require(previous['archive_sha256']==result['archive_sha256'],'Existing verification belongs to different package')
        destination=DEST/('local_verification_'+str(int(time.time()))+'.json')
    safe_write(destination,(json.dumps(result,indent=2)+'\n').encode())
    return result


def fetch(sync_light=False):
    DEST.mkdir(parents=True,exist_ok=True)
    receipt_path=DEST/'backup_archive_receipt.json'
    if receipt_path.exists():
        # Read-only remote receipt comparison authorizes reusing the prior download.
        remote=subprocess.check_output(['ssh','-T','-o','BatchMode=yes',ALIAS,'cat',REMOTE+'/backup_archive_receipt.json'])
        require(remote==receipt_path.read_bytes(),'Remote receipt changed; preserve both attempts')
    else:download('backup_archive_receipt.json')
    receipt=json.loads(receipt_path.read_text())
    require(receipt['archive']==REMOTE+'/'+ARCHIVE,'Receipt points outside the expected task package')
    download(ARCHIVE,receipt['sha256'])
    return verify(sync_light)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('fetch','verify'))
    p.add_argument('--sync-light',action='store_true')
    args=p.parse_args();print(json.dumps(fetch(args.sync_light) if args.command=='fetch' else verify(args.sync_light),indent=2))
