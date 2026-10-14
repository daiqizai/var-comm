"""Restore only archived original pixels/metric assets; no scientific imports.

Two explicit roles: prepare freezes exact backup members; restore reads them
once into the new increment. The old archives and their metadata are untouched.
"""
from pathlib import Path, PurePosixPath
from collections import defaultdict, Counter
import argparse, hashlib, json, os, stat, sys, tarfile, time, traceback

HERE=Path(__file__).resolve().parent
DRIVE=Path('D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011/quality_metrics_supplement_20261014_v1')
STATIC=Path('D:/Lu_Data/VAR_COMM_MIGRATION_20261008/required_static_v1')
ADD=Path('D:/Lu_Data/VAR_COMM_MIGRATION_20261008/runtime_code_terminal_addendum_r2')
INC=Path('D:/Lu_Data/VAR_COMM_MIGRATION_20261008/increment_20261009_v1')
OLD='/home/liulu/projects/VAR_COMM/'
RAW=OLD+'outputs/MAIN-RAW64-20261007/unified500_same_rx_parallel_images_r2/'
ROOT=DRIVE/'input_restore'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf8',newline='\n') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def safe(p):
    assert p.is_absolute() and '..' not in p.parts
    for q in (p,*p.parents):
        if os.path.lexists(q):
            s=q.lstat();assert not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&1024
def destination(name):
    p=PurePosixPath(name);assert str(p)==name and not p.is_absolute() and '\\' not in name and ':' not in name
    assert all(x not in ('','..','.') for x in p.parts)
    d=ROOT.joinpath(*p.parts);safe(d);assert d.resolve().is_relative_to(ROOT.resolve());return d
def prepare():
    assert os.name=='nt';safe(DRIVE);assert not (DRIVE/'restore_plan.json').exists()
    model=read(HERE/'context/original_modelmanifest.json');want={}
    for metric in ('dists','dreamsim','ms_ssim'):
        for key,value in model[metric].items():
            if isinstance(value,dict) and 'path' in value:
                if 'files' in value:
                    for rel,digest in value['files'].items():want[value['path']+'/'+rel]=digest
                elif 'sha256' in value:want[value['path']]=value['sha256']
    chosen=[];seen=set();alias_targets={}
    for base,glob,completion in ((STATIC,'volume_*_manifest.json','packing_completion.json'),(ADD,'*.content_manifest.json','packing_completion.json')):
        done=read(base/completion);volumes=done['volumes']
        by_name={v.get('name',v.get('file')):v for v in volumes}
        for manifest in sorted(base.glob(glob)):
            data=read(manifest);rows=data if isinstance(data,list) else data.get('files',data.get('entries'))
            assert isinstance(rows,list)
            if base==STATIC:volname='VAR_COMM_REQUIRED_STATIC_'+manifest.stem.split('_')[1]+'.tar.zst'
            else:volname=manifest.name.removesuffix('.content_manifest.json')
            assert volname in by_name
            volume=base/volname;meta=by_name[volname];assert volume.stat().st_size==meta['bytes']
            for row in rows:
                old=row['source_path'];digest=row.get('sha256',row.get('expected_sha256'))
                selected=old in want or (old.startswith(RAW) and (old.endswith('.npz') or old.endswith('.json')))
                if not selected:continue
                if row['kind']=='symlink':
                    # One historical VGG alias points to an already sealed file.
                    # Restore that verified referent as an ordinary file at the
                    # original alias spelling; never install a symlink/junction.
                    assert old=='/home/liulu/.cache/torch/hub/checkpoints/vgg16-397923af.pth'
                    assert row['referent_sha256']==want[old] and row['target'].startswith('/home/liulu/projects/')
                    alias_targets[row['target']]=dict(old_path=old,path=row['archive'],historical_alias_target=row['target'])
                    want[row['target']]=want.pop(old);continue
                if old in want:assert digest==want[old]
                assert row['kind']=='file' and old not in seen;seen.add(old)
                actual=dict(old_path=old,path=row['archive'],member=row['archive'],sha256=digest,bytes=row['bytes'],
                    volume=str(volume),volume_bytes=meta['bytes'],volume_sha256=meta['sha256'],backup_manifest=str(manifest))
                actual.update(alias_targets.get(old,{}));chosen.append(actual)
    assert set(want).issubset(seen),'Required metric asset is absent from fixed backups'
    assert sum('/images/' in r['old_path'] and r['old_path'].startswith(RAW) for r in chosen)==500
    assert sum('/sources/' in r['old_path'] and r['old_path'].startswith(RAW) for r in chosen)==500
    inv=read(INC/'inventory.json');targets=[r for r in inv['entries'] if r.get('type')=='file' and
        r['path'].startswith('results/paper_supplement_20261008/a1_bpg_adaptive/') and '/holdout/reconstructions/' in r['path'] and r['path'].endswith('.npz')]
    assert len(targets)==500
    objects={r['object']:r for r in inv['objects']}
    volumes={r['file']:r for r in read(INC/'packing_completion.json')['volumes']}
    for row in targets:
        obj=objects[row['object']];assert obj['sha256']==row['sha256'] and obj['bytes']==row['bytes']
        assert obj['local_source'] is None and obj['volume'] in volumes
        meta=volumes[obj['volume']];volume=INC/'archives'/obj['volume'];assert volume.stat().st_size==meta['bytes']
        chosen.append(dict(old_path=OLD+row['path'],path='VAR_COMM/'+row['path'],member=obj['object'],sha256=row['sha256'],bytes=row['bytes'],
            volume=str(volume),volume_bytes=meta['bytes'],volume_sha256=meta['sha256'],backup_manifest=str(INC/'inventory.json')))
    assert len({r['path'].casefold() for r in chosen})==len(chosen)
    for r in chosen:destination(r['path'])
    plan=dict(schema='QUALITY_METRIC_ORIGINAL_BYTE_RESTORE_V1',files=chosen,metric_assets=len(want),raw_source_archives=500,bpg_source_archives=500,
        bytes=sum(r['bytes'] for r in chosen),new_training=0,new_channel=0,new_reconstruction=0,new_metric_scores=0,
        archive_SHA_status='Historical verified archive receipts; every selected member is independently SHA-verified on this restore',
        old_modelmanifest=dict(path=str(HERE/'context/original_modelmanifest.json'),sha256=sha(HERE/'context/original_modelmanifest.json')))
    save(DRIVE/'restore_plan.json',plan);print(json.dumps({k:v for k,v in plan.items() if k!='files'}))
def restore(expected):
    assert os.name=='nt';safe(ROOT);assert not ROOT.exists(),'Existing restore attempt must be diagnosed, never silently repeated'
    p=DRIVE/'restore_plan.json';assert sha(p)==expected;plan=read(p);assert plan['schema']=='QUALITY_METRIC_ORIGINAL_BYTE_RESTORE_V1'
    sys.path.insert(0,str(DRIVE/'tools'));import zstandard
    ROOT.mkdir();start=time.monotonic();groups=defaultdict(list)
    for row in plan['files']:groups[row['volume']].append(row)
    saved=[]
    try:
        for index,(name,rows) in enumerate(sorted(groups.items())):
            volume=Path(name);before=volume.stat();assert before.st_size==rows[0]['volume_bytes']
            wanted={r['member']:r for r in rows};assert len(wanted)==len(rows);seen=set()
            with volume.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as stream,tarfile.open(fileobj=stream,mode='r|') as tar:
                for member in tar:
                    if member.name not in wanted:continue
                    assert member.name not in seen and member.isfile() and not member.issym() and not member.islnk()
                    row=wanted[member.name];assert member.size==row['bytes'];target=destination(row['path']);target.parent.mkdir(parents=True,exist_ok=True);safe(target.parent)
                    tmp=target.with_name(target.name+'.restore.partial');h=hashlib.sha256();count=0
                    with tar.extractfile(member) as source,tmp.open('xb') as out:
                        for b in iter(lambda:source.read(8<<20),b''):out.write(b);h.update(b);count+=len(b)
                        out.flush();os.fsync(out.fileno())
                    assert count==row['bytes'] and h.hexdigest()==row['sha256'];tmp.rename(target)
                    assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256'];seen.add(member.name);saved.append(row)
                    if seen==set(wanted):break
            after=volume.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and seen==set(wanted)
            save(ROOT/f'volume_{index:03d}_completion.json',dict(volume=name,selected_files=len(rows),selected_file_SHA_PASS=True,source_modified=False))
            print('RESTORED_VOLUME',index,len(rows),flush=True)
        assert len(saved)==len(plan['files']);save(ROOT/'completion.json',dict(status='COMPLETE_EXISTING_PIXEL_AND_METRIC_ASSET_BYTE_RESTORE',files=len(saved),bytes=plan['bytes'],
            plan_sha256=expected,all_selected_file_SHA_PASS=True,old_archives_unchanged=True,new_training=0,new_channel=0,new_reconstruction=0,new_metric_scores=0,seconds=time.monotonic()-start))
    except BaseException:
        save(ROOT/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc(),restored_files=len(saved)));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('role',choices=('prepare','restore'));a.add_argument('--plan-sha256');v=a.parse_args()
    if v.role=='prepare':prepare()
    else:assert v.plan_sha256;restore(v.plan_sha256)
