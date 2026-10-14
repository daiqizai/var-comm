"""Explicit byte-only transport of restored official metric assets."""
from pathlib import Path, PurePosixPath
import argparse,copy,hashlib,json,os,tarfile

HERE=Path(__file__).resolve().parent
LOCAL=Path('D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011/quality_metrics_supplement_20261014_v1')
REMOTE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):
    with Path(p).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def pack():
    assert os.name=='nt' and not (LOCAL/'metric_assets.tar.gz').exists()
    completion=read(LOCAL/'input_restore/completion.json');assert completion['all_selected_file_SHA_PASS'] is True
    plan=read(LOCAL/'restore_plan.json');files=[r for r in plan['files'] if not ('unified500_same_rx' in r['old_path'] or 'a1_bpg_adaptive' in r['old_path'])];assert len(files)==61
    model=read(HERE/'context/original_modelmanifest.json');models={k:copy.deepcopy(model[k]) for k in ('dists','dreamsim','ms_ssim')}
    def relocate(value):
        if isinstance(value,dict):
            if 'path' in value:
                old=value['path'];prefix='/home/liulu/projects/VAR_COMM/'
                rel='VAR_COMM/'+old[len(prefix):] if old.startswith(prefix) else 'external'+old
                value['path']=str(REMOTE/'metric_assets'/rel)
            for x in value.values():relocate(x)
        elif isinstance(value,list):
            for x in value:relocate(x)
    relocate(models)
    manifest=dict(schema='OFFICIAL_THREE_METRIC_ASSETS_RELOCATED_V1',models=models,
        files=[{k:r[k] for k in ('old_path','path','sha256','bytes')} for r in files],old_modelmanifest_sha256=sha(HERE/'context/original_modelmanifest.json'),
        original_selected_member_SHA_PASS=True,all_file_SHA_PASS=False,no_PFS_symlink=True,model_asset_restoration_only=True,scientific_scores=0)
    save(LOCAL/'metric_assets_pending_manifest.json',manifest)
    with tarfile.open(LOCAL/'metric_assets.tar.gz','w:gz',compresslevel=1) as tar:
        for row in files:
            p=LOCAL/'input_restore'/row['path'];assert not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
            tar.add(p,arcname=row['path'],recursive=False)
        tar.add(LOCAL/'metric_assets_pending_manifest.json',arcname='manifest.json',recursive=False)
    result=dict(path=str(LOCAL/'metric_assets.tar.gz'),bytes=(LOCAL/'metric_assets.tar.gz').stat().st_size,sha256=sha(LOCAL/'metric_assets.tar.gz'),files=61,source_bytes=sum(r['bytes'] for r in files),new_channel=0,new_receiver=0,new_training=0,new_metric_scores=0)
    save(LOCAL/'metric_assets_archive_receipt.json',result);print(json.dumps(result),flush=True)
def materialize(expected):
    assert os.name=='posix';archive=REMOTE/'metric_assets.tar.gz';assert sha(archive)==expected
    dest=REMOTE/'metric_assets';assert not dest.exists() and not dest.is_symlink()
    with tarfile.open(archive,'r:gz') as tar:
        meta=tar.extractfile('manifest.json').read();manifest=json.loads(meta);wanted={r['path']:r for r in manifest['files']};assert len(wanted)==61
        members=tar.getmembers();assert {m.name for m in members}==set(wanted)|{'manifest.json'} and len(members)==62
        dest.mkdir();seen=set()
        for m in members:
            if m.name=='manifest.json':continue
            assert m.isfile() and not m.issym() and not m.islnk() and m.name in wanted and m.name not in seen
            parts=PurePosixPath(m.name);assert not parts.is_absolute() and '\\' not in m.name and '..' not in parts.parts
            row=wanted[m.name];assert m.size==row['bytes'];p=dest.joinpath(*parts.parts);assert p.is_relative_to(dest);p.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();n=0
            with tar.extractfile(m) as source,p.open('xb') as out:
                for b in iter(lambda:source.read(8<<20),b''):out.write(b);h.update(b);n+=len(b)
            assert n==row['bytes'] and h.hexdigest()==row['sha256'] and sha(p)==row['sha256'];seen.add(m.name)
        assert seen==set(wanted)
    manifest.update(all_file_SHA_PASS=True,transport_archive=dict(path=str(archive),sha256=expected,bytes=archive.stat().st_size),new_scores=0)
    save(REMOTE/'metric_assets_manifest.json',manifest);print(json.dumps(dict(path=str(REMOTE/'metric_assets_manifest.json'),sha256=sha(REMOTE/'metric_assets_manifest.json'),files=61,scientific_calls=0)),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('role',choices=('pack','materialize'));a.add_argument('--archive-sha256');v=a.parse_args()
    if v.role=='pack':pack()
    else:assert v.archive_sha256;materialize(v.archive_sha256)
