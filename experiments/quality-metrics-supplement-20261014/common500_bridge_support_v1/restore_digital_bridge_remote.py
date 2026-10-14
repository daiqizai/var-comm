from pathlib import Path,PurePosixPath
import hashlib,json,os,tarfile,time
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1')
TRANSPORT=T/'digital_input_bridge_transport_v2';DEST=T/'digital_input_bridge_v2'
ARCHIVE=TRANSPORT/'digital_input_bridge_v2.tar.gz';EXPECTED='93e1b11d9b214827baf15ec1fd10126b67c9fedd5cbb005d22ff4e5552b8ad7a'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
assert ARCHIVE.stat().st_size==2377544488 and sha(ARCHIVE)==EXPECTED and not DEST.exists()
started=time.monotonic();DEST.mkdir();seen=set()
with tarfile.open(ARCHIVE,'r|gz') as tar:
    for m in tar:
        q=PurePosixPath(m.name);assert not q.is_absolute() and '..' not in q.parts and '\\' not in m.name and m.name not in seen and m.isfile() and not m.issym() and not m.islnk();seen.add(m.name)
        p=DEST.joinpath(*q.parts);p.parent.mkdir(parents=True,exist_ok=True)
        with tar.extractfile(m) as src,p.open('xb') as out:
            for b in iter(lambda:src.read(8<<20),b''):out.write(b)
            out.flush();os.fsync(out.fileno())
assert len(seen)==503
manifest=DEST/'manifest.json';assert sha(manifest)=='8f18ff4693301b2ca1cd4803a02f9623a99c0c80dae09ced14d5e4b1bfb2d6aa'
v=json.loads(manifest.read_text())
for row in v['archives']+[v['frame_index'],v['sources']]:
    p=Path(row['path']);assert p.is_relative_to(DEST) and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
done=dict(status='PASS_503_BRIDGE_FILES_ARCHIVE_AND_ALL_PAYLOAD_SHA',archive=pin(ARCHIVE),manifest=pin(manifest),files=503,source_count=500,logical_frames=13500,new_model_calls=0,new_channel_calls=0,new_reconstruction_calls=0,new_metric_scores=0,seconds=time.monotonic()-started)
with (DEST/'restoration_completion.json').open('x') as f:json.dump(done,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(pin(DEST/'restoration_completion.json')))
