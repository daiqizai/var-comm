from pathlib import Path,PurePosixPath
import hashlib,json,os,stat,zipfile
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1')
W=T/'dependency_wheels_v1';D=T/'dependency_overlay_v1';S=D/'sitepackages'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
manifest=json.loads((W/'wheel_manifest.json').read_text());assert manifest['official_SHA_PASS'] is True and not D.exists()
S.mkdir(parents=True);files=[]
for wheel in manifest['files']:
    p=W/wheel['filename'];assert p.stat().st_size==wheel['bytes'] and sha(p)==wheel['sha256']
    with zipfile.ZipFile(p) as z:
        for r in z.infolist():
            if r.is_dir():continue
            rel=PurePosixPath(r.filename);assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in r.filename and not stat.S_ISLNK(r.external_attr>>16)
            target=S.joinpath(*rel.parts);assert target.is_relative_to(S);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(z.read(r))
            files.append(pin(target))
save(D/'manifest.json',dict(status='PRIVATE_THREE_MISSING_DEPENDENCIES_READY',wheels=pin(W/'wheel_manifest.json'),files=files,packages=manifest['files'],old_environment_unchanged=True,metric_source_and_weights_unchanged=True,scientific_calls=0))
print(json.dumps(dict(manifest=pin(D/'manifest.json'),files=len(files))))
