"""Restore the 500 sealed whole-scale VAR arrays, using original byte codec."""
from pathlib import Path,PurePosixPath
import ast,hashlib,json,lzma,os,stat,tarfile,time
HERE=Path(__file__).resolve().parent
BASE=Path('D:/Lu_Data/VAR_COMM_WCL_INCREMENT_20261009_v1/final_wcl_v2')
ROOT=Path('D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011/quality_metrics_supplement_20261014_v1/whole19_restore_v1')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def long(p):return Path('\\\\?\\'+str(p.resolve()))
def save(p,v):
    with long(p).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert not ROOT.exists();ROOT.mkdir()
selected=json.loads((HERE/'whole19_selected_inventory.json').read_text(encoding='utf8'));files=selected['files'];blobs=selected['blobs'];assert len(files)==len(blobs)==500
tool=BASE/'wcl_final_increment_v2.py';nodes=[n for n in ast.parse(tool.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name in ('decode_blob','unshuffle4')];assert len(nodes)==2
ns=dict(hashlib=hashlib,lzma=lzma,RAW_CODEC='raw_outer_gzip1',SHUFFLE_CODEC='shuffle4_xz0');exec(compile(ast.Module(body=nodes,type_ignores=[]),str(tool),'exec'),ns)
volnames={v['volume'] for v in blobs.values()};assert len(volnames)==1;name=next(iter(volnames));candidates=[p for p in (BASE/name,BASE/'archives'/name) if p.exists()];assert len(candidates)==1;volume=candidates[0]
expected=next(r for r in selected['volumes'] if r['name']==name);before=volume.stat();assert before.st_size==expected['bytes'] and sha(volume)==expected['sha256']
targets={r['chunks'][0]['sha256']:r for r in files};assert len(targets)==500 and all(len(r['chunks'])==1 and r['provided_by'] is None for r in files)
start=time.monotonic();seen=set();records=[]
with tarfile.open(volume,'r|gz') as tar:
    for member in tar:
        digest=member.name.removeprefix('blobs/')
        if digest not in targets:continue
        assert member.isfile() and not member.issym() and not member.islnk() and digest not in seen
        data=tar.extractfile(member).read();raw=ns['decode_blob'](data,blobs[digest],digest);row=targets[digest]
        relative=PurePosixPath(row['restore_path']);assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in row['restore_path']
        target=long(ROOT.joinpath(*relative.parts));target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256'];seen.add(digest);records.append(dict(old_path=row['path'],restore_path=row['restore_path'],bytes=row['bytes'],sha256=row['sha256']))
        if len(seen)%100==0:print('RESTORED',len(seen),flush=True)
        if len(seen)==500:break
assert len(seen)==500;after=volume.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
save(ROOT/'completion.json',dict(status='PASS_500_ORIGINAL_WHOLE19_ARRAYS_BYTE_RESTORED',files=records,source_volume=expected,source_volume_path=str(volume),original_codec_source=dict(path=str(tool),sha256=sha(tool)),all_file_SHA_PASS=True,new_model_calls=0,new_channel_calls=0,new_reconstruction_calls=0,seconds=time.monotonic()-start))
print(json.dumps(dict(files=500,bytes=sum(r['bytes'] for r in records),completion_sha256=sha(ROOT/'completion.json'),scientific_calls=0)),flush=True)
