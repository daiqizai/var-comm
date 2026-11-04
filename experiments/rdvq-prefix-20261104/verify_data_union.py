"""Verify the fixed COCO split and exact decoded-pixel cross-split identities.

No replacement, download, re-encoding, or model inference. Missing files are a
reported transport gap. Existing verification outputs are never overwritten.
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,time,zlib
import numpy as np
from PIL import Image

def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args()
    r=json.loads(Path(a.request).read_text());O=Path(r['output']);assert O.is_dir()
    splits=json.loads(Path(r['split_names']).read_text())['splits'];entries=json.loads(Path(r['official_entries']).read_text())
    groups={'train20k':'train','development1000':'development','calibration1000':'calibration','test500':'test'}
    assert [len(splits[s]) for s in groups]==[20000,1000,1000,500]
    jobs=[(groups[s],n) for s in groups for n in splits[s]];assert len(set(n for _,n in jobs))==22500
    roots=[Path(v) for v in r['source_roots']];explicit={v['name']:Path(v['path']) for v in json.loads(Path(r['priority_manifest']).read_text())['images']}
    def check(job):
        split,n=job;e=entries[n];choices=([explicit[n]] if n in explicit else [])+[d/n for d in roots];p=None;invalid=[]
        for q in dict.fromkeys(choices):
            if not q.exists():continue
            b=q.read_bytes()
            if len(b)==e['size'] and zlib.crc32(b)==e['crc32']:p=q;break
            invalid.append(str(q))
        if p is None:return dict(name=n,split=split,status='missing_or_incomplete',invalid_paths=invalid)
        with Image.open(p) as im:rgb=np.asarray(im.convert('RGB')).copy()
        assert rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
        h=hashlib.sha256();h.update(str(rgb.shape).encode());h.update(rgb.tobytes())
        return dict(name=n,source_id=Path(n).stem,split=split,path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),crc32=e['crc32'],native_rgb_shape=list(rgb.shape),native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),shape_and_pixels_sha256=h.hexdigest(),status='verified')
    start=time.time();results=[]
    with (O/'verification.jsonl').open('x') as log,concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for v in pool.map(check,jobs):
            results.append(v);log.write(json.dumps(v)+'\n');log.flush()
            if len(results)%100==0:write(O/'progress.json',dict(checked=len(results),total=22500,seconds=time.time()-start))
    missing=[v for v in results if v['status']!='verified'];write(O/'missing.json',missing)
    if missing:raise RuntimeError(f'{len(missing)} exact data identities missing; training is not admitted')
    lookup={};duplicates=[]
    for v in results:
        h=v['shape_and_pixels_sha256']
        if h in lookup:
            prior=lookup[h];duplicates.append(dict(source_a=prior['name'],split_a=prior['split'],source_b=v['name'],split_b=v['split'],native_rgb_sha256=h))
        else:lookup[h]=v
    cross=[v for v in duplicates if v['split_a']!=v['split_b']]
    write(O/'duplicate_audit.json',dict(exact_pixel_duplicates=duplicates,cross_split_duplicates=cross,near_duplicate_detection='NOT_RUN',public_pretraining_image_manifest='UNAVAILABLE; overlap not certified'))
    if cross:raise RuntimeError('Exact native-pixel cross-split duplicates; preserve IDs and stop training admission')
    for split in groups.values():write(O/(split+'_manifest.json'),dict(sources=[v for v in results if v['split']==split]))
    write(O/'complete.json',dict(passed=True,images=22500,split_counts={s:sum(v['split']==s for v in results) for s in groups.values()},same_split_exact_duplicates=len(duplicates),cross_split_exact_duplicates=0,new_model_calls=0,seconds=time.time()-start))
if __name__=='__main__':main()
