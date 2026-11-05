"""Expand official COCO training split, preserving every held-out identity.

Only authoritative ZIP-size/CRC-verified JPEGs enter training. Native RGB hashes
deduplicate train and exclude development/calibration/test. No model calls.
Each missing download is attempted once; failures remain in a separate ledger.
"""
from pathlib import Path
import concurrent.futures, hashlib, io, json, time, urllib.request, zlib
import numpy as np
from PIL import Image
from runtime import L,OLD,write,sha
def main():
    out=L/'corpus_coco116k_v2';out.mkdir(exist_ok=False)
    entries=json.loads((OLD/'coco_official_directory/entries.json').read_text())
    split=json.loads((OLD/'asset_subset_v3/source_split_names.json').read_text())['splits']
    excluded=set(split['development1000']+split['calibration1000'])
    names=sorted(n for n in entries if n.startswith('train2017/') and n not in excluded)
    assert len(names)==116287
    reuse={};held=set()
    for kind in ['train','development','calibration','test']:
        data=json.loads((OLD/'data_union_verified_v1'/f'{kind}_manifest.json').read_text())
        for row in data['sources']:
            if kind=='train':reuse[row['name']]=row
            else:held.add(row['native_rgb_sha256'])
    write(out/'registration.json',dict(names=names,excluded_names=sorted(excluded),heldout_native_hashes=sorted(held),
          directory_sha256=sha(OLD/'coco_official_directory/entries.json'),download_attempts_per_missing=1,
          source='COCO 2017 official image archive; original image licenses apply',max_seconds=43200))
    images=L/'data/coco2017';images.mkdir(parents=True,exist_ok=True)
    begin=time.time()
    def fetch(n):
        if n in reuse:return dict(reuse[n],origin='verified_existing_train')
        if time.time()-begin>43000:raise TimeoutError('corpus window exhausted')
        e=entries[n];p=images/n;p.parent.mkdir(exist_ok=True)
        if p.exists():b=p.read_bytes()
        else:
            with urllib.request.urlopen('https://s3.amazonaws.com/images.cocodataset.org/'+n,timeout=60) as r:b=r.read()
        assert len(b)==e['size'] and zlib.crc32(b)==e['crc32'],n
        with Image.open(io.BytesIO(b)) as im:a=np.asarray(im.convert('RGB')).copy()
        if not p.exists():
            tmp=p.with_suffix('.partial');tmp.write_bytes(b);tmp.replace(p)
        return dict(name=n,source_id='coco2017:'+n,path=str(p),sha256=hashlib.sha256(b).hexdigest(),
                    native_rgb_sha256=hashlib.sha256(a.tobytes()).hexdigest(),native_rgb_shape=list(a.shape),origin='official_S3_HTTPS',**{k:e[k] for k in ['size','crc32']})
    good=[];fail=[]
    # Existing verified training data are available immediately as a clearly
    # identified bootstrap shard. It is never relabelled as the expanded corpus.
    unique={}
    for n in sorted(reuse):
        r=reuse[n];h=r['native_rgb_sha256']
        if h not in held:unique.setdefault(h,r)
    write(out/'bootstrap_train_manifest.json',dict(sources=list(unique.values()),images=len(unique),role='initial bootstrap shard, not full corpus',heldout_excluded=True))
    with (out/'journal.jsonl').open('x') as journal,concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        futures={ex.submit(fetch,n):n for n in names}
        for f in concurrent.futures.as_completed(futures):
            try:r=f.result();good.append(r);record=dict(r,status='verified')
            except Exception as e:r=dict(name=futures[f],error=repr(e));fail.append(r);record=dict(status='failed',**r)
            journal.write(json.dumps(record)+'\n');journal.flush()
            if (len(good)+len(fail))%500==0:write(out/'progress.json',dict(verified=len(good),failed=len(fail),total=len(names),seconds=time.time()-begin))
    unique={};duplicates=[];leaks=[]
    for r in sorted(good,key=lambda x:x['name']):
        h=r['native_rgb_sha256']
        if h in held:leaks.append(r['name'])
        elif h in unique:duplicates.append(dict(removed=r['name'],kept=unique[h]['name']))
        else:unique[h]=r
    write(out/'download_failures.json',fail);write(out/'deduplication.json',dict(duplicate_train=duplicates,excluded_heldout_matches=leaks))
    write(out/'train_manifest.json',dict(sources=list(unique.values()),images=len(unique),expected_candidate_images=len(names),failed=len(fail),complete_download=not fail,heldout_excluded=True))
    write(out/'complete.json',dict(complete_download=not fail,usable_unique_training=len(unique),failures=len(fail),seconds=time.time()-begin,manifest_sha256=sha(out/'train_manifest.json')))
    if fail:raise RuntimeError(f'{len(fail)} download failures retained; usable shard is explicitly incomplete')
if __name__=='__main__':main()
