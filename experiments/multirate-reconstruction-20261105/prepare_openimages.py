"""Fixed 100k Open Images training candidates for the next data allocation.

Official CVDF images are resized copies (long side <=1024), not original Flickr
pixels. Keep attribution metadata, HTTPS ETag/size, file and decoded-pixel hashes.
This job does not alter an active training manifest or download eval images.
"""
import csv,concurrent.futures,hashlib,heapq,io,json,re,time,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from runtime import L,OLD,write,sha
URL='https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
def main():
    out=L/'corpus_openimages100k_v1';out.mkdir(exist_ok=False);begin=time.time()
    write(out/'registration.json',dict(selection='100000 smallest SHA256(seed:ImageID) among official train rows with CC-BY license',seed=20261105,
        max_candidates=100000,index_url=URL,images='https://s3.amazonaws.com/open-images-dataset/train/{ImageID}.jpg',max_seconds=43200,
        references=['https://storage.googleapis.com/openimages/web/download_v6.html','https://github.com/cvdfoundation/open-images-dataset'],
        use_in_active_training=False,admission='new manifest at a later explicit training allocation; no hot change'))
    index=out/'official_train_index.csv';tmp=index.with_suffix('.partial')
    with urllib.request.urlopen(URL,timeout=90) as response,tmp.open('xb') as f:
        total=int(response.headers['Content-Length']);done=0;metadata=dict(response.headers)
        while True:
            block=response.read(1<<20)
            if not block:break
            f.write(block);done+=len(block)
            if done%(16<<20)==0:write(out/'progress.json',dict(stage='index',bytes=done,total_bytes=total))
    assert done==total;tmp.replace(index)
    write(out/'index_identity.json',dict(path=str(index),sha256=sha(index),bytes=done,headers=metadata))
    def candidates():
        with index.open(newline='') as f:
            for row in csv.DictReader(f):
                if 'creativecommons.org/licenses/by/' not in row.get('License',''):continue
                sid=row['ImageID'];assert re.fullmatch('[0-9a-fA-F]{16}',sid)
                yield (hashlib.sha256(('20261105:'+sid).encode()).hexdigest(),sid,row)
    selected=heapq.nsmallest(100000,candidates(),key=lambda v:v[0]);assert len(selected)==100000
    write(out/'selected_metadata.json',[v[2] for v in selected])
    held=set()
    for split in ['development','calibration','test']:
        held.update(r['native_rgb_sha256'] for r in json.loads((OLD/'data_union_verified_v1'/f'{split}_manifest.json').read_text())['sources'])
    dest=L/'data/openimages_cvdf_train';dest.mkdir(parents=True,exist_ok=True)
    def fetch(item):
        _,sid,meta=item
        if time.time()-begin>43000:raise TimeoutError('registered window exhausted')
        p=dest/(sid+'.jpg');assert not p.exists()
        with urllib.request.urlopen('https://s3.amazonaws.com/open-images-dataset/train/'+sid+'.jpg',timeout=60) as response:
            b=response.read();length=int(response.headers['Content-Length']);etag=response.headers.get('ETag','').strip('"')
        assert len(b)==length
        if re.fullmatch('[0-9a-f]{32}',etag):assert hashlib.md5(b).hexdigest()==etag
        with Image.open(io.BytesIO(b)) as im:pixels=np.asarray(im.convert('RGB')).copy()
        temp=p.with_suffix('.partial');temp.write_bytes(b);temp.replace(p)
        return dict(source_id='openimages:train:'+sid,name='openimages/train/'+sid+'.jpg',path=str(p),sha256=hashlib.sha256(b).hexdigest(),
           bytes=len(b),etag=etag,native_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),native_rgb_shape=list(pixels.shape),
           license=meta['License'],metadata_id=sid,origin='official CVDF resized train pixels; original attribution in selected_metadata.json')
    good=[];fail=[]
    with (out/'journal.jsonl').open('x') as log,concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures={pool.submit(fetch,item):item[1] for item in selected}
        for future in concurrent.futures.as_completed(futures):
            try:row=future.result();good.append(row);record=dict(row,status='verified')
            except Exception as e:row=dict(source_id=futures[future],error=repr(e));fail.append(row);record=dict(row,status='failed')
            log.write(json.dumps(record)+'\n');log.flush()
            if (len(good)+len(fail))%500==0:write(out/'progress.json',dict(stage='images',verified=len(good),failed=len(fail),target=100000,seconds=time.time()-begin))
    unique={};removed=[]
    for row in sorted(good,key=lambda v:v['source_id']):
        h=row['native_rgb_sha256']
        if h in held or h in unique:removed.append(row['source_id'])
        else:unique[h]=row
    write(out/'train_manifest.json',dict(sources=list(unique.values()),candidate_images=100000,verified_images=len(good),unique_images=len(unique),failed=len(fail),heldout_excluded=True,near_duplicate_detection='not run'))
    write(out/'download_failures.json',fail);write(out/'removed_duplicates.json',removed)
    write(out/'complete.json',dict(complete_download=not fail,unique_training_images=len(unique),failures=len(fail),manifest_sha256=sha(out/'train_manifest.json')))
    if fail:raise RuntimeError('download failures retained; manifest explicitly partial')
if __name__=='__main__':main()
