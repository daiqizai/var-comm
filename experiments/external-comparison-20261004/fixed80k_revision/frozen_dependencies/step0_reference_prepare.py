"""Prepare six frozen no-channel reference conditions without loading a model."""
from __future__ import annotations
import argparse
import hashlib
import io
from pathlib import Path
import sys
import numpy as np
from PIL import Image, features
import PIL
from step0_cache_export import read, require, sha, seal, identity, pixel_hash, pixels
from step0_reference_protocol import PAIR_SEED, NOISE_SEED, unrelated_pairs

CONDITIONS = ('identity','unrelated','same_class','gaussian_blur_sigma1','gaussian_noise_sigma2_255','jpeg_quality90')
MANIFEST_SHA = '49f344e8cf72960b7c164af96e05c203e24f4ae1cbfb8be9f3f02f8842b8e8b8'


def admitted(root, study):
    root=Path(root).resolve()
    path=root/'outputs/HISTORICAL-METRICS-R3-20261003/queue_registration.json'
    queue=read(path); item=queue['inherited_studies'][study]
    receipt_path=root/'outputs/HISTORICAL-METRICS-R3-20261003/inheritance'/(study+'.json')
    require(item['receipt_path']==str(receipt_path) and sha(receipt_path)==item['receipt_sha256']
            ==queue['input_proof_bindings'][str(receipt_path)],'Inherited reference source is not admitted')
    receipt=read(receipt_path)
    require(receipt['status']=='VERIFIED_COMPLETE_R2_STUDY' and receipt['study']==study
            and receipt['source_checkpoints_verified']==receipt['float_caches_verified']==100,
            'Incomplete historical source admission')
    return queue,receipt,{str(path):sha(path),str(receipt_path):sha(receipt_path)}


def targets_from_completed(root):
    """Only decompress source_rgb; no reconstruction/model replay is needed."""
    root=Path(root).resolve();study='FINAL_P2048_P3060'
    _,receipt,bindings=admitted(root,study)
    out=root/'outputs/HISTORICAL-METRICS-R2-20261003'/study
    rp=root/'results/historical_metrics_r2_20261003'/study/'registration.json'
    require(sha(rp)==receipt['bindings'][str(rp)],'Inherited source registration changed')
    bindings[str(rp)]=sha(rp);reg=read(rp)
    require(len(reg['source_identity'])==100,'Original 100-source identity missing')
    result=[]
    for i,record in enumerate(reg['source_identity']):
        cp_path=out/'source_checkpoints'/f'{i:04d}.json'; cache_path=out/'reconstructions'/f'{i:04d}.npz'
        for p in (cp_path,cache_path):
            require(sha(p)==receipt['bindings'][str(p)],'Inherited original source changed: '+str(p))
            bindings[str(p)]=sha(p)
        cp=read(cp_path)
        require(cp['payload_sha256']==identity({k:v for k,v in cp.items() if k!='payload_sha256'})
                and cp['binding']==identity(reg) and cp['source_index']==i,'Source checkpoint identity differs')
        proof=cp['float_reconstructions']
        require(proof['path']==str(cache_path) and proof['sha256']==bindings[str(cache_path)],'Source cache proof differs')
        with np.load(cache_path,allow_pickle=False) as archive:target=pixels(archive['source_rgb']).copy()
        require(pixel_hash(target,False)==proof['reference_sha256'] and all(
            r['history_reference_sha256']==pixel_hash(target) and r['history_source_id']==record['image_id']
            and r['history_true_class_index']==record['class_index'] for r in cp['rows']),'Source RGB/scientific identity differs')
        require(hashlib.sha256(np.rint(target*255).astype(np.uint8).tobytes()).hexdigest()==record['preprocessing_id'],
                'Source raw uint8 identity differs')
        result.append(dict(source_index=i,**record,rgb=target))
    return result,bindings


def choose_donors(records, registration, manifest):
    """Choose all IDs from original membership metadata before reading donors."""
    require(set(registration['train_ids']).isdisjoint(registration['calibration_ids']),'Train/calibration overlap')
    dev_ids={r['image_id'] for r in records};pool=[]
    for role,count in (('train',20000),('calibration',1000)):
        members=registration['source_image_bindings'][role]
        ids=registration['train_ids' if role=='train' else 'calibration_ids']
        require(len(members)==len(ids)==len(set(ids))==count and ids==[r['image_id'] for r in members],
                'Original complete train/calibration population differs')
        pop=[p for p in manifest['populations'] if p['name']==role]
        require(len(pop)==1 and pop[0]['count']==count,'RGB manifest population differs')
        shards={x['path']:x['sha256'] for x in pop[0]['shards']}
        for member in members:
            require(member['shard'] in shards and isinstance(member['index_in_shard'],int)
                    and member['index_in_shard']>=0,'Donor membership shard differs')
            if member['image_id'] not in dev_ids:
                pool.append(dict(**member,population=role,shard_sha256=shards[member['shard']]))
    result=[]
    for source in records:
        choices=[p for p in pool if p['class_index']==source['class_index']]
        require(choices,'No original train/calibration donor for class '+str(source['class_index']))
        def key(value):
            return hashlib.sha256(f'{PAIR_SEED}|{source["image_id"]}|{value["image_id"]}'.encode()).hexdigest(),value['image_id']
        chosen=min(choices,key=key)
        result.append(dict(source_index=source['source_index'],source_id=source['image_id'],donor=chosen,selection_hash=key(chosen)[0]))
    return result


def mild_images(source,image_id):
    source=pixels(source)
    radius=3;x=np.arange(-radius,radius+1,dtype=np.float64)
    kernel=np.exp(-.5*x*x);kernel/=kernel.sum()
    blurred=source.astype(np.float64)
    for axis in (1,2):
        pad=[(0,0)]*3;pad[axis]=(radius,radius)
        padded=np.pad(blurred,pad,mode='reflect')
        blurred=sum(weight*np.take(padded,np.arange(source.shape[axis])+offset,axis=axis)
                    for offset,weight in enumerate(kernel))
    seed=int.from_bytes(hashlib.sha256(f'{NOISE_SEED}|{image_id}'.encode()).digest()[:8],'little')
    noise=np.random.default_rng(seed).standard_normal(source.shape)*(2/255)
    noisy=np.clip(source.astype(np.float64)+noise,0,1).astype(np.float32)
    u8=np.rint(source.transpose(1,2,0)*255).astype(np.uint8)
    stream=io.BytesIO()
    Image.fromarray(u8).save(stream,format='JPEG',quality=90,subsampling=0,optimize=False,progressive=False)
    jpeg_bytes=stream.getvalue();stream.seek(0)
    with Image.open(stream) as image:jpeg=np.asarray(image.convert('RGB'),dtype=np.float32).transpose(2,0,1)/np.float32(255)
    return [np.clip(blurred,0,1).astype(np.float32),noisy,np.ascontiguousarray(jpeg)],dict(
        noise_seed_uint64=seed,jpeg_bytes_sha256=hashlib.sha256(jpeg_bytes).hexdigest(),
        pillow_version=PIL.__version__,libjpeg_version=features.version_codec('jpg'))


def save_cache(path, images):
    images=np.stack([pixels(x) for x in images])
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            require(old.files==['images'] and np.array_equal(old['images'],images),'Existing reference cache differs')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as stream:np.savez_compressed(stream,images=images)
    return dict(path=str(path),sha256=sha(path),image_sha256=[pixel_hash(x) for x in images],conditions=list(CONDITIONS))


def prepare(root,cache_root,p_registration,protocol,out):
    root,cache_root,out=Path(root).resolve(),Path(cache_root).resolve(),Path(out).resolve()
    allowed=root/'outputs/EXTERNAL-COMPARISON-20261004'
    require(out!=root and (root not in out.parents or out==allowed or allowed in out.parents),'Reference output must be separate from historical inputs')
    require('holdout' not in str(cache_root).lower() and 'holdout' not in str(p_registration).lower(),'Holdout is outside scope')
    plan=read(protocol)
    require(plan['status'].startswith('REFERENCE_CONDITIONS_') and plan['source_count']==100
            and plan['unrelated']['seed']==PAIR_SEED,'Reference preregistration differs')
    targets,bindings=targets_from_completed(root)
    ordered=[{k:v for k,v in r.items() if k!='rgb'} for r in targets]
    pairs,attempts=unrelated_pairs(ordered)
    require(plan['unrelated']['pairs']==pairs and plan['unrelated']['draws']==attempts,'Frozen unrelated pairing differs')
    mp=cache_root/'manifest.json';rp=Path(p_registration).resolve()
    require(sha(mp)==MANIFEST_SHA,'Original 20k/1k RGB manifest changed')
    manifest,registration=read(mp),read(rp)
    donors=choose_donors(ordered,registration,manifest)
    bindings.update({str(mp):sha(mp),str(rp):sha(rp),str(Path(protocol).resolve()):sha(protocol)})
    out.mkdir(parents=True,exist_ok=True)
    preselection=dict(status='SIX_REFERENCE_CONDITIONS_SELECTED_BEFORE_METRICS_AND_DONOR_PIXEL_READ',
        sources=ordered,conditions=list(CONDITIONS),rows=600,unrelated_pairs=pairs,same_class_donors=donors,
        identity_reference='Original float RGB paired with itself; PSNR is positive infinity, serialized null with explicit flag',
        input_bindings=bindings,selection_uses_quality=False,holdout_access=False,noise_triplication=False)
    seal(out/'selection.json',preselection)
    # Hash-bound, trusted original Torch files only. No model is constructed.
    import torch
    wanted={}
    for chosen in donors:wanted.setdefault(chosen['donor']['shard'],[]).append(chosen)
    donor_rgb={};donor_proofs=[]
    dev_pixels={r['preprocessing_id'] for r in targets}
    for name,selected in wanted.items():
        p=(cache_root/name).resolve()
        require(p.is_relative_to(cache_root) and sha(p)==selected[0]['donor']['shard_sha256'],'Donor shard path/hash differs')
        bindings[str(p)]=sha(p)
        shard=torch.load(p,map_location='cpu',weights_only=True)
        values,ids,labels=shard['targets_u8'],shard['image_ids'],shard['labels']
        require(values.dtype==torch.uint8 and tuple(values.shape)==(len(ids),3,256,256) and len(labels)==len(ids),'Donor RGB schema differs')
        for chosen in selected:
            d=chosen['donor'];j=d['index_in_shard']
            require(j<len(ids) and ids[j]==d['image_id'] and int(labels[j])==d['class_index'],'Donor actual membership differs')
            a=values[j].numpy();pixel_sha=hashlib.sha256(a.tobytes()).hexdigest()
            require(pixel_sha not in dev_pixels,'Same-class donor duplicates a development image')
            rgb=(((a.astype(np.float32)/np.float32(127.5))-np.float32(1))+np.float32(1))*np.float32(.5)
            donor_rgb[chosen['source_index']]=rgb
            donor_proofs.append(dict(**chosen,raw_pixels_sha256=pixel_sha,float_rgb_sha256=pixel_hash(rgb),shard_path=str(p)))
        del shard,values
    caches=[]
    for target,pair in zip(targets,pairs):
        i=target['source_index'];mild,proof=mild_images(target['rgb'],target['image_id'])
        images=[target['rgb'],targets[pair['donor_source_index']]['rgb'],donor_rgb[i],*mild]
        cached=save_cache(out/'reference_images'/f'{i:04d}.npz',images)
        caches.append(dict(source_index=i,source_id=target['image_id'],class_index=target['class_index'],
            reference_sha256=pixel_hash(target['rgb']),cache=cached,mild_generation=proof))
    value=dict(status='SIX_REFERENCE_INPUTS_COMPLETE',sources=ordered,caches=caches,rows=600,
        conditions=list(CONDITIONS),same_class_donor_proofs=sorted(donor_proofs,key=lambda x:x['source_index']),
        selection_sha256=sha(out/'selection.json'),input_bindings=bindings,model_inference=False,
        source_bindings={str(Path(__file__).with_name(name).resolve()):sha(Path(__file__).with_name(name))
                         for name in ('step0_reference_prepare.py','step0_reference_protocol.py','step0_cache_export.py')},
        holdout_access=False,wireless_claim=False,noise_triplication=False)
    seal(out/'prepared.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--p1024-registration',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    print(prepare(args.root,args.cache_root,args.p1024_registration,args.protocol,args.out)['status'],flush=True)
