"""Score only missing actual source-image pairs, preserving all logical failures."""
import csv,hashlib,json,math,pathlib,time
from identity import equivalent
def run(c):
    import numpy as np,torch
    reg=c['reg'];old=c['old'];metric=c['metric'];ledger=c['ledger'];evaluator=c['evaluator'];perceptual=c['perceptual'];pair=c['pair'];psnr=c['psnr'];metric_identity=c['metric_identity']
    E=pathlib.Path(reg['code_root']);import sys;sys.path.insert(0,str(E));from a1_owner import desc,verify,put
    from control import RUN,OUT
    def image_ok(image):
        assert image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and (image>=0).all() and (image<=1).all();return image
    prior_meta=json.loads(pathlib.Path(reg['prior_metadata']['path']).read_text());equivalent(prior_meta,c['metadata'])
    prior_id=hashlib.sha256(json.dumps(prior_meta,sort_keys=True).encode()).hexdigest();assert prior_id==reg['prior_metric_identity']
    plan=json.loads(pathlib.Path(reg['pair_plan']['path']).read_text());assert len(plan['sources'])==1000
    cached={};old_cache={x['pair_key']:x for x in reg['prior_pair_files']};refs={x['source_id']:x for x in reg['prior_references']}
    pairs_out=[];frame_rows=[];reused=0
    for source in plan['sources']:
        src=source['archive'];verify(src)
        with np.load(src['path'],allow_pickle=False) as z:pixels=z['pixels'].copy()
        assert pixels.dtype==np.uint8 and pixels.shape==(3,256,256)
        target=image_ok(pixels.astype(np.float32)/255);x=torch.as_tensor(target[None],device='cuda:0');prepared=None
        for item in source['pairs']:
            image_pin=item['image']
            if image_pin is None:image=np.full((3,256,256),.5,dtype=np.float32)
            else:
                verify(image_pin)
                with np.load(image_pin['path'],allow_pickle=False) as z:image=z['float_output'].copy()
            image_ok(image);assert hashlib.sha256(image.tobytes()).hexdigest()==item['image_float_sha256']
            key=old.pair_cache_key(target,image,metric_identity);oldkey=old.pair_cache_key(target,image,prior_id)
            origin='NEW_ACTUAL_PAIR'
            if key not in cached:
                if oldkey in old_cache:
                    pin=old_cache[oldkey]['file'];verify(pin);record=json.loads(pathlib.Path(pin['path']).read_text())
                    assert record['pair_key']==oldkey and record['metric_identity']==prior_id and record['source']==src and record['image_float_sha256']==old.array_sha(image)
                    cached[key]=record['scores'];origin='EXACT_OLD_PAIR_REUSE';reused+=1
                else:
                    if prepared is None:
                        if source['source_id'] in refs:
                            r=refs[source['source_id']];assert r['source']==src;verify(r['file'])
                            with np.load(r['file']['path'],allow_pickle=False) as z:features=z['features'].copy()
                            assert features.shape==(1,1024) and features.dtype==np.float32 and np.isfinite(features).all()
                            prepared={'digest':metric.image_digest(x),'batch':1,'evaluator_identity':evaluator.identity(),'dinov2_vitl14':torch.as_tensor(features)}
                        else:
                            prepared=ledger.call('reference_dino',str(source['source_index']),lambda:evaluator.prepare_reference(x))
                            np.savez(RUN/(str(source['source_index'])+'_reference.npz'),features=prepared['dinov2_vitl14'].numpy())
                    def compute():
                        y=torch.as_tensor(image[None],device='cuda:0')
                        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                            lp=ledger.call('lpips_pair',key,lambda:pair(perceptual,y,x))
                            di=ledger.call('reconstruction_dino',key,lambda:evaluator.score(x,y,[0],label_conditioned=False,prepared=prepared))
                        values={'psnr_db':float(psnr(target,image)),'lpips_alex':float(lp[0].item()),'dinov2_vitl14_cosine':float(di[0]['dinov2_vitl14_cosine'])}
                        assert all(math.isfinite(v) for v in values.values());return values
                    cached[key]=ledger.call('image_scores',key,compute)
                put(RUN/(key+'.json'),{'pair_key':key,'source':src,'image':image_pin,'image_float_sha256':old.array_sha(image),'metric_identity':metric_identity,'scores':cached[key],'origin':origin,'old_pair_key':oldkey if origin=='EXACT_OLD_PAIR_REUSE' else None})
            else:origin='EXACT_CURRENT_PAIR_REUSE'
            pairs_out.append({'source_id':source['source_id'],'source_index':source['source_index'],'state':item['state'],'pair_key':key,'origin':origin,**cached[key]})
            for frame in item['frames']:frame_rows.append(dict(frame,pair_key=key,**cached[key]))
        if (source['source_index']+1)%50==0:print(json.dumps({'sources_through':source['source_index']+1,'unique_pairs':len(cached),'reused_old_pairs':reused,'calls':ledger.counts}),flush=True)
    assert len(frame_rows)==18000 and len({r['key'] for r in frame_rows})==18000
    assert ledger.counts.get('image_scores',0)<=7200 and ledger.counts.get('lpips_backbone_forward',0)==2*ledger.counts.get('image_scores',0)
    assert ledger.counts.get('lpips_pair',0)==ledger.counts.get('reconstruction_dino',0)==ledger.counts.get('image_scores',0)
    for name,rows in [('actual_pair_scores.csv',pairs_out),('logical_scores.csv',frame_rows)]:
        with (OUT/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    put(OUT/'completion.json',{'status':'CAL1000_METRICS_CLOSED_PENDING_FROZEN_POLICY','logical_frames':18000,'source_count':1000,'counts':ledger.counts,'unique_pairs':len(cached),'reused_old_pairs':reused,'metric_identity':metric_identity,'old_metric_identity':prior_id,
      'metric_identity_translation':'Only DINO metadata selection-use population label changed; all other numerical, model, preprocessing, source, hardware and pair identity checked equal.',
      'unresolved':0,'new_channel':0,'VAR_calls':0,'bootstrap_calls':0,'ended_unix':time.time(),'outputs':[desc(OUT/n) for n in ['actual_pair_scores.csv','logical_scores.csv','metric_identity.json']]})
