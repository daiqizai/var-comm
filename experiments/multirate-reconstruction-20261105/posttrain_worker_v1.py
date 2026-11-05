"""One-shot TX / source-denied independent RX / native metric stages."""
from pathlib import Path
import argparse,csv,json,os,sys,time
from runtime import setup,OLD,sha,write
setup()
import numpy as np
import torch
from posttrain_codec_v1 import BUDGETS,METRICS,digest_array,source_pixels,source_tensor,numerical_route,load_models,original512_tx,original512_rx,score_native

def load_new(spec):
    from model import MultiRateCodec
    init=json.loads((OLD/'checkpoint.json').read_text());assert sha(init['path'])==init['sha256']
    assert sha(spec['path'])==spec['sha256']
    model=MultiRateCodec(init['path']);state=torch.load(spec['path'],map_location='cpu',weights_only=False)
    assert state['step']==150000 and state['world_size']==8
    model.load_state_dict(state['model'],strict=True);del state
    return model.eval()

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--stage',choices=['tx','rx','score'],required=True);a=p.parse_args()
    r=json.loads(Path(a.request).read_text());base=Path(r['output']);out=base/a.stage;out.mkdir(exist_ok=False)
    for path,h in r['bindings'].items():assert sha(path)==h,path
    begin=time.time();numerical_route();size=r['size'];budget_values=[b*(size//256)**2 for b in BUDGETS]
    counters=dict(new_encoder=0,new_rx=0,new_metric_pairs=0,baseline_encoder=0,baseline_rx=0,baseline_metric_pairs=0,baseline_cache_logical_reuse=0,baseline_verification_metric_pairs=0)
    if a.stage=='tx':
        rows=json.loads(Path(r['manifest']).read_text())['sources'][r['shard']::8]
        model=load_new(r['checkpoint']);from codec import encode_packet
        public=base/'public';public.mkdir(exist_ok=False);entries=[];private=[]
        for i,row in enumerate(rows):
            sid=row['source_id'];pixels=source_pixels(row,size);np.save(out/(sid+'_source.npy'),pixels,allow_pickle=False);x=source_tensor(pixels)
            for m,budget in enumerate(budget_values):
                key=sid+f'_new_m{m}';write(out/(key+'_started.json'),dict(time=time.time(),source=sid,mode=m+(8 if size==512 else 0)))
                wire,meta=encode_packet(model,x,m+(8 if size==512 else 0));counters['new_encoder']+=1
                assert meta['bits']==len(wire)*8<=budget
                path=public/(key+'.bin');path.write_bytes(wire)
                entry=dict(key=key,source=sid,method='new',size=size,budget_bits=budget,mode=m+(8 if size==512 else 0),endpoint=meta['endpoint'],packet_bits=meta['bits'],packet=str(path),packet_sha256=sha(path))
                expected=meta.pop('received_ids').numpy();np.save(out/(key+'_expected.npy'),expected,allow_pickle=False)
                write(out/(key+'_tx.json'),dict(meta,received_indices_sha256=digest_array(expected)))
                entries.append(entry)
            private.append(dict(source=sid,reference_sha256=digest_array(np.ascontiguousarray(pixels.transpose(2,0,1).astype(np.float32)/255))))
            write(out/'progress.json',dict(sources=i+1,total=len(rows),counters=counters))
        del model;torch.cuda.empty_cache()
        if size==512:
            init=json.loads((OLD/'checkpoint.json').read_text());assert sha(init['path'])==init['sha256'];vq,ar,cb=load_models(init['path'])
            for i,row in enumerate(rows):
                sid=row['source_id'];x=source_tensor(np.load(out/(sid+'_source.npy'),allow_pickle=False))
                write(out/(sid+'_original_started.json'),dict(time=time.time(),encoder_calls_planned=1))
                packets,idx,mask=original512_tx(vq,ar,cb,x);counters['baseline_encoder']+=1
                for m,budget in enumerate(budget_values):
                    ep=max(t for t,w in packets.items() if len(w)*8<=budget);wire=packets[ep];key=sid+f'_original_m{m}'
                    path=public/(key+'.bin');path.write_bytes(wire);np.save(out/(key+'_expected.npy'),idx[:,mask<ep],allow_pickle=False)
                    entries.append(dict(key=key,source=sid,method='original',size=size,budget_bits=budget,mode=m+8,endpoint=ep,packet_bits=len(wire)*8,packet=str(path),packet_sha256=sha(path)))
                write(out/'progress.json',dict(original_sources=i+1,total=len(rows),counters=counters))
        write(public/'entries.json',dict(entries=entries,checkpoint=r['checkpoint'],fixed_source_resolution=size))
        write(out/'references.json',dict(sources=private))
    elif a.stage=='rx':
        # Deny native source data and all TX-private trees, including sibling shards.
        denied=[str(Path(v).resolve()) for v in r['rx_denied_roots']]
        def deny(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
                path=str(Path(os.fsdecode(args[0])).absolute())
                if any(path==v or path.startswith(v+'/') for v in denied):raise RuntimeError('RX source/private access rejected: '+path)
        sys.addaudithook(deny)
        entries=json.loads((base/'public/entries.json').read_text())['entries'];model=load_new(r['checkpoint']);from codec import receive_packet
        for method in ['new','original']:
            items=[e for e in entries if e['method']==method]
            if not items:continue
            if method=='original':
                del model;torch.cuda.empty_cache();init=json.loads((OLD/'checkpoint.json').read_text());vq,ar,cb=load_models(init['path'])
            cache={}
            for e in items:
                key=e['key'];wire=Path(e['packet']).read_bytes();assert sha(e['packet'])==e['packet_sha256'] and len(wire)*8<=e['budget_bits']
                # Cache identity includes source, exact wire, model and numerical route.
                identity=(e['source'],e['packet_sha256'])
                if identity in cache:
                    prior=cache[identity]
                    for suffix in ['_rgb.npy','_ids.npy']:os.link(out/(prior+suffix),out/(key+suffix))
                    reused=True
                else:
                    write(out/(key+'_started.json'),dict(time=time.time(),packet_sha256=e['packet_sha256']))
                    if method=='new':
                        raw,ids,meta=receive_packet(model,wire);image=raw.add(1).mul(.5).clamp(0,1)[0].cpu().float().numpy();ids=ids.numpy();counters['new_rx']+=1
                    else:image,ids=original512_rx(vq,ar,cb,wire);counters['baseline_rx']+=1
                    assert image.shape==(3,size,size) and image.dtype==np.float32 and np.isfinite(image).all()
                    np.save(out/(key+'_rgb.npy'),image,allow_pickle=False);np.save(out/(key+'_ids.npy'),ids,allow_pickle=False);cache[identity]=key;reused=False
                image=np.load(out/(key+'_rgb.npy'),allow_pickle=False);ids=np.load(out/(key+'_ids.npy'),allow_pickle=False)
                write(out/(key+'_receipt.json'),dict(packet_sha256=e['packet_sha256'],rgb_sha256=digest_array(image),ids_sha256=digest_array(ids),null=e['endpoint']==0,reused=reused))
        write(out/'independent_rx.json',dict(source_or_tx_private_access=False,denied_roots=denied))
    else:
        from metrics_v2 import MetricSuite
        suite=MetricSuite(json.loads(Path(r['metric_assets']).read_text()));entries=json.loads((base/'public/entries.json').read_text())['entries']
        rows=json.loads(Path(r['manifest']).read_text())['sources'][r['shard']::8];source_map={v['source_id']:v for v in rows}
        old={ (v['source'],int(v['budget_bits'])):v for v in csv.DictReader(Path(r['baseline256']).open()) } if size==256 else {}
        if size==256:
            first=rows[0]['source_id'];cached=old[(first,budget_values[0])]
            reference=np.ascontiguousarray(np.load(base/'tx'/(first+'_source.npy'),allow_pickle=False).transpose(2,0,1).astype(np.float32)/255)
            cached_image=np.load(cached['reconstruction_path'],allow_pickle=False)
            assert digest_array(cached_image)==cached['reconstruction_sha256']
            checked=score_native(suite,reference,cached_image);counters['baseline_verification_metric_pairs']+=1
            for m in METRICS:assert abs(checked[m]-float(cached[m]))<1e-5,(m,'cached metric route mismatch')
            write(out/'baseline_metric_reuse_check.json',dict(source=first,budget_bits=budget_values[0],scores=checked,tolerance=1e-5,passed=True))
        fields=['source','size','method','budget_bits','packet_bits','endpoint','received_tokens','null','reference_sha256','reconstruction_sha256',*METRICS,'origin']
        scorecache={};records=[]
        with (out/'scores.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for e in entries:
                key=e['key'];sid=e['source'];pixels=np.load(base/'tx'/(sid+'_source.npy'),allow_pickle=False)
                reference=np.ascontiguousarray(pixels.transpose(2,0,1).astype(np.float32)/255)
                image=np.load(base/'rx'/(key+'_rgb.npy'),allow_pickle=False)
                got=np.load(base/'rx'/(key+'_ids.npy'),allow_pickle=False);expected=np.load(base/'tx'/(key+'_expected.npy'),allow_pickle=False)
                assert np.array_equal(got,expected),(key,'independent received prefix mismatch')
                receipt=json.loads((base/'rx'/(key+'_receipt.json')).read_text());rh=digest_array(reference);yh=digest_array(image)
                assert yh==receipt['rgb_sha256'] and digest_array(got)==receipt['ids_sha256']
                pair=(rh,yh)
                if pair not in scorecache:
                    write(out/(key+'_metric_started.json'),dict(time=time.time(),reference_sha256=rh,reconstruction_sha256=yh))
                    scorecache[pair]=score_native(suite,reference,image);counters['new_metric_pairs' if e['method']=='new' else 'baseline_metric_pairs']+=1
                v=dict(source=sid,size=size,method=e['method'],budget_bits=e['budget_bits'],packet_bits=e['packet_bits'],endpoint=e['endpoint'],received_tokens=int(got.size),null=int(e['endpoint']==0),reference_sha256=rh,reconstruction_sha256=yh,**scorecache[pair],origin='native_independent_RX')
                writer.writerow(v);f.flush();records.append(v)
            if size==256:
                for source in rows:
                    for budget in budget_values:
                        v=old[(source['source_id'],budget)].copy()
                        assert v['reference_sha256']==digest_array(np.ascontiguousarray(np.load(base/'tx'/(source['source_id']+'_source.npy'),allow_pickle=False).transpose(2,0,1).astype(np.float32)/255))
                        writer.writerow({k:v[k] for k in fields});f.flush();counters['baseline_cache_logical_reuse']+=1
        assert len(records)==len(rows)*8*(1 if size==256 else 2)
        write(out/'audit.json',dict(received_prefixes_exact=len(records),all_native_finite=True,budgets_valid=True,baseline256_source_native_identity_checked=size==256))
    write(out/'complete.json',dict(passed=True,stage=a.stage,size=size,shard=r['shard'],seconds=time.time()-begin,counters=counters))

if __name__=='__main__':main()
