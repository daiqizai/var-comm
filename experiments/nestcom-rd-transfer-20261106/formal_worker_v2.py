"""One-shot, resumeless formal TX, source-denied independent RX, and scoring."""
import argparse,csv,json,os,sys,time
from pathlib import Path
from runtime import *
setup()
import numpy as np
import torch
from formal_codec_v1 import *
from formal_metrics_v2 import FiveMetrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--stage',choices=['tx','rx','score'],required=True);a=p.parse_args()
    r=json.loads(Path(a.request).read_text());base=Path(r['output']);out=base/('score_v2' if a.stage=='score' else a.stage);out.mkdir(exist_ok=False)
    for path,h in r['bindings'].items():assert sha(path)==h,path
    old_paused();numerical();torch.manual_seed(20261106);torch.cuda.manual_seed_all(20261106)
    start=time.time();counts=dict(encoder=0,entropy_passes=0,qualification_reference_encodes=0,rx=0,rx_logical_reused=0,
        direct_full_qualification_renders=0,metric_pairs=0,metric_pair_reuse=0,metric_self_checks=0)
    def progress(**kw):write(out/'progress.json',dict(time=time.time(),counts=counts,**kw))
    if a.stage=='tx':
        rows=json.loads(Path(r['manifest']).read_text())['sources'][r['shard']::r['shards']]
        public=base/'public';public.mkdir(exist_ok=False);entries=[]
        for method in r['methods']:
            model=load_model(r,method)
            for i,row in enumerate(rows):
                assert time.time()<r['deadline']
                sid=str(row['source_id']);keybase=f'{i:04d}_{method}'
                raw=pixels(row);ref=np.ascontiguousarray(raw.transpose(2,0,1).astype(np.float32)/255)
                refpath=out/f'{i:04d}_reference.npy'
                if not refpath.exists():np.save(refpath,ref,allow_pickle=False)
                else:assert np.array_equal(np.load(refpath,allow_pickle=False),ref)
                write(out/(keybase+'_started.json'),dict(time=time.time(),source=sid,method=method))
                with torch.no_grad():
                    ids,_,_,full,_=model.quantize(source_tensor(raw),soft=False);counts['encoder']+=1
                    encoded,cdfhash=encode_all(model,ids);counts['entropy_passes']+=1
                    flat=torch.cat(ids,1)[0].cpu().numpy()
                    if r['qualification']:
                        from packet import encode_fit
                        for condition,wire,meta in encoded:
                            if condition=='full':continue
                            check,_=encode_fit(model,ids,int(condition));counts['qualification_reference_encodes']+=1
                            assert np.array_equal(check,wire),'Shared incremental pass changed canonical bounded packet'
                        direct=model.decoder(full)[0].cpu().float().numpy();counts['direct_full_qualification_renders']+=1
                        np.save(out/(keybase+'_directfull.npy'),direct,allow_pickle=False)
                for condition,wire,meta in encoded:
                    key=keybase+'_'+condition;path=public/(key+'.npy');np.save(path,wire,allow_pickle=False)
                    np.save(out/(key+'_expected.npy'),flat[:meta['tokens']],allow_pickle=False)
                    entries.append(dict(key=key,source=sid,method=method,reference_index=i,wire=str(path),wire_sha256=sha(path),
                        wire_native_sha256=digest(wire),tx_cdf_sha256=cdfhash,reference_sha256=digest(ref),**meta))
                progress(method=method,sources_completed=i+1,total_sources=len(rows))
            del model;torch.cuda.empty_cache()
        write(public/'entries.json',dict(entries=entries,qualification=r['qualification'],source_count=len(rows)))
    elif a.stage=='rx':
        denied=r['rx_denied_roots']
        def deny(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
                path=os.path.abspath(os.fsdecode(args[0]))
                if any(path==v or path.startswith(v+'/') for v in denied):raise RuntimeError('RX source/TX-private access denied: '+path)
        sys.addaudithook(deny)
        entries=json.loads((base/'public/entries.json').read_text())['entries']
        for method in r['methods']:
            model=load_model(r,method);cache={}
            for i,e in enumerate(v for v in entries if v['method']==method):
                assert time.time()<r['deadline']
                assert sha(e['wire'])==e['wire_sha256'];wire=np.load(e['wire'],allow_pickle=False)
                assert digest(wire)==e['wire_native_sha256']
                if e['kind']=='bounded_packet':assert wire.size==e['packet_bits']<=e['packet_budget_bits']
                identity=(e['source'],e['kind'],e['wire_sha256'])
                key=e['key'];reuse=identity in cache
                if reuse:
                    for suffix in ['_rgb.npy','_ids.npy']:os.link(out/(cache[identity]+suffix),out/(key+suffix))
                    counts['rx_logical_reused']+=1
                else:
                    write(out/(key+'_started.json'),dict(time=time.time(),wire_sha256=e['wire_sha256']))
                    rgb,ids=receive(model,wire,e['kind']);counts['rx']+=1
                    np.save(out/(key+'_rgb.npy'),rgb,allow_pickle=False);np.save(out/(key+'_ids.npy'),ids,allow_pickle=False)
                    cache[identity]=key
                rgb=np.load(out/(key+'_rgb.npy'),allow_pickle=False);ids=np.load(out/(key+'_ids.npy'),allow_pickle=False)
                assert ids.size==e['tokens']
                write(out/(key+'_receipt.json'),dict(rgb_sha256=digest(rgb),ids_sha256=digest(ids),wire_sha256=e['wire_sha256'],reused=reuse))
                progress(method=method,logical_completed=i+1)
            del model;torch.cuda.empty_cache()
        write(out/'independence.json',dict(denied_roots=denied,no_source_or_tx_private_reads=True))
    else:
        entries=json.loads((base/'public/entries.json').read_text())['entries'];suite=FiveMetrics(json.loads(Path(r['metric_assets']).read_text()))
        fields=['source','method','condition','source_capacity','packet_budget_bits','packet_bits','source_bits','tokens','null',
                'reference_sha256','reconstruction_sha256',*METRICS]
        cache={};checks=[]
        with (out/'frame_metrics.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i,e in enumerate(entries):
                assert time.time()<r['deadline']
                key=e['key'];ref=np.load(base/'tx'/f"{e['reference_index']:04d}_reference.npy",allow_pickle=False)
                rgb=np.load(base/'rx'/(key+'_rgb.npy'),allow_pickle=False);got=np.load(base/'rx'/(key+'_ids.npy'),allow_pickle=False)
                expected=np.load(base/'tx'/(key+'_expected.npy'),allow_pickle=False)
                receipt=json.loads((base/'rx'/(key+'_receipt.json')).read_text())
                assert digest(ref)==e['reference_sha256'] and digest(rgb)==receipt['rgb_sha256'] and digest(got)==receipt['ids_sha256']
                assert np.array_equal(got,expected),'Received tokens differ from transmitted prefix'
                if r['qualification'] and e['condition']=='full':
                    direct=np.load(base/'tx'/(key.rsplit('_',1)[0]+'_directfull.npy'),allow_pickle=False)
                    assert np.array_equal(rgb,direct),'Full independent RX differs from direct quantized reconstruction'
                if r['qualification'] and i==0:
                    checks.append(suite.score(ref,ref));counts['metric_self_checks']+=1
                    v=checks[-1];assert v['psnr_db']==120 and abs(v['clip_i']-1)<1e-5
                    assert abs(v['dists'])<1e-5 and abs(v['dreamsim'])<1e-5 and abs(v['lpips_alex'])<1e-5
                pair=(digest(ref),digest(rgb))
                if pair not in cache:
                    write(out/(key+'_started.json'),dict(time=time.time(),reference_sha256=pair[0],reconstruction_sha256=pair[1]))
                    cache[pair]=suite.score(ref,rgb);counts['metric_pairs']+=1
                else:counts['metric_pair_reuse']+=1
                v={k:e[k] for k in fields if k in e};v.update(reconstruction_sha256=pair[1],**cache[pair]);writer.writerow(v);f.flush()
                progress(logical_completed=i+1,total=len(entries))
        write(out/'audit.json',dict(logical_results=len(entries),all_prefixes_exact=True,full_parity_checked=r['qualification'],metric_self_checks=checks))
    torch.cuda.synchronize()
    write(out/'complete.json',dict(stage=a.stage,passed=True,counts=counts,seconds=time.time()-start,
        torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
        device_uuid=str(torch.cuda.get_device_properties(0).uuid),precision='FP32_TF32off_mathSDPA'))

if __name__=='__main__':main()
