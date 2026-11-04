from pathlib import Path
import argparse,csv,hashlib,json,os,sys,time,traceback
import numpy as np
import torch
from packet import SourcePacket,choose_longest_feasible
from codec_adapter import load_models,source_image,encode_all_prefixes,receive,SHAPES

def dump(p,v):p.write_text(json.dumps(v,indent=2))
def write_csv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['tx','rx','audit']);ap.add_argument('--root',required=True);ap.add_argument('--checkpoint',required=True);a=ap.parse_args()
    L=Path(a.root);O=L/('p0_'+a.mode+'_v1');public=L/'p0_public_inputs';start=time.time()
    if a.mode=='audit':
        tx=json.loads((L/'p0_tx_v1/complete.json').read_text());rx=json.loads((L/'p0_rx_v1/complete.json').read_text());rows=[]
        for s in tx['sources']:
            q=np.load(L/'p0_tx_v1'/s/'source_indices.npy');mask=np.load(L/'p0_tx_v1'/s/'slice_mask.npy')
            for t in range(1,29):
                r=np.load(L/'p0_rx_v1'/s/f's{t:02d}_indices.npy');m=mask<t
                ok=np.array_equal(q[:,m],r[:,m]);rows.append(dict(source=s,slices=t,tokens=int(m.sum()),exact_received_indices=ok))
                assert ok,(s,t,'clean independent received prefix mismatch')
                im=np.load(L/'p0_rx_v1'/s/f's{t:02d}_rgb.npy');assert im.shape==(3,256,256) and np.isfinite(im).all()
        write_csv(O/'roundtrip_checks.csv',rows)
        dump(O/'complete.json',dict(passed=True,independent_prefixes=len(rows),sources=len(tx['sources']),gpu_calls=0,tx_process_separate=True,rx_no_source_inputs=True,seconds=time.time()-start))
        return
    if a.mode=='rx':
        # Audit Python file reads: receiver is prohibited from accessing source JPEGs
        # or the transmitter's private indices, manifest and reference arrays.
        denied=[str((L/'data').resolve()),str((L/'p0_tx_v1').resolve()),str((L/'asset_subset_v2').resolve())]
        def hook(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
                p=str(Path(os.fsdecode(args[0])).absolute())
                if any(p==d or p.startswith(d+'/') for d in denied):raise RuntimeError('Receiver attempted private TX/source read: '+p)
        sys.addaudithook(hook)
    vq,ar,codebook=load_models(a.checkpoint)
    if a.mode=='tx':
        source_manifest=json.loads((L/'asset_subset_v2/p0_ready.json').read_text());public.mkdir(exist_ok=False)
        identities=[];allrows=[];sources=[]
        for source in source_manifest['images']:
            sid=Path(source['name']).stem;sources.append(sid);private=O/sid;private.mkdir();inbox=public/sid;inbox.mkdir()
            assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
            x,pixels,identity=source_image(source['path']);np.save(private/'source_rgb.npy',pixels,allow_pickle=False)
            packets,rows,indices,mask,timing=encode_all_prefixes(vq,ar,codebook,x)
            np.save(private/'source_indices.npy',indices.numpy(),allow_pickle=False);np.save(private/'slice_mask.npy',mask.numpy(),allow_pickle=False)
            for t,p in packets.items():(inbox/f's{t:02d}.bin').write_bytes(p.to_bytes())
            (inbox/'s00.bin').write_bytes(SourcePacket(0,b'',b'').to_bytes())
            for r in rows:allrows.append(dict(source=sid,**r))
            identities.append(dict(**source,**identity,timing=timing))
            dump(O/'progress.json',dict(completed_sources=len(sources),packets=len(allrows),encoder_calls=len(sources)))
        write_csv(O/'packet_lengths.csv',allrows);dump(O/'source_identity.json',identities)
        dump(public/'public_config.json',dict(version=1,shapes=SHAPES,codebook_size=4096,codebook_dim=32,precision=16,topk=1024,suffix='argmax',numeric_route='FP32/TF32off/mathSDPA',sources=sources))
        dump(O/'complete.json',dict(sources=sources,encoder_calls=len(sources),ar_slice_calls=28*len(sources),terminated_streams=len(allrows),seconds=time.time()-start))
    else:
        config=json.loads((public/'public_config.json').read_text());rows=[]
        for sid in config['sources']:
            out=O/sid;out.mkdir()
            for t in range(29):
                wire=(public/sid/f's{t:02d}.bin').read_bytes();idx,im,record=receive(vq,ar,codebook,wire)
                if idx is not None:np.save(out/f's{t:02d}_indices.npy',idx.numpy(),allow_pickle=False)
                np.save(out/f's{t:02d}_rgb.npy',im,allow_pickle=False)
                rows.append(dict(source=sid,slices=t,packet_sha256=hashlib.sha256(wire).hexdigest(),rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest(),**record))
            dump(O/'progress.json',dict(completed_sources=len(list(O.glob('*/'))),logical_receivers=len(rows)))
        # NULL has no decoded_tokens; normalize CSV schema explicitly.
        for r in rows:r.setdefault('decoded_tokens',0)
        write_csv(O/'receiver_rows.csv',rows)
        dump(O/'complete.json',dict(sources=config['sources'],logical_receivers=len(rows),new_model_receivers=sum(not r['null'] for r in rows),null_gray=sum(r['null'] for r in rows),seconds=time.time()-start))

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
