"""Reusable, independent TX/RX actual-bit prefix evaluation. No PHY is implied."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys,time
import numpy as np
import torch
from codec_adapter import load_models,source_image,encode_all_prefixes,receive,SHAPES
from packet import SourcePacket

def save(path,value):
    p=Path(path);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['tx','rx','audit']);ap.add_argument('--request',required=True);a=ap.parse_args()
    request=json.loads(Path(a.request).read_text());root=Path(request['output_root']);O=root/a.mode
    assert O.exists() and not (O/'complete.json').exists()
    public=root/'public';start=time.time()
    if a.mode=='audit':
        config=json.loads((public/'config.json').read_text());count=0;rows=[]
        for sid in config['sources']:
            q=np.load(root/'tx'/sid/'indices.npy');mask=np.load(root/'tx'/sid/'slice_mask.npy')
            for t in range(1,29):
                rx=root/'rx'/sid;got=np.load(rx/f's{t:02d}_indices.npy');ok=np.array_equal(q[:,mask<t],got[:,mask<t]);assert ok,(sid,t)
                im=np.load(rx/f's{t:02d}_rgb.npy');assert im.dtype==np.float32 and im.shape==(3,256,256) and np.isfinite(im).all() and im.min()>=0 and im.max()<=1
                count+=1
            rows.append(dict(source=sid,received_prefixes_exact=28))
        save(O/'complete.json',dict(passed=True,sources=len(config['sources']),prefixes=count,checks=rows,new_model_calls=0,elapsed_seconds=time.time()-start));return
    if a.mode=='rx':
        denied=[str(Path(p).resolve()) for p in request['rx_denied_roots']]+[str((root/'tx').resolve())]
        def hook(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
                p=str(Path(os.fsdecode(args[0])).absolute())
                if any(p==d or p.startswith(d+'/') for d in denied):raise RuntimeError('RX source/private read rejected: '+p)
        sys.addaudithook(hook)
    vq,ar,codebook=load_models(request['checkpoint']['path']);records=[]
    prop=torch.cuda.get_device_properties(0)
    save(O/'execution_identity.json',dict(torch_version=torch.__version__,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),device_order=os.environ.get('CUDA_DEVICE_ORDER'),device_name=prop.name,cuda_uuid=str(prop.uuid),dtype='float32',tf32=False,attention='math SDPA',checkpoint=request['checkpoint']))
    if a.mode=='tx':
        # The TX-only manifest is deliberately separate from the RX request.
        manifest=json.loads(Path(request['source_manifest']).read_text());sources=manifest['sources']
        public.mkdir(exist_ok=False);done=[]
        for source in sources:
            sid=source['source_id'];private=O/sid;private.mkdir();inbox=public/sid;inbox.mkdir()
            p=Path(source['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
            save(private/'call_started.json',dict(source=sid,encoder_calls_planned=1,ar_slices_planned=28,started=time.time()))
            x,pixels,identity=source_image(p);np.save(private/'source_rgb.npy',pixels,allow_pickle=False)
            packets,rows,idx,mask,timing=encode_all_prefixes(vq,ar,codebook,x)
            np.save(private/'indices.npy',idx.numpy(),allow_pickle=False);np.save(private/'slice_mask.npy',mask.numpy(),allow_pickle=False)
            for t,packet in packets.items():(inbox/f's{t:02d}.bin').write_bytes(packet.to_bytes())
            (inbox/'s00.bin').write_bytes(SourcePacket(0,b'',b'').to_bytes())
            save(private/'lengths.json',rows);save(private/'identity.json',dict(**source,**identity,timing=timing,checkpoint=request['checkpoint']))
            save(private/'complete.json',dict(passed=True,encoder_calls=1,ar_calls=28,terminations=28));done.append(sid)
            save(O/'progress.json',dict(sources=len(done),total=len(sources)))
        save(public/'config.json',dict(sources=done,shapes=SHAPES,model=request['checkpoint'],precision=16,topk=1024,numerical_route='FP32/TF32off/mathSDPA'))
        save(O/'complete.json',dict(passed=True,sources=len(done),encoder_calls=len(done),ar_slice_calls=28*len(done),terminations=28*len(done),elapsed_seconds=time.time()-start))
    else:
        config=json.loads((public/'config.json').read_text())
        for sid in config['sources']:
            out=O/sid;out.mkdir()
            for t in range(29):
                wire=(public/sid/f's{t:02d}.bin').read_bytes();save(out/f's{t:02d}_started.json',dict(started=time.time(),packet_sha256=hashlib.sha256(wire).hexdigest()))
                idx,im,row=receive(vq,ar,codebook,wire)
                if idx is not None:np.save(out/f's{t:02d}_indices.npy',idx.numpy(),allow_pickle=False)
                np.save(out/f's{t:02d}_rgb.npy',im,allow_pickle=False)
                record=dict(source=sid,slices=t,packet_bits=8*len(wire),packet_sha256=hashlib.sha256(wire).hexdigest(),rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest(),**row)
                save(out/f's{t:02d}_complete.json',record);records.append(record)
            save(O/'progress.json',dict(sources=len(records)//29,total=len(config['sources']),receivers=len(records)))
        save(O/'complete.json',dict(passed=True,sources=len(config['sources']),receiver_calls=len(config['sources'])*28,null_gray=len(config['sources']),elapsed_seconds=time.time()-start))

if __name__=='__main__':main()
