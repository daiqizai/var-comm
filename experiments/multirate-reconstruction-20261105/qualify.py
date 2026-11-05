"""Fixed small engineering gate; no quality-ranking prerequisite."""
import argparse,json,time,os,random
from pathlib import Path
from runtime import setup,L,OLD,sha,write,CHECKPOINT_ROOT
setup()
import numpy as np
import torch
from model import MultiRateCodec,shapes_for
from codec import encode_packet,receive_packet,unpack
from codec_adapter import source_image,tensor_sha
from training_core import load_training_lpips

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(exist_ok=False)
    manifest=json.loads((OLD/'data_union_verified_v1/train_manifest.json').read_text());sources=manifest['sources'][:2]
    write(out/'registration.json',dict(source_ids=[s['source_id'] for s in sources],seed=20261105,
        checks=['600M..1B active parameters','finite gradient groups','256 and 512 forward/backward','exact packet independent RX','CRC rejection','optimizer/RNG checkpoint roundtrip'],
        max_gpu_hours=2,main_training=False))
    torch.set_num_threads(2);start=time.time()
    init=json.loads((OLD/'checkpoint.json').read_text());assert sha(init['path'])==init['sha256']
    model=MultiRateCodec(init['path']);counts=model.parameter_counts();write(out/'parameter_counts.json',counts)
    assert 600_000_000<=counts['total']<=1_000_000_000 and counts['total']==counts['trainable'],counts
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True)
    lp=load_training_lpips(json.loads((OLD/'metric_assets.json').read_text()))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    records=[]
    for i,size in enumerate([256,512]):
        x,_,_=source_image(sources[i]['path'])
        if size==512:
            from PIL import Image
            from torchvision.transforms import functional as TF
            im=Image.open(sources[i]['path']).convert('RGB');im=TF.center_crop(TF.resize(im,512,interpolation=TF.InterpolationMode.BICUBIC),[512,512])
            x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)[None].cuda().float()/127.5-1
        mode=torch.tensor([7 if size==256 else 15],device='cuda');model.train();opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats();t=time.time()
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,record,rec=model(x,mode,full=True,perceptual=lp)
        assert torch.isfinite(loss);loss.backward()
        gradients={}
        for name,module in model.named_children():
            ps=[p for p in module.parameters() if p.grad is not None]
            assert ps and all(torch.isfinite(p.grad).all() for p in ps),name
            norm=sum(float(p.grad.float().square().sum()) for p in ps)**.5
            assert norm>0,name;gradients[name]=norm
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
        records.append(dict(size=size,loss=float(loss),gradient_norms=gradients,seconds=time.time()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),tokens=sum(h*w for h,w in shapes_for(size))))
        write(out/'gradient_checks.json',records);del loss,rec,x
    opt.zero_grad(set_to_none=True);del lp
    # Persist the complete optimizer state and the identical model used by TX.
    CHECKPOINT_ROOT.mkdir(parents=True,exist_ok=True);save=CHECKPOINT_ROOT/'qualification_v1.pt'
    assert not save.exists();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),steps=2),save)
    checkpoint_identity=dict(path=str(save),sha256=sha(save),bytes=save.stat().st_size);write(out/'checkpoint.json',checkpoint_identity)
    loaded=torch.load(save,map_location='cpu',weights_only=False)
    for n,p in model.state_dict().items():assert torch.equal(p.cpu(),loaded['model'][n]),n
    assert len(loaded['optimizer']['state'])==len(opt.state)
    for k,state in opt.state_dict()['state'].items():
        for n,v in state.items():
            if torch.is_tensor(v):assert torch.equal(v.cpu(),loaded['optimizer']['state'][k][n])
    del loaded,opt
    # This is actual new-model packet coding, not a replay of an old codec.
    model.eval();x,_,_=source_image(sources[0]['path'])
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    wire,tx=encode_packet(model,x,7)
    (out/'packet.bin').write_bytes(wire);torch.save(tx['received_ids'],out/'tx_received_ids.pt')
    tx={k:v for k,v in tx.items() if k!='received_ids'};write(out/'tx.json',tx)
    rx,ids,meta=receive_packet(model,wire)
    assert torch.equal(ids,torch.load(out/'tx_received_ids.pt',weights_only=True))
    assert torch.isfinite(rx).all();torch.save(rx.cpu(),out/'inprocess_rx.pt')
    corrupt=bytearray(wire);corrupt[-1]^=1
    try:unpack(bytes(corrupt));raise RuntimeError('CRC failed to reject')
    except AssertionError:pass
    write(out/'complete.json',dict(passed=True,parameters=counts,gradient_checks=records,checkpoint=checkpoint_identity,
          packet_bits=len(wire)*8,endpoint=meta['endpoint'],inprocess_rx_exact=True,crc_rejected=True,
          independent_process_rx='pending separate process',seconds=time.time()-start,old_P2_stop=(OLD/'STOP_P2').exists()))
if __name__=='__main__':main()
