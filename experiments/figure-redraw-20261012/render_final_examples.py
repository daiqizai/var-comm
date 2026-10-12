"""Eight authorized fixed-source inference frames; never trains or scores images.

Run once. A failure leaves its case reservation intact and is never retried by
this entry. TX waveforms and actual RX observations are saved under the tmp root.
"""
from pathlib import Path
import dataclasses, fcntl, hashlib, json, os, sys, time, traceback
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/baseline-strength-20261011'))
import baseline_runtime as b
import run_continue as training
import numpy as np
import torch
from PIL import Image

OUT = ROOT/'results/figure_redraw_20261012/final_examples_v1'
ASSETS = b.TMP/'figure_redraw_final_examples_v1'
INDICES = [21, 92, 95, 75]

def ash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'owner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists(), 'Already attempted; inspect receipts, do not retry'
    training.resource_check(startup=True)
    sources = b.read(b.RESULT/'timing/fixed16_source_bindings_20261012.json')['records']
    sources = [next(r for r in sources if r['source_index']==i) for i in INDICES]
    selected = b.read(ROOT/'paper/figures/user_selected_examples_20261012/selection_and_integrity.json')
    checkpoints = b.read(b.RESULT/'automatic_followthrough_v1/registration.json')['checkpoints']
    bindings = {str(Path(__file__).resolve()): b.sha(__file__),
                str(Path(b.__file__)): b.sha(b.__file__), str(Path(training.__file__)): b.sha(training.__file__)}
    pixels = {}
    for letter, row in zip('abcd', sources):
        for name in ['archive','checkpoint']:
            pin=row[name]; assert b.sha(pin['path'])==pin['sha256']; bindings[pin['path']]=pin['sha256']
        with np.load(row['archive']['path'],allow_pickle=False) as z: x=z['pixels'].copy()
        assert x.dtype==np.uint8 and x.shape==(3,256,256)
        assert hashlib.sha256(x.tobytes()).hexdigest()==row['preprocessing_id']
        old=ROOT/'paper/figures/user_selected_examples_20261012/individual'/f'example_{letter}_original.png'
        assert b.sha(old)==selected['individual_sha256'][old.name]
        assert np.array_equal(np.asarray(Image.open(old)),x.transpose(1,2,0))
        pixels[row['source_index']]=x
    for name in ['P100000','Swin200000']:
        p=checkpoints[name]; assert b.sha(p['path'])==p['sha256']; bindings[p['path']]=p['sha256']
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
    reg=dict(status='REGISTERED_EIGHT_FIXED_FRAMES',source_indices=INDICES,sources=sources,
             N=1024,snr_db=13,noise_seed=2001,frames=8,checkpoints={k:checkpoints[k] for k in ['P100000','Swin200000']},
             source_bindings=bindings,source_selection='User selected after viewing 16 development examples',
             training_updates=0,metric_calls=0,bootstrap_calls=0,old_observations_reused=False,
             digital_output_reused=True,gpu=os.environ['CUDA_VISIBLE_DEVICES'],batch=1)
    b.write(OUT/'registration.json',reg)
    ASSETS.mkdir(parents=True,exist_ok=False)
    b.write(OUT/'launch.json',dict(pid=os.getpid(),time=time.time(),registration_sha256=b.sha(OUT/'registration.json')))
    numerical=b.configure(); receipts=[]; case=None
    try:
        for branch, name in [('P','P100000'),('Swin','Swin200000')]:
            training.resource_check(startup=True)
            state=training.load(branch)
            if branch=='P':
                del state['lp']; torch.cuda.empty_cache()
                # Fresh frozen encoder, not a latent borrowed from another source.
                from models.vqvae import VQVAE
                vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,test_mode=True,share_quant_resi=4)
                weight=b.RT/'assets/model_seed_v1/external/home/liulu/projects/VAR-MAP-GATE0/checkpoints/vae_ch160v4096z32.pth'
                assert b.sha(weight)=='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c'
                vae.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
                vae.cuda().eval().requires_grad_(False)
            cp=checkpoints[name]
            payload=torch.load(cp['path'],map_location='cpu',weights_only=True)
            state['model'].load_state_dict(payload['model'],strict=True); del payload
            model=state['model'].eval().requires_grad_(False)
            for row in sources:
                training.resource_check()
                assert not (OUT/'STOP').exists()
                i=row['source_index']; case=dict(method=name,source_index=i,N=1024,snr_db=13,noise_seed=2001)
                b.write(OUT/'pending_case.json',case)
                x=pixels[i]
                with torch.inference_mode():
                    if branch=='P':
                        F=vae.quant_conv(vae.encoder(torch.as_tensor(x[None],device='cuda',dtype=torch.float32)/127.5-1))
                        wave=model.transmit(F)[0].cpu().numpy()
                        noise=state['ns']['seeded_noise']('VAR-CONTINUOUS-1024|'+row['source_id'],2001,(1024,2)).astype(np.float32)
                        observed=wave+noise*np.float32(10**(-13/20))
                        received=model.receive(torch.as_tensor(observed[None],device='cuda'),torch.tensor([13],device='cuda',dtype=torch.float32))
                        image=state['dc'](received)[0].cpu().numpy()
                        link=dict(header_accepted=True,fallback='none')
                    else:
                        wave=state['sr'].transmit(model,torch.as_tensor(x[None],device='cuda',dtype=torch.float32)/255,1024,13)
                        noise=state['sp'].standard_noise(row['source_id'],2001,1024,13)
                        observed=wave+noise/(10**(13/20))
                        # RX only accepts the actual observed frame, public N/SNR.
                        image,context=state['sr'].receive(model,observed,1024,13)
                        link=dict(header_accepted=context.accepted,header_crc_accepted=context.crc_accepted,
                                  header_fields_legal=context.fields_legal,fallback='none' if context.accepted else 'fixed_gray_0.5')
                assert wave.shape==observed.shape==(1024,2)
                assert image.shape==(3,256,256) and image.dtype==np.float32 and np.isfinite(image).all()
                assert ((image>=0)&(image<=1)).all()
                display=np.rint(image.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
                archive=ASSETS/f'{name}_{i:04d}.npz'
                np.savez_compressed(archive,wave=wave,observed=observed,reconstruction=image,display=display)
                png=ASSETS/f'{name}_{i:04d}.png'; Image.fromarray(display).save(png)
                assert np.array_equal(np.asarray(Image.open(png)),display)
                receipts.append(dict(case,source_id=row['source_id'],source_pixels_sha256=row['preprocessing_id'],
                    checkpoint_sha256=cp['sha256'],wave_sha256=ash(wave),observation_sha256=ash(observed),
                    reconstruction_sha256=ash(image),display_sha256=ash(display),
                    actual_energy=float(np.square(wave.astype(np.float64)).sum()),**link,
                    archive=dict(path=str(archive),sha256=b.sha(archive)),png=dict(path=str(png),sha256=b.sha(png))))
                b.write(OUT/'receipts.json',receipts)
                b.write(OUT/'status.json',dict(status='INFERENCE',completed=len(receipts),total=8))
                (OUT/'pending_case.json').unlink(); case=None
                print('FRAME_COMPLETE',name,i,flush=True)
            del model,state
            if branch=='P':del vae,F,received
            torch.cuda.empty_cache()
        assert len(receipts)==8 and len({(r['method'],r['source_index']) for r in receipts})==8
        for p,h in bindings.items():assert b.sha(p)==h,p
        b.write(OUT/'completion.json',dict(status='COMPLETE_EIGHT_FIXED_FRAMES',frames=8,numerical=numerical,
            receipts_sha256=b.sha(OUT/'receipts.json'),training_updates=0,metric_calls=0,bootstrap_calls=0,
            display_conversion='round(255 * reconstruction) to uint8; no enhancement',elapsed_end=time.time()))
    except BaseException:
        b.write(OUT/'failure.json',dict(status='STOPPED_NO_AUTO_RETRY',pending=case,completed=len(receipts),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
