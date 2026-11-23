"""Test the V4 wireless-interface split on a synthetic input, outside the image cohorts."""
import ast
import math
import torch
from qam import (EX, K, M, ROOT, SWIN, TEMP, blob, encode_with_ct, get_pam_levels,
                 load_swinjscc_backbone, producer_functions, record, ste_pam_quantize)
from dir2_zpac.universal_blind_v4 import physical_blind_channel_v4


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    dev=torch.device("cpu")
    model=load_swinjscc_backbone(SWIN,model_mode="SwinJSCC_w/_SA",c_bottleneck=96,
                                 device="cpu",vendor_root=TEMP/"cpu_vendor").eval()
    generator=torch.Generator().manual_seed(6112323)
    x=torch.rand((2,3,256,256),generator=generator)
    src=blob(EX+"/scripts/evaluate_universal_blind_v4.py").decode()
    node=next(z for z in ast.parse(src).body if isinstance(z,ast.FunctionDef) and z.name=="forward_pilot4_qam_reference")
    context=dict(torch=torch,encode_with_ct=encode_with_ct,get_pam_levels=get_pam_levels,
                 ste_pam_quantize=ste_pam_quantize,physical_blind_channel_v4=physical_blind_channel_v4)
    exec(compile(ast.get_source_segment(src,node),"original_pilot_reference","exec"),context)
    f=producer_functions()
    noise=f["make_paired_iq_noise"](K,[0,1],6112323,dev)
    h=f["make_paired_h_rb"](M,.85,[0,1],6112323,dev)
    ref=context["forward_pilot4_qam_reference"](model,x,64,10.,"ofdm_complex",noise,h,9181)
    tx,*aux=f["pilot_tx"](model,x)
    hr,hi=h.repeat_interleave(K//M,dim=1).unbind(-1)
    real,imag=tx.unbind(-1)
    clean=torch.stack((hr*real-hi*imag,hr*imag+hi*real),dim=-1)
    e=tx.square().sum(-1).mean(1)
    var=e*.1
    y=clean+noise*(.5*var).sqrt()[:,None,None]
    rec=f["pilot_rx"](model,x,y,var,h,9181,*aux)
    diff=(rec-ref).abs().amax().item()
    assert diff<1e-5
    # Native units and demonstrably fixed absolute noise after a constant public radio gain.
    gain=1.35
    wire=gain*clean+math.sqrt(.05)*noise
    residual=(wire-gain*clean)-math.sqrt(.05)*noise
    effective=(wire/gain)-(clean+math.sqrt(.05)*noise/gain)
    assert residual.abs().amax().item()<1e-6 and effective.abs().amax().item()<1e-6
    record(ROOT/"channel_check.json",dict(synthetic_sources=2,calibration_images=0,test_images=0,
           native_receiver_max_abs=diff,absolute_noise_max_abs=residual.abs().amax().item(),
           equivalent_receiver_units_max_abs=effective.abs().amax().item(),test_gain=gain,seed=6112323))


if __name__=="__main__":
    main()
