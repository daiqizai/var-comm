"""Discarded-weight training/restore qualification; not a P2 experiment arm."""
from pathlib import Path
import argparse,json,time,hashlib,os
import numpy as np
import torch
from training_core import PrefixModel,load_training_lpips,rng_state,restore_rng,state_sha

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--output',required=True);a=a.parse_args()
    L=Path(a.root);O=Path(a.output);checkpoint=json.loads((L/'checkpoint.json').read_text())['path']
    source=json.loads((L/'p0_tx_v2/complete.json').read_text())['sources'][0]
    x=torch.from_numpy(np.load(L/'p0_tx_v2'/source/'source_rgb.npy')).permute(2,0,1).float().div(127.5).sub(1).unsqueeze(0).cuda()
    model=PrefixModel(checkpoint);lp=load_training_lpips(json.loads((L/'metric_assets.json').read_text()))
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    # v1 reached its first backward only after the three native-route checks.
    # Preserve that diagnosis; do not repeat its successful generation calls.
    old=json.loads((L/'training_qualification_v1/actual_wait.json').read_text())
    assert old['returncode']==1 and old['child_gone']
    failure=(L/'training_qualification_v1/stderr.log').read_text()
    assert 'upsample_bicubic2d_backward_out_cuda does not have a deterministic implementation' in failure
    qualification=[dict(endpoint=t,indices_equal=True,max_abs_rgb_difference_upper_bound=1e-6,origin='v1 assertions completed before first backward failure') for t in [4,16,28]]
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.deterministic=True
    model.train();gen=torch.Generator().manual_seed(2026110401)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.95))
    directory=Path(checkpoint).parent/'qualification_training_v2';directory.mkdir(exist_ok=False)
    records=[]
    def step(arm,accumulate):
        opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();values=[]
        for _ in range(accumulate):
            loss,values_step=model.objective(x,arm,lp,gen);assert torch.isfinite(loss);(loss/accumulate).backward();values.append(float(loss.detach()))
        grads={name:sum(float(p.grad.detach().square().sum()) for p in group.parameters() if p.requires_grad and p.grad is not None)**.5 for name,group in [('E',model.vq.encoder),('D',model.vq.decoder),('P',model.ar)]}
        assert all(np.isfinite(v) and v>0 for v in grads.values()),grads
        assert model.vq.quantize.embedding.weight.grad is None
        opt.step();torch.cuda.synchronize()
        r=dict(arm=arm,effective_batch=accumulate,microbatch=1,seconds=time.perf_counter()-t,mean_loss=float(np.mean(values)),gradient_norm=grads,peak_bytes=torch.cuda.max_memory_allocated(),last_branches=values_step)
        records.append(r);(O/'progress.json').write_text(json.dumps(dict(completed_qualification_updates=len(records),last=r),indent=2));return r
    step('R3',16)
    saved=dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(gen),step=1,data_cursor=16,discard_after_qualification=True)
    torch.save(saved,directory/'step1.pt');del saved
    step('R3',16);reference=state_sha(model);torch.save(model.state_dict(),directory/'step2_reference.pt')
    saved=torch.load(directory/'step1.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
    # CPU generator and CPU RNG tensors are serialized as CPU tensors.
    for key in ['torch','endpoint_generator']:saved['rng'][key]=saved['rng'][key].cpu()
    saved['rng']['cuda']=[v.cpu() for v in saved['rng']['cuda']]
    restore_rng(saved['rng'],gen);del saved
    step('R3',16);resumed=state_sha(model)
    expected=torch.load(directory/'step2_reference.pt',map_location='cpu',weights_only=True)
    max_difference=max(float((v.detach().cpu().float()-expected[k].float()).abs().max()) for k,v in model.state_dict().items())
    assert max_difference<=1e-6,('resume exceeds preregistered FP32 absolute tolerance',max_difference)
    del expected
    step('R1',16);step('R2',16)
    (O/'complete.json').write_text(json.dumps(dict(passed=True,original_native_indices_equal=True,receiver_qualification=qualification,
        fp32=True,tf32=False,codebook_frozen=True,trainable_modules=['E','D','P'],resume_step2_bitwise_equal=(resumed==reference),resume_max_abs_difference=max_difference,resume_tolerance=1e-6,
        reference_step2_sha256=reference,replayed_step2_sha256=resumed,qualification_updates=len(records),
        replay_updates=1,records=records,weights_used_for_P2=False,checkpoint=str(directory/'step1.pt')),indent=2))

if __name__=='__main__':main()
