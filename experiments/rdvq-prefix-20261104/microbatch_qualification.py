from pathlib import Path
import json,sys,time
import numpy as np
import torch
from training_core import PrefixModel,load_training_lpips
L=Path(sys.argv[1]);O=L/'microbatch4_qualification_v1'
assert json.loads((L/'training_qualification_v2/complete.json').read_text())['passed']
ids=json.loads((L/'p0_tx_v2/complete.json').read_text())['sources'][:4]
x=torch.stack([torch.from_numpy(np.load(L/'p0_tx_v2'/s/'source_rgb.npy')).permute(2,0,1).float()/127.5-1 for s in ids]).cuda()
model=PrefixModel(json.loads((L/'checkpoint.json').read_text())['path']);model.train()
lp=load_training_lpips(json.loads((L/'metric_assets.json').read_text()))
torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.deterministic=True
opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.95));gen=torch.Generator().manual_seed(2026110401)
records=[]
for arm in ['R2','R3','R3']:
 opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();values=[]
 for _ in range(4):
  loss,r=model.objective(x,arm,lp,gen);assert torch.isfinite(loss);(loss/4).backward();values.append(float(loss.detach()))
 grad={k:sum(float(p.grad.detach().square().sum()) for p in m.parameters() if p.grad is not None)**.5 for k,m in [('E',model.vq.encoder),('D',model.vq.decoder),('P',model.ar)]};assert all(np.isfinite(v) and v>0 for v in grad.values())
 opt.step();torch.cuda.synchronize();records.append(dict(arm=arm,global_batch=16,microbatch=4,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),gradient_norms=grad,losses=values))
 (O/'progress.json').write_text(json.dumps(records,indent=2))
(O/'complete.json').write_text(json.dumps(dict(passed=True,records=records,optimizer_updates=3,discarded_weights=True,not_training_comparison=True),indent=2))
