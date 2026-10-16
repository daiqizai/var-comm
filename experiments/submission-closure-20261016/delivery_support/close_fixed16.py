from pathlib import Path
import sys,json,subprocess
E=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/submission-closure-20261016');sys.path.insert(0,str(E));from common import *
role='fixed16_score';d=T/(role+'_control_v1');w=read(d/'actual_wait.json');assert w['returncode']==0 and w['actual_wait']and not w['timed_out']and gone(w['process'])and gone(w['parent_process'])
c=read(T/'fixed16_score/completion.json');assert c['counts']['unresolved']==0 and c['new_raw_metric_scalars']==48 and c['new_psnr_scalars']==64
out=O/'fixed16';out.mkdir(exist_ok=False)
for p in [T/'fixed16_score'/n for n in ['frame_scores.csv','summary.csv','paired_descriptive.csv','completion.json','runtime_identity.json']]+[d/'actual_wait.json',d/'launch.json',d/'stdout.txt',d/'stderr.txt']:
 target=out/p.name
 with target.open('xb')as f:f.write(p.read_bytes())
 assert sha(target)==sha(p)
save(out/'closure_audit.json',dict(status='CLOSED_ALL16_NATIVE_FOUR_METHODS',wait=pin(d/'actual_wait.json'),science=pin(T/'fixed16_score/completion.json'),parent_child_gone=True,new_scalars=112,reused_scalars=144,outputs=[pin(p)for p in out.iterdir()]))
print((out/'summary.csv').read_text())
