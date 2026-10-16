from pathlib import Path
import sys,csv,sqlite3,shutil
E=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/submission-closure-20261016');sys.path.insert(0,str(E));from common import *
d=T/'source_audit_control_v1';run=T/'m10_source';w=read(d/'actual_wait.json')
assert type(w['returncode'])is int and w['returncode']==0 and w['actual_wait']and not w['timed_out']and gone(w['process'])and gone(w['parent_process'])
assert not(run/'failure.json').exists()
c=read(run/'completion.json');assert c['new_TX']==1000 and c['new_RX']==8 and c['new_prior_scales']==10080 and c['counts']['unresolved']==0
for v in c['outputs']:checked(v)
db=sqlite3.connect(f'file:{run}/calls.sqlite?mode=ro',uri=True);counts=db.execute('SELECT kind,status,COUNT(*) FROM calls GROUP BY kind,status').fetchall();db.close();assert all(s=='COMPLETE'for _,s,_ in counts)
out=O/'m10';out.mkdir(exist_ok=False)
for p in [run/'m10_lengths.csv',run/'source_roundtrip_report.json',run/'completion.json',d/'actual_wait.json',d/'launch.json',d/'stdout.txt',d/'stderr.txt']:
 with (out/p.name).open('xb')as f:f.write(p.read_bytes())
save(out/'closure_audit.json',dict(status='CLOSED_M10_SOURCE_ONLY',parent_child_gone=True,counts=counts,ledger=pin(run/'calls.sqlite'),outputs=[pin(p)for p in out.iterdir()],new_channel_frames=0,new_render=0))
import numpy as np
l=np.array([int(x['arithmetic_bits'])for x in csv.DictReader((out/'m10_lengths.csv').open())]);print('M10',dict(n=len(l),min=int(l.min()),median=float(np.median(l)),p95=float(np.percentile(l,95)),max=int(l.max()),fit4751=int((l<=4751).sum()),above8191=int((l>8191).sum()),actual_seconds=w['elapsed_seconds']))
print('CLOSURE',pin(out/'closure_audit.json'))
p=R/'outputs/WCL-ENTROPY-PARTIAL-20261010/H800_PHY457_v2';print('PHYFILES',[(q.name,q.stat().st_size)for q in p.iterdir()if q.is_file()]);x=read(p/'completion.json');print('PHYCOMPKEY',list(x));print('PHYCOMPREFS',{k:v for k,v in x.items()if any(t in k for t in ['catalog','request','layout'])})
