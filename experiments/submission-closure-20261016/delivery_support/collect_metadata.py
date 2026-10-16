from pathlib import Path
import sys,json,hashlib,subprocess
E=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/submission-closure-20261016');sys.path.insert(0,str(E));from common import *
out=O/'facts';out.mkdir(exist_ok=False)
cat=R/'outputs/WCL-ENTROPY-PARTIAL-20261010/H800_PHY457_v2/constructor_catalogue.json';x=read(cat)
layouts=x['layouts'];assert len(layouts)==12
rows=[]
for y in layouts:
 rates=[p['nominal_rate']for p in x['profiles']if(p['k'],p['n'],p['q'])==(y['k'],y['n'],y['q'])];assert len(set(rates))==1
 rows.append(dict(q=y['q'],nominal_rate=rates[0],k=y['k'],n=y['n'],capacity=y['k']-29,bg=y['bg'],z=y['z'],k_ldpc=y['k_ldpc'],k_filler=y['k_filler'],mother_bits=y['mother_bits'],puncturing_first_2Z=y['puncturing_first_2Z'],puncturing_remaining=y['puncturing_remaining'],shortening_bits=y['shortening_bits'],repetition_bits=y['repetition_bits'],num_iter=y['decoder_config']['num_iter'],cn_update=y['decoder_config']['cn_update'],early_stop=y['decoder_config']['early_stop'],layout_id=y['layout_id']))
csvsave(out/'entropy_12_actual_ldpc_layouts.csv',rows);save(out/'entropy_backend_evidence.json',dict(catalogue=pin(cat),qualification=pin(cat.parent/'completion.json'),backend=read(cat.parent/'completion.json')['backend_identity']))
dev=RT/'assets/timing_fixed16_seed_v1/VAR_COMM/outputs/CONTENT-REAL-64QAM-20261006/H/development100_assets/manifest.json';x=read(dev);save(out/'development100_manifest.json',x);print('DEVKEY',list(x))
ns=read(R/'experiments/entropy-partial-mismatch-20261010/reports/h800_ep_new100_source_v1.json');print('NEWREFS',{k:v for k,v in ns.items()if any(t in k for t in ['request','registration','path','source'])and k not in ['notes']})
print('QUALDIRS',[p.name for p in (RT/'qualification').iterdir()if 'new100_source' in p.name or 'new100_content' in p.name])
print('LAYOUTS',rows)
