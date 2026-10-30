import sys,csv
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
sources=read(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json');byid={s['source_id']:s for s in sources};old=R/'paper/manuscripts/nestcom_wcl_focused_20261029_v6/figure3_metrics.csv'
with old.open()as f:rows=list(csv.DictReader(f))
selected=[]
for index,label in [(95,'Pastries'),(21,'Dog')]:
 v=next(v for v in rows if int(v['source_index'])==index);s=byid.get(v['source_id']);assert s is not None,'Known example not in common500; do not substitute'
 assert s['preprocessing_id']==v['preprocessing_id']and s['reference_sha256']==v['source_array_sha256']
 selected.append(dict(label=label,source_id=s['source_id'],source_index=s['source_index'],preprocessing_id=s['preprocessing_id'],reference_sha256=s['reference_sha256']))
save(D/'display_source_rule.json',dict(status='FROZEN_BEFORE_FINAL_RECONSTRUCTION_METRICS',known_development_examples=selected,prior_identity=pin(old),common500_manifest=pin(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json'),cells=[[1024,7],[1024,10]],channel_repeat=0,method_native_noise_preserved=True,gray_displayed=True,full_view=True,quality_based_selection=False,NEW200_main_contact_sheet='BLOCKED_NO_UNUSED_RAW_POOL',C_mechanism=dict(source_ids=[s['source_id']for s in read(L/'inputs/fit100.json')[:2]],cell=[1024,7],rule='First2 in already frozen FIT hash order, fixed development diagnostics, no quality-based filtering')))
print(selected)
