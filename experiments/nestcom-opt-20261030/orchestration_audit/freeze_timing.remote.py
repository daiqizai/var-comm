import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
sources=read(L/'inputs/fit100.json')[:16]
save(D/'timing_source_rule.json',dict(status='PREPARED_BEFORE_TIMING',source_selection='First16 in already frozen FIT hash order; no quality-based replacement.',sources=[dict(source_id=s['source_id'],source_index=s['source_index'],source_pixels_sha256=s['preprocessing_id'],reference_sha256=s['reference_sha256'],archive=s['archive'])for s in sources],warmup_repeats=2,measured_repeats=3,batch=1,source_count=16,same_gpu_per_comparison=True,CPU_affinity=[22,23],metrics=['mean','median','p95','peak_memory','model_storage','termination_attempts'],live_load_observation_required=True,not_executed=True))
print('TIMING_RULE_FROZEN')
from collect_progress import main
main()
