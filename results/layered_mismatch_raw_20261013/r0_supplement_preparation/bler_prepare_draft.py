"""Metadata-only supplement draft and historical wall-clock estimate; no PHY import."""
from pathlib import Path
import argparse
import json
import sys

parser=argparse.ArgumentParser();parser.add_argument('--revision',type=int,default=2);args=parser.parse_args()
assert args.revision>=2
suffix=f'_v{args.revision}'

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/layered-mismatch-raw-20261013'
O=R/'results/layered_mismatch_raw_20261013/r0_supplement_preparation'
RT=R.parent.parent/'var_comm_runtime_20261010'
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu')
sys.path.insert(0,str(E));import r0_bler as b
groups=json.loads((O.parent/'candidates_cpu_v1/groups_by_phy_key.json').read_text())
cells=b.validate_cells(groups,json.loads((O/'screening_missing_bler_keys.json').read_text()),json.loads((O/'screening_historical_bler.json').read_text()))
request=RT/'qualification/h800_mismatch_phy_v1_attempt1/registered/request.json'
raw=json.loads(request.read_text())
pins={
 'groups':b.pin(O.parent/'candidates_cpu_v1/groups_by_phy_key.json'),
 'missing':b.pin(O/'screening_missing_bler_keys.json'),
 'historical':b.pin(O/'screening_historical_bler.json'),
 'raw_request':b.pin(request),'r0_bler':b.pin(E/'r0_bler.py'),
 'screening_source':b.pin(E/'r0_screening.py'),
 'screening_contract':b.pin(O/'screening_contract.json'),
 'authorization':b.pin(O/'screening_approval_addendum.json'),
 'server_relocation':b.pin(O/'server_input_relocation.json')}
for key,h in b.SCREEN_PINS.items():assert pins[key]['sha256']==h
historical=RT/'qualification/h800_mismatch_phy_v1_attempt1'
wait=json.loads((historical/'run_owner_actual_wait.json').read_text())
done=json.loads((historical/'registered/run/worker_completion.json').read_text())
assert wait['returncode']==0 and wait['actual_wait'] and done['results']['physical_count']==4500
assert done['results']['packet_ledger']==dict(cap=9000,total=9000,unresolved=0)
estimate=dict(status='COARSE_HISTORICAL_ONLY_NOT_NEW_PILOT',
 historical_actual_wait=b.pin(historical/'run_owner_actual_wait.json'),
 historical_completion=b.pin(historical/'registered/run/worker_completion.json'),
 historical_seconds=wait['elapsed_seconds'],historical_same_backend_CPU_threads=2,
 historical_full_frames=4500,historical_body_trials=4500,historical_header_decodes=4500,
 scaled_seconds=wait['elapsed_seconds']/4500*b.CAP,
 scaled_hours=wait['elapsed_seconds']/4500*b.CAP/3600,
 limitation='Different mix of body dimensions/SNR and persistence overhead; old full-frame overhead includes headers. This is an estimate, not a runtime guarantee.',
 first_update='Observe only the first already-budgeted64 trials, then estimate throughput without any extra pilot.',
 new_timing_or_scientific_calls=0)
draft=dict(schema=b.SCHEMA,status='DRAFT_NOT_REGISTERED',namespace=b.NAMESPACE,body_session=b.BODY_SESSION,
 snrs_db=list(b.SNRS),trials_per_point=b.TRIALS,point_count=len(cells),body_trial_cap=b.CAP,
 header_trials=0,training_updates=0,automatic_retry=False,automatic_successor=False,
 ordered_cells_sha256=b.digest(cells),pins=pins,
 runtime=dict(python=raw['CPU_python'],threads=2,cpu_affinity=[2,3],CUDA_VISIBLE_DEVICES='',
  storage_root=str(TMP),output_directory=str(TMP/'liulu-var-comm/layered_mismatch_raw_20261013/r0_bler_v1'),
  timing_v6_directory=str(R/'results/baseline_strength_20261011/timing/six_method_v6'),
  timing_v6_run_directory=str(TMP/'liulu-var-comm/baseline_strength_20261011/six_method_timing_v6'),
  max_seconds=21600,deadline_unix=None),
 pending='Root must review, select a live deadline, confirm CPU/resource coordination and freeze an actual registration; no real registration or launch was created by this script.',
 immutable_scientific_grid=True,proposed_max_wall_hours=6,
 scientific_execution='NOT_RUN',header_or_model_calls=0)
for p in pins.values():b.checkpin(p)
O.mkdir(parents=True,exist_ok=True)
b.save(O/f'bler_registration_draft{suffix}.json',draft)
b.save(O/f'bler_wallclock_estimate{suffix}.json',estimate)
b.save(O/f'bler_preparation_receipt{suffix}.json',dict(status='METADATA_READY_NO_SCIENCE',
 missing_points=len(cells),retained_historical_selection_points=672,
 retained_historical_four_snr_points=896,body_trial_cap=b.CAP,header_trials=0,
 draft=b.pin(O/f'bler_registration_draft{suffix}.json'),wallclock_estimate=b.pin(O/f'bler_wallclock_estimate{suffix}.json'),
 source=b.pin(E/'r0_bler.py'),test_source=b.pin(E/'test_r0_bler.py'),
 torch_imported='torch' in sys.modules,real_runtime_constructed=False,scientific_calls=0))
print(json.dumps(dict(status='DRAFT_ONLY_NO_SCIENCE',points=len(cells),trials=b.CAP,estimated_hours=estimate['scaled_hours'],draft=b.pin(O/f'bler_registration_draft{suffix}.json'))))
