"""Read-only post-selection snapshot; historical failures remain separate."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,time
R=Path('/home/liulu/projects/VAR_COMM');O=R/'outputs/CONTENT-REAL-64QAM-20261006';H=O/'H';E=R/'experiments/content-real-64qam-20261006'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):return dict(path=str(p),sha256=sha(p),status=read(p).get('status'))
s=importlib.util.spec_from_file_location('closed_selection_owner',E/'stage_owner.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
d=H/'initial_selection_execution_r2';cfg=read(d/'owner_config.json');reg=read(d/'execution_registration.json')
a.validate_config(cfg,reg,sha(d/'owner_config.json'));done=read(d/'completion.json')
assert done['status']=='REGISTERED_H_STAGE_BATCH_COMPLETE' and done['registration_sha256']==sha(d/'execution_registration.json')
assert done['allowed_stage_ids']==['freeze'] and not (d/'failure.json').exists()
for x in done['completed']:assert sha(Path(x['completion']))==x['sha256']
wd=d/'stages/freeze/workers/initial_selection';exit=read(wd/'exit_receipt.json');launch=read(wd/'launch.json')
assert exit['exit_code']==0 and a.same_identity(exit['identity'],launch['identity'])
assert sha(wd/'worker.log')==exit['closed_log_sha256']
assert sha(Path(exit['completion']))==exit['completion_sha256']
assert all(a.raw_process_state(x['pid']) is None for x in (read(d/'owner_identity.json'),exit['identity']))
job=cfg['stages'][0]['jobs'][0];a.receipt(job['completion'],job['accepted_statuses'],sha(d/'execution_registration.json'),job['receipt_expect'],job['out'])
budget=a.budget_snapshot(cfg['budget_path'],cfg['budget_registration_sha256'],cfg['phase_limits'],quiescent=True)
assert budget==done['budget'] and budget['charged']==69960 and budget['failed']==budget['unresolved']==0
close=H/'render_closeout_v1';cd=read(close/'closeout_completion.json');a.verify(cd['outputs'])
assert cd['status']=='H_RENDER_CLOSEOUT_COMPLETE_STOP_RELEASED' and cd['original_owner_success'] is False
assert not (H/'STOP').exists() and not (H/'initial_true200_render_r1/STOP').exists()
selected=read(H/'initial_selection200_r2/selected_whole.json');assert selected['selected_count']==8
value=dict(status='H_INITIAL200_SELECTED_FULL1000_NOT_STARTED',observed_unix=time.time(),bindings_verified=True,
 closeout=record(close/'closeout_completion.json'),closeout_registration=record(close/'execution_registration.json'),
 selection_owner=record(d/'completion.json'),selection_registration=record(d/'execution_registration.json'),
 selection=record(H/'initial_selection200_r2/selected_whole.json'),budget=budget,
 current_processes=[],original_render_owner_success=False,scientific_render_outputs_complete=True,
 new_failures=[],H_full_delivery_claimed=False,full1000_calibration_started=False,main_full1000_calibration_complete=False,
 C_started=False,holdout_started=False,partial_calibration_complete=False,partial_reference_count=6,
 GPU=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader'],text=True).strip(),
 GPU_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),
 free_disk_GiB=shutil.disk_usage(R).free/2**30,remote_sha=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=R,text=True).split()[0],
 selected_summary=[{k:r[k] for k in ('snr_db','arm','target_m','nominal_rate','initial_true200_mean_psnr_db','actual_m_histogram')} for r in selected['selected_candidates']])
assert not value['GPU_processes']
out=O/f'closeout_selection_observation_{int(time.time())}.json'
with out.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(out),status=value['status'],GPU=value['GPU'],charged=budget['charged'])))
