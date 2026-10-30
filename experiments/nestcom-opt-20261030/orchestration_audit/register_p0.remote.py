import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
assert not(D/'p0_registration.json').exists()
defaults=read(D/'protocol_defaults.json');models=pin(R/'results/checkpoint_relocation_20261028_v1/model_routes.json');routes=cr(models)
for d in routes['files']+routes['author_files']+[routes['dist']]:checked(d)
anchors=[]
for k in ['primary_checkpoint','verification_checkpoint']:
 v=defaults['routes']['A'][k];p=Path(v['path']);assert sha(p)==v['sha256']and p.stat().st_size==v['size_bytes'];anchors.append(pin(p))
donors=read(L/'inputs/p0_donors.json')
sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'));from n512_phy import received_projection
N=B/'nestcom_n512_20261028_v1'
for p in sorted((N/'pilot_phy_00/frames').glob('*.json')):
 f=read(p);rx=f.get('actual_RX')
 if rx is None:
  print('FRAME_KEYS',list(f));break
 projection=received_projection(rx)
 assert projection==f['projection']
 if projection['status']!='SOURCE_INPUT':
  donors['gray']=dict(input=dict(key=digest(projection),projection=projection),frame=pin(p),expected_pixel_sha256=asha(np.full((3,256,256),.5,np.float32)));break
assert set(donors)=={'whole','partial','full','gray'},('P0 missing donor',list(donors))
save(L/'inputs/p0_donors_v2.json',donors)
code=[pin(E/f)for f in ['opt_common.py','p0_replay.py','control_once.py']]
# Bind all imported scientific modules, separately from unrelated old controllers.
deps=['experiments/kodak-current-20261026/nest_v1/kn_common.py','experiments/kodak-current-20261026/nest_v1/kn_native.py','experiments/t1b-m10-calibration-20261016/tb_native.py','experiments/t1b-m10-calibration-20261016/runtime_boundary.py','experiments/t1c-m10-test-20261016/tc_receiver.py','experiments/content-real-64qam-20261006/h_source_driver.py','experiments/scale-causal-partial-residual-20261002/partial_receiver.py','src/var_comm/next_scale_prior.py','experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py']
code.extend(pin(R/x)for x in deps)
proto=read(R/'results/nestcom_n512_20261028_v1/protocol.json')
resolved=dict(defaults=pin(D/'protocol_defaults.json'),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),models=models,anchors=anchors,sources=pin(D/'source_manifest.json'),policy_N512=pin(R/'results/nestcom_n512_20261028_v1/frozen_policies.json'),policy_N1024=pin(R/'results/t1b_m10_calibration_20261016_v1/delivery/frozen_policies.json'),policy_7db=pin(R/'results/nestcom_7db_20261030_v1/frozen_policies.json'),catalogue_N512=proto['catalogue'],catalogue_N1024=pin(R/'results/t1b_m10_calibration_20261016_v1/public_catalogue.json'),metrics=pin(R/'results/nestcom_evidence_20261029_v1/three_metrics_registration.json'),clip=pin(R/'results/nestcom_evidence_20261029_v1/clip_metrics_path_v2_registration.json'),noise_seeds=dict(FIT=[4101],CONFIRM=[4101,4102,4103],common500=[6201,6202,6203],NEW200=[6201,6202,6203]),B_generation_seed_recipe='fixed per-image reset to each of the three registered seeds; no hidden source ID',FIT_denominators='not measured; freeze after original baseline before candidate gate',NEW200='not yet located; no new-test outputs read',no_automatic_commit_push=True,scientific_call_count_before_P0=0)
save(D/'protocol_resolved_p0.json',resolved)
r=dict(code=code,donors=pin(L/'inputs/p0_donors_v2.json'),model_routes=models,static=proto['nest_static'],output=str(L/'p0_replay'),cpu_affinity=[20,21],pci='00000000:64:00.0',seconds=3500,caps=dict(model_load=1,prior_scale=60,decoder_forward=3,source_decode=3,render=3))
save(D/'p0_registration.json',r);rsha=sha(D/'p0_registration.json')
control=dict(code=code,control=str(L/'p0_control'),command=[str(RT/'envs/frozen_visual_py310/bin/python'),'-B',str(E/'p0_replay.py'),str(D/'p0_registration.json'),rsha],environment=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3',PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'),hard_seconds=3590,gpu=True)
save(D/'p0_control_registration.json',control)
print('PREPARED',rsha,sha(D/'p0_control_registration.json'))
