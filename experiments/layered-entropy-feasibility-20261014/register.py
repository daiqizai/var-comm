"""New LE1 registration; never interprets user PLAN_ONLY as an executed registration."""
from common import *
import subprocess
def run():
 assert not (O/'registration.json').exists() and not (T/'science/launch.json').exists()
 scope=read(E/'user_package/scope.json');assert scope['default_handoff_scope']==['LE0_read_only_audit','LE1_source_feasibility']
 for name in ('asset_audit_resume_control_v1','tests_control_v1'):
  w=read(T/name/'actual_wait.json');assert type(w['returncode']) is int and w['returncode']==0 and not any(w[x] for x in ('terminated','killed','timed_out')) and gone(w['process']) and gone(w['parent_process'])
 gap=read(O/'assets/gap.json');assert gap['reused']==750 and gap['missing_count']==450
 sourcebindings=[pin(p) for p in sorted(E.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.md')]
 legacy=['src/var_comm/next_scale_prior.py','experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py','experiments/scale-causal-partial-residual-20261002/partial_receiver.py','experiments/content-real-64qam-20261006/h_source_driver.py','experiments/entropy-partial-mismatch-20261010/scripts/ep_source_codec.py','experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py','experiments/wcl-evidence-closure-20261009/scripts/t1_codec_runtime.py','experiments/wcl-evidence-closure-20261009/scripts/t1_entropy_core.py','experiments/entropy-partial-mismatch-20261010/scripts/h800_ep_pilot_metrics_v1.py']
 sourcebindings.extend(pin(R/n) for n in legacy)
 for p in (R/'src').rglob('*'):
  if p.is_file() and p.suffix in ('.py','.cpp','.h','.hpp') and str(p) not in {r['path'] for r in sourcebindings}:sourcebindings.append(pin(p))
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();assert head=='ccf9497e9661b75b9b229c4bf4e7d08b6a9c0b3e'
 csvtext=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 gpu=next([v.strip() for v in row.split(',')] for row in csvtext.splitlines() if row.split(',')[0].strip()=='3')
 assert int(gpu[3])>=8192 and int(gpu[4])<=50
 caps=dict(model_load=1,source_tx=50,source_rx=24,clean_render=450,decoder_forward=450,prior_scale=5240,dino_model_load=1,source_embedding=50,reconstruction_dino=450,tokenizer=0,wireless_frames=0,packet_trials=0,training_updates=0)
 save(O/'registration.json',dict(schema='NESTCOM_LAYERED_ENTROPY_LE01_ACTUAL_REGISTRATION_V1',status='REGISTERED_LE0_LE1_ONLY',user_plan_scope=pin(E/'user_package/scope.json'),authorized_stages=['LE0','LE1'],later_stages='NOT_AUTHORIZED_NOT_REGISTERED',source_bindings=sourcebindings,
  pilot50=pin(O/'assets/pilot50.json'),model_routes=pin(O/'assets/model_routes.json'),clean_cache=pin(T/'assets/admitted_clean_cache.json'),gap=pin(O/'assets/gap.json'),caps=caps,models=MODELS,wireless_frames=0,training_updates=0,probability_passes_per_source=1,source_endpoint_domain=list(ENDS),base_targets=[6,7],actual_base_cache=[4,5,6,7],numeric_flags=read(O/'assets/model_routes.json')['numeric_flags'],cpu_affinity=[0,1],gpu_index=3,gpu_uuid=gpu[1],gpu_pci=gpu[2],admission=dict(free_MiB=int(gpu[3]),utilization=int(gpu[4]),startup_free_MiB=8192,allocator_cap_GiB=12,running_free_MiB=2048,exclusive_GPU_required=False,not_timing=True),created_unix=time.time(),deadline_unix=time.time()+21600,max_seconds=21600,head_before_execution=head,
  independent_roundtrip_domain=dict(first_sources=6,base_depths=[6,7],ends=[[8,0],[8,42]],total=24,truth_audit_after_fixed_output=True),clean_cache_reuse=750,new_clean_gap=450,missing_tokenizer_sources=0,automatic_retry=False,old_raw_R1_gate='NO_GO_STOP'))
 print(canonical(dict(status='REGISTERED_NEW_LE1_NOT_EXECUTED',registration=pin(O/'registration.json'))),flush=True)
if __name__=='__main__':run()
