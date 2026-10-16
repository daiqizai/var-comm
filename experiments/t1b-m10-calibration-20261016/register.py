"""Metadata-only T1-B registration after exact closed source/reuse qualification."""
from tb_common import *
import csv,subprocess,ast

def main():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==HEAD
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=R,text=True).strip()
 assert not(O/'registration.json').exists()
 for name in ('control','phy','receiver','metrics','selection'):assert not(T/name).exists()
 controls=[]
 C=B/'t1b_m10_preparation_control_v1'
 assert read(C/'completion.json')['scientific_calls']==0 and not(C/'failure.json').exists()
 assert gone(read(C/'supervisor.json')['process'])
 for role in ('test_plan','prepare'):
  w=read(C/role/'actual_wait.json');assert type(w['returncode'])is int and w['returncode']==0 and gone(w['process'])
  controls+=[pin(C/role/'actual_wait.json')]
 X=B/'t1b_m10_execution_qualification_v1';w=read(X/'actual_wait.json')
 assert type(w['returncode'])is int and w['returncode']==0 and w['actual_wait'] and gone(w['process']) and not(X/'failure.json').exists()
 controls.append(pin(X/'actual_wait.json'))
 p=T/'preparation';done=read(p/'completion.json');manifest=checkread(done['manifest'])
 assert done['new_scientific_calls']==0 and manifest['unique_new_physical']==22139 and manifest['new_packet_total_cap']==44278
 sr=read(FIRST_O/'source_registration.json');assert sha(FIRST_O/'source_registration.json')=='13c9057b42cdc9d36a8934b2479e8a9385073d5205402852a355e32803c60845'
 sources=checkread(manifest['sources']);published=FIRST_O/'m10/m10_lengths.csv'
 assert sha(published)=='b94bd74527c763fab35287e4fe25ddd50f00e013188029dda63195dc58e33de7'
 with published.open(newline='',encoding='utf8')as f:csvrows=list(csv.DictReader(f))
 assert len(csvrows)==len(sources)==1000
 for i,(s,c)in enumerate(zip(sources,csvrows)):
  assert s['source_index']==int(c['source_index'])==i and s['source_id']==c['source_id'] and s['preprocessing_id']==c['preprocessing_id'] and s['tokens_sha256']==c['tokens_sha256']
  assert s['current_streams']['sha256']==c['archive_sha256'] and c['old24_byte_exact']=='True'
  ep=next(x for x in s['stream_endpoints']if(x['m'],x['K'])==(10,0))
  assert ep['bits']==int(c['arithmetic_bits']) and ep['payload_sha256']==c['bits_sha256']
  assert all(s[k]==sr['sources'][i][k]for k in sr['sources'][i])
 inh=read(p/'inheritance.json');requests={k:checkread(d)for k,d in inh['requests'].items()}
 for d in inh['controls']:checked(d)
 rv=requests['full_visual'];device=rv['device_identity_binding'];assert device==requests['full_metrics']['device_identity_binding']
 assert device['nvml_gpu_index']==sr['gpu_index']==2 and device['nvml_uuid']==sr['gpu_uuid']
 code={}
 def add(d):
  checked(d);current=pin(d['path']);prior=code.setdefault(d['path'],current);assert current==prior
 for d in sr['code']:add(d)
 for req in requests.values():
  for name,h in req.get('tool_bindings',{}).items():add(dict(path=str(S/name),sha256=h))
 qual=checkread(requests['full_phy']['phy_identity']['qualification'])
 for path,h in qual['source_bindings'].items():add(dict(path=path,sha256=h))
 for d in inh['metric_config']['sources'].values():add(d)
 for name in ('leo_whole_math_v1.py',):add(pin(S/name))
 for name in ('h64_catalog.py','h64_source.py'):add(pin(R/'experiments/content-real-64qam-20261006'/name))
 for f in sorted(E.glob('*.py')):
  ast.parse(f.read_text(encoding='utf8'),filename=str(f));add(pin(f))
 assert not any(x in sys.modules for x in('torch','tensorflow','sionna'))
 r=dict(schema='NESTCOM_T1B_M10_CALIBRATION_REGISTRATION_V1',scope='T1-B_ONLY_STOP_AFTER_DELIVERY',basis_HEAD=HEAD,
  user_authorization='Execute T1-B; deliver and stop. Symmetric whole/partial m10; verify reuse and fill missing events only; retain old candidates.',
  candidate_delta=done['manifest'],catalogue=pin(O/'public_catalogue.json'),old_receive_domain_audit=pin(O/'old_receive_domain_audit.json'),
  sources=manifest['sources'],frames=manifest['frames'],old_images=pin(p/'old_image_index.json'),old_pairs=pin(p/'old_pair_index.json'),inheritance=pin(p/'inheritance.json'),
  model_routes=sr['model_routes'],static=rv['static'],code=sorted(code.values(),key=lambda x:x['path']),
  source_boundary_closure=pin(FIRST_O/'m10/closure_audit.json'),published_lengths=pin(published),preparation_controls=controls,
  gpu_index=2,gpu_pci=sr['gpu_pci'],gpu_uuid=sr['gpu_uuid'],gpu_cuda_uuid=device['cuda_uuid'],
  resource_policy=dict(start_free_MiB=15360,start_util_max=25,private_allocator_GiB=12,running_free_floor_MiB=2048,CPU_affinity=[0,1],FP32=True,TF32=False,math_threads=6,interop_threads=2,exclusive_GPU=False,old_resource_conditions_not_claimed_identical=True),
  caps=dict(physical=22139,packets=44278,source_rx=22139,renders=22139,metric_pairs=36000,source_TX=0,tokenizer=0,logical_frames=36000),
  seconds=dict(phy=21600,receiver=21600,metrics=14400,selection=1800),
  selection_rule=manifest['selection_rule'],tie_break='legacy_before_new_on_exact_tie_then_candidate_id_lexical',
  averaging='source mean over seeds4101/4102/4103 then mean over original1000 sources',
  receiver_inputs='actual paid public profile, actual received arithmetic bits, shared frozen models; source truth only after fixed output',
  N=1024,header_symbols=68,body_symbols=956,length_bits=13,CRC_bits=16,Es=2,SNRs=list(SNRS),noise_seeds=list(SEEDS),
  T1C=False,LE3=False,training=False,bootstrap=False,backup_deferred_by_user=True,automatic_followup=False,automatic_retry=False)
 save(O/'registration.json',r)
 save(O/'registration_evidence.json',dict(registration=pin(O/'registration.json'),published_lengths_matched_sources=1000,new_scientific_calls=0,model_imports=[]))
 print(json.dumps(pin(O/'registration.json')),flush=True)
if __name__=='__main__':main()
