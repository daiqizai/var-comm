from shared import *
import numpy as np
import pair_protocol as p
def main():
 a=read(O/'qualification/native_audit.json');assert a['status']=='NUMERICAL_PASS_VISUAL_REVIEW_PENDING'
 assert a['preview']['sha256']=='1c2da3672614d2b90768493ebb539fc15803c44b41098cd719cf4a07582e31f2';verify(a['preview'])
 for role in ('qualification_close_saved','qualification_audit_saved','qualification_native','audit_b1_native','verify_cal100_cache'):
  w=read(L/'controls'/role/'actual_wait.json');assert w['returncode']==0 and gone(w['process'])
 put(O/'qualification/review.json',dict(status='PASS',reviewer='root Codex',actual_opened=True,opened_local='C:/Users/11946/Documents/ChatGPT/comm/.research/arpc_n2048_adaptive_20261019/qualification_contact_sheet.png',preview=a['preview'],native_audit=pin(O/'qualification/native_audit.json'),transport_audit=pin(O/'qualification/phy_completion.json'),observations=['All six decoded/reused images are finite native RGB256 with aligned full framing.','NeST qualifying prefix images are blurred but retain gross scene layout.','ARPC short e4 changes object details markedly; this is retained, not filtered or corrected.','No TX truth or CDF supplied to new native worker; two new received token sequences matched only after output and process exit.'],resource_review='New NeST2 model jobs completed on admitted H800 GPU0 with original FP32/TF32off/6-2 threads and12GiB allocator/2GiB floor. ARPC3 and NeST1 reused only exact closed source objects.',original_RX_metadata_failure_preserved=True,original_RX_returncode=1,recovery_added_packet_calls=0,B2_requires_cal100_nomination=True))
 r=registration('4d07344ecd2008f3663558a596afa3a879eb13797bccb4c2a41b3bff5c683ae9');root=L/'cal100_clean_v1';root.mkdir();inputs=root/'wire_inputs';inputs.mkdir()
 gaps=read(L/'cal100_verified_cache_v1/missing.json');previous={x['source_input_key']:x for x in read(L/'qualification/native/rx/completion.json')['outputs']};jobs={};mapping=[]
 for s in gaps:
  assert s['codec']==0;verify(s['wire'])
  with np.load(s['wire']['path'],allow_pickle=False)as z:bits=z[s['array_key']].copy()
  assert hashlib.sha256(bits.tobytes()).hexdigest()==s['bits_sha256'];data=p.pack_bits(bits);key=p.input_key(0,s['endpoint'],s['bit_length'],data)
  mapping.append(dict(source=s,key=key,qualification_reuse=previous.get(key)))
  if key in previous:continue
  if key not in jobs:
   ip=inputs/(key+'.bin')
   with ip.open('xb')as f:f.write(data)
   jobs[key]=dict(key=key,codec=0,endpoint=s['endpoint'],bit_length=s['bit_length'],generation_seed=0,input=pin(ip))
 assert len(gaps)==1793 and sum(x['qualification_reuse'] is not None for x in mapping)==2 and len(jobs)<=1791
 put(inputs/'jobs.json',list(jobs.values()));put(root/'private_mapping.json',mapping)
 gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=pci.bus_id,uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',');assert int(gpu[2])>=15360
 cfg=dict(run=str(root),model_routes=r['model_routes'],static=r['static'],jobs=pin(inputs/'jobs.json'),gpu_index=0,gpu_pci=gpu[0].strip(),gpu_uuid=gpu[1].strip(),new_jobs_cap=len(jobs),max_seconds=3600)
 put(root/'public.json',cfg)
 put(O/'cal100_clean_registration.json',dict(status='REGISTERED_MISSING_CLEAN_RX_ONLY',source=pin(E/'nest_prefix_rx.py'),preparation_source=pin(E/'review_and_prepare_cal100.py'),b1_review=pin(O/'qualification/review.json'),original_registration=pin(REG),public=pin(root/'public.json'),jobs=pin(inputs/'jobs.json'),private_mapping=pin(root/'private_mapping.json'),exact_cal100_cache=pin(L/'cal100_verified_cache_v1/completion.json'),new_source_RX_and_render=len(jobs),qualified_reuse=2,existing_image_and_score_reuse=1407,new_DINO_upper=1793,new_source_TX=0,new_tokenizer=0,new_channel=0,new_training=0,scope='Only cal100 prefix gaps. Never score test data or select on test.') )
 print(json.dumps(dict(new_RX_and_render=len(jobs),qualification_reuse=2,existing_reuse=1407,registration=pin(O/'cal100_clean_registration.json'))))
if __name__=='__main__':main()
