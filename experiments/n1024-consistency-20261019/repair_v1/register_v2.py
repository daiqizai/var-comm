"""Register the reviewed finite repair; no scientific role is called here."""
from nc_common import *
from nc_protocol import closed_wait,validate_repair_grid
import subprocess,argparse
def main(a):
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==a.expected_head
 assert not(O/'registration.json').exists()
 for role in ('phy','receiver','metrics','delivery'):assert not(T/role).exists() and not(T/(role+'_control')).exists()
 rev=checkread(dict(path=a.review_path,sha256=a.review_sha256,bytes=a.review_bytes))
 assert rev['status']=='PASS' and rev['scope']==SCOPE and rev['scientific_launch_authorized'] is True
 prep=read(BASE/'repair_preparation_v2/completion.json');assert prep['scope']==SCOPE
 testdir=BASE/'repair_tests_v1';closed_wait(read(testdir/'actual_wait.json'),gone)
 test=read(testdir/'completion.json');assert test['status']=='PASS_SYNTHETIC_ONLY' and test['production_scientific_calls']==0
 old=checkread(prep['original_registration']);grid=validate_repair_grid(checkread(prep['frames']))
 code=[];seen=set()
 for d in old['code']+[pin(p) for p in sorted(E.iterdir()) if p.is_file()]:
  checked(d)
  if d['path'] not in seen:code.append(d);seen.add(d['path'])
 assert rev['code']==[pin(p) for p in sorted(E.iterdir()) if p.is_file()]
 gpu=subprocess.check_output(['nvidia-smi','-i',a.gpu_uuid,'--query-gpu=pci.bus_id,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert gpu[1].strip()==a.gpu_uuid and int(gpu[2])>=15360 and int(gpu[3])<=25
 for k in ('sources','frames','reused_frames','receiver_cache','verified_inputs','cross_family_noise_proof'):checked(prep[k])
 for d in checkread(prep['verified_inputs']):checked(d)
 for k in ('model_routes','catalogue','phy_qualification','supplemental_registration'):checked(old[k])
 sr=checkread(old['supplemental_registration'])
 for k in ('loader','metric_assets','dependency_overlay'):checked(sr[k])
 overlay=checkread(sr['dependency_overlay'])
 for d in overlay['files']:checked(d)
 O.mkdir(exist_ok=True);T.mkdir(exist_ok=True)
 common=dict(scope=SCOPE,model_routes=old['model_routes'],static=old['static'],gpu_index=a.gpu_uuid,gpu_pci=gpu[0].strip(),gpu_uuid=a.gpu_uuid,gpu_cuda_uuid=a.gpu_uuid,max_seconds=14400)
 save(O/'receiver_registration.json',dict(schema='N1024_REPAIR_RX_CONTEXT_V1',**common,allowed_input='actual_RX_projection_only',TX_source_truth=False,TX_CDF=False))
 r=dict(schema='N1024_SOURCE_VERSION_REPAIR_REGISTRATION_V1',**common,basis_HEAD=a.expected_head,user_authorization='2026-10-19 manuscript completion: minimal changed common500 WHOLE 4/10 events only; no policy reselection',review=pin(a.review_path),preparation=pin(BASE/'repair_preparation_v2/completion.json'),original_T1C_registration=prep['original_registration'],frozen_policies=old['frozen_policies'],code=code,sources=prep['sources'],frames=prep['frames'],reused_frames=prep['reused_frames'],receiver_cache=prep['receiver_cache'],receiver_registration=pin(O/'receiver_registration.json'),cross_family_noise_proof=prep['cross_family_noise_proof'],exact_grid=grid,phy_qualification=old['phy_qualification'],backend_identity=old['backend_identity'],catalogue=old['catalogue'],metric_config=old['metric_config'],supplemental_registration=old['supplemental_registration'],caps=dict(physical=2421,packets=4842,logical=2421,source_TX_additional=0,tokenizer=0,receiver_unique_upper_bound=2421,metric_new_pairs_upper_bound=2421),full_grid=dict(logical=9000,reused=6579,changed=2421,sources=500,populations=['common500'],families=['WHOLE','PARTIAL'],snrs=[4,10,19],noise_seeds=[6201,6202,6203]),N=1024,header_symbols=68,body_symbols=956,Es=2,length_bits=13,CRC_bits=16,bootstrap=prep['bootstrap'],statistics=dict(source_average_three_noise=True,directly_paired=True,reuse_exact_saved_index_bank=True,new_draws=0),source_utilization=pin(R/'results/n1024_consistency_20261019_v1/source_records_3000.csv'),tests=[pin(testdir/n) for n in ('actual_wait.json','completion.json','stdout.txt','stderr.txt')],automatic_retry=False,automatic_successor=False,training=0,selection=0,other_methods=False,other_SNRs=False,old_results_untouched=True)
 save(O/'registration.json',r);print(json.dumps(pin(O/'registration.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);p.add_argument('--expected-head',required=True);p.add_argument('--review-path',required=True);p.add_argument('--review-sha256',required=True);p.add_argument('--review-bytes',required=True,type=int);main(p.parse_args())
