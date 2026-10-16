"""Freeze exact changed test-event budget after source completion and CPU qualification."""
from tc_common import *
from tc_protocol import validate_grid,closed_wait
import subprocess
def main():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==HEAD
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=R,text=True).strip()
 assert not(O/'link_registration.json').exists()
 for stage in ('source','tests'):
  closed_wait(read(T/(stage+'_control/actual_wait.json')),gone);assert not(T/(stage+'_control/failure.json')).exists()
 for stage in ('phy','receiver','metrics','delivery'):assert not(T/stage).exists() and not(T/(stage+'_control')).exists()
 src=registration('source','84891e935d19ce8016546a4cb0da5e9302fa10678aa2f800d53b410f45023466');prep=read(T/'link_preparation_v3/completion.json');grid=validate_grid(checkread(prep['frames']))
 assert grid==dict(logical=3600,reused=2454,changed=1146,new_physical=1125,packet_cap=2250)
 assert sha(FROZEN)=='81dfc416ccd82bad538885e8f71f394a6f7f94bb21e403b510721d78582b3e02'
 old=read(R/'results/t1b_m10_calibration_20261016_v1/registration.json');inherit=checkread(old['inheritance']);phy=checkread(inherit['requests']['full_phy'])['phy_identity']
 gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=pci.bus_id,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert int(gpu[2])>=15360 and int(gpu[3])<=25
 code=src['code']+[pin(E/n) for n in ('tc_protocol.py','tc_tests.py','prepare_link_v2.py','prepare_cross_family_reuse.py','tc_phy.py','tc_receiver.py','tc_metrics.py','tc_delivery.py','register_link.py')]
 code+=[pin(R/'experiments/t1b-m10-calibration-20261016/metric_device_recovery_v2.py')]
 for d in code:checked(d)
 sr=pin(R/'results/quality_metrics_supplement_20261014_v1/common500/registration.json');sreg=checkread(sr)
 for key in ('loader','metric_assets','dependency_overlay'):checked(sreg[key])
 r=dict(schema='T1C_LINK_REGISTRATION_V1',scope='T1C_19dB_ONLY_FROZEN_TWO_ENTROPY_FAMILIES',basis_HEAD=HEAD,user_authorization='2026-10-16: frozen corrected entropy whole and partial at19dB; common500 plus independentnew100; three original noise seeds; only missing or changed actual events; deliver and stop',source_registration=pin(O/'source_registration.json'),source_completion=pin(T/'source/completion.json'),frozen_policies=pin(FROZEN),code=code,sources=prep['sources'],frames=prep['frames'],exact_grid=grid,preparation=pin(T/'link_preparation_v3/completion.json'),model_routes=old['model_routes'],static=old['static'],catalogue=old['catalogue'],phy_qualification=phy['qualification'],backend_identity=phy['backend_identity'],metric_config=inherit['metric_config'],supplemental_registration=sr,gpu_index=2,gpu_pci=gpu[0].strip(),gpu_uuid=gpu[1].strip(),gpu_cuda_uuid=old['gpu_cuda_uuid'],max_seconds=7200,caps=dict(physical=1125,packets=2250,logical=3600,source_TX_additional=0,receiver_unique_upper_bound=1125,metric_new_pairs_upper_bound=1125),N=1024,header_symbols=68,body_symbols=956,Es=2,length_bits=13,CRC_bits=16,q=6,k=4780,n=5736,arithmetic_capacity=4751,source_rounding_or_truncation=False,noise_seeds={'common500':[6201,6202,6203],'new100':[9301,9302,9303]},statistics=dict(source_average_three_noise=True,bootstrap=10000,seed=2026100701,paired_difference_before_resampling=True,populations_never_pooled=True),prior_common500_whole_4090_bytes_differ=True,old_results_untouched=True,old_timing_stays_old_version=True,training=0,selection=0,LE3=False,other_SNRs=False,other_methods=False,backup_deferred_by_user=True,automatic_retry=False,automatic_successor=False,CPU_qualification=[pin(T/'tests_control/actual_wait.json'),pin(T/'tests_control/stderr.txt')])
 save(O/'link_registration.json',r);save(O/'link_review.json',dict(status='PASS_ENGINEERING_REVIEW_NOT_SCIENTIFIC_COMPLETION',reviewer='root',registration=pin(O/'link_registration.json'),CPU_tests=15,source_calls_inherited_without_repeat=600,review=['Exact source/order/counter/noise/profile/byte delta;1125 unique new transmissions','Paid68+956 and13+16 fields retained; frozen public catalogue372','Independent RX actual received payload only; privatecache and qualified runtime firewall','All decoded outputs fixed before source audit and scoring','Legacy output denominators retained; source-level paired bootstrap; no selection','Stage exclusive lock and real child wait; no automatic retry or followup']))
 print(json.dumps(pin(O/'link_registration.json')))
if __name__=='__main__':main()
