"""Registration only, consumes cached complete objects; no source/model calls."""
from shared import *
import pair_protocol as p
def main():
 assert not REG.exists() and not L.exists() and not O.exists()
 head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip();assert head=='5f9baf7c7258168e8aef56e5ec4d7ee0406c3ffa'
 prior=R/'results/arpc_n2048_feasibility_20261019_v1'
 assets=read(E/'inputs/asset_audit.json');sources=read(E/'inputs/arpc_sources.json');lookup={(x['cohort'],x['method'],x['source_id']):x for x in assets['assets']}
 qual=[];formal=[]
 for cohort,rows in [('cal100',sources['cal100'][:6]),('external_common100',sources['external_common100_v1'])]:
  for rank,s in enumerate(rows):
   for codec,method in enumerate(['NeST-Com','ARPC']):
    asset=lookup[cohort,method,s['source_id']];verify(asset['wire']);verify(asset['source_archive'])
    cases=[('clean',19,0),('corrupt',19,0)] if cohort=='cal100' else [('awgn',snr,seed) for snr in [4,10,19] for seed in [6201,6202,6203]]
    dest=qual if cohort=='cal100' else formal
    for case,snr,seed in cases:
     i=len(dest);pairid=i if cohort=='cal100' else 10000+i
     dest.append(dict(key=('q' if cohort=='cal100' else 'f')+f'{i:04d}',pair_id=pairid,codec=codec,source_id=s['source_id'],source_index=s['source_index'],rank=rank,snr_db=snr,noise_seed=seed,case=case,asset=asset))
 assert len(qual)==24 and len(formal)==1800
 clean=read(OLD_O/'a2_clean_v1/registration.json');tc=read(R/'results/t1c_m10_test_20261016_v1/link_registration.json')
 code={}
 for d in clean['source_files']+tc['code']:
  path=Path(d['path'])
  if path.suffix in ['.py','.json','.md'] and path.is_file():verify(d);code[str(path)]=pin(path)
 for root in [E,OLD_E/'phy_v1',OLD_E/'a3_rx_v1',R/'experiments/prior-aware-grouped-mcs-v1',R/'experiments/main-raw64-20261007']:
  for f in root.glob('*.py'):code[str(f)]=pin(f)
 # Explicit bindings for unchanged PHY primitives and native receiver helpers.
 for f in [R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py',R/'experiments/t1c-m10-test-20261016/tc_receiver.py',R/'experiments/t1b-m10-calibration-20261016/tb_native.py',OLD_E/'a1_native.py'] :code[str(f)]=pin(f)
 catalogue=read(E/'inputs/existing_arpc_catalogue.json')
 O.mkdir();L.mkdir();(L/'qualification').mkdir();(L/'controls').mkdir()
 r=dict(schema='N2048_EQUAL_COMPLETE_REPRESENTATIONS_V1',basis_HEAD=head,created_unix=time.time(),user_authorization='2026-10-19 latest human: 启动吧; execute the delivered finite N2048 plan, qualification before formal grid',code=list(code.values()),weights=clean['weights'],model_routes=tc['model_routes'],static=tc['static'],qualification_jobs=qual,formal_jobs=formal,backend_identity=catalogue['backend_identity'],proposal=pin(prior/'proposed_scope.json'),asset_audit=pin(E/'inputs/asset_audit.json'),sources=pin(E/'inputs/arpc_sources.json'),caps=dict(qualification_objects=24,qualification_subframes=48,formal_objects=1800,formal_subframes=3600,total_subframes=3648,total_header_body_decodes=7296,source_RX=1812,TX_probability=0,tokenizer=0,training=0),noise=dict(namespace=p.SESSION,seeds=[6201,6202,6203],shape=[2048,2],same_standard_noise_across_methods=True,distinct_contiguous_slices=True),profile_catalogue=p.profiles(),failure_policy='Unknown failure preserved; no retry or completed-call replay; protocol/source rejects gray, accepted wrong outputs retained',resources=dict(nest_host='DevMachineModel-H100-4C-P1660-01',nest_gpu=0,nest_start_free_mib=15360,nest_run_free_mib=2048,arpc_host='leo-tactile',arpc_gpu=4,arpc_start_free_mib=65536,arpc_run_free_mib=8192),stage_seconds=dict(qualification_tx=600,qualification_rx=600,qualification_audit=300,formal_tx=3600,formal_rx=7200,formal_audit=600,qualification_nest=1800,qualification_arpc=1800,formal_nest=7200,formal_arpc=7200),statistics=dict(replicates=10000,seed=2026101901,source_mean_over_three_noises=True,direct_paired=True,once=True),backup='DEFERRED',old_N1024_untouched=True)
 put(REG,r);print(json.dumps(pin(REG)))
if __name__=='__main__':main()
