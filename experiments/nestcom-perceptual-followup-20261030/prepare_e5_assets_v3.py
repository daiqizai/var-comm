"""Explicitly resolve the already authorized classifier-weight relocation."""
from followup_common import *

def main():
 src=L/'E5_plan_runtime_v2';out=L/'E5_plan_assets_v3';out.mkdir(exist_ok=False)
 cfg=read(src/'config.json');old=cfg['ada_native']['classifier']['weight'];assert not Path(old['path']).exists()
 sys.path.insert(0,str(R/'tools'));from checkpoint_relocation_20261028 import resolve,RECEIPT
 moved=pin(resolve(old['path']));assert moved['sha256']==old['sha256']
 cfg['ada_native']['classifier']['weight']=moved
 for name in ['sources.json','expected_TX.json','groups.json']:save(out/name,read(src/name))
 save(out/'config.json',cfg)
 save(out/'relocation_identity.json',dict(original=old,actual=moved,prior_migration_receipt=pin(RECEIPT),same_weight_bytes=True,weight_updates=0,scientific_calls_before_correction=0,old_metadata_untouched=True))
 save(D/'E5_asset_preflight_v1_failure.json',dict(status='PRELAUNCH_MISSING_OLD_WEIGHT_PATH',missing=old,CPU_tests=pin(D/'E5_CPU_checks.json'),model_calls=0,PHY_calls=0,first_timing_launches=0,correction=pin(out/'relocation_identity.json')))
 for name in ['P_checkpoint','Swin_checkpoint','model_routes','phy_qualification','n512_qualification']:checked(cfg[name])
 for desc in cfg['ada_native']['weights'].values():checked(desc)
 for desc in cfg['ada_native']['classifier'].values():
  if isinstance(desc,dict)and'path'in desc:checked(desc)
 for g in read(out/'groups.json'):
  if'catalogue'in g:checked(g['catalogue'])
 save(out/'completion.json',dict(status='PREPARED',groups=16,sources=32,all_asset_hashes_checked=True,CPU_tests=pin(D/'E5_CPU_checks.json'),new_model_calls=0,new_PHY=0))
 print('E5_ASSETS_V3_VERIFIED')
if __name__=='__main__':main()
