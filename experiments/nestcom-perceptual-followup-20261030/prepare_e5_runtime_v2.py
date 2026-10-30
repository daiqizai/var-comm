"""Preserve each original public directory in timing; no model or PHY execution."""
from followup_common import *

def main():
 old=L/'E5_plan';out=L/'E5_plan_runtime_v2';out.mkdir(exist_ok=False)
 cfg=read(old/'config.json');groups=read(old/'groups.json')
 original512=pin(R/'results/nestcom_n512_20261028_v1/public_catalogue.json')
 original1024=pin(R/'results/t1b_m10_calibration_20261016_v1/public_catalogue.json')
 original7=pin(B/'nestcom_7db_20261030_v1/cal_phy_nest/catalogue.json')
 changes=[]
 for g in groups:
  new=None
  if g['method']=='WHOLE':new=original512 if g['N']==512 else original7 if g['snr_db']==7 else original1024
  if g['method']=='Ada'and g['N']==512:new=original512
  if g['method']=='Ada'and g['snr_db']==7:new=original7
  if new is not None:
   changes.append(dict(group=g['name'],before=g['catalogue'],after=new,reason='Retain original paid directory; token extension belongs only to the new grid.'))
   oldprofiles={p['profile_id']:p for p in cr(g['catalogue'])['profiles']}
   assert all(oldprofiles[p['profile_id']]==p for p in cr(new)['profiles'])
   g['catalogue']=new
 cfg['environments']=dict(NeST='frozen_visual_py310',Ada='frozen_visual_py310',ARPC='/usr/bin/python3 with original environment_v1 package route',continuous='frozen_visual_py310',BPG='frozen_ldpc_py311 CPU-only; same host/two-core affinity, no GPU arithmetic')
 cfg['catalogue_scope']='Original WHOLE/Ada directories preserved; token extension only for token and controlled finer grids.'
 for name in ['sources.json','expected_TX.json']:
  save(out/name,read(old/name))
 save(out/'groups.json',groups);save(out/'config.json',cfg)
 save(out/'metadata_correction.json',dict(original_plan=pin(old/'completion.json'),changes=changes,scientific_calls_before_revision=0,training=0,source_bits_changes=0,old_files_untouched=True))
 save(out/'completion.json',dict(status='PREPARED',groups=16,sources=32,new_scientific_calls=0,expected_TX=pin(out/'expected_TX.json')))
 print('E5_RUNTIME_METADATA_V2',len(changes))
if __name__=='__main__':main()
