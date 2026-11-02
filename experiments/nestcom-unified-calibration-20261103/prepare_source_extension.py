from study_common import *
from launch_extra import launch_extra

def main():
 cal=read(B/'nestcom_n512_20261028_v1/qualified_sources/cal1000.json');assert len(cal)==1000
 sm={s['source_id']:s for s in cal};reuse={};lineage=[]
 sys.path.insert(0,str(R/'experiments/nestcom-perceptual-followup-20261030'))
 from prepare_e2_receivers import wait_for
 for role in ['fit_D_source','confirm_D_source']:
  wait=wait_for(OL,role);assert wait is not None
  rows=read(OL/role/'outputs.json');used=0
  for v in rows:
   if v['source_id']not in sm:continue
   s=sm[v['source_id']];assert s['preprocessing_id']==v['source_pixels_sha256'] and v['original_all25_exact']
   checked(v['streams']);assert len(v['lengths'])==651
   if v['source_id']in reuse:assert reuse[v['source_id']]['array_sha256']==v['array_sha256']
   else:reuse[v['source_id']]=v;used+=1
  lineage.append(dict(role=role,reused=used,wait=wait,index=pin(OL/role/'outputs.json'),identity=pin(OL/role/'actual_native_identity.json')))
 assert len(reuse)==200
 out=L/'source_plan';out.mkdir(exist_ok=False);missing=[s for s in cal if s['source_id']not in reuse];assert len(missing)==800
 save(out/'reuse.json',reuse);save(out/'missing.json',missing);save(out/'completion.json',dict(status='COMPLETE_PLAN',total=1000,reuse=200,missing=800,lineage=lineage))
 r=read(OLD/'final_D_source_registration.json');r={k:r[k]for k in ['code','model_routes','catalogues','route']};r.update(sources=pin(out/'missing.json'),cpu_affinity=[36,38],new_tokenizer=0,new_training=0)
 launch_extra('cal_token_source800',r,'source_extension.py',VPY,3600,gpu=3)
if __name__=='__main__':main()
