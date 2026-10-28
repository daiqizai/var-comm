"""Reuse only closed same-device metric identities; register missing formal pairs."""
from n512_common import *
from pipeline_lineage_v3 import pair_plan,closed
from stage_operations_v3 import budget
from dispatch import launch
import csv,subprocess

def closed_donor(role,reg):
 root=B/'adatoken_response_20261026_v1'/role;d=read(root/'completion.json');assert d['status']=='COMPLETE'
 w=read(root.parent/(role+'_control')/'actual_wait.json');assert w['actual_wait']and w['returncode']==0 and not w['timed_out']and gone(w['process'])and gone(w['parent_process'])
 assert not(root/'failure.json').exists()and not(root.parent/(role+'_control')/'failure.json').exists()
 identity=read(root/'metric_identity.json');rr=reg['resources'][role]
 assert identity['torch']==rr['torch']and identity['pci'].lower()==rr['pci']and identity['fp32']and not identity['tf32']and identity['batch']==1
 assert identity['config']==reg['frozen_metric_registration']and identity['loader']==reg['config']['clip']['loader']
 with sqlite3.connect('file:'+str(root/'ledger.sqlite')+'?mode=ro',uri=True)as c:
  assert c.execute("select count(*) from calls where status!='COMPLETE'").fetchone()[0]==0
  features={k:json.loads(v)for k,v in c.execute("select id,receipt from calls where kind='image_embedding'")}
 inherited=cr(reg['feature_cache'][role]);inherited.update(features)
 return root,d,identity,inherited

def dispatch_clip(role,rp):
 r=read(rp);gpu=1;line=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();v=[x.strip()for x in line.split(',')]
 assert int(v[2])>=60000 and int(v[3])<=25 and v[1].lower().lstrip('0')==r['pci'].lower().lstrip('0')
 assert not(L/'controls'/role).exists();rq=L/'control_requests'/(role+'.json')
 save(rq,dict(control=str(L/'controls'/role),argv=['/usr/bin/python','-B',str(E/'formal_metric_worker.py'),str(rp),sha(rp)],environment={'CUDA_VISIBLE_DEVICES':'1'},seconds=r['max_seconds']+60,code=r['code']+[pin(E/'control.py')]))
 with(L/(role+'_owner.log')).open('xb')as f:p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control.py'),str(rq),sha(rq)],cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
 save(L/(role+'_dispatch.json'),dict(process=ident(p.pid),request=pin(rq),host=os.uname().nodename,GPU_observation=line));return read(L/(role+'_dispatch.json'))

def main():
 assert read(O/'formal_post_output_audit.json')['status']=='PASS';p=pair_plan('formal');plan=cr(p['pairs']);seconds,spent=budget(True,2)
 root=L/'formal_metric_preparation';root.mkdir(exist_ok=False);sources={s['source_id']:s for s in plan['sources']};wanted={x['pair_key']:x for x in plan['pairs']};wanted_pixels={x['pixel_sha256']for x in plan['pairs']}|{s['reference_sha256']for s in sources.values()};regs={}
 for mode in ('three','clip'):
  oldrole=f'c_predclass_common500_{mode}_metrics';oldrp=R/'results/adatoken_response_20261026_v1'/(oldrole+'_registration.json');old=read(oldrp);donor,done,identity,features=closed_donor(oldrole,old)
  features={h:d for h,d in features.items()if h in wanted_pixels};[checked(d)for d in features.values()];save(root/(mode+'_features.json'),features)
  metrics=('psnr_db','lpips_alex','dreamsim')if mode=='three'else('clip_image_cosine',);pairs={}
  with checked(done['scores']).open()as f:
   for row in csv.DictReader(f):
    sid=row['source_id'];s=sources[sid];pk=digest([s['reference_sha256'],row['pixel_sha256']])
    if pk not in wanted:continue
    record=dict(pair_key=pk,source_id=sid,source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=row['pixel_sha256'],origin='EXACT_NATIVE_PAIR_REUSE',**{m:float(row[m])for m in metrics})
    if pk in pairs:assert pairs[pk]==record
    else:pairs[pk]=record
  save(root/(mode+'_pairs.json'),pairs)
  refs=[]
  if mode=='clip':
   for s in plan['sources']:
    h=s['reference_sha256'];assert h in features;refs.append(dict(source_id=s['source_id'],pixel_sha256=h,feature=features[h]))
  save(root/(mode+'_references.json'),refs);role='formal_'+mode+'_metrics';resource=old['resources'][oldrole]
  r=dict(mode=mode,plan=p['pairs'],output=str(L/role),config=old['config'],frozen_metric_registration=old['frozen_metric_registration'],original_four=old['original_four'],prior_clip_identity=old['prior_clip_identity'],clip_cache=pin(root/(mode+'_references.json')),feature_cache=pin(root/(mode+'_features.json')),pair_cache=pin(root/(mode+'_pairs.json')),cache_identity=pin(donor/'metric_identity.json'),cache_registration=pin(oldrp),cache_completion=pin(donor/'completion.json'),torch=resource['torch'],pci=resource['pci'],max_seconds=seconds,GPU_device_seconds_already_spent=spent,caps=dict(embedding=len(plan['pairs'])+500,pair=len(plan['pairs'])),code=[pin(E/z)for z in ['n512_common.py','formal_metric_worker.py','prepare_formal_metrics.py','pipeline_lineage_v3.py','rx_lineage.py']]+old['code'])
  r['caps'].update(dict(load_lpips=1,load_dreamsim=1)if mode=='three'else dict(load_clip=1));rp=O/(role+'_registration.json');save(rp,r);regs[mode]=pin(rp)
 save(root/'completion.json',dict(status='REGISTERED_FORMAL_METRIC_MISSING_PAIRS',registrations=regs,pair_plan=p,old_clip_211_features_not_used=True))
 print(launch('formal_three_metrics',Path(regs['three']['path']),E/'formal_metric_worker.py',gpu=6),flush=True)
 print(dispatch_clip('formal_clip_metrics',Path(regs['clip']['path'])),flush=True)

if __name__=='__main__':main()
