"""Prepare the explicitly authorized raw64 CPU phase from actual pinned assets."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/home/liulu/projects/VAR_COMM');N=R/'outputs/MAIN-RAW64-20261007';O=R/'outputs/CONTENT-REAL-64QAM-20261006';P=N/'prepared_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def verify(d):
 for p,h in d.items():assert sha(p)==h,(p,h)
def historical_admission(plan):
 evidence={}; phases=[]
 for name,obsfile in [('qualification','MAIN_QUALIFICATION_NORMAL_CLOSED_OBSERVATION.json'),('proxy','MAIN_PROXY_NORMAL_CLOSED_OBSERVATION.json')]:
  ep=O/'MAIN'/f'{name}_execution_r2';sp=O/'MAIN'/f'{name}_r2'/'completion.json';ob=O/'prepared_v1'/obsfile;obs=read(ob)
  assert obs['processes_exited'] is True
  if name=='qualification':
   assert obs['original_closed_phase_validated'] and sha(ep/'completion.json')==obs['owner_completion_sha256'] and sha(sp)==obs['science_completion_sha256']
  else:
   assert obs['original_owner_success'] and obs['original_closed_phase_passed']
   for k in ['owner_completion','science_completion','registration']:assert sha(obs[k]['path'])==obs[k]['sha256']
  owner=read(ep/'completion.json');science=read(sp)
  assert 'COMPLETE' in owner['status'] and 'COMPLETE' in science['status']
  verify(owner['outputs']);verify(science['outputs'])
  for p in [ep/'completion.json',sp,ob]:evidence[str(p)]=sha(p)
  launches=list(ep.rglob('launch.json'))
  for p in launches:
   x=read(p);ident=x.get('identity',x);pid=ident.get('pid')
   if pid:
    proc=Path('/proc')/str(pid)
    if proc.exists():
     s=(proc/'stat').read_text();start=int(s[s.rfind(')')+2:].split()[19]);assert start!=ident.get('start_ticks'),('Old admitted process still present',pid)
   evidence[str(p)]=sha(p)
  phases.append({'phase':name,'science':str(sp),'owner':str(ep/'completion.json'),'outputs_verified':len(owner['outputs'])+len(science['outputs'])})
 table=O/'MAIN/proxy_r2/points.json';assert read(O/'MAIN/proxy_r2/completion.json')['outputs'][str(table)]==sha(table)
 save(P/'cpu_reuse_admission_v1.json',{'status':'MAIN_RAW64_REUSE_ADMITTED','normal_owners_verified':True,'source_images_read':False,'plan_sha256':sha(plan),'proofs':phases,'bindings':evidence,'historical_maps_reverified':True,'old_ledger_opened':False})
 return evidence
def main():
 sys.path[:0]=[str(P/'runner'),str(P/'phy')]
 import main_raw64_cpu_runner as runner
 old=O/'MAIN/plan_only_v1/result'
 materials={'schedule':str(N/'PHY_SCHEDULE_V1.json'),'catalogue':str(N/'assemble_v1/completion.json'),'science_registration':str(N/'SCIENCE_REGISTRATION_V1.json'),'legacy_bler':str(R/'outputs/PRIOR-AWARE-UEP-20261004-V1/bler_refined_02.json'),'legacy_MAIN_proxy':str(O/'MAIN/proxy_r2/points.json')}
 materials['bindings']={v:sha(v) for v in materials.values()};save(P/'cpu_plan_materials.json',materials)
 plan=P/'cpu_plan_v1.json';runner.build_plan(P/'cpu_plan_materials.json',plan)
 history=historical_admission(plan)
 r=read(P/'runner/cpu_request.template.json');r.pop('_template_only',None);r['deadline_unix']=1791651239.4938014;r['plan']=str(plan);r['adapter_config']['catalogue']=str(N/'assemble_v1/completion.json')
 assert set(sum(r['resources']['affinities'],[]))<=os.sched_getaffinity(0)
 source_paths=[str(p) for folder in ['runner','phy'] for p in (P/folder).glob('*.py')]
 ip=[str(N/'prepare_cpu.py'),str(P/'cpu_cpu_qualification.json'),*history]
 requestmaterials={'request':r,'source_paths':source_paths,'input_paths':ip};save(P/'cpu_request_materials.json',requestmaterials)
 request=P/'cpu_request_v1.json';runner.build_request(P/'cpu_request_materials.json',request)
 reg=runner.register(request);save(P/'cpu_preparation_completion.json',{'status':'CPU_PREPARED_REGISTERED_NOT_STARTED','registration':reg,'registration_sha256':sha(reg),'plan_sha256':sha(plan),'new_decoder_calls':0})
 print(reg,flush=True)
if __name__=='__main__':main()
