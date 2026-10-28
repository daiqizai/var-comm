"""One-time bounded registration. Does not launch or import models."""
from evidence_common import *
import collections,subprocess,csv
NL=B/'nestcom_n512_20261028_v1';AL=B/'adatoken_response_20261026_v1'
def main():
 prep=L/'input_inventory_v2';assert read(prep/'completion.json')['status']=='INPUT_IDENTITY_COMPLETE'
 jobs=read(prep/'jobs.json');logical=read(prep/'logical.json');groups={}
 for j in jobs:
  identity=cr(j['numeric_identity']);pci=identity.get('pci',identity.get('actual_PCI_bus_id')).lower();g=(j['kind'],pci);groups.setdefault(g,[]).append(j)
 assert len(groups)<=8 and sum(len(x)for x in groups.values())==485
 # Every reused receiver role must have genuine exit0 completion, with no live old PID.
 closures=[]
 for role in sorted({str(Path(j['donor_image']['path']).parent.parent)for j in jobs}):
  p=Path(role);c=p.parent/(p.name+'_control');c=c if c.exists()else p.parent/'controls'/p.name
  w=read(c/'actual_wait.json');assert w['returncode']==0 and w['actual_wait']and not(c/'failure.json').exists()and not(p/'failure.json').exists()
  process=w.get('process',w.get('child'));assert process and gone(process)
  completion=pin(p/'completion.json');closures.append(dict(role=role,completion=completion,actual_wait=pin(c/'actual_wait.json'),process_gone=True))
 save(O/'donor_closure_audit.json',closures)
 protocol=dict(schema='NEST_EVIDENCE_20261029_V1',basis_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),population=pin(O/'source_manifest.json'),cache_identity=pin(O/'cache_identity.json'),E1=dict(workpoints=[[512,4],[512,10],[1024,10]],sources=100,repeats=3,logical_per_arm=900,arms=['Ada-sampling','Ada-argmax','NeST'],only_change='Unreceived tail argmax of original CFG4 logits; fresh KV on received prefix, then greedy context',source_entropy_CDF='unchanged conditional branch',paired=['Ada-argmax minus Ada-sampling','NeST minus Ada-argmax','NeST minus Ada-sampling']),E2=dict(workpoints=[[512,4],[512,10]],sources=100,repeats=3,logical_per_arm=600,arms=['R0-coarse-only','R1-VAR-completed'],only_change='Omit unreceived scale residual contributions; original S10 C_s and 16x16 latent, identical Dc',paired=['R1 minus R0']),statistics=dict(metrics=['psnr_db','lpips_alex','dreamsim','clip_image_cosine'],source_aggregation='Arithmetic mean of three original repetitions, then equal sources',bootstrap='Original 10000x100 bank in source_manifest, full order checked',interval='Direct paired source differences, percentile 2.5/97.5',new_bank_draws=0),budget=dict(max_unique_E1=285,max_unique_E2=200,max_qualification_replays_E1=4,max_qualification_replays_E2=4,max_new_metric_pairs=485,max_metric_embeddings_per_network=485,GPU_device_hours=12,CPU_process_hours=12,receiver_stage_seconds=3600,metric_stage_seconds=3600,new_PHY=0,new_training=0,new_tokenizer=0,new_source_encoding=0),no_automatic_retry=True,qualifications='First bank-ordered accepted input per original physical device for each experiment. Valid new results retained.',figure_selection=pin(O/'figure3_selection.json'))
 save(O/'protocol.json',protocol);registrations=[]
 bank_order=read(O/'source_manifest.json')['source_order'];rank={s:i for i,s in enumerate(bank_order)}
 first_use={}
 for row in sorted(logical,key=lambda x:(rank[x['source_id']],x['N'],x['snr_db'],x['noise_seed'])):
  if 'new_key'in row:first_use.setdefault(row['new_key'],row)
 for gi,((kind,pci),group)in enumerate(sorted(groups.items())):
  group.sort(key=lambda j:(rank[first_use[j['key']]['source_id']],first_use[j['key']]['N'],first_use[j['key']]['snr_db']))
  name=kind+'_'+pci.split(':')[1].lower();out=L/name;assert not out.exists()and not(L/(name+'_control')).exists()
  identity=cr(group[0]['numeric_identity']);assert all(cr(j['numeric_identity'])==identity for j in group)
  if kind=='e1':
   qualifiers=[group[0]['key']]
   base=read(R/'results/nestcom_n512_20261028_v1/formal_rx_ada_00_registration.json')
   config={k:base[k]for k in ('weights','upstream_code')};caps=dict(model_load=1,receiver_render=len(group),qualification_replay=len(set(qualifiers)))
  else:
   qualifiers=[group[0]['key']];base=read(R/'results/nestcom_n512_20261028_v1/formal_rx_nest_00_registration.json');relocated=R/'results/checkpoint_relocation_20261028_v1/model_routes.json';old_routes=cr(base['model_routes']);new_routes=read(relocated);assert old_routes['models']==new_routes['models'] and [d['sha256']for d in old_routes['files']]==[d['sha256']for d in new_routes['files']];[checked(d)for d in new_routes['files']];config=dict(model_routes=pin(relocated),original_model_routes=base['model_routes']);caps=dict(model_load=1,receiver_render=len(group),qualification_replay=1,prior_scale=10,decoder_forward=len(group)+1)
  paths={d['path']:d for d in base['code']}
  for p in E.glob('*.py'):paths[str(p)]=pin(p)
  inp=prep/(name+'_received_only.json');save(inp,group)
  request=dict(kind=kind,count=len(group),inputs=pin(inp),output=str(out),donor_identity=identity,pci=pci,cpu_affinity=[0,1]if kind=='e2'else[16,17,18,19,20,21],qualification_keys=list(dict.fromkeys(qualifiers)),caps=caps,max_seconds=3600,protocol=pin(O/'protocol.json'),code=list(paths.values()),**config)
  path=O/(name+'_registration.json');save(path,request)
  gpu={'61':0,'62':1,'63':2,'64':3,'a1':4,'a2':5,'a3':6,'a4':7}[pci.split(':')[1].lower()]
  ctrl=dict(control=str(L/(name+'_control')),code=[pin(E/'control.py'),pin(E/'evidence_common.py')],argv=[str(RT/'envs/frozen_visual_py310/bin/python'),'-B',str(E/'receiver_worker.py'),str(path),sha(path)],environment=dict(CUDA_VISIBLE_DEVICES=str(gpu)),seconds=3660)
  cp=O/(name+'_control_registration.json');save(cp,ctrl);registrations.append(dict(role=name,request=pin(path),control=pin(cp),gpu=gpu,pci=pci,count=len(group),qualifications=len(set(qualifiers))))
 save(O/'receiver_stage_manifest.json',registrations);print(json.dumps(registrations))
if __name__=='__main__':main()
