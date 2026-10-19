"""Closed actual N2048 pixels, frozen seven metrics and one source bootstrap."""
from shared import *
import argparse,csv,gc,importlib.util,math,fcntl
import numpy as np
from analysis_data import METRICS,METHODS,source_means,summarize
MOUT=O/'metrics_v1';MRUN=L/'metrics_v1';MREG=MOUT/'registration.json'

def csvwrite(p,rows):
 assert rows
 with p.open('x',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())

def load_image(pair):
 if pair['image'] is None:y=np.full((3,256,256),.5,np.float32)
 else:
  with np.load(verify(pair['image']),allow_pickle=False)as z:y=z[pair['array_key']].copy()
 assert y.dtype==np.float32 and y.shape==(3,256,256) and np.isfinite(y).all() and y.min()>=0 and y.max()<=1
 assert hashlib.sha256(y.tobytes()).hexdigest()==pair['image_float_sha256'];return y

def prepare(expected):
 r=registration(expected);assert not MOUT.exists() and not MRUN.exists()
 for stage in ['formal_rx','formal_nest']:
  w=read(L/'controls'/stage/'actual_wait.json');assert type(w['returncode'])is int and w['returncode']==0 and gone(w['process'])
 assert read(O/'formal/arpc_process_closure.json')['status']=='GONE'
 arpauth=read(O/'formal_arpc_resource_authorization.json')
 for k in ['source','registration','public','jobs']:verify(arpauth[k])
 nc=read(L/'formal/nest/rx/completion.json');ac=read(L/'formal/arpc_resource_v1/rx/completion.json')
 assert nc['ledger']['unresolved']==0 and not ac['blocked_reads']
 nd={x['source_input_key']:x for x in nc['outputs']};ad={x['key']:x for x in ac['outputs']}
 frames=read(O/'formal/logical_frames.json');jobs={x['key']:x for x in r['formal_jobs']};groups={};logical=[]
 gray=hashlib.sha256(np.full((3,256,256),.5,np.float32).tobytes()).hexdigest()
 for x in frames:
  j=jobs[x['key']];sid=x['source_id'];codec=x['codec'];image=None;arraykey=None;h=gray;status=x['status']
  if x['accepted']:
   out=(nd if codec==0 else ad)[x['source_input_key']]
   if codec==0:
    assert not out['source_truth_used'] and not out['TX_CDF_used'];image=out['image'];arraykey='image';h=out['float_sha256'];status=out['status']
   elif out.get('status')!='SOURCE_PARSE_FAIL':image=pin(out['path']);arraykey='float_output';h=out['float_sha256'];status='SOURCE_DECODED'
   else:status='SOURCE_PARSE_FAIL'
  frame=dict(key=x['key'],method=METHODS[codec],source_id=sid,source_index=j['source_index'],snr_db=j['snr_db'],noise_seed=j['noise_seed'],pair_status=x['status'],receiver_status=status,accepted_wrong_post_exit=x['accepted_wrong'])
  logical.append(dict(**frame,image=image,array_key=arraykey,image_float_sha256=h))
  if sid not in groups:groups[sid]=dict(source_id=sid,source_index=j['source_index'],rank=j['rank'],archive=j['asset']['source_archive'],pairs={})
  g=groups[sid];assert g['archive']==j['asset']['source_archive']
  if h not in g['pairs']:g['pairs'][h]=dict(image=image,array_key=arraykey,image_float_sha256=h,frames=[])
  g['pairs'][h]['frames'].append(frame)
 plan=[]
 for s in sorted(groups.values(),key=lambda x:x['rank']):
  verify(s['archive']);s['pairs']=list(s['pairs'].values());assert sum(len(p['frames'])for p in s['pairs'])==18
  for pair in s['pairs']:load_image(pair)
  plan.append(s)
 assert len(plan)==100 and len(logical)==1800
 # Read only the completed prior seven-metric pair cache; never reuse source means.
 oldreg=OLD_O/'a3_metrics_v1/registration.json';pr=read(oldreg)
 for d in pr['source_files']:verify(d)
 olddone=read(OLD_O/'a3_metrics_v1/score_completion.json');assert olddone['unresolved']==0
 for d in olddone['outputs']:verify(d)
 cache={}
 with (OLD_O/'a3_metrics_v1/actual_pair_scores.csv').open()as f:
  for row in csv.DictReader(f):
   key=(row['source_id'],row['image_float_sha256']);assert key not in cache;cache[key]=row
 oldplan=read(OLD_O/'a3_metrics_v1/pair_plan.json')['sources'];oldsrc={s['source_id']:s for s in oldplan};reuse=0
 for s in plan:
  assert s['archive']==oldsrc[s['source_id']]['archive']
  for p in s['pairs']:
   row=cache.get((s['source_id'],p['image_float_sha256']))
   if row:
    op=next(q for q in oldsrc[s['source_id']]['pairs']if q['image_float_sha256']==p['image_float_sha256'])
    if op['image']:
     with np.load(verify(op['image']),allow_pickle=False)as z:oldim=z['float_output'].copy()
    else:oldim=np.full((3,256,256),.5,np.float32)
    assert np.array_equal(load_image(p),oldim)
    p['reuse']=dict(scores={k:float(row[k])for k in METRICS},old_pair_key=row['pair_key'],old_image=op['image']);reuse+=1
 total=sum(len(s['pairs'])for s in plan);assert total<=1800
 MOUT.mkdir();MRUN.mkdir();put(MOUT/'actual_logical_pixels.json',logical);put(MOUT/'pair_plan.json',dict(sources=plan))
 supplemental=read(pr['supplemental_registration']['path'])
 for k in ['loader','metric_assets','dependency_overlay']:verify(supplemental[k])
 inputs=[pin(REG),pin(O/'qualification_review.json'),pin(O/'formal/phy_completion.json'),pin(O/'formal/arpc_process_closure.json'),pin(O/'formal_arpc_resource_authorization.json'),pin(L/'formal/nest/rx/completion.json'),pin(L/'formal/arpc_resource_v1/rx/completion.json'),pin(MOUT/'pair_plan.json'),pin(oldreg),pin(OLD_O/'a3_metrics_v1/score_completion.json'),pin(OLD_O/'a3_metrics_v1/actual_pair_scores.csv'),pin(OLD_O/'a3_metrics_v1/pair_plan.json'),pin(OLD_O/'a3_metrics_v1/four_metric_identity.json'),pin(OLD_O/'a3_metrics_v1/supplemental_identity.json')]
 put(MREG,dict(schema='N2048_TWO_METHOD_SEVEN_METRICS_V1',inputs=inputs,code=[pin(E/n)for n in ['metrics.py','analysis_data.py','test_analysis.py','shared.py']],original_metric_sources=pr['source_files'],original_four_metrics=pr['original_four_metrics'],metric_adapter=pin(OLD_E/'a3_metrics_v1/metric_adapter.py'),metric_config=pr['metric_config'],supplemental_registration=pr['supplemental_registration'],supplemental_overlay_root=pr['supplemental_overlay_root'],resource=pr['resource'],source_count=100,logical_frames=1800,unique_pairs=total,exact_prior_seven_metric_pairs=reuse,new_seven_metric_pairs=total-reuse,statistics=r['statistics'],stage_seconds=dict(score=3600,statistics=300),no_policy_selection=True,training=0,TX_probability=0))
 print(json.dumps(dict(registration=pin(MREG),pairs=total,reused=reuse,new=total-reuse)))

class ScoreLedger:
 def __init__(self,guard,pairs):
  self.guard=guard;self.counts={};caps=dict(model_constructions=3,reference_preparations=100,image_scores=pairs,dinov2_vitl14_reference=100,dinov2_vitl14_reconstruction=pairs,convnext_reference=100,convnext_reconstruction=pairs,lpips_pair=pairs,lpips_alexnet_backbone_forward=2*pairs,load_dists=1,load_dreamsim=1,load_ms_ssim=1,dreamsim_reference=100,dists_pair=pairs,dreamsim_reconstruction=pairs,ms_ssim_pair=pairs)
  self.raw=Ledger(MRUN/'calls.sqlite',caps)
 def call(self,kind,fn,**metadata):
  self.guard();n=self.counts.get(kind,0)+1;self.counts[kind]=n
  return self.raw.call(kind,str(n),fn)

def score(expected):
 assert sha(MREG)==expected;reg=read(MREG)
 for d in reg['inputs']+reg['code']+reg['original_metric_sources']:verify(d)
 cache(MRUN)
 import ctypes,torch
 sys.path.insert(0,str(OLD_E/'cuda_recovery_v1'));from runtime import check_mapping
 mapping=check_mapping(ctypes.CDLL('libcuda.so.1'),reg['resource']['cuda_admission'],os.environ.get('CUDA_VISIBLE_DEVICES'));put(MRUN/'actual_cuda_mapping.json',mapping)
 assert torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8';torch.cuda.set_device(0)
 assert torch.cuda.mem_get_info()[0]>=8*1024**3;torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
 start=time.time()
 def guard():assert time.time()-start<3500 and torch.cuda.mem_get_info()[0]>=2*1024**3 and not (MRUN/'STOP').exists()
 plan=read(MOUT/'pair_plan.json')['sources'];npairs=reg['new_seven_metric_pairs'];ledger=ScoreLedger(guard,npairs)
 def reference(s):
  with np.load(verify(s['archive']),allow_pickle=False)as z:x=z['pixels'].copy()
  assert x.dtype==np.uint8 and x.shape==(3,256,256);return x.astype(np.float32)/np.float32(255)
 sys.path.insert(0,str(Path(reg['original_four_metrics']['path']).parent));sys.path.insert(0,str(OLD_E/'a3_metrics_v1'))
 from metric_adapter import load
 metric=load(verify(reg['original_four_metrics']));ev=metric.FourMetrics(reg['metric_config'],ledger,guard)
 # Preserve the historical metric metadata identity solely to validate cached scores.
 ev.metadata.update(population='external_common100_v1',selection_objective=None,policy_selection=False,ARPC_source_native_float=True,metric_adapter=reg['metric_adapter']);ev.identity=metric.core.digest(ev.metadata)
 priorid=read(OLD_O/'a3_metrics_v1/four_metric_identity.json');assert ev.identity==priorid['identity']
 put(MOUT/'four_metric_identity.json',dict(identity=ev.identity,metadata=ev.metadata,legacy_ARPC_float_label_applies_to_both_actual_native_float_arrays=True))
 pairsdir=MRUN/'pairs';pairsdir.mkdir();values={}
 for s in plan:
  x=reference(s);new=[p for p in s['pairs']if 'reuse'not in p]
  prepared=ev.prepare(x)if new else None
  for p in s['pairs']:
   key=metric.pair_cache_key(x,load_image(p),ev.identity);p['score_key']=key;assert key not in values
   v=({k:p['reuse']['scores'][k]for k in METRICS[:4]} if 'reuse'in p else ev.score(x,load_image(p),0,prepared));values[key]=v
   put(pairsdir/(key+'_four.json'),dict(scores=v,source=s['archive'],image=p['image'],image_float_sha256=p['image_float_sha256'],reuse=p.get('reuse'),metric_identity=ev.identity))
  print(json.dumps(dict(stage='FOUR',source=s['rank'])),flush=True)
 del ev,prepared;gc.collect();torch.cuda.empty_cache()
 supplemental=read(verify(reg['supplemental_registration']));assets=read(verify(supplemental['metric_assets']));overlay=read(verify(supplemental['dependency_overlay']))
 for d in overlay['files']:verify(d)
 sys.path.insert(0,reg['supplemental_overlay_root']);spec=importlib.util.spec_from_file_location('_n2048_frozen_supplement',verify(supplemental['loader']));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 torch.manual_seed(2026101401);models={}
 with m.no_network():
  for name,loader in [('dists',m.load_dists),('dreamsim',m.load_dreamsim),('ms_ssim',m.load_ms_ssim)]:models[name]=ledger.call('load_'+name,lambda name=name,loader=loader:loader(assets['models'][name],torch.device('cuda:0')))
 oldsupp=read(OLD_O/'a3_metrics_v1/supplemental_identity.json');assert oldsupp['protocol']==supplemental['protocol'] and oldsupp['model_metadata']==assets['models'] and oldsupp['versions']==m.package_versions() and oldsupp['FP32'] and not oldsupp['TF32']
 put(MOUT/'supplemental_identity.json',dict(original_registration=reg['supplemental_registration'],model_metadata=assets['models'],protocol=supplemental['protocol'],versions=m.package_versions(),FP32=True,TF32=False,seed=2026101401,mapping=mapping,reused_numerical_identity=pin(OLD_O/'a3_metrics_v1/supplemental_identity.json')))
 logical=[];pairrows=[]
 with torch.inference_mode():
  for s in plan:
   x=reference(s);xt=torch.from_numpy(x[None]).cuda();new=[p for p in s['pairs']if 'reuse'not in p]
   feat=ledger.call('dreamsim_reference',lambda:models['dreamsim'].embed(m.preprocess_dreamsim(xt)).float().detach().cpu())if new else None
   for p in s['pairs']:
    key=p['score_key']
    if 'reuse'in p:v={k:p['reuse']['scores'][k]for k in METRICS[4:]}
    else:
     yt=torch.from_numpy(load_image(p)[None]).cuda()
     dv=ledger.call('dists_pair',lambda:models['dists'](xt,yt,require_grad=False,batch_average=False).reshape(-1)[0].item())
     fy=ledger.call('dreamsim_reconstruction',lambda:models['dreamsim'].embed(m.preprocess_dreamsim(yt)).float())
     ds=(1-torch.nn.functional.cosine_similarity(feat.cuda(),fy,dim=-1))[0].item()
     ms=ledger.call('ms_ssim_pair',lambda:models['ms_ssim'](xt,yt,data_range=1.0,size_average=False,win_size=11,win_sigma=1.5,weights=supplemental['protocol']['ms_ssim']['weights'],K=(.01,.03))[0].item());v=dict(dists=dv,dreamsim=ds,ms_ssim=ms)
    allv={**values[key],**v};assert set(allv)==set(METRICS) and all(math.isfinite(x)for x in allv.values())
    put(pairsdir/(key+'_supplement.json'),dict(scores=v,reused='reuse'in p,identity=pin(MOUT/'supplemental_identity.json')))
    pairrows.append(dict(source_id=s['source_id'],source_index=s['source_index'],pair_key=key,image_float_sha256=p['image_float_sha256'],reused='reuse'in p,**allv));logical.extend([dict(**f,pair_key=key,**allv)for f in p['frames']])
   print(json.dumps(dict(stage='SUPPLEMENT',source=s['rank'])),flush=True)
 assert len(logical)==1800 and len(pairrows)==reg['unique_pairs'];assert ledger.raw.summary()['unresolved']==0
 for k in ['image_scores','dists_pair','dreamsim_reconstruction','ms_ssim_pair']:assert ledger.counts.get(k,0)==npairs
 csvwrite(MOUT/'logical_scores.csv',logical);csvwrite(MOUT/'actual_pair_scores.csv',pairrows)
 put(MOUT/'score_completion.json',dict(status='N2048_SEVEN_METRICS_COMPLETE',logical_frames=1800,source_count=100,unique_pairs=len(pairrows),new_pairs=npairs,reused_pairs=reg['exact_prior_seven_metric_pairs'],ledger=ledger.raw.summary(),bootstrap=0,outputs=[pin(MOUT/n)for n in ['logical_scores.csv','actual_pair_scores.csv','four_metric_identity.json','supplemental_identity.json']]))

def statistics(expected):
 assert sha(MREG)==expected;reg=read(MREG);done=read(MOUT/'score_completion.json');assert done['ledger']['unresolved']==0
 for d in done['outputs']:verify(d)
 with (MOUT/'logical_scores.csv').open()as f:logical=list(csv.DictReader(f))
 ids=[s['source_id']for s in read(MOUT/'pair_plan.json')['sources']];rows=source_means(logical,ids);assert len(ids)==100
 b=MRUN/'bootstrap';b.mkdir();put(b/'reserved.json',dict(seed=2026101901,replicates=10000,source_ids=ids,draw_calls=1))
 draws=np.random.default_rng(2026101901).integers(0,100,size=(10000,100),dtype=np.int64);np.save(b/'source_indices.npy',draws,allow_pickle=False);put(b/'complete.json',dict(indices=pin(b/'source_indices.npy'),draw_calls=1))
 summary,paired=summarize(rows,ids,draws)
 for n,a in [('source_noise_means.csv',rows),('summary.csv',summary),('paired_differences.csv',paired)]:csvwrite(MOUT/n,a)
 put(MOUT/'completion.json',dict(status='N2048_TWO_METHOD_COMPARISON_COMPLETE',methods=2,sources=100,logical_frames=1800,metrics=list(METRICS),bootstrap=dict(seed=2026101901,replicates=10000,draw_calls=1,indices=pin(b/'source_indices.npy'),unit='source after mean over three noises',interval='pointwise percentile95; direct NeST-minus-ARPC differences; no multiplicity adjustment'),outputs=[pin(MOUT/n)for n in ['score_completion.json','source_noise_means.csv','summary.csv','paired_differences.csv']]))

def run(role,expected):
 assert sha(MREG)==expected;r=read(MREG)
 for d in r['inputs']+r['code']:verify(d)
 root=L/'controls'/('metrics_'+role);assert not root.exists();lock=(L/'execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if role=='statistics':
  w=read(L/'controls/metrics_score/actual_wait.json');assert type(w['returncode'])is int and w['returncode']==0 and gone(w['process'])
 root.mkdir();put(root/'supervisor.json',dict(process=identity(os.getpid()),parent_process=identity(os.getppid()),argv=sys.argv,registration=pin(MREG)))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'if role=='score'else'',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6');env.pop('PYTHONPATH',None)
 try:wait_child(root,[str(PY),'-B',str(E/'metrics.py'),'_'+role,'--expected',expected],env,r['stage_seconds'][role]);put(root/'completion.json',dict(status='COMPLETE',wait=pin(root/'actual_wait.json')))
 except BaseException:put(root/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('role',choices=['prepare','score','statistics','_score','_statistics']);a.add_argument('--expected',required=True);s=a.parse_args()
 if s.role=='prepare':prepare(s.expected)
 elif s.role.startswith('_'):globals()[s.role[1:]](s.expected)
 else:run(s.role,s.expected)
