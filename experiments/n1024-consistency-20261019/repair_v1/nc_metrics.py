"""Score only genuinely new fixed pixel pairs; keep both full test denominators."""
from nc_common import *
import numpy as np,collections,traceback,subprocess,importlib.util,gc,math
from nc_protocol import closed_wait
FOUR=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction')
class AdapterLedger:
 def __init__(self,l):self.l=l;self.count=collections.Counter()
 def call(self,kind,fn,**metadata):
  self.count[kind]+=1;return self.l.call(kind,self.count[kind],fn)
def run(expected):
 r=registration('link',expected);out=T/'metrics';out.mkdir();runtime_cache(out);os.sched_setaffinity(0,{0,1});start=time.monotonic();ledger=None;last=[0.]
 try:
  w=read(T/'receiver_control/actual_wait.json');closed_wait(w,gone)
  vd=read(T/'receiver/completion.json');pd=read(T/'phy/completion.json');assert vd['registration_sha256']==pd['registration_sha256']==expected
  images=checkread(vd['images']);logical=checkread(pd['logical']);sources=checkread(r['sources']);reg=checkread(r['supplemental_registration'])
  cap=r['caps']['physical'];caps=dict(model_constructions=3,reference_preparations=500,image_scores=cap,dinov2_vitl14_reference=500,dinov2_vitl14_reconstruction=cap,convnext_reference=500,convnext_reconstruction=cap,lpips_pair=cap,lpips_alexnet_backbone_forward=2*cap,load_dists=1,load_dreamsim=1,load_ms_ssim=1,dreamsim_reference=500,dists_pair=cap,dreamsim_reconstruction=cap,ms_ssim_pair=cap)
  ledger=Ledger(out/'calls.sqlite',caps);counted=AdapterLedger(ledger)
  def guard():
   assert time.monotonic()-start<r['max_seconds'] and not(T/'STOP').exists()
   if time.monotonic()-last[0]>10:
    assert int(subprocess.check_output(['nvidia-smi','-i',str(r['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>=2048;last[0]=time.monotonic()
  def loadref(j):
   d=sources[j];checked(d['archive'])
   with np.load(d['archive']['path'],allow_pickle=False) as z:u=z['pixels'].copy()
   assert u.dtype==np.uint8 and hashlib.sha256(u.tobytes()).hexdigest()==d['preprocessing_id'];return u.astype(np.float32)/np.float32(255)
  def loadimage(rec):
   checked(rec['image_archive'])
   with np.load(rec['image_archive']['path'],allow_pickle=False) as z:a=z[rec['image_key']].copy()
   if rec['image_slot'] is not None:a=a[int(rec['image_slot'])]
   assert a.dtype==np.float32 and a.shape==(3,256,256) and asha(a)==rec['image_sha256'];return a
  oldpairs={};ambiguous=set()
  for item in logical:
   e=item['event'];old=e['legacy'];key=canonical([e['source_global_index'],old['image']['image_sha256']]);v=old['metrics']
   if key in oldpairs and oldpairs[key]['metrics']!=v:ambiguous.add(key)
   else:oldpairs[key]=dict(metrics=v,evidence=old['image']['old_row'])
  for prior in checkread(r['reused_frames']):
   key=canonical([prior['source_index'],prior['reconstruction']['image_sha256']]);v=prior['metrics']
   if key in oldpairs and oldpairs[key]['metrics']!=v:ambiguous.add(key)
   else:oldpairs[key]=dict(metrics=v,evidence=prior['physical_frame'])
  for key in ambiguous:oldpairs.pop(key,None)
  cache={};needs={};resultrows=[];audits=[];audit_seen=set()
  for item in logical:
   e=item['event'];j=e['source_global_index'];legacy=e['legacy'];ref=e['execution_reference'];rec=ref['image'] if e['reuse'] else (item['cached_reconstruction'] or images[item['source_input_key']]);key=canonical([j,rec['image_sha256']])
   if e['reuse']:
    value=dict(metrics=ref['metrics'],origin='UNCHANGED_ORIGINAL_LOGICAL_PAIR' if e['original_logical_pair_unchanged'] else 'EXACT_OTHER_FAMILY_CLOSED_EVENT_AND_PIXEL_PAIR',evidence=ref['image']['old_row'])
   elif key in oldpairs:value=dict(oldpairs[key],origin='EXACT_OLD_FIXED_PIXEL_PAIR')
   else:value=None;needs.setdefault(key,dict(source_global_index=j,reconstruction=rec,population=e['population']))
   if not e['reuse'] and item['source_input_key'] not in audit_seen:
    audit_seen.add(item['source_input_key']);ep=checkread(rec['evidence']) if 'path' in rec['evidence'] else dict(evidence=rec['evidence']);pkt=checkread(item['physical_frame']);rx=pkt['actual_RX'];ee=ep['evidence'];b=rx['body'];p=rx['rx_profile'];a=dict(source_input_key=item['source_input_key'],fixed_output=rec['evidence'],truth_entered_RX=False)
    if b and b['crc_accepted'] and b['parser_accepted'] and p['m']==e['m'] and p['K']==e['K'] and asha(np.array(b['payload'],dtype=np.uint8))==pkt['payload_sha256']:
     checked(sources[j]['archive'])
     with np.load(sources[j]['archive']['path'],allow_pickle=False) as z:truth=z['tokens'][:OFFSETS[p['m']]+p['K']]
     assert ee['source_status']=='ARITHMETIC_SOURCE_DECODED' and np.array_equal(truth,np.array(ee['received_tokens'],dtype=np.int64)),('BLOCKED_ACTUAL_RX_CDF_MISMATCH',j)
     a['exact_stream_and_prefix_after_fixed_output']=True
    else:a['exact_stream_and_prefix_after_fixed_output']=False
    audits.append(a)
   resultrows.append(dict(event=e,physical_frame=item['physical_frame'],reconstruction=rec,pair_key=key,value=value))
  save(out/'post_output_source_audit.json',audits);save(out/'missing_pairs.json',needs)
  import torch
  import platform
  observed=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
  assert observed=={k:v for k,v in r['metric_config']['runtime_identity']['expected_runtime'].items() if k not in ('threads','interop_threads')}
  from metric_device_recovery_v2 import cuda_uuid_equal
  assert torch.cuda.device_count()==1 and cuda_uuid_equal(str(torch.cuda.get_device_properties(0).uuid),r['gpu_cuda_uuid']);torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
  import h800_ep_pilot_metrics_v1 as metric
  config=dict(r['metric_config']);config['runtime_identity']=dict(config['runtime_identity'],N1024_repair_population='common500_only',physical_gpu=r['gpu_index'],policy_selection=False)
  ev=metric.FourMetrics(config,counted,guard);ev.metadata.update(population='N1024_repair_common500_only',selection_objective=None,policy_selection=False);ev.identity=digest(ev.metadata)
  save(out/'four_metric_identity.json',dict(identity=ev.identity,metadata=ev.metadata));(out/'pairs').mkdir();refs={}
  for key,item in needs.items():
   guard();j=item['source_global_index'];x=loadref(j);y=loadimage(item['reconstruction'])
   if j not in refs:refs[j]=ev.prepare(x)
   v=ev.score(x,y,sources[j]['evaluation_class_index'],refs[j]);cache[key]=dict(metrics=v,origin='ACTUAL_NEW_FIXED_PIXEL_PAIR',source_global_index=j,reconstruction=item['reconstruction'])
   p=out/'pairs'/f'{len(cache)-1:04d}_four.json';save(p,dict(pair_key=key,metrics=v,metric_identity=ev.identity,source=sources[j]['archive'],reconstruction=item['reconstruction']));cache[key]['four_evidence']=pin(p)
   if len(cache)%25==0:progress(out,dict(four_scored=len(cache),total=len(needs)))
  del ev,refs;gc.collect();torch.cuda.empty_cache()
  # Only common500 already has these three supplementary metrics. new100 keeps its original four.
  extra={k:v for k,v in needs.items() if v['population']=='common500'}
  if extra:
   assets=checkread(reg['metric_assets']);overlay=checkread(reg['dependency_overlay']);checked(reg['loader'])
   for p in overlay['files']:checked(p)
   sys.path.insert(0,str(B/'quality_metrics_supplement_20261014_v1/dependency_overlay_v1/sitepackages'))
   sp=importlib.util.spec_from_file_location('_t1c_frozen_official_loader',reg['loader']['path']);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
   torch.manual_seed(2026101401);models={}
   for name,loader in [('dists',m.load_dists),('dreamsim',m.load_dreamsim),('ms_ssim',m.load_ms_ssim)]:models[name]=ledger.call('load_'+name,'once',lambda name=name,loader=loader:loader(assets['models'][name],torch.device('cuda:0')))
   save(out/'supplemental_identity.json',dict(original_registration=r['supplemental_registration'],model_metadata=assets['models'],protocol=reg['protocol'],versions=m.package_versions(),physical_gpu=r['gpu_index'],FP32=True,TF32=False,no_selection=True))
   features={}
   with torch.inference_mode():
    for key,item in extra.items():
     guard();j=item['source_global_index'];x=loadref(j);y=loadimage(item['reconstruction']);xt=torch.from_numpy(x[None]).cuda();yt=torch.from_numpy(y[None]).cuda()
     if j not in features:features[j]=ledger.call('dreamsim_reference',j,lambda:models['dreamsim'].embed(m.preprocess_dreamsim(xt)).float().detach().cpu())
     dv=ledger.call('dists_pair',key,lambda:models['dists'](xt,yt,require_grad=False,batch_average=False).reshape(-1)[0].item())
     fy=ledger.call('dreamsim_reconstruction',key,lambda:models['dreamsim'].embed(m.preprocess_dreamsim(yt)).float());ds=(1-torch.nn.functional.cosine_similarity(features[j].cuda(),fy,dim=-1))[0].item()
     ms=ledger.call('ms_ssim_pair',key,lambda:models['ms_ssim'](xt,yt,data_range=1.0,size_average=False,win_size=11,win_sigma=1.5,weights=reg['protocol']['ms_ssim']['weights'],K=(0.01,0.03))[0].item())
     v=dict(dists=dv,dreamsim=ds,ms_ssim=ms);assert all(math.isfinite(vv) for vv in v.values());cache[key]['metrics'].update(v)
     p=out/'pairs'/f'{item["source_global_index"]:04d}_{hashlib.sha256(key.encode()).hexdigest()[:12]}_supp.json';save(p,dict(pair_key=key,metrics=v,identity=pin(out/'supplemental_identity.json')));cache[key]['supplemental_evidence']=pin(p)
  rows=[]
  for item in resultrows:
   e=item['event'];v=item['value'] if item['value'] is not None else cache[item['pair_key']];assert set(v['metrics'])==set(FOUR+(('dists','dreamsim','ms_ssim') if e['population']=='common500' else ()))
   rows.append(dict(population=e['population'],source_global_index=e['source_global_index'],source_index=e['source_index'],source_id=e['source_id'],snr_db=e['snr_db'],noise_seed=e['noise_seed'],strategy_family=e['strategy_family'],candidate_id=e['candidate_id'],actual_m=e['m'],actual_K=e['K'],arithmetic_bits=e['arithmetic_bits'],physical_reuse=e['reuse'],physical_frame=item['physical_frame'],reconstruction=item['reconstruction'],metrics=v['metrics'],score_origin=v['origin'],legacy_metrics=e['legacy']['metrics'],legacy_image_sha256=e['legacy']['image']['image_sha256']))
  assert len(rows)==2421 and ledger.summary()['unresolved']==0;save(out/'metric_rows.json',rows);save(out/'completion.json',dict(status='COMPLETE_N1024_REPAIR_FIXED_OUTPUT_METRICS',registration_sha256=expected,logical_frames=2421,actual_new_four_pairs=len(needs),actual_new_supplemental_pairs=len(extra),rows=pin(out/'metric_rows.json'),post_output_audit=pin(out/'post_output_source_audit.json'),ledger=ledger.summary(),seconds=time.monotonic()-start,policy_selection=False))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=None if ledger is None else ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':run(sys.argv[1])
