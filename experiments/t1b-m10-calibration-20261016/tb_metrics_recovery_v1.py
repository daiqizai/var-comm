"""Score only missing fixed image pairs; retain failures and original four metrics."""
from tb_common import *
import numpy as np,platform,subprocess,traceback,collections
class AdapterLedger:
 def __init__(self,ledger):self.ledger=ledger;self.n=collections.Counter()
 def call(self,kind,fn,**metadata):
  self.n[kind]+=1;return self.ledger.call(kind,str(self.n[kind]),fn)

def run(expected):
 from recovery_binding import verify_recovery
 verify_recovery(sys.argv[2])
 r,out=stage_start('metrics',expected);ledger=None;start=time.monotonic();last=[0.]
 try:
  vd=read(T/'receiver_recovery_v1/completion.json');pd=read(T/'phy/completion.json');assert vd['registration_sha256']==pd['registration_sha256']==expected
  images=checkread(vd['images']);logical=checkread(pd['logical_frames']);sources=checkread(r['sources']);oldpairs=checkread(r['old_pairs']);inh=checkread(r['inheritance'])
  caps=dict(model_constructions=3,reference_preparations=1000,image_scores=36000,dinov2_vitl14_reference=1000,dinov2_vitl14_reconstruction=36000,convnext_reference=1000,convnext_reconstruction=36000,lpips_pair=36000,lpips_alexnet_backbone_forward=72000)
  ledger=Ledger(out/'calls.sqlite',caps);counted=AdapterLedger(ledger)
  def guard():
   assert time.monotonic()-start<r['seconds']['metrics'] and not(T/'STOP').exists()
   if time.monotonic()-last[0]>10:
    free=int(subprocess.check_output(['nvidia-smi','-i',str(r['gpu_index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=2048;last[0]=time.monotonic()
  import torch
  assert torch.cuda.device_count()==1;torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
  assert str(props.uuid)==r['gpu_cuda_uuid'];torch.cuda.set_per_process_memory_fraction((12<<30)/props.total_memory)
  observed=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
  exp=inh['metric_config']['runtime_identity']['expected_runtime'];assert observed=={k:v for k,v in exp.items()if k not in('threads','interop_threads')}
  import h800_ep_pilot_metrics_v1 as metrics
  from h800_ep_pilot_core_v1 import reference_pixels
  ev=metrics.FourMetrics(inh['metric_config'],counted,guard);assert ev.identity==inh['old_metric_identity']['identity']
  save(out/'metric_identity.json',dict(identity=ev.identity,metadata=ev.metadata,observed_runtime=observed,actual_gpu_uuid=str(props.uuid),scope='T1B_original_calibration1000'))
  (out/'pairs').mkdir();rows=[];cache={};verified_pixels={};references={};prepared={};new=0;reuse=collections.Counter();fixed_audits={};physical_cache={}
  for ix,item in enumerate(logical):
   guard();event=item['event'];i=event['source_index'];rec=images[item['source_input_key']];key=canonical([i,rec['image_sha256']])
   # The receiver has exited. Only now may the separate scorer read source truth.
   if i not in references:references[i]=reference_pixels(sources[i])
   if rec['image_archive']['path'] not in verified_pixels:
    checked(rec['image_archive'])
    with np.load(rec['image_archive']['path'],allow_pickle=False)as z:image=z['image'].copy()
    assert image.dtype==np.float32 and image.shape==(3,256,256) and asha(image)==rec['image_sha256'];verified_pixels[rec['image_archive']['path']]=True
   else:image=None
   if rec['origin']=='ACTUAL_NEW' and item['source_input_key']not in fixed_audits:
    proof=checkread(rec['evidence']);pkt=checkread(item['physical_frame']);rx=pkt['actual_RX'];e=proof['evidence'];audit=dict(source_input_key=item['source_input_key'],fixed_output=rec['evidence'],source_truth_entered_receiver=False)
    p=rx['rx_profile'];b=rx['body']
    if e['source_decode_called'] and b and b['payload']is not None and asha(np.asarray(b['payload'],dtype=np.uint8))==pkt['payload_sha256'] and p['m']==event['m'] and p['K']==event['K']:
     with np.load(sources[i]['archive']['path'],allow_pickle=False)as z:truth=z['tokens'][:OFFSETS[p['m']]+p['K']].copy()
     assert e['source_status']=='ARITHMETIC_SOURCE_DECODED' and np.array_equal(truth,np.asarray(e['received_tokens'],dtype=np.int64)),('BLOCKED_EXACT_RECEIVED_STREAM_MODEL_CDF_MISMATCH',i)
     audit.update(exact_transmitted_stream=True,received_prefix_equal_after_output_fixed=True)
    else:audit.update(exact_transmitted_stream=False,truth_not_used_for_error_correction=True)
    fixed_audits[item['source_input_key']]=audit
   if key not in cache:
    if key in oldpairs:
     old=oldpairs[key];pair=checkread(old['pair']);assert pair['metric_identity']==ev.identity and pair['metrics']==old['metrics'] and pair['reference_archive']==sources[i]['archive']
     assert old['metric_pair_key']==digest(dict(reference_sha256=asha(references[i]),reconstruction_sha256=rec['image_sha256'],metric_identity=ev.identity))
     cache[key]=dict(metrics=old['metrics'],pair=old['pair'],origin='CLOSED_EXACT_PIXEL_PAIR')
    else:
     if image is None:
      with np.load(rec['image_archive']['path'],allow_pickle=False)as z:image=z['image'].copy()
     if i not in prepared:prepared[i]=ev.prepare(references[i])
     values=ev.score(references[i],image,sources[i]['evaluation_class_index'],prepared[i]);p=out/'pairs'/f'{new:05d}.json'
     save(p,dict(metrics=values,metric_identity=ev.identity,metric_pair_key=metrics.pair_cache_key(references[i],image,ev.identity),reference_archive=sources[i]['archive'],reconstruction_archive=rec['image_archive'],source_index=i,image_sha256=rec['image_sha256']))
     cache[key]=dict(metrics=values,pair=pin(p),origin='ACTUAL_NEW');new+=1
   v=cache[key];reuse[v['origin']]+=1
   rows.append(dict(source_index=i,source_id=event['source_id'],snr_db=event['snr_db'],noise_seed=event['noise_seed'],candidate_id=event['candidate_id'],strategy_family=event['strategy_family'],actual_m=event['m'],actual_K=event['K'],**v['metrics'],metric_pair_result=v['pair'],reuse=v['origin'],source_input_key=item['source_input_key'],image_sha256=rec['image_sha256']))
   if(ix+1)%300==0:progress(out,dict(logical=ix+1,new_pairs=new,ledger=ledger.summary()))
  assert len(rows)==36000 and ledger.summary()['unresolved']==0
  save(out/'metric_rows.json',rows);save(out/'post_output_source_audit.json',list(fixed_audits.values()))
  save(out/'completion.json',dict(status='COMPLETE_T1B_METRICS',registration_sha256=expected,logical_frames=36000,new_pairs=new,unique_pairs=len(cache),reuse=dict(reuse),metric_rows=pin(out/'metric_rows.json'),post_output_source_audit=pin(out/'post_output_source_audit.json'),ledger=ledger.summary(),seconds=time.monotonic()-start))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=None if ledger is None else ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':run(sys.argv[1])
