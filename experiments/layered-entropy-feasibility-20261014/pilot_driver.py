"""LE1 only: source50 TX passes, source24 independent RX proofs,450 missing clean states."""
from common import *
import sys,tempfile,traceback,fcntl,subprocess
def run():
 r=read(O/'registration.json');assert r['authorized_stages']==['LE0','LE1'] and r['wireless_frames']==0
 for p in r['source_bindings']:checked(p)
 checked(r['pilot50']);checked(r['clean_cache']);checked(r['gap']);routes=read(checked(r['model_routes']))
 assert time.time()<r['deadline_unix'];out=T/'science';assert not (out/'launch.json').exists();out.mkdir(exist_ok=True)
 tempfile.tempdir=os.environ['TMPDIR'];os.sched_setaffinity(0,set(r['cpu_affinity']))
 save(out/'launch.json',dict(process=pid_identity(os.getpid()),parent_process=pid_identity(os.getppid()),registration=pin(O/'registration.json'),actual_argv=sys.argv))
 lock=(T/'science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ledger=Ledger(out/'calls.sqlite',r['caps']);latest=[0]
 def boundary():
  assert time.time()<r['deadline_unix'] and not (T/'STOP').exists() and not (O/'STOP').exists()
  if time.monotonic()-latest[0]>10:
   text=subprocess.check_output(['nvidia-smi','-i',str(r['gpu_index']),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();uuid,free,util=[s.strip() for s in text.split(',')]
   assert uuid==r['gpu_uuid'] and int(free)>=2048,('BLOCKED_RESOURCE_FLOOR',text);latest[0]=time.monotonic()
 boundary()
 import numpy as np
 from native import build
 native,renderer=ledger.call('model_load','once',lambda:build(routes,ledger,boundary,out))
 from source_split_codec import SplitSourceCodec
 from packet_two_group import proof_catalogue,pack_group,intbits
 codec=SplitSourceCodec(R,native=native);catalogue=proof_catalogue();save(O/'source_proof_catalogue.json',dict(status='SOURCE_ONLY_PAID_PROFILE_DIRECTORY_NO_WIRELESS_QUALIFICATION',profiles=list(catalogue.values())))
 sources=read(r['pilot50']['path'])['records'];cache=read(r['clean_cache']['path']);missing=read(r['gap']['path'])['missing'];streams={};lengthrows=[];roundtrips=[]
 (out/'streams').mkdir();(out/'received_proofs').mkdir();(out/'clean_images').mkdir()
 for source in sources:
  boundary();slot=source['pilot_slot'];sid=source['source_id']
  with np.load(source['archive']['path'],allow_pickle=False) as z:tokens=z['tokens'].copy()
  encoded=ledger.call('source_tx',sid,lambda:codec.encode_all(tokens));arrays={}
  for (m,K),v in encoded['single'].items():arrays[f'single_{m}_{K}']=v['bits']
  for j,v in encoded['base'].items():arrays[f'base_{j}']=v['bits']
  for (j,m,K),v in encoded['enhancement'].items():
   arrays[f'enhancement_{j}_{m}_{K}']=v['bits'];single=encoded['single'][(m,K)];base=encoded['base'][j]
   lengthrows.append(dict(source_id=sid,pilot_slot=slot,j=j,m=m,K=K,total_tokens=OFFSETS[m]+K,base_tokens=OFFSETS[j],b_single=single['arithmetic_bits'],b_base=base['arithmetic_bits'],b_enhancement=v['arithmetic_bits'],single_finish_bits=single['finish_bits'],base_finish_bits=base['finish_bits'],enhancement_finish_bits=v['finish_bits'],extra_arithmetic_bits=base['arithmetic_bits']+v['arithmetic_bits']-single['arithmetic_bits'],extra_length_field_bits=13,extra_CRC_bits=16,extra_information_bits=base['arithmetic_bits']+v['arithmetic_bits']-single['arithmetic_bits']+29))
  p=out/'streams'/f'{slot:02d}.npz';np.savez(p,**arrays)
  record=dict(source=source,streams=pin(p),cdf_trace=encoded['cdf_trace']);save(out/'streams'/f'{slot:02d}.json',record);streams[sid]=record
  if slot<6:
   for j in (6,7):
    for m,K in ((8,0),(8,42)):
     key=f'{slot}:{j}:{m}:{K}';profile=next(p for p in catalogue.values() if not p['null_profile'] and (p['j'],p['m'],p['K'])==(j,m,K))
     B=pack_group(encoded['base'][j]['bits'],profile['k_B']);Ebits=pack_group(encoded['enhancement'][(j,m,K)]['bits'],profile['k_E'])
     rx=ledger.call('source_rx',key,lambda:codec.decode_received(intbits(profile['profile_id'],12),B,Ebits,catalogue))
     # Output is persisted before any token/CDF truth-side comparison.
     rp=out/'received_proofs'/f'{slot:02d}_{j}_{m}_{K}.npz';np.savez(rp,received_tokens=rx['received_tokens'])
     info={k:v for k,v in rx.items() if k!='received_tokens'};save(out/'received_proofs'/f'{slot:02d}_{j}_{m}_{K}.json',dict(actual_RX_output=pin(rp),receiver=info,profile_id=profile['profile_id'],base_information_bits=len(B),enhancement_information_bits=len(Ebits)))
     token_equal=np.array_equal(rx['received_tokens'],tokens[:OFFSETS[m]+K]);cdf_equal=rx['cdf_trace']==encoded['cdf_trace'][:m+int(K>0)]
     audit=dict(source_id=sid,j=j,m=m,K=K,status='PASS' if token_equal and cdf_equal and rx['status']=='FULL_RECEIVED_PREFIX' else 'BLOCKED',token_exact=bool(token_equal),independent_CDF_exact=cdf_equal,actual_output=pin(rp),truth_audit_after_fixed_output=True,receiver_truth_used=False,TX_CDF_used=False)
     save(out/'received_proofs'/f'{slot:02d}_{j}_{m}_{K}_audit.json',audit);roundtrips.append(audit)
     assert audit['status']=='PASS',('BLOCKED_INDEPENDENT_SOURCE_ROUNDTRIP',audit)
  save(out/f'progress_source_{slot:02d}.json',dict(completed_TX_sources=slot+1,total=50,independent_roundtrips=len(roundtrips),counts=ledger.summary()))
  print(f'TX_SOURCE_COMPLETE {slot+1}/50; RX_PROOFS {len(roundtrips)}/24',flush=True)
 assert len(lengthrows)==3400 and len(roundtrips)==24
 csvsave(O/'source_lengths.csv',lengthrows);save(O/'source_roundtrip_report.json',dict(status='PASS_ALL_24_ACTUAL_INDEPENDENT_TWO_STREAM_SOURCE_PROOFS',proofs=roundtrips,wireless_frames=0,TX_probability_traversals=50,extra_real_qualifications=0))
 # Only the exact missing source/endpoints are reconstructed. Existing outputs
 # retain their full provenance; no old model or quality call is replayed.
 for cell in missing:
  boundary();source=sources[cell['pilot_slot']];sid=source['source_id'];m,K=cell['m'],cell['K'];key=f'{cell["pilot_slot"]}:{m}:{K}'
  with np.load(source['archive']['path'],allow_pickle=False) as z:tokens=z['tokens'].copy()
  with native.torch.no_grad():im=ledger.call('clean_render',key,lambda:renderer(native,tokens[:OFFSETS[m]+K].copy(),m,K))
  assert im.dtype==np.float32 and im.shape==(3,256,256) and np.isfinite(im).all() and im.min()>=0 and im.max()<=1
  ip=out/'clean_images'/f'{cell["pilot_slot"]:02d}_{m}_{K}.npz';np.savez(ip,image=im)
  rec=dict(cell,image_archive=pin(ip),image_key='image',image_array_sha256=asha(im),model_identity=digest(MODELS),origin='NEW_ONLY_REGISTERED_MISSING_CLEAN_ENDPOINT')
  save(out/'clean_images'/f'{cell["pilot_slot"]:02d}_{m}_{K}.json',rec);cache.append(rec)
  if len(cache)%50==0:print(f'CLEAN_OUTPUTS_AVAILABLE {len(cache)}/1200',flush=True)
 assert len(cache)==1200
 save(out/'all_clean_images.json',cache)
 # Original official DINO-L implementation and asset resolver, source/endpoint
 # metrics only; DISTS/DreamSim/ConvNeXt/LPIPS are never constructed.
 sys.path.insert(0,str(R/'experiments/entropy-partial-mismatch-20261010/scripts'))
 import h800_ep_pilot_metrics_v1 as metric
 assets=metric.bound_assets(routes['metric_config']);original=metric.load_module(routes['metric_config']['sources']['metric_models.py'],'_LE1_original_dino')
 with original.no_network():evaluator=ledger.call('dino_model_load','once',lambda:original.MetricEvaluator(assets['dinov2_manifest'],device='cuda:0',requested=('dinov2_vitl14',)))
 assert set(evaluator.models)=={'dinov2_vitl14'}
 targets={};prepared={};qrows=[]
 for source in sources:
  with np.load(source['archive']['path'],allow_pickle=False) as z:target=z['pixels'].astype(np.float32)/255.
  targets[source['source_id']]=target
 for rec in sorted(cache,key=lambda x:(next(s['pilot_slot'] for s in sources if s['source_id']==x['source_id']),x['m'],x['K'])):
  boundary();sid=rec['source_id'];source=next(s for s in sources if s['source_id']==sid)
  with np.load(checked(rec['image_archive']),allow_pickle=False) as z:im=z[rec['image_key']].copy()
  assert asha(im)==rec['image_array_sha256']
  mse=float(np.mean((im.astype(np.float64)-targets[sid].astype(np.float64))**2));psnr=float(-10*np.log10(mse))
  if 'dino' in rec:dino=rec['dino']
  else:
   t=native.torch;x=t.from_numpy(im[None]).to('cuda:0');y=t.from_numpy(targets[sid][None]).to('cuda:0')
   if sid not in prepared:prepared[sid]=ledger.call('source_embedding',sid,lambda:evaluator.prepare_reference(y,[source['evaluation_class_index']],label_conditioned=False))
   value=ledger.call('reconstruction_dino',f'{source["pilot_slot"]}:{rec["m"]}:{rec["K"]}',lambda:evaluator.score(x,y,[source['evaluation_class_index']],label_conditioned=False,prepared=prepared[sid]))[0];dino=float(value['dinov2_vitl14_cosine'])
  qrows.append(dict(source_id=sid,pilot_slot=source['pilot_slot'],m=rec['m'],K=rec['K'],tokens=OFFSETS[rec['m']]+rec['K'],psnr_db=psnr,dinov2_vitl14_cosine=dino,origin=rec['origin'],image_array_sha256=rec['image_array_sha256'],image_path=rec['image_archive']['path'],model_identity=digest(MODELS)))
  save(out/'clean_images'/f'quality_{source["pilot_slot"]:02d}_{rec["m"]}_{rec["K"]}.json',qrows[-1])
 csvsave(O/'clean_quality.csv',qrows);save(out/'completion.json',dict(status='COMPLETE_LE1_SOURCE50_RX24_CLEAN1200_NO_WIRELESS',registration=pin(O/'registration.json'),length_rows=3400,quality_rows=1200,reused_clean=750,new_clean=450,TX_sources=50,source_roundtrips=24,tokenizer_calls=0,wireless_frames=0,training_updates=0,counts=ledger.summary(),outputs=[pin(O/n) for n in ('source_lengths.csv','clean_quality.csv','source_roundtrip_report.json')]))
 assert ledger.summary()['unresolved']==0
 print('COMPLETE_LE1_SOURCE50_RX24_CLEAN1200_NO_WIRELESS',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException:
  p=T/'science/failure.json'
  if not p.exists():save(p,dict(status='BLOCKED_PRESERVE_ACTUAL_PREFIX_NO_RETRY',traceback=traceback.format_exc(),wireless_frames=0,training_updates=0))
  raise
