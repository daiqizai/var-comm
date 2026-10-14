"""Explicit closed-prefix resource recovery; never repeat completed science."""
from common import *
import ast,copy,sys,tempfile,traceback,fcntl,subprocess,gc

REC=O/'resource_resume_v1/registration.json'
OUT=T/'resource_resume_v1'
SCI=T/'science'

def adapted_native_source():
 original=(E/'native.py').read_text(encoding='utf8')
 replacements=[("read(O/'registration.json')['gpu_pci']","read(REC)['gpu_pci']"),('(12<<30)','(8<<30)'),('allocator_cap_bytes=12<<30','allocator_cap_bytes=8<<30'),("'prior_scale',str(counts['prior'])","'prior_scale','resume1:'+str(counts['prior'])"),("'decoder_forward',str(counts['decoder'])","'decoder_forward','resume1:'+str(counts['decoder'])")]
 adapted=original
 for old,new in replacements:
  assert adapted.count(old)==1,(old,adapted.count(old));adapted=adapted.replace(old,new)
 restored=adapted
 for old,new in reversed(replacements):
  assert restored.count(new)==1;restored=restored.replace(new,old)
 assert restored==original and ast.dump(ast.parse(restored))==ast.dump(ast.parse(original))
 return adapted,replacements

def old_length_rows(source,arrays):
 rows=[]
 for j in (4,5,6,7):
  for m,K in ENDS:
   if OFFSETS[m]+K<=OFFSETS[j]:continue
   s,b,e=[arrays[n] for n in (f'single_{m}_{K}',f'base_{j}',f'enhancement_{j}_{m}_{K}')]
   assert all(x.dtype.name=='uint8' and x.ndim==1 for x in (s,b,e))
   delta=len(b)+len(e)-len(s)
   rows.append(dict(source_id=source['source_id'],pilot_slot=source['pilot_slot'],j=j,m=m,K=K,total_tokens=OFFSETS[m]+K,base_tokens=OFFSETS[j],b_single=len(s),b_base=len(b),b_enhancement=len(e),single_finish_bits=None,base_finish_bits=None,enhancement_finish_bits=None,extra_arithmetic_bits=delta,extra_length_field_bits=13,extra_CRC_bits=16,extra_information_bits=delta+29,finish_increment_metadata='NOT_PERSISTED_FIRST38_FULL_TERMINATED_LENGTHS_EXACT'))
 assert len(rows)==68
 return rows

def validate_closed_prefix(reg):
 old=read(checked(reg['original_registration']));assert old['authorized_stages']==['LE0','LE1']
 for p in old['source_bindings']:checked(p)
 for p in reg['source_bindings']:checked(p)
 wait=read(checked(reg['failed_actual_wait']));assert type(wait['returncode']) is int and wait['returncode']==1
 assert not any(wait[k] for k in ('timed_out','terminated','killed')) and wait['metadata_error'] is None
 assert gone(wait['process']) and gone(wait['parent_process'])
 failure=read(checked(reg['original_failure']));assert 'BLOCKED_RESOURCE_FLOOR' in failure['traceback']
 diagnosis=read(checked(reg['diagnosis']));assert diagnosis['remaining_TX']==12 and diagnosis['remaining_RX']==0
 checked(reg['closed_ledger_before_resume'])
 db=sqlite3.connect('file:'+str(SCI/'calls.sqlite')+'?mode=ro',uri=True)
 rows=list(db.execute('SELECT kind,key,status FROM calls'));db.close()
 assert all(s=='COMPLETE' for k,key,s in rows)
 assert {k:sum(a==k for a,b,c in rows) for k in ('model_load','source_tx','source_rx','prior_scale')}==dict(model_load=1,source_tx=38,source_rx=24,prior_scale=584)
 sources=read(checked(old['pilot50']))['records'];completed=[];proofs=[]
 for slot in range(38):
  rec=read(SCI/'streams'/f'{slot:02d}.json');assert rec['source']==sources[slot]
  checked(rec['streams']);completed.append(rec)
 assert len(list((SCI/'streams').glob('*.json')))==38
 for p in sorted((SCI/'received_proofs').glob('*_audit.json')):
  a=read(p);assert a['status']=='PASS' and a['token_exact'] and a['independent_CDF_exact'];checked(a['actual_output']);proofs.append(a)
 assert len(proofs)==24 and not list((SCI/'clean_images').iterdir())
 return old,sources,completed,proofs

def run():
 reg=read(REC);assert sha(REC)==sys.argv[1]
 old,sources,completed,proofs=validate_closed_prefix(reg)
 assert time.time()<old['deadline_unix'] and reg['deadline_unix']==old['deadline_unix']
 OUT.mkdir();save(OUT/'launch.json',dict(process=pid_identity(os.getpid()),parent_process=pid_identity(os.getppid()),argv=sys.argv,registration=pin(REC)))
 lock=(T/'science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 tempfile.tempdir=os.environ['TMPDIR'];os.sched_setaffinity(0,set(old['cpu_affinity']))
 ledger=Ledger(SCI/'calls.sqlite',reg['effective_caps']);latest=[0]
 def snapshot():
  line=subprocess.check_output(['nvidia-smi','-i',str(reg['gpu_index']),'--query-gpu=uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
  uuid,pci,free,util=[s.strip() for s in line.split(',')]
  assert uuid==reg['gpu_uuid'] and pci.lower()==reg['gpu_pci'].lower();return dict(raw=line,free=int(free),util=int(util))
 admission=snapshot();assert admission['free']>=8192 and admission['util']<=50;save(OUT/'admission.json',admission)
 def boundary():
  assert time.time()<old['deadline_unix'] and not (T/'STOP').exists() and not (O/'STOP').exists()
  if time.monotonic()-latest[0]>10:
   s=snapshot();assert s['free']>=2048,('BLOCKED_RESOURCE_FLOOR',s);latest[0]=time.monotonic()
 boundary()
 import numpy as np
 code,replacements=adapted_native_source();ns=dict(REC=REC,__name__='_LE01_resource_resume_native')
 exec(compile(code,str(E/'native.py')+' [explicit resource resume]', 'exec'),ns)
 save(OUT/'native_resource_adaptation.json',dict(original=pin(E/'native.py'),inverse_AST_exact=True,replacements=replacements,scientific_function_definitions_unchanged=True))
 native,renderer=ledger.call('source_model_reload','resource_resume_once',lambda:ns['build'](read(checked(old['model_routes'])),ledger,boundary,OUT))
 from source_split_codec import SplitSourceCodec
 codec=SplitSourceCodec(R,native=native);lengthrows=[]
 for rec in completed:
  with np.load(checked(rec['streams']),allow_pickle=False) as arrays:lengthrows.extend(old_length_rows(rec['source'],arrays))
 for source in sources[38:]:
  boundary();slot=source['pilot_slot'];sid=source['source_id']
  with np.load(checked(source['archive']),allow_pickle=False) as z:tokens=z['tokens'].copy()
  encoded=ledger.call('source_tx',sid,lambda:codec.encode_all(tokens));arrays={};newrows=[]
  for (m,K),v in encoded['single'].items():arrays[f'single_{m}_{K}']=v['bits']
  for j,v in encoded['base'].items():arrays[f'base_{j}']=v['bits']
  for (j,m,K),v in encoded['enhancement'].items():
   arrays[f'enhancement_{j}_{m}_{K}']=v['bits'];s=encoded['single'][(m,K)];b=encoded['base'][j]
   newrows.append(dict(source_id=sid,pilot_slot=slot,j=j,m=m,K=K,total_tokens=OFFSETS[m]+K,base_tokens=OFFSETS[j],b_single=s['arithmetic_bits'],b_base=b['arithmetic_bits'],b_enhancement=v['arithmetic_bits'],single_finish_bits=s['finish_bits'],base_finish_bits=b['finish_bits'],enhancement_finish_bits=v['finish_bits'],extra_arithmetic_bits=b['arithmetic_bits']+v['arithmetic_bits']-s['arithmetic_bits'],extra_length_field_bits=13,extra_CRC_bits=16,extra_information_bits=b['arithmetic_bits']+v['arithmetic_bits']-s['arithmetic_bits']+29,finish_increment_metadata='PERSISTED_ACTUAL_ENCODER_FINISH'))
  p=SCI/'streams'/f'{slot:02d}.npz';assert not p.exists();np.savez(p,**arrays)
  save(SCI/'streams'/f'{slot:02d}.json',dict(source=source,streams=pin(p),cdf_trace=encoded['cdf_trace'],length_rows=newrows,resume_registration=pin(REC)))
  lengthrows.extend(newrows);save(OUT/f'progress_source_{slot:02d}.json',dict(counts=ledger.summary(),source=source['source_id']))
  print(f'TX_SOURCE_COMPLETE {slot+1}/50; ORIGINAL_RX_PROOFS_REUSED 24/24',flush=True)
 assert len(lengthrows)==3400
 csvsave(O/'source_lengths.csv',lengthrows)
 save(O/'source_roundtrip_report.json',dict(status='PASS_ALL_24_ACTUAL_INDEPENDENT_TWO_STREAM_SOURCE_PROOFS',proofs=proofs,wireless_frames=0,TX_probability_traversals=50,extra_real_qualifications=0,qualification_device='Original GPU3 H800, original numeric recipe; resource-resume GPU1 performed zero additional roundtrips',individual_finish_increment_missing_first38=True,all_terminated_lengths_exact=True))
 cache=read(checked(old['clean_cache']));missing=read(checked(old['gap']))['missing']
 for cell in missing:
  boundary();source=sources[cell['pilot_slot']];m,K=cell['m'],cell['K'];key=f'{cell["pilot_slot"]}:{m}:{K}'
  with np.load(checked(source['archive']),allow_pickle=False) as z:tokens=z['tokens'].copy()
  with native.torch.no_grad():im=ledger.call('clean_render',key,lambda:renderer(native,tokens[:OFFSETS[m]+K].copy(),m,K))
  assert im.dtype==np.float32 and im.shape==(3,256,256) and np.isfinite(im).all() and im.min()>=0 and im.max()<=1
  ip=SCI/'clean_images'/f'{cell["pilot_slot"]:02d}_{m}_{K}.npz';assert not ip.exists();np.savez(ip,image=im)
  rec=dict(cell,image_archive=pin(ip),image_key='image',image_array_sha256=asha(im),model_identity=digest(MODELS),origin='NEW_ONLY_REGISTERED_MISSING_CLEAN_ENDPOINT')
  save(SCI/'clean_images'/f'{cell["pilot_slot"]:02d}_{m}_{K}.json',rec);cache.append(rec)
  if len(cache)%50==0:print(f'CLEAN_OUTPUTS_AVAILABLE {len(cache)}/1200',flush=True)
 assert len(cache)==1200;save(SCI/'all_clean_images.json',cache)
 # Release already completed source/reconstruction models before DINO; no math change.
 torch=native.torch;del codec,native,encoded;gc.collect();torch.cuda.empty_cache();boundary()
 routes=read(old['model_routes']['path']);sys.path.insert(0,str(R/'experiments/entropy-partial-mismatch-20261010/scripts'))
 import h800_ep_pilot_metrics_v1 as metric
 assets=metric.bound_assets(routes['metric_config']);original=metric.load_module(routes['metric_config']['sources']['metric_models.py'],'_LE1_original_dino_resume')
 with original.no_network():evaluator=ledger.call('dino_model_load','once',lambda:original.MetricEvaluator(assets['dinov2_manifest'],device='cuda:0',requested=('dinov2_vitl14',)))
 assert set(evaluator.models)=={'dinov2_vitl14'}
 targets={};prepared={};qrows=[];source_by_id={s['source_id']:s for s in sources}
 for s in sources:
  with np.load(checked(s['archive']),allow_pickle=False) as z:targets[s['source_id']]=z['pixels'].astype(np.float32)/255.
 for rec in sorted(cache,key=lambda x:(source_by_id[x['source_id']]['pilot_slot'],x['m'],x['K'])):
  boundary();sid=rec['source_id'];s=source_by_id[sid]
  with np.load(checked(rec['image_archive']),allow_pickle=False) as z:im=z[rec['image_key']].copy()
  assert asha(im)==rec['image_array_sha256'];mse=float(np.mean((im.astype(np.float64)-targets[sid].astype(np.float64))**2));psnr=float(-10*np.log10(mse))
  if 'dino' in rec:dino=rec['dino']
  else:
   x=torch.from_numpy(im[None]).to('cuda:0');y=torch.from_numpy(targets[sid][None]).to('cuda:0')
   if sid not in prepared:prepared[sid]=ledger.call('source_embedding',sid,lambda:evaluator.prepare_reference(y,[s['evaluation_class_index']],label_conditioned=False))
   value=ledger.call('reconstruction_dino',f'{s["pilot_slot"]}:{rec["m"]}:{rec["K"]}',lambda:evaluator.score(x,y,[s['evaluation_class_index']],label_conditioned=False,prepared=prepared[sid]))[0];dino=float(value['dinov2_vitl14_cosine'])
  row=dict(source_id=sid,pilot_slot=s['pilot_slot'],m=rec['m'],K=rec['K'],tokens=OFFSETS[rec['m']]+rec['K'],psnr_db=psnr,dinov2_vitl14_cosine=dino,origin=rec['origin'],image_array_sha256=rec['image_array_sha256'],image_path=rec['image_archive']['path'],model_identity=digest(MODELS))
  qrows.append(row);save(SCI/'clean_images'/f'quality_{s["pilot_slot"]:02d}_{rec["m"]}_{rec["K"]}.json',row)
 csvsave(O/'clean_quality.csv',qrows)
 summary=ledger.summary();assert summary['unresolved']==0
 save(OUT/'completion.json',dict(status='COMPLETE_LE1_SOURCE50_RX24_CLEAN1200_NO_WIRELESS',registration=pin(O/'registration.json'),resource_resume_registration=pin(REC),length_rows=3400,quality_rows=1200,reused_clean=750,new_clean=450,TX_sources=50,source_roundtrips=24,tokenizer_calls=0,wireless_frames=0,training_updates=0,original_resource_failure_preserved=True,counts=summary,outputs=[pin(O/n) for n in ('source_lengths.csv','clean_quality.csv','source_roundtrip_report.json')]))
 print('COMPLETE_LE1_SOURCE50_RX24_CLEAN1200_NO_WIRELESS',flush=True)

if __name__=='__main__':
 try:run()
 except BaseException:
  if OUT.exists() and not (OUT/'failure.json').exists():save(OUT/'failure.json',dict(status='BLOCKED_PRESERVE_ACTUAL_PREFIX_NO_RETRY',traceback=traceback.format_exc(),wireless_frames=0))
  raise
