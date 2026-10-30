"""Fresh actual TX/RX boundaries. No score networks, output caches or timed log I/O."""
from followup_common import *
import base64,traceback,tempfile,subprocess,contextlib,shutil

class MemoryCalls:
 def __init__(self): self.counts={}
 def call(self,kind,*args):
  if isinstance(kind,dict):kind=kind['kind']
  self.counts[kind]=self.counts.get(kind,0)+1
  return args[-1]()
 def decode(self,phase,key,kind,request,fn):return self.call(kind,fn)

def historical():
 sys.path.insert(0,str(R/'experiments/five-method-timing-20261020'))
 import native,worker
 return native,worker

def bpg_runtime(cfg,out):
 """Native source search unchanged; CPU-count shim reflects the two-core affinity."""
 import bpg
 shim=B/'five_method_timing_20261020_v1/bpg_threads_qualification_v1/affinity_cpu_count.so'
 assert sha(shim)=='953375c8aa9c3ddd8f3313f99425ee2fe33a08b838492b0fee965dd04211494d'
 old_run=bpg.subprocess.run
 # Only codec child environment changes. Never preload the Python/model process.
 def child_run(argv,**kwargs):
  if str(argv[0])==cfg['bpg_config']['bpgenc']['path']:
   kwargs['env']=dict(os.environ,LD_PRELOAD=str(shim))
  return old_run(argv,**kwargs)
 bpg.subprocess=type('CodecProcesses',(),{'run':staticmethod(child_run)})
 @contextlib.contextmanager
 def workspace(root):
  path=Path(tempfile.mkdtemp(dir=root))
  try:yield str(path)
  except BaseException:
   dest=out/'failed_codec_evidence'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(path,dest);raise
  else:shutil.rmtree(path)
 bpg.workspace=workspace
 return bpg

def setup(cfg,g,out,role,pci):
 pci_parts=pci.split(':');pci=f'{int(pci_parts[0],16):04x}:{pci_parts[1].lower()}:{pci_parts[2].lower()}'
 if g['method']=='ARPC':sys.path.insert(0,cfg['arpc_public']['environment'])
 native,worker=historical();native.paths();calls=MemoryCalls();method=g['method']
 cfg=dict(cfg,resource=dict(gpu_pci=pci))
 # The original mathematical loader already requires two CPU cores.
 def visual(r,d):
  import tb_native
  n,render=tb_native.build(cr(r['model_routes']),calls,lambda:None,d,pci)
  return n,render,calls
 native.visual=visual
 n=c=classifier=None
 if method in ['WHOLE','Token_NeST']:
  n,_,_=visual(cfg,out)
  from grid_codec import codec
  from h800_ep48_link_core_v1 import bind_static
  c=codec(n);bind_static(c,cfg['static']);t=n.torch
 elif method=='Ada':
  sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'))
  from c_predclass_native import load_native
  ar=dict(cfg['ada_native'],pci=pci.lower())
  n=load_native(ar,out,calls);t=n.torch
  if role=='tx':
   from c_predclass_classifier import load_classifier
   classifier=load_classifier(ar,out,calls)
  from c_arithmetic import load_codec
  enc,dec,cdf,proof=load_codec();save(out/'arithmetic_identity.json',proof);c=(enc,dec,cdf)
 elif method in ['Latent_JSCC','SwinJSCC']:
  n,calls=native.continuous(cfg,out,'P100000'if method=='Latent_JSCC'else'Swin200000');t=n.t
 elif method=='ARPC':n=native.arpc(cfg,out,role);t=n.torch
 else:
  assert method=='BPG_LDPC'
  import torch
  native.flags(torch);t=torch
  n=bpg_runtime(cfg,out);cfg['_tmpfs']=str(Path('/dev/shm')/('nest_followup_'+out.name));Path(cfg['_tmpfs']).mkdir(exist_ok=False)
 # Construct uncached PHY objects outside the measured boundaries; no decoding.
 phy=rt=None
 if method in ['WHOLE','Token_NeST','Ada']:
  sys.path[:0]=[str(R/'experiments/nestcom-n512-20261028'),str(R/'experiments/adatoken-response-20261026')]
  import n512_phy as n5
  if g['N']==512:
   qual=cr(cfg['n512_qualification']);assert qual['status']=='COMPLETE_REAL_N512_PHY'and qual['frame_count']==72 and qual['header_ID_count']==3384 and qual['ledger']['unresolved']==0
   phy,_=n5.engine();rt=phy.Runtime(R,configure_threads=False)
  else:
   import c_predclass_frame as cp
   phy=cp.isolated_engine();ns=phy.pack_body.__globals__;ns['FAMILIES']=(*ns['FAMILIES'][:-2],cp.FAMILY,cp.NULL)
   rt=phy.create_runtime(R,str(checked(cfg['phy_qualification'])),configure_threads=False)
  cat=cr(g['catalogue']);rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']}
 elif method=='ARPC':
  module=native.load(R/'experiments/arpc-external-baseline-20261019/phy_v1/link.py','_e5_arpc_phy')
  cls=native.replace_function(module,'Runtime',[('t.set_num_threads(2);t.set_num_interop_threads(1)','assert t.get_num_threads()==6 and t.get_num_interop_threads()==2')]);rt=cls()
 elif method=='BPG_LDPC':
  path=B/'five_method_timing_20261020_v1/assets/bpg_support';sys.path.insert(0,str(path))
  module=native.load(path/'bpg_phy.py','_e5_bpg_phy');cls=native.replace_function(module,'Link',[("root='/home/liulu/projects/VAR_COMM'",'root='+repr(str(R)))]);rt=cls(cfg['bpg_config'])
 if rt is not None:rt.header.s.ROOT=B/'baseline_strength_20261011/native_build';rt.header.s.load_native()
 assert len(os.sched_getaffinity(0))==2 and t.get_num_threads()==6 and t.get_num_interop_threads()==2
 return dict(cfg=cfg,g=g,n=n,c=c,t=t,rt=rt,phy=phy,calls=calls,classifier=classifier,worker=worker)

def counters(g,index,profile_id=0):
 if g['N']==512:return 90000000+index*3,None
 if g['snr_db']==7:
  public=26000000+16384*index
  return public+profile_id,public
 return (3*500+index)*3,None

def tx(pixels,source_id,index,ctx):
 g=ctx['g'];m=g['method'];n=ctx['n'];rt=ctx['rt'];t=ctx['t'];cfg=ctx['cfg'];worker=ctx['worker'];meter=worker.Clock(None);proof={}
 if m in ['WHOLE','Token_NeST','Ada']:
  profiles=[p for p in rt.profiles.values()if(p['q'],p['nominal_rate'])==(g['policy']['q'],g['policy']['rate'])]
  cap=profiles[0]['k']-(39 if m=='Ada'else 29)
  if m=='Ada':
   x=pixels.astype(np.float32)/np.float32(255);cls=ctx['classifier'].predict(x);tokens=n.tokenize(x);n.start(cls);Encoder,_,cdf=ctx['c'];coder=Encoder();streams={}
   for i,v in enumerate(tokens):
    table=n.cdf(n.logits(i),cdf);coder.encode(np.array([v],np.int64),table);n.previous=int(v);streams[i+1]=coder.finish()
   ends=list(range(256,0,-1));fits=[e for e in ends if len(streams[e])<=cap]
   if fits:
    e=fits[0];bits=streams[e];p=next(p for p in profiles if p['family']=='ADA_PREDICTED_CLASS_PREFIX'and p['m']==e)
    from c_predclass_protocol import container
    payload=container(cls,bits);proof=dict(endpoint=[e,0],_arithmetic_bits=bits,predicted_class=cls)
   else:p=next(p for p in profiles if p['family']=='EXPLICIT_NULL_FIXEDFRAME');payload=np.zeros(2,np.uint8);proof=dict(endpoint=[0,0],NULL=True)
  else:
   from kn_native import tokenizer
   tokens=tokenizer(n,pixels);ends=[tuple(e)for e in g['endpoints']];streams=ctx['c'].encode_endpoints(tokens,ends);fits=[e for e in ends if len(streams[e]['bits'])<=min(cap,8191)]
   if fits:
    e=fits[0];payload=streams[e]['bits'];p=next(p for p in profiles if(p['m'],p['K'],p['family'])==(*e,'EC_VAR_PARTIAL'if e[1]else'EC_VAR_WHOLE'));proof=dict(endpoint=list(e),_arithmetic_bits=payload,predicted_class=None)
   else:
    assert g['N']==512,'N1024 no feasible source endpoint'
    p=next(p for p in profiles if p['family']=='EXPLICIT_NULL_FIXEDFRAME');payload=np.zeros(2,np.uint8);proof=dict(endpoint=[0,0],NULL=True)
  proof.update(terminated_candidates=len(ends),length_checks=next((i+1 for i,e in enumerate(ends)if fits and e==fits[0]),len(ends)),_source_tokens=tokens)
  counter,public=counters(g,index,p['profile_id'])
  if g['N']==512:
   from n512_phy import WIRE
   wave,wire=ctx['phy'].transmit_frame(rt,payload,p['profile_id'],counter,WIRE)
  else:wave,wire=rt.transmit(p['profile_id'],payload,counter)
  return wave,dict(counter=counter,public_base=public,source_id=source_id,source_index=index),dict(proof,transmission=wire)
 counter,_=counters(g,index)
 if m in ['Latent_JSCC','SwinJSCC']:
  wave,witness=n.encode(pixels,g['snr_db'],counter,meter)
  return wave,dict(counter=counter,public_base=None,source_id=source_id,source_index=index),dict(native_TX=True)
 if m=='ARPC':
  result,proof=worker.arpc_tx(pixels,cfg,g['snr_db'],n,meter);p=rt.profile(result['endpoint'],result['q'],result['nominal_rate']);wave,wire=rt.transmit(base64.b64decode(result['payload_b64']),p['profile_id'],counter)
 else:
  p=next(p for p in cfg['bpg_config']['catalogue']if p['profile_id']==cfg['bpg_policies'][str(g['snr_db'])]);payload,proof=n.encode(pixels,cfg['bpg_config'],p['capacity_bytes'],cfg['_tmpfs']);assert payload is not None,'BPG no feasible packet';wave=rt.transmit(payload,p,counter)
  proof['thread_policy']='Frozen x265 settings; inherited two-core affinity and verified cpu-count shim'
 return wave,dict(counter=counter,public_base=None,source_id=source_id,source_index=index),proof

def noise(wave,meta,ctx):
 g=ctx['g'];m=g['method'];snr=g['snr_db']
 if m in ['Latent_JSCC','SwinJSCC']:return ctx['n'].noise(wave,meta['source_id'],snr)
 if m=='BPG_LDPC':return ctx['rt'].noisy(wave,'A3_DEVELOPMENT_TIMING',snr,meta['source_index'],2001)
 if m=='ARPC':
  import ep_phy
  phy=ep_phy
 else:phy=ctx['phy']
 seed=2001 if m=='Ada'and snr==7 else 6201
 return wave+phy.standard_noise(meta['source_id'],seed)*10**(-snr/20)

def rx(observed,meta,ctx,event):
 g=ctx['g'];m=g['method'];n=ctx['n'];rt=ctx['rt'];snr=g['snr_db'];meter=ctx['worker'].Clock(None);calls=ctx['calls'];gray=np.full((3,256,256),.5,np.float32)
 if m in ['Latent_JSCC','SwinJSCC']:
  image,actual=n.receive(observed,snr,meta['counter'],meter,event)
  return image,dict(status=actual['status'],success=bool(actual['header']['header_ok']))
 if m in ['WHOLE','Token_NeST','Ada']:
  phy=ctx['phy']
  if snr==7:
   sys.path.insert(0,str(R/'experiments/wcl-adatoken-20261027'));from public_counter_rx import receive
   actual=receive(phy,rt,observed,snr,meta['public_base'],calls,event,phase='E5_TIMING')
  elif g['N']==512:
   from n512_phy import WIRE
   actual=phy.receive_frame(rt,observed,snr,meta['counter'],calls,event,phase='E5_TIMING',session=WIRE)
  else:actual=rt.receive(observed,snr,meta['counter'],rt.profiles,calls,event,phase='E5_TIMING')
  from n512_phy import received_projection
  p=received_projection(actual)
  if p['status']!='SOURCE_INPUT':return gray,dict(status=p['reason'],success=False)
  bits=np.array(p['bits'],np.uint8)
  if m=='Ada':
   from c_predclass_protocol import generation_seed
   sys.path.insert(0,str(R/'experiments/adatoken-response-20261026/c_predclass_production_rx_v1'));from protocol_rx import realize
   enc,dec,cdf=ctx['c'];v=realize(n,enc,dec,cdf,bits,p['m'],p['received_class'],generation_seed(bits,p['m'],p['received_class']))
   return v['image'],dict(status=v['status'],success=v['status']=='ACCEPTED_SOURCE',_received_tokens=v['received_tokens'])
  from ep_source_codec import InvalidSourceStream
  try:v=ctx['c'].decode(p['family'],bits,p['m'],p['K'])
  except InvalidSourceStream:return gray,dict(status='SOURCE_PARSE_REJECT',success=False)
  from receiver_variants import completed_latent,decode_image
  latent,_=completed_latent(n,v['received_tokens'],p['m'],p['K']);image=decode_image(n,latent)
  return image,dict(status='ACCEPTED_SOURCE',success=True,_received_tokens=v['received_tokens'])
 actual=rt.receive(observed,snr,meta['counter'],calls,event)if m=='ARPC'else rt.receive(observed,snr,meta['counter'],calls,'timing',event)
 b=actual['body'];ok=bool(actual['header']['header_ok']and b and b['crc_accepted']and b['parser_accepted'])
 if not ok:return gray,dict(status=actual['status'],success=False)
 if m=='ARPC':
  projected=dict(accepted=True,status=actual['status'],profile=actual['profile'],payload_b64=base64.b64encode(b['payload']).decode())
  image,status=ctx['worker'].native_rx(projected,ctx['cfg'],'ARPC',n,meter)
 else:
  image,status=n.decode(base64.b64decode(b['payload_b64']),ctx['cfg']['bpg_config'],ctx['cfg']['_tmpfs']);image=gray if image is None else image
 return image,dict(status=status,success=status in ['ARPC_DECODED','BPG_DECODED'])

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 cfg=cr(r['config']);g=r['group'];role=r['direction'];out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(cfg['cpu_affinity']));beg=time.monotonic();ledger=Ledger(out/'calls.sqlite',dict(timing_trial=160));rows=[]
 (out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp');os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
 try:
  sources=cr(r['sources']);assert len(sources)==32
  if role=='tx':
   truth={(x['group'],x['source_id']):x for x in cr(r['expected_TX'])};(out/'received_symbols').mkdir();(out/'witnesses').mkdir()
  else:
   inputrows=cr(r['received_symbols']);assert len(inputrows)==160
   denied={str(Path(s['archive']['path']).resolve())for s in sources};denied.add(str((L/'E5_plan/expected_TX.json').resolve()))
   def firewall(event,args):
    if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
     p=str(Path(os.fsdecode(args[0])).resolve())
     if p in denied or '/witnesses/'in p:raise PermissionError('RX cannot read source/TX truth: '+p)
   sys.addaudithook(firewall)
  ctx=setup(cfg,g,out,role,r['pci']);t=ctx['t'];uses_gpu=g['method']!='BPG_LDPC';sync=t.cuda.synchronize if uses_gpu else lambda:None
  sync()
  if uses_gpu:assert str(t.cuda.get_device_properties(0).uuid)==r['gpu_uuid']
  save(out/'environment.json',dict(torch=t.__version__,cuda=t.version.cuda,GPU_UUID=str(t.cuda.get_device_properties(0).uuid)if uses_gpu else None,registered_UUID=r['gpu_uuid'],pci=r['pci'],CPU_only=not uses_gpu,cpu_affinity=sorted(os.sched_getaffinity(0)),threads=t.get_num_threads(),interop=t.get_num_interop_threads(),batch=1,setup_excluded=True))
  inputs=[];fingerprints={}
  for si,s in enumerate(sources):
   if role=='tx':
    with np.load(checked(s['archive']),allow_pickle=False)as z:pixels=z['pixels'].copy()
    assert pixels.dtype==np.uint8 and hashlib.sha256(pixels.tobytes()).hexdigest()==s['preprocessing_id']
   for repetition in range(5):
    if time.monotonic()-beg>=r['seconds']or(L/'STOP').exists():raise BudgetStop('E5 bounded stage')
    event=f'{si:02d}_{repetition}';phase='warmup'if repetition<2 else'measured'
    if role=='rx':
     inp=inputrows[si*5+repetition];assert inp['source_id']==s['source_id']and inp['event']==event
     with np.load(checked(inp['symbols']),allow_pickle=False)as z:observed=z['observed'].copy()
     assert asha(observed)==inp['observation_sha256']
    def trial():
     with t.no_grad():
      sync();start=time.perf_counter()
      value=tx(pixels,s['source_id'],s['source_index'],ctx)if role=='tx'else rx(observed,inp['metadata'],ctx,event)
      sync();elapsed=time.perf_counter()-start
     return value,elapsed
    value,elapsed=ledger.call('timing_trial',event,trial)
    if role=='tx':
     wave,meta,witness=value;assert wave.shape==(g['N'],2)and np.isfinite(wave).all()
     for raw,key in [('_arithmetic_bits','arithmetic_bits_sha256'),('_source_tokens','source_tokens_sha256')]:
      if raw in witness:witness[key]=asha(witness.pop(raw))
     if (g['name'],s['source_id'])in truth:
      expected_tx=truth[g['name'],s['source_id']]
      for k in ['endpoint','arithmetic_bits_sha256','predicted_class']:assert witness[k]==expected_tx[k],('Fresh TX differs',g['name'],s['source_id'],k)
     wh=asha(wave)
     if si in fingerprints:assert fingerprints[si]==wh
     else:fingerprints[si]=wh
     observed=noise(wave,meta,ctx);path=out/'received_symbols'/(event+'.npz')
     with path.open('xb')as f:np.savez(f,observed=observed)
     inputs.append(dict(event=event,source_id=s['source_id'],symbols=pin(path),observation_sha256=asha(observed),metadata=meta))
     save(out/'witnesses'/(event+'.json'),dict(witness,emitted_symbols_sha256=wh,actual_energy=float(np.square(wave.astype(np.float64)).sum())))
     identity=dict(emitted_symbols_sha256=wh,success=None)
    else:
     image,identity=value;assert image.dtype==np.float32 and image.shape==(3,256,256)and np.isfinite(image).all()and image.min()>=0 and image.max()<=1;identity['image_sha256']=asha(image)
     if '_received_tokens'in identity:identity['received_tokens_sha256']=asha(identity.pop('_received_tokens'))
     if si in fingerprints:assert fingerprints[si]==identity['image_sha256']
     else:fingerprints[si]=identity['image_sha256']
    row=dict(group=g['name'],method=g['method'],N=g['N'],snr_db=g['snr_db'],source_id=s['source_id'],source_index=s['source_index'],event=event,phase=phase,repetition=repetition,direction=role,seconds=elapsed,**identity);rows.append(row);save(out/(event+'.json'),row)
   print('E5_TIMING',g['name'],role,si+1,32,flush=True)
  save(out/'rows.json',rows)
  if role=='tx':save(out/'received_symbols.json',inputs)
  save(out/'completion.json',dict(status='COMPLETE',group=g['name'],direction=role,rows=160,warmups=64,measured=96,ledger=ledger.summary(),in_memory_calls=ctx['calls'].counts,seconds=time.monotonic()-beg,cached_scientific_calls=0,quality_trials=0,same_two_core_affinity=True,peak_GPU_allocated=t.cuda.max_memory_allocated()if uses_gpu else 0,peak_GPU_reserved=t.cuda.max_memory_reserved()if uses_gpu else 0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),completed_rows=len(rows),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
