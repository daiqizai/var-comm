"""Assemble unchanged frozen original definitions; no alternative VAR/Dc math."""
from __future__ import annotations
from tb_common import *
import ast,copy,sys,types,platform
def definitions(path,names,ns):
 nodes=[n for n in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
def build(routes,ledger,boundary,out,registered_pci):
 import numpy as np,torch
 torch.set_num_threads(6);torch.set_num_interop_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.cuda.set_device(0)
 assert len(os.sched_getaffinity(0))==2 and torch.cuda.device_count()==1
 pci_ns=dict(ctypes=__import__('ctypes'));definitions(R/'experiments/entropy-partial-mismatch-20261010/scripts/leo_whole_math_v1.py',{'cuda_driver_pci_bus_id'},pci_ns)
 actual_pci=pci_ns['cuda_driver_pci_bus_id']()
 assert actual_pci.lower().lstrip('0')==registered_pci.lower().lstrip('0'),('Wrong physical CUDA device',actual_pci,registered_pci)
 props=torch.cuda.get_device_properties(0);torch.cuda.set_per_process_memory_fraction((12<<30)/props.total_memory)
 observed=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
 expected=routes['native_qualification']['numeric_settings']['runtime'];assert observed==expected,(observed,expected)
 paths={}
 for p in routes['files']:
  checked(p)
  if p['path'].endswith('vae_ch160v4096z32.pth'):paths['vae_checkpoint']=p['path']
  elif p['path'].endswith('var_d16.pth'):paths['var_checkpoint']=p['path']
  else:paths['decoder_checkpoint']=p['path']
 for p in routes['author_files']:checked(p)
 checked(routes['dist']);paths['var_source']=routes['author_root']
 ns=dict(torch=torch,Path=Path,sys=sys,np=np,hashlib=hashlib,PATCH_NUMS=SIZES,VOCAB_SIZE=4096)
 definitions(R/'src/var_comm/next_scale_prior.py',{'load_models','state_sha256'},ns)
 vae,var=ns['load_models'](paths,torch.device('cuda:0'))
 dn=dict(torch=torch,nn=torch.nn,copy=copy);definitions(R/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py',{'ContinuousDecoder'},dn)
 checkpoint=torch.load(paths['decoder_checkpoint'],map_location='cpu',weights_only=True)
 decoder=dn['ContinuousDecoder'](vae).to('cuda:0');decoder.load_state_dict(checkpoint['decoder'],strict=True);del checkpoint
 models=dict(vae=vae,var=var,decoder=decoder)
 for m in models.values():m.eval().requires_grad_(False)
 states={k:ns['state_sha256'](v) for k,v in models.items()};assert states==MODELS,('BLOCKED_MODEL_STATE',states)
 pn=dict(torch=torch,np=np,SIZES=SIZES);definitions(R/'experiments/scale-causal-partial-residual-20261002/partial_receiver.py',{'_Prior'},pn)
 OriginalPrior=pn['_Prior'];counts=dict(prior=0,decoder=0)
 class CountedPrior(OriginalPrior):
  def logits(self,k):
   boundary();counts['prior']+=1
   return ledger.call('prior_scale',str(counts['prior']),lambda:super(CountedPrior,self).logits(k))
 class CountedDecoder:
  def __call__(self,latent):
   counts['decoder']+=1;return ledger.call('decoder_forward',str(counts['decoder']),lambda:decoder(latent))
 flags=dict(deterministic=True,matmul_tf32=False,cudnn_tf32=False,precision='highest')
 n=types.SimpleNamespace(torch=torch,receiver=types.SimpleNamespace(_Prior=CountedPrior),loaded=dict(vae=vae,var=var,decoder=CountedDecoder(),device=torch.device('cuda:0'),identity=dict(models=states)),flags=flags)
 hp=dict(np=np,time=time,SIZES=SIZES,require=lambda ok,msg:None if ok else (_ for _ in ()).throw(RuntimeError(msg)),token_count=lambda m,K=0:OFFSETS[m]+K)
 definitions(R/'experiments/content-real-64qam-20261006/h_source_driver.py',{'render_received'},hp)
 save(out/'actual_native_identity.json',dict(models=states,runtime=observed,numeric_flags=routes['numeric_flags'],GPU_name=props.name,actual_CUDA_UUID=str(props.uuid),actual_PCI_bus_id=actual_pci,registered_PCI_bus_id=registered_pci,visible_GPU=os.environ['CUDA_VISIBLE_DEVICES'],allocator_cap_bytes=12<<30,cpu_affinity=sorted(os.sched_getaffinity(0)),source_mathematics='Unchanged SHA-bound original function/class definitions',sampling='unconditional_class1000_greedy_no_CFG',training_updates=0))
 return n,hp['render_received']
