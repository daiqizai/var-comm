"""Existing NeST mathematics with a catalog-only complete-m10 extension."""
from common import *
import sys,types,copy
sys.path[:0]=[str(E),str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/entropy-partial-mismatch-20261010/scripts'),str(R/'experiments/wcl-evidence-closure-20261009/scripts'),str(R/'src')]
import ep_source_codec as ep
from t6_source_codec import SourceCodec10
def endpoint(m,K):
 if (m,K)==(10,0) and type(m)is int and type(K)is int:return m,K
 return ep.endpoint(m,K)
def token_count(m,K):endpoint(m,K);return OFFSETS[m]+K
tree=ast.parse(pathlib.Path(ep.__file__).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='PartialSourceCodec')
node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='encode_endpoints')
namespace=dict(vars(ep),endpoint=endpoint,token_count=token_count)
# Same AST, same arithmetic/probability code. Only the endpoint catalog gains(10,0).
exec(compile(ast.Module(body=[node],type_ignores=[]),ep.__file__+'::m10_catalog_only','exec'),namespace)
class EndpointCodec(ep.PartialSourceCodec):
 encode_endpoints=namespace['encode_endpoints']
 def decode(self,family,bits,m,K=0):
  if m==10:
   assert family=='EC_VAR_WHOLE' and K==0
   return SourceCodec10.decode(self,family,bits,10)
  return super().decode(family,bits,m,K)
class Counter:
 def __init__(self):self.counts={}
 def call(self,kind,key,fn):self.counts[kind]=self.counts.get(kind,0)+1;return fn()
def load_runtime(reg,out,boundary,pci):
 import tb_native,torch
 counter=Counter();n,render=tb_native.build(read(checked(reg['model_routes'])),counter,boundary,out,pci)
 capture={};original=n.loaded['decoder']
 class Capture:
  def __call__(self,z):capture['latent']=z.detach().float().cpu().clone();return original(z)
 n.loaded['decoder']=Capture()
 return n,render,EndpointCodec(R,native=n),capture,counter
def load_lpips(reg):
 import torch,lpips
 m=lpips.LPIPS(net='alex',pnet_rand=True,model_path=str(checked(reg['lpips_linear'])),verbose=False)
 w=torch.load(checked(reg['alexnet']),map_location='cpu',weights_only=True)
 m.net.load_state_dict({n:w['features.'+n.split('.',1)[1]] for n in m.net.state_dict()},strict=True)
 m=m.cuda().eval().requires_grad_(False)
 ns=dict(torch=torch,hashlib=hashlib);definitions(R/'src/var_comm/next_scale_prior.py',{'state_sha256'},ns)
 assert ns['state_sha256'](m)==reg['lpips_state_sha256'];return m
