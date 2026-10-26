"""Original RX mathematics, with the registered nested-grid endpoint validator."""
from common import *
import numpy as np,copy

def endpoint(m,K):
 if type(m)is not int or type(K)is not int or(m,K)not in grids()['8']:raise ValueError('Outside registered49 endpoint grid')
 return m,K

def receiver_class(root=R):
 root=Path(root);p=root/'experiments/entropy-partial-mismatch-20261010/scripts/ep_source_codec.py'
 sys.path.insert(0,str(p.parent));import ep_source_codec as old
 from t6_source_codec import SourceCodec10
 tree=ast.parse(p.read_text());original=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='PartialSourceCodec')
 decode=next(x for x in original.body if isinstance(x,ast.FunctionDef)and x.name=='decode')
 node=ast.ClassDef(name='NestedPartial',bases=[ast.Name(id='SourceCodec',ctx=ast.Load())],keywords=[],body=[copy.deepcopy(decode)],decorator_list=[])
 namespace=dict(vars(old));namespace['endpoint']=endpoint
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<unchanged-partial-RX-with-grid-validator>','exec'),namespace)
 Base=namespace['NestedPartial']
 class Receiver(Base):
  def decode(self,family,payload,m,K=0):
   endpoint(m,K);self.cdf_trace=[]
   if m==10:
    assert family==old.WHOLE_FAMILY and K==0
    return SourceCodec10.decode(self,family,payload,m)
   return super().decode(family,payload,m,K)
  def _cdf(self,family,provider,scale):
   table=super()._cdf(family,provider,scale);self.cdf_trace.append(dict(scale=scale,cdf_sha256=asha(table)));return table
 return Receiver,dict(original=pin(p),decode_function_AST_sha256=hashlib.sha256(ast.dump(decode,include_attributes=False).encode()).hexdigest(),original_decode_body_unchanged=True,only_endpoint_validator_generalized=True,whole10_original=True)
