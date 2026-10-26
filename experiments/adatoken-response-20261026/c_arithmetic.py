"""Exact original integer arithmetic with only the independent vocabulary bound changed."""
from common import *
import copy,numpy as np
def load_codec(root=R):
 root=Path(root);files={'entropy.py':'f3adf956fba05d60e687472fa153e0bf684119ef6c8885a75fe4ea21962b2b2b','whole_entropy.py':'5d08d2f8c218f0bdf5b6d2a8ca7b3aa729baa75e274531afbfd8f31c95b24a5b'}
 nodes=[];counts={4096:0,4097:0}
 class Vocab(ast.NodeTransformer):
  def visit_Constant(self,n):
   if type(n.value)is int and n.value in counts:counts[n.value]+=1;return ast.copy_location(ast.Constant({4096:16384,4097:16385}[n.value]),n)
   return n
 for name,expected in files.items():
  p=root/'src/var_comm'/name;assert sha(p)==expected
  names={'validate_bits','validate_cdf','probability_cdf'}if name=='entropy.py'else{'ArithmeticEncoder','ArithmeticDecoder'}
  old=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names];assert len(old)==len(names)
  new=Vocab().visit(copy.deepcopy(ast.Module(body=old,type_ignores=[])))
  class Undo(ast.NodeTransformer):
   def visit_Constant(self,n):
    if type(n.value)is int and n.value in(16384,16385):return ast.copy_location(ast.Constant({16384:4096,16385:4097}[n.value]),n)
    return n
  assert ast.dump(Undo().visit(copy.deepcopy(new)),include_attributes=False)==ast.dump(ast.Module(body=old,type_ignores=[]),include_attributes=False)
  nodes+=new.body
 ns=dict(np=np,TOTAL=1<<24,FULL=1<<32,HALF=1<<31,QUARTER=1<<30);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<bound-vocabulary-adapter>','exec'),ns)
 Base=ns['ArithmeticDecoder']
 class BoundedDecoder(Base):
  def read_bit(self):
   if self.position>=len(self.bits)+30:raise ValueError('Arithmetic terminal lookahead exceeded')
   return super().read_bit()
 return ns['ArithmeticEncoder'],BoundedDecoder,ns['probability_cdf'],dict(original_files=files,constant_substitutions=counts,vocab=16384,total=1<<24,state_bits=32,inverse_AST_equal=True)
def generation_seed(bits,prefix):
 assert type(prefix)is int and 1<=prefix<=256
 b=np.asarray(bits);assert b.dtype==np.uint8 and b.ndim==1 and np.isin(b,(0,1)).all()
 return int.from_bytes(hashlib.sha256(b'LLAMAGEN_FIXEDFRAME_GENERATION_V1|'+str(prefix).encode()+b'|'+b.tobytes()).digest()[:8],'little')%(1<<31)
def canonical(decoder,encoder,bits):
 assert np.array_equal(encoder.finish(),bits),'Noncanonical arithmetic length/termination'
 assert decoder.position-len(bits)==30,'Arithmetic terminal lookahead mismatch'
