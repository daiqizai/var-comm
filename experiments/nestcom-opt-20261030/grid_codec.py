"""Only the permitted endpoint grammar expands; arithmetic/CDF math is original."""
from opt_common import *
import ast,types
def endpoint(m,K):
 assert type(m)is int and type(K)is int and 4<=m<=10
 assert (m==10 and K==0)or(m<10 and 0<=K<SIZES[m]**2)
 return m,K
def endpoints(grid):
 if grid=='token':return tuple((m,K)for m in range(4,10)for K in range(SIZES[m]**2))+((10,0),)
 den={'quarter':4,'eighth':8}[grid]
 return tuple((m,K)for m in range(4,10)for K in sorted({0,*(SIZES[m]**2*j//den for j in range(1,den))}))+((10,0),)
def codec(native):
 sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'))
 import kn_common
 import ep_source_codec as original
 from t6_source_codec import SourceCodec10
 p=R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_source_codec.py'
 tree=ast.parse(p.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='PartialSourceCodec')
 ns=dict(original.__dict__);ns.update(endpoint=endpoint,token_count=lambda m,K:OFFSETS[m]+K)
 # The entire original class body is preserved byte-for-AST; only its two
 # endpoint validation/count helpers are resolved in this private namespace.
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(p)+'::OPT_TOKEN_GRAMMAR', 'exec'),ns)
 OriginalExpanded=ns['PartialSourceCodec']
 class Receiver(OriginalExpanded):
  def decode(self,family,payload,m,K=0):
   if m==10:
    assert family=='EC_VAR_WHOLE'and K==0
    return SourceCodec10.decode(self,family,payload,m)
   return super().decode(family,payload,m,K)
 return Receiver(R,native=native)
