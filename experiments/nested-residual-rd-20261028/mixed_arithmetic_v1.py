"""Alphabet-width adapter; interval arithmetic/termination remain frozen verbatim."""
import ast
from pathlib import Path
import numpy as np
FULL=1<<32
HALF=1<<31
QUARTER=1<<30
TOTAL=1<<24

def validate_cdf(cdf,rows=None):
    a=np.asarray(cdf)
    if a.ndim!=2 or a.shape[1]not in(3,4097)or(rows is not None and len(a)!=rows):raise ValueError('Mixed CDF shape')
    if not np.issubdtype(a.dtype,np.integer):raise ValueError('CDF must be integer')
    a=a.astype(np.int64,copy=False)
    if np.any(a[:,0]!=0)or np.any(a[:,-1]!=TOTAL)or np.any(np.diff(a,axis=1)<=0):raise ValueError('Invalid positive CDF')
    return a

def validate_bits(bits,name='bits'):
    a=np.asarray(bits)
    if a.ndim!=1 or not np.isin(a,(0,1)).all():raise ValueError(name)
    return a.astype(np.uint8,copy=False)

def classes(path):
    tree=ast.parse(Path(path).read_text());nodes=[x for x in tree.body if isinstance(x,ast.ClassDef)and x.name in('ArithmeticEncoder','ArithmeticDecoder')]
    assert len(nodes)==2
    class Width(ast.NodeTransformer):
        def __init__(self):self.count=0
        def visit_Constant(self,node):
            if node.value==4096:
                self.count+=1
                return ast.copy_location(ast.parse('cdf.shape[1]-1',mode='eval').body,node)
            return node
    adapter=Width();module=adapter.visit(ast.Module(body=nodes,type_ignores=[]));assert adapter.count==2
    ns=dict(np=np,FULL=FULL,HALF=HALF,QUARTER=QUARTER,TOTAL=TOTAL,validate_cdf=validate_cdf,validate_bits=validate_bits)
    exec(compile(ast.fix_missing_locations(module),str(path)+'[mixed-width-adapter]','exec'),ns)
    Encoder=ns['ArithmeticEncoder'];Decoder=ns['ArithmeticDecoder']
    class BoundedDecoder(Decoder):
        def read_bit(self):
            if self.position>=len(self.bits)+30:raise ValueError('Terminal lookahead exceeded')
            return super().read_bit()
    return Encoder,BoundedDecoder
