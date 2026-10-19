"""Same four metric operations; only fixed evaluation population metadata changes."""
import ast,hashlib,types
from common import PE,PO,read
SHA='6bde19df620a7dbf391f5d406daa3b4c1de2a66d051c990c26ae9a2474c1ec8d'
def load(path):
    old=PE/'a3_metrics_v1/metric_adapter.py';reg=read(PO/'a3_metrics_v1/registration.json');pin=next(d for d in reg['source_files']if d['path']==str(old));raw=old.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==pin['sha256']and len(raw)==pin['bytes']
    namespace={'__name__':'_qualified_arpc_metric_adapter'};exec(compile(raw,str(old),'exec'),namespace)
    text=namespace['adapt'](path.read_bytes())
    text=text.replace('ARPC_external_common100_v1_fixed_evaluation','ARPC_common500_remaining400_fixed_evaluation').replace('Fixed external_common100 evaluation','Fixed common500 remaining400 evaluation')
    # Population labels do not occur in any scoring/preprocessing operation.
    before=ast.parse(namespace['adapt'](path.read_bytes()));after=ast.parse(text)
    for name in ('validate_image','prepare','score'):
        nodes=[]
        for t in (before,after):
            cls=next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name=='FourMetrics')
            nodes.append(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name==name))
        assert ast.dump(nodes[0])==ast.dump(nodes[1])
    m=types.ModuleType('_arpc_common500_four_metrics');m.__file__=str(path);exec(compile(text,str(path)+'::common500_population','exec'),m.__dict__);return m
