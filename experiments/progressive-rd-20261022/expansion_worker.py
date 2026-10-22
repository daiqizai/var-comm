"""Closed seed science, two output paths, and an indexed accounting ledger."""
from common import *
import argparse,traceback,seed_worker_v2

class IndexedLedger(Ledger):
    def __init__(self,path):super().__init__(path);self.keys=set()
    def call(self,kind,key,fn):
        assert (kind,key)not in self.keys,(kind,key)
        self.keys.add((kind,key));item=dict(kind=kind,key=key,status='STARTED',started_unix=time.time());self.calls.append(item);self.event(item)
        start=time.perf_counter()
        try:value=fn()
        except BaseException as e:
            self.event(dict(item,status='FAILED',error=repr(e),elapsed_seconds=time.perf_counter()-start));raise
        item.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-start);self.event(item);return value

def bind_engine():
    tree=ast.parse((E/'seed_worker_v2.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    replacements={'seed_registration_v2.json':'expansion_registration.json','seed_cache_v2':'expansion_cache_v1'}
    counts={k:0 for k in replacements}
    class Paths(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str)and n.value in replacements:
                counts[n.value]+=1
                return ast.copy_location(ast.Constant(replacements[n.value]),n)
            return n
    node=Paths().visit(node);assert counts=={'seed_registration_v2.json':2,'seed_cache_v2':1},counts
    ns=dict(vars(seed_worker_v2),Ledger=IndexedLedger);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(E/'seed_worker_v2.py')+'::expansion_paths_only','exec'),ns)
    return ns['main']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration-sha256',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    try:bind_engine()(a.registration_sha256,a.shard)
    except BaseException as e:
        d=L/'expansion_cache_v1'/f'shard_{a.shard:02d}';d.mkdir(parents=True,exist_ok=True)
        if not(d/'failure.json').exists():save(d/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),process=process(os.getpid())))
        raise
