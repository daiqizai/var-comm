"""Original DINO evaluator, only previously registered weight paths resolved."""
from study_common import *
import ast,copy
sys.path[:0]=[str(R/'experiments/entropy-partial-mismatch-20261010/scripts'),str(R/'tools'),str(R/'experiments/nestcom-n512-20261028')]
import h800_ep_pilot_metrics_v1 as original
import checkpoint_relocation_20261028 as relocation
p=Path(original.__file__);node=copy.deepcopy(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='bound_assets'))
matches=[n for n in ast.walk(node)if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='path']
assert len(matches)==1 and ast.unparse(matches[0].value)=="Path(row['actual_path'])"
matches[0].value.func=ast.Name(id='_checkpoint_path',ctx=ast.Load());ns=dict(original.__dict__);ns['_checkpoint_path']=relocation.resolve
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p)+'::CHECKED_STORAGE_RELOCATION','exec'),ns)
original.bound_assets=ns['bound_assets']
import dino_worker
dino_worker.L=L
dino_worker.main(*sys.argv[1:])
