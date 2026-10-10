"""GPU0-only four-metric owner, after actual visual v2 closure.

The frozen metric mathematics, paired arithmetic and finite budgets are reused
in a private namespace. GPU0 is part of the actual metric runtime identity;
no GPU2 score, reconstruction or numerical-equivalence assertion is admitted.
"""
from __future__ import annotations
import ast
import inspect
import types
import h800_mismatch_metric_owner_v1 as base
import h800_mismatch_visual_v2 as visual_v2

BASE_SHA='55c3957d2b3de73051f831e243b3028f7558d802e5da43fcdd4194635c5d68ff'
base.g.require(base.g.sha(base.__file__)==BASE_SHA,'Frozen raw metric owner v1 changed')
SCHEMA='H800_RAW100_ONE_BIN_MISMATCH_METRIC_OWNER_V2_GPU0'
PASS='PASS_H800_RAW100_MISMATCH_FOUR_METRICS_GPU0'
TOOLS=visual_v2.TOOLS+('h800_mismatch_visual_v2.py','h800_ep_pilot_metric_owner_v1.py',
    'h800_ep_pilot_metrics_v1.py','h800_mismatch_metrics_v1.py','h800_mismatch_metric_owner_v1.py')
ns=dict(vars(base));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,TOOLS=TOOLS,visual=visual_v2)
for name,value in vars(base).items():
    if isinstance(value,types.FunctionType) and value.__module__==base.__name__:
        ns[name]=base.gate.source.host.clone_function(value,ns)


def admission_function(name,expected_twos,expected_paths):
    tree=ast.parse(inspect.getsource(getattr(base,name)))
    ints=[n for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) is int and n.value==2]
    paths=[n for n in ast.walk(tree) if isinstance(n,ast.Constant) and n.value=='qualification/h800_mismatch_visual_v1_attempt1']
    base.g.require(len(ints)==expected_twos and len(paths)==expected_paths,'Frozen metric admission AST changed')
    for node in ints:node.value=0
    for node in paths:node.value='qualification/h800_mismatch_visual_v2_attempt1'
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            node.value=node.value.replace('Registered single GPU2 required','Registered single GPU0 required')
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),ns)


admission_function('prepare',2,0)
admission_function('main',1,0)
admission_function('visual_closure',0,1)
original_registration=ns['registration']
def registration(path,digest,runner):
    result=original_registration(path,digest,runner)
    base.g.require(result['target_index']==0,'GPU0-only metric v2 admission required')
    return result
ns['registration']=registration
for name,value in ns.items():
    if not name.startswith('__'):globals()[name]=value


if __name__=='__main__':main()
