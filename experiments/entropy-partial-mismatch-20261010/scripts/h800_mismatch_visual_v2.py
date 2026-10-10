"""Explicit GPU0 admission for unchanged raw mismatch reconstruction mathematics.

GPU2's pre-launch free-memory refusal involved no scientific call. This separate
owner uses the already observed GPU0 identity, retains16GiB and all original call
caps, and admits no image from GPU2 or another host. The frozen v1 is unchanged.
"""
from __future__ import annotations
import ast
import inspect
from pathlib import Path
import types
import h800_mismatch_visual_v1 as base

BASE_SHA='de547876187cf8e69a68b964f8e8170a0326ea126355e85d583640a89d15e59b'
base.g.require(base.g.sha(base.__file__)==BASE_SHA,'Frozen raw visual v1 changed')
SCHEMA='H800_RAW100_ONE_BIN_MISMATCH_VISUAL_V2_GPU0'
PASS='PASS_H800_RAW100_MISMATCH_RECONSTRUCTIONS_ONLY_GPU0'
ns=dict(vars(base));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,
    TOOLS=base.TOOLS+('h800_mismatch_visual_v1.py',))
for name,value in vars(base).items():
    if isinstance(value,types.FunctionType) and value.__module__==base.__name__:
        ns[name]=base.gate.source.host.clone_function(value,ns)


def device_only_function(name,expected_twos):
    tree=ast.parse(inspect.getsource(getattr(base,name)))
    constants=[n for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) is int and n.value==2]
    base.g.require(len(constants)==expected_twos,'Frozen GPU2 admission AST changed')
    for node in constants:node.value=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            node.value=node.value.replace('Registered shared single GPU2 required','Registered shared single GPU0 required')
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),ns)


device_only_function('prepare',2)
device_only_function('main',1)
original_registration=ns['registration']
def registration(path,digest,runner):
    result=original_registration(path,digest,runner)
    base.g.require(result['target_index']==0,'GPU0-only visual v2 admission required')
    return result
ns['registration']=registration
for name,value in ns.items():
    if not name.startswith('__'):globals()[name]=value


if __name__=='__main__':main()
