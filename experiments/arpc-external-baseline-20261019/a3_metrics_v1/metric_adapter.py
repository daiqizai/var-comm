"""Private population metadata adaptation; original four-metric math stays exact."""
import ast,hashlib,types
SHA='6bde19df620a7dbf391f5d406daa3b4c1de2a66d051c990c26ae9a2474c1ec8d'
EDITS=[("POPULATION='original_calibration_pilot'","POPULATION='ARPC_external_common100_v1_fixed_evaluation'"),
 ('DINOv2-L only: registered original-calibration pilot objective; no holdout selection','Fixed external_common100 evaluation; no policy or checkpoint selection')]
def adapt(data):
    assert hashlib.sha256(data).hexdigest()==SHA
    original=data.decode();text=original
    for a,b in EDITS:assert text.count(a)==1;text=text.replace(a,b,1)
    reverse=text
    for a,b in reversed(EDITS):assert reverse.count(b)==1;reverse=reverse.replace(b,a,1)
    assert reverse==original
    before=ast.parse(original);after=ast.parse(text)
    for tree in [before,after]:
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                for a,b in EDITS:
                    if node.value in (a,b):node.value='POPULATION_LABEL'
    bclass=next(n for n in before.body if isinstance(n,ast.ClassDef) and n.name=='FourMetrics')
    aclass=next(n for n in after.body if isinstance(n,ast.ClassDef) and n.name=='FourMetrics')
    for name in ['validate_image','prepare','score']:
        b=next(n for n in bclass.body if isinstance(n,ast.FunctionDef) and n.name==name)
        a=next(n for n in aclass.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b)
    return text
def load(path):
    m=types.ModuleType('_arpc_fixed_eval_four_metrics');m.__file__=str(path)
    exec(compile(adapt(path.read_bytes()),str(path)+'::fixed_eval_metadata', 'exec'),m.__dict__)
    return m
