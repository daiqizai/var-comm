"""Actual source signatures, stub bodies; no backbone or image inference."""
from common import *
import ast,copy,sys
routes=read(O/'assets/model_routes.json');metric=checked(routes['metric_config']['sources']['metric_models.py']);tree=ast.parse(metric.read_text(encoding='utf8'))
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MetricEvaluator');nodes=[]
for name in ('prepare_reference','score'):
 n=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name))
 n.decorator_list=[];n.body=[ast.Return(ast.Constant(value=None))];nodes.append(n)
c=ast.ClassDef(name='SignatureOnly',bases=[],keywords=[],body=nodes,decorator_list=[]);ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[c],type_ignores=[])),str(metric)+' [signature-only]', 'exec'),ns)
v=ns['SignatureOnly']();ref=object();reconstruction=object();v.prepare_reference(ref);v.score(reference=ref,reconstruction=reconstruction,labels=[0],label_conditioned=False,prepared={})
try:v.prepare_reference(ref,[0],label_conditioned=False)
except TypeError:pass
else:raise AssertionError('Old failure must be reproduced at argument binding only')
score=ast.parse((E/'score_completion.py').read_text());call=next(n for n in ast.walk(score) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='score')
assert {k.arg:ast.unparse(k.value) for k in call.keywords}==dict(reference='ref',reconstruction='reconstruction',labels="[s['evaluation_class_index']]",label_conditioned='False',prepared='prepared[sid]')
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
print(canonical(dict(status='PASS_ORIGINAL_METRIC_API_SIGNATURE_AND_PAIRED_ORDER',metric_source=pin(metric),checks=5,model_imports=[],model_forward_calls=0,scientific_calls=0,old_failure_reproduced_before_function_body=True)),flush=True)
