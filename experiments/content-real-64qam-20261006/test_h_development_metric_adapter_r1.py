"""R1 CPU fixtures: same frozen adapter assertions, explicitly bound module paths.

Only the two historical-location assumptions are replaced. The original test
file and all scientific code remain byte-identical. There is no path fallback.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
import numpy as np
import h_development_metric_adapter as m
import test_h_development_metric_adapter as original
from test_h_development_metric_adapter import Tensor, nullcontext

ORIGINAL_TEST_SHA='46db0b0145080a4e8d644d61e78c86b14f888e58037cc0fec9584e6b876a4e25'
MODULE_SHA={
    'replay_module':'9582ce2f89c430369b7b19fd7810d1f90e56240f6d7b9845506fc57ded00bcbd',
    'validation_module':'05f1b82cd5186ce047fe377ccc79975f5986cfe6097e4670d312b7a65f3dc6dc',
}
ASSET_KEYS={'suite_module','validation_module','replay_module','metrics_registration','modelmanifest',
    'models_qualification','independent_binding','convnext_weights'}


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:raise RuntimeError(message)


def bound_fixture_module(name):
    require(name in MODULE_SHA,'Only two registered historical fixture modules are allowed')
    require(file_sha(original.__file__)==ORIGINAL_TEST_SHA,'Original adapter test source changed')
    path=os.environ.get('H_METRIC_TEST_ASSETS','');expected=os.environ.get('H_METRIC_TEST_ASSETS_SHA256','')
    require(path and Path(path).is_absolute() and Path(path).is_file() and not Path(path).is_symlink(),
        'Explicit absolute metric asset manifest required; no fallback')
    require(re.fullmatch(r'[0-9a-f]{64}',expected) is not None and file_sha(path)==expected,
        'Exact metric asset manifest SHA required')
    value=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    require(value.get('status')=='H_ORIGINAL_METRIC_ASSETS_BOUND' and set(value.get('paths',{}))==ASSET_KEYS,
        'Original metric asset manifest schema required')
    bound={}
    for key in ('source_bindings','input_bindings'):
        require(isinstance(value.get(key),dict),'Explicit metric asset bindings required')
        for p,s in value[key].items():
            require(p not in bound or bound[p]==s,'Metric source/input binding conflict')
            bound[p]=s
    p=Path(value['paths'][name])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(),'Bound fixture file missing')
    require(bound.get(str(p))==MODULE_SHA[name] and file_sha(p)==MODULE_SHA[name],
        'Historical fixture source SHA differs')
    return p


class AdapterTests(original.AdapterTests):
    def test_original_derangement_code_exact_source_and100population(self):
        path=bound_fixture_module('replay_module')
        tree=ast.parse(path.read_text(encoding='utf-8'))
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ReplayEngine')
        fn=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_derangement'));fn.decorator_list=[]
        namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
        replay=SimpleNamespace(__file__=str(path),ReplayEngine=SimpleNamespace(_derangement=namespace['_derangement']))
        ids=[f's{i}' for i in range(100)];got=m.frozen_mismatch(replay,ids)
        self.assertEqual(got['permutation'],namespace['_derangement']());self.assertEqual(got['seed'],20260930)
        with self.assertRaises(RuntimeError):m.frozen_mismatch(replay,ids[:99])

    def test_batch_one_actual_suite_and_independent_API_bridge_with_fakes(self):
        calls=[];flags={'threads':6,'matmul_tf32':False}
        def cosine(a,b,dim):
            return Tensor((a.a*b.a).sum(axis=dim)/(np.linalg.norm(a.a,axis=dim)*np.linalg.norm(b.a,axis=dim)))
        torch=SimpleNamespace(from_numpy=Tensor,inference_mode=nullcontext,autocast=lambda **k:nullcontext(),
            nn=SimpleNamespace(functional=SimpleNamespace(cosine_similarity=cosine)))
        metric=SimpleNamespace(no_network=nullcontext,preprocess_resnet50=lambda x:x)
        class Eval:
            device=SimpleNamespace(type='cpu');metadata={'metrics':{'clip':{'status':'READY'}}}
            models={'resnet50':lambda image:Tensor([[0,1,2]])}
            def identity(self):return m.identity(self.metadata)
            def prepare_reference(self,x):calls.append(('prepare',x.a.shape));return {'resnet50':Tensor([1])}
            def score(self,ref,image,labels,label_conditioned,prepared):
                calls.append(('score',image.a.shape,labels,label_conditioned))
                d={k:.4 for k in ('clip_image_cosine','dists','dreamsim','dinov2_vitl14_cosine','ms_ssim')}
                d.update(resnet50_prediction=2,resnet50_source_prediction=1,resnet50_top1_label=False,
                    resnet50_top1_source_prediction=False,resnet50_source_top1_label=True,label_conditioned=False);return [d]
        def quality(ref,images,*args):
            self.assertEqual(len(images),1);calls.append(('legacy',1));return [dict(psnr_db=6.02,lpips_alex=.2,dino_cosine=0.)],np.array([1,0],np.float32),np.array([[0,1]],np.float32)
        native=SimpleNamespace(dino_features=lambda d,x:Tensor(np.array([[1,0]],np.float32)),quality_metrics=quality)
        validation=bound_fixture_module('validation_module')
        sp=importlib.util.spec_from_file_location('h_test_validation',validation);v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
        class Conv:
            identity=dict(weights_sha256=m.CONVNEXT_SHA,classification_batch_size=1,used_for_selection=False)
            def predict(self,rgb):calls.append(('conv-predict',rgb.shape));return 1
            def score(self,rgb,*,true_label,source_prediction):
                calls.append(('conv-score',rgb.shape));return dict(v.diagnostics(true_label,source_prediction,2),convnext_used_for_selection=False)
        b=m.SuiteBackend((Eval(),metric,native,None,None,{},flags),Conv(),torch,lambda _:dict(flags))
        ref=np.zeros((3,256,256),np.float32);prepared=b.prepare(ref)
        values=b.score(ref,np.full_like(ref,.5),1,prepared,np.array([0,1],np.float32))
        self.assertEqual(values['dino_mismatched'],1.);self.assertTrue(values['convnext_source_correct_to_wrong'])
        self.assertIn(('score',(1,3,256,256),[1],False),calls);self.assertIs(values['label_conditioned'],False)
        b.evaluator.metadata['changed']=True
        with self.assertRaisesRegex(RuntimeError,'identity changed'):b.check()


if __name__=='__main__':unittest.main()
