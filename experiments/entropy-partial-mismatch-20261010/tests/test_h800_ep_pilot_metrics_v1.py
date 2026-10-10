"""CPU-only AST, four-metric pairing and finite-budget checks; no real models."""
import ast
from contextlib import nullcontext
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_pilot_metrics_v1 as m
import h800_ep_pilot_phy_v1 as physical
import leo_whole_gate_v1 as g

QUALITY='''
def load_quality_models(paths, device):
    linear_weights = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
    if not linear_weights.is_file():
        raise FileNotFoundError(linear_weights)
    perceptual = lpips.LPIPS(net="alex", pnet_rand=True, model_path=str(linear_weights), verbose=False)
    alex_weights = torch.load(paths["alexnet_checkpoint"], map_location="cpu", weights_only=True)
    translated = {name: alex_weights["features." + name.split(".", 1)[1]] for name in perceptual.net.state_dict()}
    perceptual.net.load_state_dict(translated, strict=True)
    del alex_weights, translated
    dino = forbidden_extra_model()
    return perceptual.to(device).eval().requires_grad_(False), dino, linear_weights
def quality_metrics(source, images, perceptual, dino, device):
    perceptual_values = perceptual(reconstructed * 2 - 1, reference * 2 - 1).reshape(-1)
'''
PSNR='''
def original_run():
    mse = float(np.square(image.astype(np.float64) - target.astype(np.float64)).mean())
    require(mse > 0 and math.isfinite(mse), 'Unexpected infinite BPG PSNR; explicit statistics adaptation needed')
    metric = dict(psnr_db=float(-10 * math.log10(mse)))
'''


class MetricTests(unittest.TestCase):
    def test_registered_metric_caps_match_predeclared_PHY_request(self):
        self.assertEqual(m.CAPS,physical.METRIC_CAPS)
        self.assertEqual(m.CAPS['model_constructions'],3)
        self.assertEqual(m.FLAGS['threads'],6);self.assertEqual(m.FLAGS['interop_threads'],2)
        self.assertFalse(m.FLAGS['matmul_tf32']);self.assertTrue(m.FLAGS['deterministic'])

    def test_metric_cache_never_reuses_gray_pair_across_different_reference(self):
        gray=np.full((3,256,256),.5,np.float32);a=np.zeros_like(gray);b=np.ones_like(gray)
        self.assertNotEqual(m.pair_cache_key(a,gray,'a'*64),m.pair_cache_key(b,gray,'a'*64))
        self.assertNotEqual(m.pair_cache_key(a,gray,'a'*64),m.pair_cache_key(a,gray,'b'*64))
        self.assertEqual(m.pair_cache_key(a,gray,'a'*64),m.pair_cache_key(a.copy(),gray.copy(),'a'*64))

    def test_PSNR_extract_preserves_float64_precision(self):
        with mock.patch.object(m,'source_ast',return_value=ast.parse(PSNR)):
            score,proof=m.psnr_projection('bound_original.py')
        target=np.zeros((3,256,256),np.float32);image=np.full_like(target,.1)
        expected=-10*math.log10(float(np.square(image.astype(np.float64)-target.astype(np.float64)).mean()))
        self.assertEqual(score(target,image),expected)
        self.assertNotEqual(score(target,image),float(-10*np.log10(np.square(image-target).mean())))
        self.assertEqual(len(proof['statement_asts']),3)
        with self.assertRaises(RuntimeError):score(target,target)

    def test_PSNR_ambiguous_original_expression_is_rejected(self):
        with mock.patch.object(m,'source_ast',return_value=ast.parse(PSNR+PSNR.replace('original_run','second'))),\
             self.assertRaisesRegex(RuntimeError,'Unique released'):
            m.psnr_projection('ambiguous.py')

    def test_LPIPS_keeps_original_prefix_and_pair_order_without_DINO_constructor(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'weights/v0.1').mkdir(parents=True);(root/'weights/v0.1/alex.pth').write_bytes(b'fixture')
            calls=[]
            class Perceptual:
                net=types.SimpleNamespace(state_dict=lambda:{'slice1.0.weight':None},
                    load_state_dict=lambda values,strict:calls.append(('weights',values,strict)))
                def to(self,d):calls.append(('device',d));return self
                def eval(self):return self
                def requires_grad_(self,value):calls.append(('requires_grad',value));return self
            def build(**kwargs):calls.append(('constructor',kwargs));return Perceptual()
            namespace=dict(Path=Path,lpips=types.SimpleNamespace(__file__=str(root/'__init__.py'),LPIPS=build),
                torch=types.SimpleNamespace(load=lambda *a,**k:{'features.0.weight':'frozen'}),
                forbidden_extra_model=lambda:(_ for _ in ()).throw(AssertionError('DINO-S was loaded')))
            with mock.patch.object(m,'source_ast',return_value=ast.parse(QUALITY)):
                loader,pair,proof=m.lpips_projection('quality.py',namespace)
            loader(dict(alexnet_checkpoint='actual_alexnet'),'cuda:0')
            self.assertEqual(calls[0][1]['net'],'alex');self.assertTrue(calls[0][1]['pnet_rand'])
            self.assertEqual(calls[1],('weights',{'slice1.0.weight':'frozen'},True))
            self.assertEqual(len(proof['kept_prefix_ast']),7)
            x=np.array([.2,.3]);y=np.array([.7,.8]);observed=[]
            pair(lambda a,b:(observed.append((a,b)) or np.ones(1)),x,y)
            self.assertTrue(np.array_equal(observed[0][0],x*2-1));self.assertTrue(np.array_equal(observed[0][1],y*2-1))

    def test_ConvNeXt_private_admission_cannot_impersonate_holdout(self):
        tree=ast.parse('class ConvNeXtValidation:\n    def __init__(self,stage,path,digest):\n        admission(stage,path,digest)\n')
        policy=dict(path='frozen_policy.json',sha256=m.core.link.POLICY_SHA)
        with mock.patch.object(m,'source_ast',return_value=tree),mock.patch.object(m.core,'sha',return_value=m.core.link.POLICY_SHA):
            cls,proof=m.convnext_projection('validation.py',policy)
            cls(m.POPULATION,policy['path'],policy['sha256'])
            with self.assertRaises(RuntimeError):cls('holdout',policy['path'],policy['sha256'])
        self.assertFalse(proof['holdout_impersonated'])

    def test_four_scores_are_source_prediction_agreement_and_correct_DINO_L_column(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=g.Ledger(Path(td)/'calls',lambda:None,m.CAPS)
            obj=m.FourMetrics.__new__(m.FourMetrics);obj.ledger=ledger;obj.identity='a'*64;obj.device='cuda:0'
            obj.torch=types.SimpleNamespace(inference_mode=nullcontext,autocast=lambda **k:nullcontext())
            obj.validate_image=lambda a:a[None];obj.psnr=lambda a,b:20.
            obj.lpips_pair=lambda *args:np.array([.25],np.float32);obj.perceptual=object()
            obj.evaluator=types.SimpleNamespace(score=lambda *a,**k:[dict(dinov2_vitl14_cosine=.75,dino_cosine=-1.)])
            obj.classifier=types.SimpleNamespace(predict=lambda image:8)
            target=np.zeros((3,256,256),np.float32);image=np.ones_like(target)
            prepared=dict(reference_sha256=m.array_sha(target),evaluator_identity=obj.identity,dino={},convnext_prediction=8)
            result=obj.score(target,image,999,prepared)
            self.assertEqual(result['convnext_top1_source_prediction'],1)
            self.assertEqual(result['dinov2_vitl14_cosine'],.75)
            self.assertEqual(set(result),set(m.core.METRICS))
            self.assertEqual(ledger.completed['image_scores'],1)
            self.assertEqual(ledger.completed['dinov2_vitl14_reconstruction'],1)

    def test_reference_identity_mismatch_stops_with_unresolved_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            obj=m.FourMetrics.__new__(m.FourMetrics);obj.ledger=g.Ledger(Path(td)/'calls',lambda:None,m.CAPS)
            obj.identity='a'*64;obj.validate_image=lambda a:a[None]
            a=np.zeros((3,256,256),np.float32)
            with self.assertRaisesRegex(RuntimeError,'Reference cache'):
                obj.score(a,a,0,dict(reference_sha256='wrong',evaluator_identity=obj.identity))
            self.assertEqual(obj.ledger.summary()['unresolved'],1)
            self.assertEqual(obj.ledger.completed['image_scores'],0)


if __name__=='__main__':unittest.main()
