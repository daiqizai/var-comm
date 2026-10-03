"""Asset-free engineering regressions; no real model/GPU acceptance claimed."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

import historical_selected_optional as s


def rows(methods, snrs):
    return [dict(method=m, source_index=i, source_id='image'+str(i),
                 snr_db=v, noise_seed=k, N=4084, E=8168,
                 psnr_db=20., lpips_alex=.2, dino_cosine=.7)
            for i in range(100) for m in methods for v in snrs for k in s.SEEDS]


class ScopeTests(unittest.TestCase):
    def test_only_requested_points_retained(self):
        allrows = rows(('B2','A1_policy','A2_policy','V_policy','V_lambda1','V_tok'), (-5,-2,1,4,13))
        chosen = s.selected_rows(allrows, ('B2','A1_policy','A2_policy','V_policy'), (-5,-2))
        self.assertEqual(len(chosen),2400)
        self.assertEqual({x['snr_db'] for x in chosen},{-5,-2})
        self.assertNotIn('V_lambda1',{x['method'] for x in chosen})

    def test_duplicate_and_missing_keys_rejected(self):
        data = rows(s.H6,(13,))
        with self.assertRaisesRegex(ValueError,'incomplete or duplicated'):
            s.selected_rows(data[:-1],s.H6,(13,))
        with self.assertRaisesRegex(ValueError,'incomplete or duplicated'):
            s.selected_rows(data+[data[0]],s.H6,(13,))

    def test_extra_training_seeds_not_admitted(self):
        data=rows(s.H6+('H6-V_N4084_seed2026092404',),(1,4,7,13,19))
        chosen=s.selected_rows(data,s.H6,(13,))
        self.assertEqual(len(chosen),600)
        self.assertEqual({r['method'] for r in chosen},set(s.H6))

    def test_true_bypass_has_no_candidate(self):
        level={'methods':{p:dict(policy_action='BYPASS',raw_selected_lambda=.25) for p in ('A1','A2','V')}}
        self.assertEqual(s.policy_plan(level,'P_low'),
                         [('P_low',None),('A1_policy',None),('A2_policy',None),('V_policy',None)])
        level['methods']['V']['policy_action']='CORRECT'
        self.assertEqual(s.policy_plan(level,'B2')[-1],('V_policy',('V',.25)))

    def test_unknown_policy_action_rejected(self):
        level={'methods':{p:dict(policy_action='BYPASS',raw_selected_lambda=.25) for p in ('A1','A2','V')}}
        level['methods']['V']['policy_action']='DEVELOPMENT_SELECTED'
        with self.assertRaisesRegex(ValueError,'unknown frozen action'):
            s.policy_plan(level,'P_low')

    def test_original_float_alias_exact_not_similar(self):
        a=rows(('original_1024_Dc',),(1,))[0]
        b={**a,'method':'m8_plus_latent_1024'}
        s.assert_alias(a,b)
        for field,value in [('noise_seed',2002),('source_id','other'),('N',3572),('lpips_alex',.20000000001)]:
            wrong={**b,field:value}
            with self.subTest(field=field),self.assertRaises(ValueError):s.assert_alias(a,wrong)

    def test_lookup_cannot_silently_add_endpoint(self):
        actions={str(float(v)):('m8_plus_latent_512_fold_N4084' if v==1 else 'm8_plus_latent_1024') for v in (1,4,7,13,19)}
        s.validate_lookup_actions({'actions':actions})
        actions['4.0']='raw_N4084_m8_Dc'
        with self.assertRaisesRegex(ValueError,'lookup actions differ'):
            s.validate_lookup_actions({'actions':actions})

    def test_h6_execution_never_visits_excluded_points_or_models(self):
        # A deliberately failing endpoint for anything but the authorised two
        # models/13dB catches late CSV filtering after broad native inference.
        chosen=s.selected_rows(rows(s.H6,(13,)),s.H6,(13,))
        image=np.ones((3,256,256),np.float32)*.25
        calls=[]; batches=[]
        def execute(record,model,meta,snr,seed,*args):
            self.assertIn(model,s.H6);self.assertEqual(snr,13)
            calls.append((model,snr,seed))
            return image,dict(ledger={},rx={},waveform_sha256='wave',observation_sha256='obs')
        def quality(target, images, *args):
            self.assertEqual(len(images),15)
            self.assertEqual([j for j,v in enumerate(images) if np.array_equal(v,image)], [9,10,11])
            self.assertTrue(all(np.array_equal(v,target) for j,v in enumerate(images) if j not in (9,10,11)))
            batches.append([len(images[:8]),len(images[8:])])
            return ([dict(original_slot=j) for j in range(15)],)
        n=types.SimpleNamespace(ready=True,torch=types.SimpleNamespace(no_grad=nullcontext),
            safe=types.SimpleNamespace(check=lambda:None),records=[dict(pixels=np.zeros((3,256,256),np.uint8))]*100,
            native=types.SimpleNamespace(execute=execute,quality_metrics=quality),
            models={m:m for m in s.H6},model_metadata={m:{} for m in s.H6},
            vae=None,var=None,decoder=None,device=None,lp=None,dino=None,
            metadata=lambda row:{'original_location':{'test':'synthetic'}})
        view=object.__new__(s.H6View);view.native=n;view.study='OPTIONAL_H6_13DB';view._inventory(chosen)
        def parity(original, actual):
            self.assertEqual(actual['original_slot'],9+s.SEEDS.index(original['noise_seed']))
            return {'synthetic':True}
        with patch.object(s.budget,'parity_check',side_effect=parity):
            outputs=list(view.iterate_source(0))
        self.assertEqual(len(outputs),6)
        self.assertEqual(calls,[(m,13,k) for m in s.H6 for k in s.SEEDS])
        self.assertEqual(batches,[[8,7],[8,7]])
        self.assertTrue(all(out[3]['synthetic'] for out in outputs))

    def test_fold_quality_proxy_uses_first_three_original_slots(self):
        chosen=rows(('calibration_frozen_resource_lookup',),(1,))[:3]
        target=np.zeros((3,256,256),np.float32)
        selected=[np.full_like(target,(j+1)/10) for j in range(3)]
        calls=[]
        def quality(reference, images, *args):
            self.assertEqual(len(images),15)
            self.assertTrue(all(np.array_equal(images[j],selected[j]) for j in range(3)))
            self.assertTrue(all(np.array_equal(x,target) for x in images[3:]))
            calls.append([8,7])
            return ([{'slot':j} for j in range(15)],'source_feature',np.arange(15)[:,None])
        original=types.SimpleNamespace(quality_metrics=quality,other='unchanged')
        proxy=s._SelectedQuality(original,chosen)
        result=proxy.quality_metrics(target,selected,None,None,None)
        self.assertEqual(result[0],[{'slot':0},{'slot':1},{'slot':2}])
        self.assertEqual(result[1],'source_feature')
        self.assertEqual(result[2].tolist(),[[0],[1],[2]])
        self.assertEqual(calls,[[8,7]])
        self.assertIs(original.quality_metrics,quality)
        self.assertEqual(proxy.other,'unchanged')

    def test_quality_slots_reject_mixed_methods_and_duplicates(self):
        chosen=rows(('m',),(13,))[:3]
        self.assertEqual(s.original_quality_slots(chosen),[9,10,11])
        for bad in (chosen+[chosen[0]], [chosen[0],{**chosen[1],'method':'other'}],
                    [{**chosen[0],'snr_db':-5}]):
            with self.assertRaises(ValueError):s.original_quality_slots(bad)

    def test_fold_proxy_is_restored_on_generator_failure(self):
        view=object.__new__(s._FoldOnly)
        original=types.SimpleNamespace(quality_metrics=lambda *args:None)
        view.q=original
        view.expected_rows=lambda i:rows(('calibration_frozen_resource_lookup',),(1,))[:3]
        def fail(native, index):
            self.assertIsInstance(native.q,s._SelectedQuality)
            raise RuntimeError('deliberate native failure')
            yield
        with patch.object(s.latent.LatentAdapter,'iterate_source',fail):
            with self.assertRaisesRegex(RuntimeError,'deliberate'):
                list(view.iterate_source(0))
        self.assertIs(view.q,original)

    def test_optional_weight_relocation_checks_bytes_and_preserves_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            old=root/'original/lpips/weights/v0.1/alex.pth'
            new=root/'metric_env/lpips/weights/v0.1/alex.pth'
            for p in (old,new):
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(b'unchanged-original-LPIPS-linear-weights')
            digest=s.budget.sha256(old)
            registered={'bindings':{str(old.resolve()):digest},'other':{'keep':['original']}}
            before=deepcopy(registered)
            native=object.__new__(s.budget.BudgetAdapter)
            native.ready=False; native.bindings={}; native.context=registered
            native.registration={'context':registered}
            fake=types.SimpleNamespace(__file__=str(new.parents[2]/'__init__.py'))
            with patch.dict('sys.modules',{'lpips':fake}):
                s.prepare_budget_quality_path(native)
            self.assertEqual(native.registration['context'],before)
            self.assertIs(native.registration['context'],registered)
            self.assertIsNot(native.context,registered)
            self.assertIsNot(native.context['other'],registered['other'])
            self.assertNotIn(str(new.resolve()),registered['bindings'])
            self.assertEqual(native.context['bindings'][str(new.resolve())],digest)
            self.assertEqual(native.bindings,{str(old.resolve()):digest,str(new.resolve()):digest})
            relocation=native.weight_path_relocations[str(new.resolve())]
            self.assertEqual(relocation,dict(original=str(old.resolve()),sha256=digest,content_identical=True))
            # A different file at an otherwise matching venv suffix must stop
            # before the runtime context is admitted; matching names are not proof.
            new.write_bytes(b'corrupted-linear-weights')
            wrong=object.__new__(s.budget.BudgetAdapter)
            wrong.ready=False; wrong.bindings={}; wrong.context=registered
            wrong.registration={'context':registered}
            with patch.dict('sys.modules',{'lpips':fake}):
                with self.assertRaisesRegex(s.budget.ReplayMismatch,'Frozen file hash differs'):
                    s.prepare_budget_quality_path(wrong)
            self.assertIs(wrong.context,registered)
            self.assertEqual(registered,before)

    def test_both_optional_budget_setups_prove_relocation_before_native_setup(self):
        visited=[]
        def prepare(native):
            native.path_proved=True
            visited.append('proof')
        def native():
            value=types.SimpleNamespace(records=[],bindings={},vae=None,var=None,decoder=None,
                                        scale=None,lp=None,dino=None,path_proved=False)
            def setup():
                self.assertTrue(value.path_proved)
                visited.append('setup')
            value.setup=setup
            return value
        h6=object.__new__(s.H6View);h6.native=native()
        phase=object.__new__(s.Phase2View);phase.native=native()
        phase.fold=types.SimpleNamespace(setup=lambda:None,records=[],bindings={})
        phase.cache=types.SimpleNamespace(records=[],bindings={})
        with patch.object(s,'prepare_budget_quality_path',side_effect=prepare):
            h6.setup();phase.setup()
        self.assertEqual(visited,['proof','setup','proof','setup'])

    def test_scope_counts_and_order_are_frozen(self):
        data=rows(s.PHASE2, (1,4,7,13,19))
        selected=s.selected_rows(data,s.PHASE2,(1,4,7,13,19))
        self.assertEqual(len(selected),4500)
        self.assertEqual(selected,data)
        self.assertIsNot(selected[0],data[0])


if __name__=='__main__':unittest.main()
