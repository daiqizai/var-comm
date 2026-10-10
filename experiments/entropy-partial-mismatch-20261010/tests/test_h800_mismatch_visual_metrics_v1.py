"""Bounded fake reconstruction and metric-metadata tests; no neural execution."""
from pathlib import Path
import copy
import sys
import time
import types
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import h800_mismatch_visual_v1 as visual
import h800_mismatch_metrics_v1 as metric
import h800_mismatch_metric_owner_v1 as owner


class ReconstructionTests(unittest.TestCase):
    def test_only_VAR_render_budget_and_exact_ten_scales(self):
        caps=visual.CAPS
        self.assertEqual(caps['model_load'],1);self.assertEqual(caps['var_render'],4500)
        self.assertEqual(caps['prior_scale'],10*caps['var_render']);self.assertEqual(caps['decoder_forward'],caps['var_render'])
        self.assertEqual(caps['source_rx']+caps['source_tx']+caps['encoder'],0)

    def test_fixed_gray_does_not_invoke_model_or_ledger(self):
        backend=mock.Mock();ledger=mock.Mock();boundary=mock.Mock()
        image=visual.render_actual(dict(kind='gray'),None,backend,ledger,boundary)
        self.assertEqual(image.dtype,np.float32);self.assertEqual(image.shape,(3,256,256));self.assertTrue((image==.5).all())
        backend.render_function.assert_not_called();ledger.call.assert_not_called();boundary.assert_called_once()

    def test_actual_prefix_and_partial_only_reach_original_render(self):
        tokens=np.array([8,9],dtype=np.int64);state=dict(kind='tokens',m=1,K=1)
        class Context:
            def __enter__(self):pass
            def __exit__(self,*a):pass
        backend=types.SimpleNamespace(source_index=4,native=types.SimpleNamespace(torch=types.SimpleNamespace(no_grad=Context)))
        backend.render_function=mock.Mock(return_value=np.zeros((3,256,256),dtype=np.float32))
        ledger=mock.Mock();ledger.call.side_effect=lambda name,fn,**kw:fn()
        visual.render_actual(state,tokens,backend,ledger,lambda:None)
        args=backend.render_function.call_args.args
        self.assertIs(args[0],backend.native);np.testing.assert_array_equal(args[1],tokens)
        self.assertIsNot(args[1],tokens);self.assertEqual(args[2:],(1,1));self.assertEqual(ledger.call.call_args.args[0],'var_render')

    def test_cache_never_erases_device_identity(self):
        state=dict(kind='tokens',prefix=[[8]],partial_values=[9],m=1,K=1,order='raster')
        first=visual.actual_state_key(state,dict(device='one'));second=visual.actual_state_key(state,dict(device='two'))
        self.assertNotEqual(first,second)

    def test_CPU_actual_wait_failure_stops_before_any_model(self):
        root=visual.gate.RT/'qualification/h800_mismatch_phy_v1_attempt1'
        pins=dict(completion=dict(path=str(root/'registered/run/completion.json'),sha256='a'*64),
            owner_actual_wait=dict(path=str(root/'run_owner_actual_wait.json'),sha256='b'*64))
        done=dict(status=visual.physical.PASS,actual_wait=dict(success=True),actual_children_waited=True,worker_exit_codes=[0])
        with mock.patch.object(visual.gate,'readpin',side_effect=[done,dict(actual_wait=True,returncode=1)]),\
            self.assertRaisesRegex(RuntimeError,'Actual mismatch CPU owner wait'):
            visual.cpu_closure(pins)

    def test_private_engine_keeps_prior_owner_unchanged(self):
        before=visual.original.engine().CAPS.copy();runner=visual.engine()
        self.assertEqual(runner.CAPS,visual.CAPS);self.assertEqual(visual.original.engine().CAPS,before)
        self.assertEqual(runner.worker_identity.__globals__['__file__'],visual.__file__)


class MetricTests(unittest.TestCase):
    def test_metric_owner_wait_failure_cannot_open_scores(self):
        root=owner.gate.RT/'qualification/h800_mismatch_visual_v1_attempt1'
        pins=dict(completion=dict(path=str(root/'registered/run/completion.json'),sha256='a'*64),
            owner_actual_wait=dict(path=str(root/'run_owner_actual_wait.json'),sha256='b'*64))
        done=dict(status=visual.PASS,actual_wait=dict(success=True),actual_children_waited=True,worker_exit_codes=[0])
        with mock.patch.object(owner.gate,'readpin',side_effect=[done,dict(actual_wait=True,returncode=1)]),\
            self.assertRaisesRegex(RuntimeError,'Actual raw visual owner'):
            owner.visual_closure(pins)

    def test_metric_owner_private_scope_keeps_old_caps(self):
        before=owner.original.CAPS.copy();runner=owner.engine()
        self.assertEqual(runner.CAPS['image_scores'],4500);self.assertEqual(owner.original.CAPS,before)
        self.assertEqual(runner.worker_identity.__globals__['__file__'],owner.__file__)

    def test_metric_registration_cannot_admit_bootstrap_or_policy_selection(self):
        e=owner.execution(time.time()+600,600);runner=mock.Mock();runner.inherited_spec.return_value={}
        request=dict(schema=owner.SCHEMA,status='REGISTERED_NOT_EXECUTED',execution=e,caps=owner.CAPS,spec={},
            records=[],logical_events=[],reconstruction_results=[],metric_config={},visual_closed={},PHY_calls=0,source_calls=0,
            VAR_calls=0,decoder_calls=0,training_updates=0,bootstrap_calls=0,policy_selection=False,automatic_successor=False,tool_bindings={})
        for field in ('bootstrap_calls','policy_selection','automatic_successor','VAR_calls'):
            changed=dict(request);changed[field]=True
            with mock.patch.object(owner.g,'inside',side_effect=Path),mock.patch.object(owner.g,'checked_json',return_value=changed),\
                mock.patch.object(owner,'visual_closure',return_value=({},dict(results=dict(logical_frames=[])),dict(records=[],input_closure=dict(original_policy={})),[])),\
                mock.patch.object(owner.visual.original.source,'prerequisites',return_value=({},)),\
                mock.patch.object(owner,'configuration',return_value={}),self.subTest(field=field),\
                self.assertRaisesRegex(RuntimeError,'Metric-only diagnostic scope'):
                owner.registration('request','a'*64,runner)

    def test_compute_functions_identical_new_population_without_global_change(self):
        original_population=metric.original.POPULATION;provider,proof=metric.private_provider()
        self.assertTrue(all(proof['methods_same_code_object'].values()));self.assertTrue(all(proof['functions_same_code_object'].values()))
        self.assertEqual(provider.__init__.__globals__['POPULATION'],metric.POPULATION)
        self.assertEqual(metric.original.POPULATION,original_population)
        self.assertEqual(metric.CAPS['image_scores'],4500);self.assertEqual(metric.CAPS['lpips_alexnet_backbone_forward'],9000)

    def rows(self):
        p=metric.inputs.frozen;rows=[]
        for i in range(100):
            for s in p.ACTUAL_SNRS:
                for k in p.LOOKUPS[s]:
                    for f in p.FAMILIES:
                        for n in p.SEEDS:
                            off=k!=s
                            rows.append(dict(source_index=i,source_id='source'+str(i),actual_snr_db=s,config_snr_db=k,family=f,noise_seed=n,
                                psnr_db=30+(-1 if off else 0),lpips_alex=.2+(.1 if off else 0),dinov2_vitl14_cosine=.8+(-.1 if off else 0),
                                convnext_top1_source_prediction=int(not off),actual_header_ok=True,body_attempted=True,
                                actual_crc_accepted=not off,gray=False))
        return rows

    def test_three_noise_paired_direction_percentage_points_and_zero_baseline(self):
        result=metric.paired_outputs(self.rows());self.assertEqual(result['bootstrap_calls'],0)
        self.assertEqual(len(result['source_paired_differences']),7200);self.assertEqual(len(result['mean_differences']),72)
        for row in result['mean_differences']:
            if row['actual_snr_db']==row['config_snr_db']:self.assertEqual(row['mean'],0)
            elif row['metric']=='lpips_alex':self.assertAlmostEqual(row['mean'],.1)
            elif row['metric']=='convnext_top1_source_prediction':self.assertEqual(row['mean'],-100);self.assertEqual(row['unit'],'percentage_points')
            self.assertIsNone(row['ci_low']);self.assertIsNone(row['ci_high'])

    def test_missing_or_duplicate_metric_frame_rejected(self):
        rows=self.rows()
        with self.assertRaisesRegex(RuntimeError,'Complete unique5400'):metric.paired_outputs(rows[:-1])
        rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(RuntimeError,'Complete unique5400'):metric.paired_outputs(rows)

    def test_header_body_and_gray_failures_remain_separate(self):
        rows=self.rows();rows[0].update(actual_header_ok=False,body_attempted=False,actual_crc_accepted=False,gray=True)
        output=metric.paired_outputs(rows);cell=output['failure_counts'][0]
        self.assertEqual(cell['header_failures'],1);self.assertEqual(cell['body_attempted'],299)
        self.assertEqual(cell['body_CRC_rejections'],299);self.assertEqual(cell['fixed_gray'],1)

    def test_metric_pair_key_includes_reference_and_identity(self):
        image=np.zeros((3,2,2),dtype=np.float32);other=np.ones_like(image)
        key=metric.pair_cache_key(image,image,'a'*64)
        self.assertNotEqual(key,metric.pair_cache_key(other,image,'a'*64))
        self.assertNotEqual(key,metric.pair_cache_key(image,image,'b'*64))


if __name__=='__main__':unittest.main()
