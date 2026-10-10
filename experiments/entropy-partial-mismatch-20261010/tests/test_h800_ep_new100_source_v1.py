"""CPU metadata/ownership checks; synthetic placeholders, no source pixels/models."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_new100_source_v1 as h


class SourceOwnerTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);values={}
        def put(name,value):
            path=root/(name+'.json');path.write_text(json.dumps(value),encoding='utf8')
            pin=dict(path=str(path),sha256=h.g.sha(path),bytes=path.stat().st_size);values[str(path)]=value;return pin
        self.put=put;self.values=values
        binary=root/'synthetic-placeholder.bin';binary.write_bytes(b'not an image or npz')
        payload=dict(path=str(binary),sha256=h.g.sha(binary),bytes=binary.stat().st_size)
        records=[dict(source_index=i,source_id=f'synthetic-{i}',canonical_source_id=f'synthetic-{i}',
            evaluation_class_index=i,original_JPEG=payload,pixels_archive=payload,preprocessing_id='a'*64,
            canonical_pixel_sha256='a'*64,horizontal_flip_pixel_sha256='b'*64) for i in range(100)]
        selected=dict(source_count=100,N=1024,SNRs=[4,10,19],methods=h.population.METHODS,noise_seeds=h.population.NOISE_SEEDS,
            policy_selection_uses_new100=False,source_reselection_allowed=False,source_ids=[r['source_id'] for r in records],
            records=[dict(source_id=r['source_id'],canonical_source_id=r['canonical_source_id']) for r in records])
        policy=dict(schema='EP_NEW100_FOUR_FROZEN_POLICY_PROJECTION_V1',strategy_selection_uses_new100=False,
            full_calibration=dict(complete_grid_verified=True),policies={method:{str(s):{} for s in (4,10,19)} for method in h.population.METHODS})
        preprocess=dict(path='synthetic-original-preprocess',sha256='c'*64)
        receipt=dict(original_preprocess=preprocess,records=[dict(original_file_sha256=payload['sha256'],
            pixel_sha256='a'*64,horizontal_flip_pixel_sha256='b'*64) for _ in records])
        gate=dict(status='EP_NEW100_COMPLETE_CONTENT_DEDUP_PASS',conflicts=[],Encoder_calls_allowed=True,source_reselection_allowed=False)
        results=dict(selection=put('selection',selected),policy_bundle=put('policy',policy),registry=put('registry',{}),
            content_receipt=put('receipt',receipt),content_gate=put('gate',gate))
        manifest=dict(schema='EP_NEW100_SOURCE_CONTENT_ASSETS_V1',source_count=100,records=records,model_calls=0,
            original_preprocess=preprocess,**results)
        results['source_manifest']=put('manifest',manifest)
        tools={'ep_new100_content_owner_v1.py':'d'*64}
        request=dict(schema='EP_NEW100_CONTENT_OWNER_V1',status='REGISTERED_NOT_EXECUTED',tool_bindings=tools,
            original_preprocess=preprocess)
        request_pin=put('request',request)
        worker=dict(schema=request['schema'],status='EP_NEW100_CONTENT_ASSETS_READY_NO_ENCODER',request_sha256=request_pin['sha256'],
            results=results,canonical_reserved=100,canonical_completed=100,unresolved=0,model_calls=0,packet_calls=0,CUDA_initialized=False)
        worker_pin=put('worker',worker)
        wait=dict(actual_wait=True,returncode=0,interrupted_or_timeout=False)
        put('actual_child_wait',wait)
        done=dict(status=worker['status'],actual_children_waited=True,worker_exit_codes=[0],actual_child_wait=wait,
            worker_completion=worker_pin,request=request_pin,results=results,model_calls=0,packet_calls=0,automatic_successor=False)
        pins=dict(completion=put('completion',done),owner_actual_wait=put('outer_wait',dict(actual_wait=True,returncode=0,timeout=False)))
        implementation=types.SimpleNamespace(SCHEMA=request['schema'],tools=lambda:tools)
        return pins,done,worker,request,manifest,policy,gate,implementation

    def patches(self,gate,implementation):
        context=ExitStack();context.enter_context(mock.patch.object(h.g,'inside',side_effect=Path))
        context.enter_context(mock.patch.object(h,'readpin',side_effect=lambda p:self.values[p['path']]))
        context.enter_context(mock.patch.object(h,'content_implementation',return_value=implementation))
        context.enter_context(mock.patch.object(h.population,'check_content',return_value=gate))
        return context

    def test_exact_closed_content_can_be_bound_without_decoding_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            pins,done,worker,request,manifest,policy,gate,impl=self.fixture(folder)
            with self.patches(gate,impl),mock.patch.object(h.core.np,'load',side_effect=AssertionError('No arrays before worker')):
                records,inputs=h.content_closure(pins)
            self.assertEqual(len(records),100);self.assertEqual(inputs,done['results'])

    def test_unclosed_content_stops_before_source_or_implementation_access(self):
        with tempfile.TemporaryDirectory() as folder:
            pins,done,*_=self.fixture(folder);done['actual_children_waited']=False
            with mock.patch.object(h,'readpin',side_effect=lambda p:self.values[p['path']]), \
                 mock.patch.object(h,'content_implementation') as impl,self.assertRaisesRegex(RuntimeError,'Actually closed'):
                h.content_closure(pins)
            impl.assert_not_called()

    def test_nonzero_content_scientific_calls_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            pins,done,worker,request,manifest,policy,gate,impl=self.fixture(folder);done['model_calls']=1
            with self.patches(gate,impl),self.assertRaisesRegex(RuntimeError,'before any scientific'):
                h.content_closure(pins)

    def test_content_worker_unresolved_or_wrong_request_rejected(self):
        for key,value in [('unresolved',1),('canonical_completed',99),('request_sha256','f'*64),('CUDA_initialized',True)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as folder:
                pins,done,worker,request,manifest,policy,gate,impl=self.fixture(folder);worker[key]=value
                with self.patches(gate,impl),self.assertRaisesRegex(RuntimeError,'owner/worker output closure'):
                    h.content_closure(pins)

    def test_actual_child_wait_file_must_equal_embedded_wait(self):
        with tempfile.TemporaryDirectory() as folder:
            pins,done,worker,request,manifest,policy,gate,impl=self.fixture(folder)
            self.values[str(Path(folder)/'actual_child_wait.json')]=dict(actual_wait=True,returncode=1,interrupted_or_timeout=False)
            with self.patches(gate,impl),self.assertRaisesRegex(RuntimeError,'Actual child wait file'):
                h.content_closure(pins)

    def test_frozen_content_tool_binding_required(self):
        with tempfile.TemporaryDirectory() as folder:
            pins,done,worker,request,manifest,policy,gate,impl=self.fixture(folder);request['tool_bindings']={}
            with self.patches(gate,impl),self.assertRaisesRegex(RuntimeError,'request/tool identities'):
                h.content_closure(pins)

    def test_policy_cannot_use_new100_or_omit_fourth_arm(self):
        for change in ('selection','omit'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                pins,done,worker,request,manifest,policy,gate,impl=self.fixture(folder)
                if change=='selection':policy['strategy_selection_uses_new100']=True
                else:policy['policies'].pop('EC_VAR_PARTIAL')
                with self.patches(gate,impl),self.assertRaisesRegex(RuntimeError,'four original frozen policies'):
                    h.content_closure(pins)

    def test_manifest_cannot_rebind_content_or_reorder_fixed_ids(self):
        for change in ('pin','order'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                pins,done,worker,request,manifest,policy,gate,impl=self.fixture(folder)
                if change=='pin':manifest['content_gate']=dict(path='other',sha256='f'*64)
                else:manifest['records'][0],manifest['records'][1]=manifest['records'][1],manifest['records'][0]
                with self.patches(gate,impl),self.assertRaises(RuntimeError):h.content_closure(pins)

    def test_private_owner_scope_preserves_old_caps_and_model_functions(self):
        before=copy.deepcopy(h.original.CAPS);runner=h.engine()
        self.assertEqual(h.original.CAPS,before);self.assertEqual(h.changed,dict(count=1,field=1,resource_evidence=1))
        self.assertEqual(runner.CAPS,dict(model_load=1,encoder=100,source_tx=100,source_rx=0,var_render=0,prior_scale=1000,decoder_forward=0))
        self.assertEqual(runner.check_tools.__globals__['__file__'],h.__file__)
        self.assertEqual(h.run.__globals__['registration'],h.registration)
        self.assertEqual(h.run.__globals__['__file__'],h.__file__)

    def test_shared_resource_guard_is_bound_to_owner_worker_and_request(self):
        runner=h.engine()
        self.assertIs(runner.DeviceAdapter,h.resources.DeviceAdapter)
        self.assertIs(runner.device_from_request.__globals__['DeviceAdapter'],h.resources.DeviceAdapter)
        self.assertEqual(h.resources.POLICY['own_allocator_cap_bytes'],16*(1<<30))
        self.assertEqual(h.resources.POLICY['prelaunch_free_bytes'],20*(1<<30))
        self.assertEqual(h.resources.POLICY['runtime_free_bytes'],4*(1<<30))
        self.assertEqual(h.resources.POLICY['runtime_free_plus_own_reserved_bytes'],20*(1<<30))
        self.assertIn('bind_resource_evidence',h.run.__code__.co_names)
        self.assertIn('bind_resource_evidence',h.worker.__code__.co_names)

    def test_resource_guard_bytes_cannot_change(self):
        with mock.patch.object(h,'RESOURCE_SHA','f'*64),self.assertRaisesRegex(RuntimeError,'Frozen source owner'):
            h.engine()

    def test_execution_window_is_bounded_independently_of_input_contract(self):
        self.assertEqual(h.gate.source.execution(time.time()+3600,3600)['max_seconds'],3600)
        with self.assertRaises(RuntimeError):h.gate.source.execution(time.time()+9999,3601)

    def test_frozen_implementation_is_mandatory(self):
        with mock.patch.object(h,'CONTENT_OWNER_SHA','0'*64),self.assertRaisesRegex(RuntimeError,'Final frozen content-owner'):
            h.content_implementation()

    def test_source_owner_replay_does_not_launch_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'run').mkdir();path=root/'request.json';path.write_text('{}')
            shared=mock.Mock();shared.owner_lock.return_value=ExitStack();runner=mock.Mock();runner.device_from_request.return_value.index=2
            args=types.SimpleNamespace(request=str(path),request_sha256='f'*64)
            with mock.patch.object(h.g.sys,'platform','linux'),mock.patch.dict(h.run.__globals__,registration=lambda *a:dict(spec={},execution={})), \
                 mock.patch.object(h.g,'inside',side_effect=Path),mock.patch.object(h.g,'helper',return_value=shared), \
                 mock.patch.object(h.original.subprocess,'Popen') as popen,self.assertRaisesRegex(RuntimeError,'already claimed'):
                h.run(args,runner)
            popen.assert_not_called()


if __name__=='__main__':unittest.main()
