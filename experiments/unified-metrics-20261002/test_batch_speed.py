"""Synthetic engineering checks; fake CUDA/model calls are never result data."""
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('batch_speed_engineering_target',HERE/'batch_speed.py')
batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)


def row(value=.4):
    return dict({k:value for k in batch.FLOAT_FIELDS},resnet50_prediction=7,resnet50_source_prediction=7,
                resnet50_top1_label=True,resnet50_top1_source_prediction=True,resnet50_source_top1_label=True,
                label_conditioned=False)


def candidate(size,seconds,*,eligible=True):
    return dict(batch_size=size,eligible=eligible,status='PASS' if eligible else 'REJECT_CUDA_OOM',
                comparison=batch.compare_rows([row()],[row()]),peak_reserved_bytes=800,
                repeat_seconds_per_image=[seconds]*3,median_seconds_per_image=seconds)


def receipt():
    binding=dict(metric_evaluator_identity='ENGINEERING_ONLY',modelmanifest_sha256='b'*64,
                 source_bindings={},device=dict(total_memory_bytes=1000),runtime_context={})
    value=dict(status=batch.STATUS,binding=binding,protocol=copy.deepcopy(batch.PROTOCOL),
               synthetic_images=True,scientific_result=False,real_model_weights=True,
               source_or_development_images_used=False,rng_state_preserved=True,
               metric_evaluator_identity=binding['metric_evaluator_identity'],
               modelmanifest_sha256=binding['modelmanifest_sha256'],source_bindings={},
               candidates=[candidate(size,1/size) for size in (1,2,4,8,16)],
               chosen_batch_size=16,qualified_batch_sizes=[1,2,4,8,16])
    value['payload_sha256']=batch.digest(value)
    return value,binding


def reseal(value):
    value.pop('payload_sha256',None);value['payload_sha256']=batch.digest(value)


class Tensor:
    def __init__(self,data):self.data=np.asarray(data,dtype=np.float32)
    def __len__(self):return len(self.data)
    def __getitem__(self,index):return Tensor(self.data[index])


class EngineeringOOM(RuntimeError):pass


class FakeCUDA:
    OutOfMemoryError=EngineeringOOM
    def __init__(self):self.peak=700
    def is_available(self):return True
    def current_device(self):return 0
    def get_device_properties(self,device):
        return SimpleNamespace(name='ENGINEERING_FAKE_CUDA',total_memory=1000,major=9,minor=0,uuid='FAKE')
    def memory_allocated(self,device):return 650
    def memory_reserved(self,device):return 700
    def synchronize(self,device):pass
    def empty_cache(self):pass
    def reset_peak_memory_stats(self,device):self.peak=700
    def max_memory_reserved(self,device):return self.peak
    def max_memory_allocated(self,device):return self.peak-10


class FakeEvaluator:
    def __init__(self,cuda):self.cuda=cuda;self.calls=0;self.prepare_calls=0;self.batch_calls=[]
    def identity(self):return 'ENGINEERING_FAKE_ID'
    def prepare_reference(self,source):self.prepare_calls+=1;return dict(resnet50=[7])
    def expand_reference(self,source,prepared,count):return source,prepared
    def score(self,source,images,labels,prepared,label_conditioned):
        self.calls+=1;size=len(images);self.batch_calls.append(size)
        if size==16:
            self.cuda.peak=950;raise EngineeringOOM('ENGINEERING deliberate OOM')
        self.cuda.peak=max(self.cuda.peak,900 if size==4 else 800)
        return [row(.401 if size==8 else .4) for _ in range(size)]


class BatchEngineering(unittest.TestCase):
    def test_float_absolute_threshold_and_exact_classification(self):
        expected=row();actual=row();actual['dists']+=1e-5
        self.assertTrue(batch.compare_rows([actual],[expected])['passed'])
        actual['dists']+=2e-5
        self.assertFalse(batch.compare_rows([actual],[expected])['passed'])
        for field in batch.EXACT_FIELDS:
            actual=row();actual[field]=False if isinstance(actual[field],bool) else 8
            if field=='label_conditioned':actual[field]=True
            self.assertFalse(batch.compare_rows([actual],[expected])['passed'],field)
        actual=row();actual['resnet50_top1_label']=1
        self.assertFalse(batch.compare_rows([actual],[expected])['passed'])

    def test_invalid_missing_extra_nonfinite_and_count_rejected(self):
        for actual in (dict(row(),surprise=.1),{k:v for k,v in row().items() if k!='dreamsim'},
                       dict(row(),dists=float('nan')),dict(row(),dists=True)):
            with self.assertRaises(ValueError):batch.compare_rows([actual],[row()])
        for actual,expected in (([],[]),([row(),row()],[row()])):
            with self.assertRaises(ValueError):batch.compare_rows(actual,expected)

    def test_fastest_valid_wins_but_scalar_is_required(self):
        values=[candidate(1,.4),candidate(2,.3),candidate(4,.01,eligible=False),candidate(8,.3)]
        self.assertEqual(batch.select_candidate(values),2)
        self.assertEqual(batch.select_candidate([candidate(1,.4),candidate(2,.01,eligible=False)]),1)
        with self.assertRaises(RuntimeError):batch.select_candidate([candidate(1,.4,eligible=False),candidate(2,.01)])

    def test_tail_decomposition_skips_failed_sizes_and_handles_zero(self):
        self.assertEqual(batch.qualified_chunks(7,8,[1,4,8]),[4,1,1,1])
        self.assertEqual(batch.qualified_chunks(15,8,[1,2,4,8,16]),[8,4,2,1])
        self.assertEqual(batch.qualified_chunks(0,1,[1]),[])
        for count,chosen,qualified in ((-1,1,[1]),(True,1,[1]),(3,4,[1,2]),(3,4,[4]),(3,3,[1,3]),(2,2,[1,True,2])):
            with self.assertRaises(ValueError):batch.qualified_chunks(count,chosen,qualified)

    def test_valid_receipt_roundtrip_and_binding_strictness(self):
        value,binding=receipt()
        self.assertEqual(batch.validate_receipt(json.loads(json.dumps(value)),binding),value)
        for key,new in (('metric_evaluator_identity','changed'),('modelmanifest_sha256','c'*64),
                        ('runtime_context',{'old_model':'changed'}),('source_bindings',{'changed':'d'*64}),
                        ('device',{'total_memory_bytes':2000})):
            changed=copy.deepcopy(binding);changed[key]=new
            with self.assertRaises(RuntimeError):batch.validate_receipt(value,changed)
        value['scientific_result']=True
        with self.assertRaises(RuntimeError):batch.validate_receipt(value,binding)

    def test_receipt_hash_candidate_coverage_and_claims_checked(self):
        for change in (lambda x:x.update(chosen_batch_size=8),
                       lambda x:x.update(qualified_batch_sizes=[1,16]),
                       lambda x:x['candidates'].pop(),
                       lambda x:x['candidates'][1].update(peak_reserved_bytes=881),
                       lambda x:x['candidates'][1].update(median_seconds_per_image=.123),
                       lambda x:x['candidates'][1]['comparison'].update(classification_exact=False),
                       lambda x:x['candidates'][1]['comparison']['max_absolute_deltas'].update(dists=.01),
                       lambda x:x.update(rng_state_preserved=False)):
            value,binding=receipt();change(value);reseal(value)
            with self.assertRaises(RuntimeError):batch.validate_receipt(value,binding)
        value,binding=receipt();value['elapsed_seconds']=2
        with self.assertRaises(RuntimeError):batch.validate_receipt(value,binding)

    def test_full_control_flow_rejects_memory_numeric_oom_and_reuses_bound_receipt(self):
        with tempfile.TemporaryDirectory(prefix='ENGINEERING_BATCH_') as temporary:
            root=Path(temporary);manifest=root/'manifest.json';manifest.write_text('{"ENGINEERING":true}')
            modelcode=root/'fake_metric_models.py';modelcode.write_text('# ENGINEERING only\n')
            cuda=FakeCUDA();evaluator=FakeEvaluator(cuda)
            torch=SimpleNamespace(cuda=cuda,device=lambda _:SimpleNamespace(type='cuda',index=0),
                tensor=lambda values,dtype:np.asarray(values),int64=np.int64,
                random=SimpleNamespace(fork_rng=lambda **_:contextlib.nullcontext()))
            models=SimpleNamespace(__file__=str(modelcode),no_network=contextlib.nullcontext,
                image_digest=lambda tensor:batch.digest(tensor.data.tolist()))
            fixtures=(Tensor(np.zeros((1,3,2,2))),Tensor(np.ones((16,3,2,2))))
            with mock.patch.dict(sys.modules,{'torch':torch,'metric_models':models}), \
                 mock.patch.object(batch,'synthetic_fixtures',return_value=fixtures):
                result=batch.qualify_and_select_batch(evaluator,'cuda:0',manifest,root,runtime_context={'old':'fixed'})
                self.assertEqual(result['qualified_batch_sizes'],[1,2])
                self.assertIn(result['chosen_batch_size'],(1,2))
                self.assertEqual([c['status'] for c in result['candidates']],
                    ['PASS','PASS','REJECT_MEMORY_RESERVATION','REJECT_NUMERICAL_AGREEMENT','REJECT_CUDA_OOM'])
                self.assertEqual(evaluator.prepare_calls,1)
                # Scalar reference is 16 calls; each passing or rejecting
                # non-OOM candidate does one warmup and three identical passes.
                self.assertEqual(evaluator.batch_calls.count(1),16+16*4)
                self.assertEqual(evaluator.batch_calls.count(2),8*4)
                before=evaluator.calls
                again=batch.qualify_and_select_batch(evaluator,'cuda:0',manifest,root,runtime_context={'old':'fixed'})
                self.assertEqual(again,result);self.assertEqual(evaluator.calls,before)
                self.assertFalse(result['scientific_result'])
                for kwargs in ({'runtime_context':{'old':'different'}},):
                    with self.assertRaisesRegex(RuntimeError,'binding'):
                        batch.qualify_and_select_batch(evaluator,'cuda:0',manifest,root,**kwargs)
                manifest.write_text('{"ENGINEERING":"changed"}')
                with self.assertRaisesRegex(RuntimeError,'binding'):
                    batch.qualify_and_select_batch(evaluator,'cuda:0',manifest,root,runtime_context={'old':'fixed'})

    def test_scalar_memory_failure_writes_failure_and_no_pass_receipt(self):
        with tempfile.TemporaryDirectory(prefix='ENGINEERING_BATCH_FAIL_') as temporary:
            root=Path(temporary);manifest=root/'manifest.json';manifest.write_text('{}')
            modelcode=root/'fake.py';modelcode.write_text('#ENGINEERING')
            cuda=FakeCUDA();evaluator=FakeEvaluator(cuda)
            cuda.max_memory_reserved=lambda device:950
            torch=SimpleNamespace(cuda=cuda,device=lambda _:SimpleNamespace(type='cuda',index=0),
                tensor=lambda values,dtype:values,int64='int64',
                random=SimpleNamespace(fork_rng=lambda **_:contextlib.nullcontext()))
            models=SimpleNamespace(__file__=str(modelcode),no_network=contextlib.nullcontext,image_digest=lambda _:'a'*64)
            with mock.patch.dict(sys.modules,{'torch':torch,'metric_models':models}), \
                 mock.patch.object(batch,'synthetic_fixtures',return_value=(Tensor([0]),Tensor([1]*16))):
                with self.assertRaisesRegex(RuntimeError,'Scalar batch'):
                    batch.qualify_and_select_batch(evaluator,'cuda',manifest,root,runtime_context={})
            self.assertFalse((root/batch.RECEIPT_NAME).exists())
            self.assertEqual(json.loads((root/'metric_batch_qualification_failed.json').read_text())['status'],'FAILED')


if __name__=='__main__':unittest.main(verbosity=2)
