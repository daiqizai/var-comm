"""CPU-only finite metric ownership and reference-sensitive reuse tests."""
import copy
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_pilot_metric_owner_v1 as h


class MetricOwnerTests(unittest.TestCase):
    def test_private_owner_does_not_modify_running_visual_caps(self):
        caps=copy.deepcopy(h.visual.CAPS);owner=h.engine()
        self.assertEqual(h.visual.CAPS,caps);self.assertEqual(owner.CAPS,h.metric.CAPS)
        self.assertEqual(owner.worker_identity.__globals__['__file__'],h.__file__)
        self.assertEqual(owner.CAPS['model_constructions'],3)
        self.assertNotIn('source_rx',owner.CAPS)

    def test_changed_frozen_metric_source_rejects_before_execution(self):
        with mock.patch.object(h,'METRIC_SHA','0'*64),self.assertRaisesRegex(RuntimeError,'implementation changed'):
            h.engine()

    def test_independent_finite_window(self):
        self.assertEqual(h.execution(time.time()+21600,21600)['allocator_cap_bytes'],16*(1<<30))
        with self.assertRaises(RuntimeError):h.execution(time.time()+21600,21601)
        with self.assertRaises(RuntimeError):h.execution(time.time()-1,21600)

    def test_unclosed_visual_attempt_cannot_admit_metrics(self):
        root=h.gate.RT/'qualification/h800_ep_pilot_visual_v1_attempt1'
        pins=dict(completion=dict(path=str(root/'registered/run/completion.json'),sha256='a'*64),
            owner_actual_wait=dict(path=str(root/'run_owner_actual_wait.json'),sha256='b'*64))
        done=dict(status=h.visual.PASS,actual_wait=dict(success=True),actual_children_waited=True,worker_exit_codes=[0])
        with mock.patch.object(h.gate,'readpin',side_effect=[done,dict(actual_wait=True,returncode=1)]),\
             self.assertRaisesRegex(RuntimeError,'visual owner must close'):
            h.visual_closure(pins)

    def test_metric_native_addition_preserves_all_original_PFS_rows(self):
        rows={'/frozen/runtime.so':dict(path='/frozen/runtime.so',sha256='a'*64,bytes=7)}
        old=dict(path='/usr/lib/old.so',sha256='b'*64,bytes=4);new=dict(path='/usr/lib/new.so',sha256='c'*64,bytes=5)
        environment=types.SimpleNamespace(rows=copy.deepcopy(rows),host_native=[old])
        result=(None,environment,None,None,{})
        with mock.patch.object(h.g,'validate_spec',return_value=result),mock.patch.object(h,'native_binding',return_value=({},[old,new])):
            actual=h.metric_environment({},verify_environment=False)
        self.assertEqual(actual[1].rows,rows);self.assertEqual(actual[1].host_native,[old,new])

    def test_conflicting_external_native_receipt_rejects(self):
        old=dict(path='/usr/lib/old.so',sha256='b'*64,bytes=4);new=old|dict(sha256='c'*64)
        environment=types.SimpleNamespace(rows={},host_native=[old])
        with mock.patch.object(h.g,'validate_spec',return_value=(None,environment,None,None,{})),\
             mock.patch.object(h,'native_binding',return_value=({},[new])),self.assertRaisesRegex(RuntimeError,'Conflicting actual native'):
            h.metric_environment({},verify_environment=False)

    def test_runtime_identity_is_derived_from_owner_device_and_receipts(self):
        device=types.SimpleNamespace(device_identity_binding=lambda:dict(UUID='actual'))
        with mock.patch.object(h,'native_binding',return_value=({'sha256':'a'*64},[])):
            identity=h.runtime_identity({'environment':{'sha256':'b'*64}},device)
        self.assertEqual(identity['device_identity'],dict(UUID='actual'))
        self.assertEqual(identity['numeric_flags'],h.metric.FLAGS)
        self.assertFalse(identity['old_host_numeric_identity_claimed'])

    def test_incomplete_images_reject_before_reference_preparation(self):
        records=[dict(source_index=i) for i in range(100)]
        with tempfile.TemporaryDirectory() as td,self.assertRaisesRegex(RuntimeError,'Complete actual reconstruction'):
            h.scores(dict(records=records,logical_events=[],reconstruction_results=[]),None,None,lambda:None,Path(td)/'out')

    def test_gray_image_pair_cache_is_reference_specific_and_preserves_each_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);image=np.full((3,256,256),.5,np.float32);archive=root/'gray.npz';np.savez(archive,image=image)
            archive_pin=h.core.descriptor(archive);image_sha=h.metric.array_sha(image);records=[dict(source_index=i,source_id=f's{i}',evaluation_class_index=i,
                archive={'source':i}) for i in range(100)]
            events=[dict(source_index=i//432,source_id=f's{i//432}',snr_db=4,noise_seed=4101,candidate_id=f'c{i%432}') for i in range(43200)]
            request=dict(records=records,logical_events=events,reconstruction_results=[{'path':f'logical/{i}'} for i in range(43200)],
                metric_config=dict(whole_policy={'path':'policy'}))
            ledger=h.g.Ledger(root/'ledger',lambda:None,h.CAPS)
            for _ in range(3):ledger.call('model_constructions',lambda:None)
            class Evaluator:
                identity='d'*64
                def prepare(self,target):
                    def operation():
                        ledger.call('dinov2_vitl14_reference',lambda:None);ledger.call('convnext_reference',lambda:None)
                    return ledger.call('reference_preparations',operation)
                def score(self,target,reconstruction,class_index,prepared):
                    def operation():
                        for kind in ('dinov2_vitl14_reconstruction','convnext_reconstruction','lpips_pair'):
                            ledger.call(kind,lambda:None)
                        for _ in range(2):ledger.call('lpips_alexnet_backbone_forward',lambda:None)
                        return dict(psnr_db=20.,lpips_alex=.2,dinov2_vitl14_cosine=class_index/100,
                            convnext_top1_source_prediction=class_index%2)
                    return ledger.call('image_scores',operation)
            def read(pin):
                if pin['path']=='policy':return {'policy':'fixture'}
                i=int(pin['path'].split('/')[-1])
                return dict(frame_index=i,logical_event=events[i],reconstruction=dict(image_archive=archive_pin,
                    image_sha256=image_sha),actual_RX_status='HEADER_REJECT' if i%2 else 'CRC_REJECT',
                    actual_received_profile=None,actual_header_ok=bool(i%2==0),actual_crc_accepted=False,actual_parser_accepted=False)
            with mock.patch.object(h.gate,'readpin',side_effect=read),mock.patch.object(h.g,'inside',side_effect=Path),\
                 mock.patch.object(h.core,'reference_pixels',side_effect=lambda r:np.full((3,256,256),r['source_index']/100,np.float32)),\
                 mock.patch.object(h.core,'finalists',return_value={'status':'fixture complete'}) as shortlist:
                result=h.scores(request,Evaluator(),ledger,lambda:None,root/'out')
            rows=h.core.checked(result['metric_rows'])
            self.assertEqual(result['actual_unique_pairs'],100);self.assertEqual(result['reused_pairs'],43100)
            self.assertEqual(result['counts']['completed']['lpips_alexnet_backbone_forward'],200)
            self.assertEqual(rows[0]['metric_pair_key'],rows[1]['metric_pair_key'])
            self.assertNotEqual(rows[0]['metric_pair_key'],rows[432]['metric_pair_key'])
            self.assertNotEqual(rows[0]['actual_RX_status'],rows[1]['actual_RX_status'])
            self.assertEqual(len(shortlist.call_args.args[0]),43200)
            self.assertEqual(rows[0]['dinov2_vitl14_cosine'],0.)
            self.assertEqual(rows[432]['dinov2_vitl14_cosine'],.01)


if __name__=='__main__':unittest.main()
