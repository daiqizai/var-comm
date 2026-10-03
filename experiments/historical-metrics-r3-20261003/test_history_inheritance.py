"""CPU inheritance fixtures; no model setup, inference or real metric claims."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

import history_inheritance as h
from history_common import NEW_METRICS, identity, read, write, write_csv, sha, row_ids
from history_float_cache import SourceImages, scored_fingerprint
from historical_selected_references import SelectedReferences, STUDY


class InheritanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name).resolve()
        cls.study = 'FINAL_P2048_P3060'
        cls.out, cls.result = h.paths(cls.root, cls.study, previous=True)
        cls.out.mkdir(parents=True); cls.result.mkdir(parents=True)
        source = cls.root/'original.py'; source.write_text('# original fixture\n')
        cls.previous = dict(source_bindings={str(source):sha(source)})
        cls.queue_path = cls.root/'outputs'/h.PREVIOUS/'queue_registration.json'
        write(cls.queue_path, cls.previous)
        cls.publication_path = cls.queue_path.with_name('source_publication.json')
        cls.publication = dict(source_bindings=cls.previous['source_bindings'],published_files=cls.previous['source_bindings'])
        write(cls.publication_path, cls.publication)
        cls.expected = [[dict(method='P',image_id=f'source{i}',psnr_db='20.0')] for i in range(100)]
        cls.ids = [row_ids(cls.study,i,rows) for i,rows in enumerate(cls.expected)]
        numerical = {'fixture_flag':False}
        metric = cls.root/'results/unified_metrics_20261002/metrics_registration.json'
        write(metric, dict(metric_evaluator_identity='evaluator',modelmanifest_sha256='model',numerical_runtime=numerical))
        batch = dict(status='METRIC_BATCH_QUALIFICATION_PASS',metric_evaluator_identity='evaluator',
            modelmanifest_sha256='model',chosen_batch_size=1,qualified_batch_sizes=[1],rng_state_preserved=True,
            binding=dict(runtime_context=dict(study=cls.study,native_bindings={},numerical_runtime=numerical,
                                             native_numerical_runtime=numerical)))
        batch['payload_sha256'] = identity(batch)
        write(cls.out/'metric_batch_qualification.json',batch)
        cls.reg = dict(status='REGISTERED',synthetic=False,study=cls.study,adapter='historical_selected_budget',
            source_bindings=cls.previous['source_bindings'],queue_registration_sha256=sha(cls.queue_path),
            source_publication_sha256=sha(cls.publication_path),source_count=100,frame_count=100,
            expected_ids_sha256=identity(cls.ids),source_ids=[f'source{i}' for i in range(100)],
            source_identity=[dict(image_id=f'source{i}',class_index=1,preprocessing_id=f'pixels{i}') for i in range(100)],
            modelmanifest_sha256='model',evaluator_identity='evaluator',numerical_runtime=numerical,
            native_numerical_runtime=numerical,original_bindings={},original_metric_registration_sha256=sha(metric),
            metric_batch_size=1,metric_qualified_batch_sizes=[1],
            batch_qualification_sha256=sha(cls.out/'metric_batch_qualification.json'))
        write(cls.result/'registration.json', cls.reg); write(cls.result/'model_metadata.json',{'fixture':True})
        evaluation = {k:cls.reg[k] for k in h.EVALUATION_FIELDS}
        rgb = np.full((3,256,256),.3,dtype=np.float32); target = np.full((3,256,256),.2,dtype=np.float32)
        rows, baselines = [], []
        for i in range(100):
            original = cls.expected[i][0]; rid=cls.ids[i][0]
            row = dict(original,history_row_id=rid,history_study=cls.study,history_source_index=i,
                history_source_id=f'source{i}',history_preprocessing_id=f'pixels{i}',history_true_class_index=1,
                history_original_row_sha256=identity(original),history_replay_parity_passed=True,
                history_modelmanifest_sha256='model',history_image_sha256=scored_fingerprint(rgb),
                history_reference_sha256=scored_fingerprint(target))
            row.update({'new_'+k:.5 for k in NEW_METRICS})
            baseline = dict(source_id=f'source{i}',preprocessing_id=f'pixels{i}',true_class_index=1,
                            reference_sha256=scored_fingerprint(target))
            images = SourceImages(target); images.add(rid,rgb)
            cp = dict(binding=identity(cls.reg),source_index=i,rows=[row],
                parity=[dict(history_row_id=rid,replay_parity_passed=True,synthetic=False)],baseline=baseline,
                evaluation_identity=evaluation,metric_batch_sizes_used=[1],unique_images=1,
                float_reconstructions=images.save(cls.out/'reconstructions'/f'{i:04d}.npz',[rid]))
            cp['payload_sha256'] = identity(cp)
            write(cls.out/'source_checkpoints'/f'{i:04d}.json',cp)
            rows.append(row); baselines.append(baseline)
        write_csv(cls.result/'metrics_per_frame.csv',rows);write_csv(cls.result/'source_baseline.csv',baselines)
        cp0 = cls.out/'source_checkpoints/0000.json'
        write(cls.out/'first_source_qualification.json',dict(status='REAL_FIRST_SOURCE_PARITY_PASS',
            synthetic=False,parity_passed=True,source_index=0,rows=1,checkpoint=str(cp0),checkpoint_sha256=sha(cp0),
            registration_sha256=sha(cls.result/'registration.json'),original_values_preserved=True,
            evaluation_identity=evaluation,original_rows_sha256=identity([identity(cls.expected[0][0])])))
        done = dict(status='HISTORICAL_STUDY_METRICS_COMPLETE',study=cls.study,parity_passed=True,synthetic=False,
            training_updates=0,policy_selection_updates=0,frames=100,sources=100,
            source_bindings=cls.previous['source_bindings'],
            inputs={str(p):sha(p) for p in (cls.queue_path,cls.publication_path,metric)},
            outputs={str(p):sha(p) for p in cls.result.iterdir()})
        write(cls.out/'completion.json',done)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.patcher = patch.dict(h.INHERITED,{self.study:dict(adapter='historical_selected_budget',frames=100,sources=100)},clear=True)
        self.patcher.start();self.addCleanup(self.patcher.stop)
        self.adapter = SimpleNamespace(expected_rows=lambda i:deepcopy(self.expected[i]))

    def admit(self):
        return h.admit_study(self.root,self.study,self.adapter,self.queue_path,self.previous,self.publication_path,self.publication)

    def alter(self,path,update):
        original = path.read_bytes();self.addCleanup(path.write_bytes,original)
        value=read(path);update(value);write(path,value)

    def test_all_100_sources_verified_without_model_setup_or_old_identity_changes(self):
        original = (self.result/'registration.json').read_bytes()
        receipt = self.admit()
        self.assertEqual(receipt['source_checkpoints_verified'],100)
        self.assertEqual(receipt['float_caches_verified'],100)
        self.assertFalse(receipt['metrics_recomputed'])
        self.assertEqual((self.result/'registration.json').read_bytes(),original)
        for i in range(100):
            self.assertIn(str(self.out/'source_checkpoints'/f'{i:04d}.json'),receipt['bindings'])
            self.assertIn(str(self.out/'reconstructions'/f'{i:04d}.npz'),receipt['bindings'])

    def test_wrong_completed_scope_rejected(self):
        self.alter(self.out/'completion.json',lambda v:v.update(frames=99))
        with self.assertRaisesRegex(RuntimeError,'completion differs'):self.admit()

    def test_original_rows_are_required_even_when_checkpoint_is_resealed(self):
        def changed(value):
            value['rows'][0]['psnr_db']='99.0'
            value['payload_sha256']=identity({k:v for k,v in value.items() if k!='payload_sha256'})
        self.alter(self.out/'source_checkpoints/0000.json',changed)
        with self.assertRaisesRegex(RuntimeError,'Original historical row fields changed'):self.admit()

    def test_metric_change_cannot_disagree_with_original_export(self):
        def changed(value):
            value['rows'][0]['new_dists']=.8
            value['payload_sha256']=identity({k:v for k,v in value.items() if k!='payload_sha256'})
        self.alter(self.out/'source_checkpoints/0000.json',changed)
        with self.assertRaisesRegex(RuntimeError,'exported row differs'):self.admit()

    def test_missing_checkpoint_and_corrupt_cache_are_rejected(self):
        path=self.out/'source_checkpoints/0000.json';raw=path.read_bytes();path.unlink()
        try:
            with self.assertRaisesRegex(RuntimeError,'checkpoints are incomplete'):self.admit()
        finally:path.write_bytes(raw)
        image=self.out/'reconstructions/0000.npz';raw=image.read_bytes();image.write_bytes(raw+b'changed')
        try:
            with self.assertRaisesRegex(RuntimeError,'cache identity differs'):self.admit()
        finally:image.write_bytes(raw)

    def test_first_source_qualification_cannot_be_rebound(self):
        self.alter(self.out/'first_source_qualification.json',lambda v:v.update(registration_sha256='changed'))
        with self.assertRaisesRegex(RuntimeError,'first-source qualification differs'):self.admit()

    def test_source_baseline_must_match_original_registration(self):
        def changed(value):
            value['baseline']['true_class_index']=2
            value['payload_sha256']=identity({k:v for k,v in value.items() if k!='payload_sha256'})
        self.alter(self.out/'source_checkpoints/0000.json',changed)
        with self.assertRaisesRegex(RuntimeError,'baseline identity differs'):self.admit()

    def test_row_source_annotation_must_match_sealed_source(self):
        def changed(value):
            value['rows'][0]['history_true_class_index']=2
            value['payload_sha256']=identity({k:v for k,v in value.items() if k!='payload_sha256'})
        self.alter(self.out/'source_checkpoints/0000.json',changed)
        with self.assertRaisesRegex(RuntimeError,'row/source/reference identity differs'):self.admit()

    def test_inherited_path_is_fixed_to_r2_and_new_study_is_r3(self):
        item=h.map_entry(self.root,self.study);queue={'inherited_studies':{self.study:item}}
        self.assertEqual(h.study_paths(self.root,queue,self.study),(self.out,self.result))
        self.assertIn(h.CURRENT,str(h.study_paths(self.root,queue,STUDY)[0]))
        item['result']=str(self.root/'injected')
        with self.assertRaisesRegex(RuntimeError,'path substitution'):h.study_paths(self.root,queue,self.study)

    def test_reference_native_and_scored_archives_cannot_collide(self):
        with patch.object(SelectedReferences,'_read_originals'):
            adapter=SelectedReferences(self.root,STUDY)
        scored=h.paths(self.root,STUDY)[0]/'reconstructions'/'0000.npz'
        native,_=adapter._cache(0)
        self.assertNotEqual(native,scored)
        self.assertEqual(native.parent.name,'native_reference_reconstructions')
        self.assertIn(h.CURRENT,str(native))


if __name__=='__main__':
    unittest.main()
