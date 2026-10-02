"""ENGINEERING/SYNTHETIC CPU runner contracts; no real images or model weights.

The full main test mocks model/replay modules and writes ONLY to a temporary
ENGINEERING fixture directory. Its outputs are discarded and are not results.
"""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load('metrics_runner_engineering_target', 'runner.py')
replay = load('metrics_replay_runner_engineering', 'replay.py')
analysis = load('metrics_analysis_runner_engineering', 'analysis.py')
batch_speed = load('batch_speed', 'batch_speed.py')


def oldrow(**changes):
    row = dict(source_id='ENGINEERING_000', source_index='0',
        preprocessing_id='ENGINEERING_PREPROCESS', N='512', snr_db='1', noise_seed='2001',
        phy_family='continuous', method='P512', decoder_id='Dc', class_condition='unconditional',
        psnr_db='20', lpips_alex='.2', dino_cosine='.7', dino_mismatched='.1')
    row.update(changes)
    return row


class ArrayTensor:
    def __init__(self, array):
        self.value = np.asarray(array)

    def unsqueeze(self, axis):
        return ArrayTensor(np.expand_dims(self.value, axis))

    def __len__(self):
        return len(self.value)


class FakeEvaluator:
    calls = 0
    prepare_calls = 0

    def __init__(self, manifest, device):
        self.metadata = dict(ENGINEERING_SYNTHETIC=True,
            metrics={name: {'status': 'READY'} for name in
                     ('clip', 'dists', 'dreamsim', 'ms_ssim', 'resnet50', 'dinov2_vitl14')})

    def identity(self):
        return 'a' * 64

    def prepare_reference(self, tensor):
        type(self).prepare_calls += 1
        return dict(resnet50=np.array([0]))

    def expand_reference(self, reference, prepared, count):
        return ArrayTensor(np.repeat(reference.value,count,axis=0)),dict(resnet50=np.repeat(prepared['resnet50'],count))

    def score(self, target, image, labels, label_conditioned, prepared):
        type(self).calls += 1
        return [dict(clip_image_cosine=.8, dinov2_vitl14_cosine=.9, dists=.1, dreamsim=.2, ms_ssim=.6,
            resnet50_prediction=0, resnet50_source_prediction=0,
            resnet50_top1_label=True, resnet50_top1_source_prediction=True,
            resnet50_source_top1_label=True, label_conditioned=False) for _ in range(len(image))]


def fake_batch_qualification(evaluator, device, manifest, out, *, runtime_context):
    path=out/'metric_batch_qualification.json'
    receipt=dict(status=batch_speed.STATUS,chosen_batch_size=1,qualified_batch_sizes=[1],
                 ENGINEERING_SYNTHETIC=True,scientific_result=False,
                 metric_evaluator_identity=evaluator.identity(),modelmanifest_sha256=runner.sha(manifest),
                 runtime_context=runtime_context)
    if path.exists() and runner.read(path)!=receipt:
        raise RuntimeError('ENGINEERING batch receipt changed')
    runner.write(path,receipt)
    return receipt


class DistinctFakeEvaluator(FakeEvaluator):
    def __init__(self):
        super().__init__(None,None)
        self.used_batches=[]

    def score(self, target, image, labels, label_conditioned, prepared):
        self.used_batches.append(len(image))
        rows=super().score(target,image,labels,label_conditioned,prepared)
        for index,row in enumerate(rows):
            row['clip_image_cosine']=float(image.value[index,0,0,0])
        return rows


class FakeEngine:
    def __init__(self, directory):
        self.records = [dict(image_id=f'ENGINEERING_{i:03d}', class_index=0,
                             preprocessing_id='ENGINEERING_PREPROCESS') for i in range(100)]
        self.loaded = {'device': 'ENGINEERING_CPU'}
        self.rows = {k: [] for k in replay.STUDIES}
        self.by_source = {k: {} for k in replay.STUDIES}
        self.layouts = {}
        for i in range(100):
            common = dict(source_id=f'ENGINEERING_{i:03d}', source_index=str(i))
            n512 = [oldrow(**common), oldrow(**common, method='D_C_QPSK', phy_family='QPSK',
                class_condition='C', header_ok='False'),
                oldrow(**common, method='D_C_QPSK_D0', phy_family='QPSK',
                       class_condition='C', header_ok='False', decoder_id='D0')]
            m1 = [oldrow(**common, phy_family=fam, historical_reference='True')
                  for fam in ('QPSK', '16QAM')]
            m2 = [oldrow(**common, N='', method='', control='DIRECT', projection='g8_c32',
                snr_db='clean', noise_seed='0', phy_family='', stage='full_noiseless')]
            for study, rows in (('N512', n512), ('M1', m1), ('M2_ORACLE', m2)):
                self.rows[study].extend(rows); self.by_source[study][i] = rows
        for study, rows in self.rows.items():
            path = directory / (study + '.json')
            path.write_text(json.dumps({'ENGINEERING_SYNTHETIC': True, 'rows': rows}))
            self.layouts[study] = SimpleNamespace(rows=path)
        self.bindings = {str(x.rows): runner.sha(x.rows) for x in self.layouts.values()}

    def metadata(self, row, study):
        return replay.metadata(row, study)

    def setup_bound_files(self):
        return dict(self.bindings)

    def manifest(self):
        return {'ENGINEERING_SYNTHETIC': True, 'rows': {k: len(v) for k, v in self.rows.items()}}

    def target(self, index):
        return np.zeros((3, 256, 256), np.float32)

    def iterate_source(self, study, index):
        for row in self.by_source[study].get(index, []):
            image = np.full((3, 256, 256), .5, np.float32)
            diag = dict(replay_parity_passed=True, synthetic=False, metric_deltas={},
                        rgb_sha256=replay.rgb_fingerprint(image))
            yield dict(row, **self.metadata(row, study)), image, diag

    def verify_frozen(self):
        runner.verify(self.bindings)
        return self.manifest()


class RunnerEngineering(unittest.TestCase):
    def queue(self,batch,qualified):
        evaluator=DistinctFakeEvaluator()
        torch=SimpleNamespace(from_numpy=ArrayTensor,tensor=lambda x,dtype=None:np.asarray(x),int64=np.int64)
        source=ArrayTensor(np.zeros((1,3,256,256),np.float32))
        prepared=evaluator.prepare_reference(source)
        return runner.PendingMetricScorer(evaluator,torch,source,prepared,0,batch,qualified),evaluator

    def test_pending_duplicates_update_all_original_row_objects_in_order(self):
        scorer,evaluator=self.queue(4,[1,2,4])
        rows=[]
        for i,key in enumerate(['a','a','b','c','a','d','b','e','e','f','g']):
            row=dict(row_id=i,label_conditioned=bool(i%2),dino_cosine='.7',old_parity=True)
            rows.append(row)
            scorer.add(key,np.full((3,256,256),(ord(key)-96)/10,np.float32),row)
            self.assertLessEqual(len(scorer.pending),4)
        self.assertNotIn('clip_image_cosine',rows[-1])
        scorer.flush()
        self.assertEqual([row['row_id'] for row in rows],list(range(11)))
        self.assertEqual(evaluator.used_batches,[4,2,1])
        self.assertEqual(len(scorer.cache),7)
        self.assertEqual(scorer.max_pending_unique_images,4)
        self.assertFalse(scorer.pending)
        for i,(key,row) in enumerate(zip(['a','a','b','c','a','d','b','e','e','f','g'],rows)):
            self.assertAlmostEqual(row['clip_image_cosine'],(ord(key)-96)/10,places=6)
            self.assertEqual(row['label_conditioned'],bool(i%2))
            self.assertEqual(row['dino_cosine'],'.7')
            self.assertTrue(row['old_parity'])
        # Already scored repeats use the permanent cache without a new batch.
        extra={};scorer.add('a',np.ones((3,256,256),np.float32),extra);scorer.flush()
        self.assertAlmostEqual(extra['clip_image_cosine'],.1)
        self.assertEqual(evaluator.used_batches,[4,2,1])

    def test_tail_uses_only_qualified_sizes_and_reference_is_prepared_once(self):
        before=DistinctFakeEvaluator.prepare_calls
        scorer,evaluator=self.queue(8,[1,4,8])
        for i in range(7):scorer.add(str(i),np.full((3,256,256),i/10,np.float32),{})
        scorer.flush()
        self.assertEqual(evaluator.used_batches,[4,1,1,1])
        self.assertEqual(DistinctFakeEvaluator.prepare_calls-before,1)

    def test_pending_owns_pixels_and_caches_are_source_local(self):
        scorer,evaluator=self.queue(2,[1,2])
        image=np.full((3,256,256),.2,np.float32);row={}
        scorer.add('same-key',image,row);image.fill(.9);scorer.flush()
        self.assertAlmostEqual(row['clip_image_cosine'],.2)
        second,evaluator2=self.queue(2,[1,2]);secondrow={}
        second.add('same-key',image,secondrow);second.flush()
        self.assertAlmostEqual(secondrow['clip_image_cosine'],.9)
        self.assertEqual(evaluator.used_batches,[1]);self.assertEqual(evaluator2.used_batches,[1])

    def test_batch_failure_does_not_retry_or_cache_partial_output(self):
        for behavior in ('exception','wrong-count','bad-metric'):
            scorer,evaluator=self.queue(2,[1,2]);row_a={};row_b={}
            ordinary=evaluator.score
            def fail(*args,**kwargs):
                if behavior=='exception':raise RuntimeError('ENGINEERING OOM')
                result=ordinary(*args,**kwargs)
                if behavior=='wrong-count':return result[:1]
                result[-1]['dists']=float('nan');return result
            evaluator.score=fail
            scorer.add('a',np.zeros((3,256,256),np.float32),row_a)
            with self.assertRaises(RuntimeError):
                scorer.add('b',np.ones((3,256,256),np.float32),row_b)
            self.assertEqual(scorer.cache,{})
            self.assertEqual(row_a,{})
            self.assertEqual(row_b,{})
            self.assertEqual(len(scorer.pending),2)
            self.assertEqual(scorer.batch_sizes_used,[])

    def test_c_header_erasure_stays_class_conditioned_supplement(self):
        row = oldrow(method='D_C_QPSK', class_condition='C', header_ok='False')
        meta = replay.metadata(row, 'N512')
        self.assertFalse(meta['replay_received_class_used_by_generator'])
        value = runner.normalized(row, 'N512', meta)
        self.assertTrue(value['label_conditioned'])
        self.assertFalse(value['is_main_conclusion'])
        self.assertEqual(value['classification_interpretation'], 'label_conditioned_descriptive_only')

    def test_d0_metadata_exact_decoder_and_reference_scope(self):
        for method in ('D_C_QPSK_D0', 'D_U_16QAM_D0'):
            row = oldrow(method=method, decoder_id='D0', class_condition='C' if method.startswith('D_C') else 'U')
            value = runner.normalized(row, 'N1024', replay.metadata(row, 'N1024'))
            self.assertEqual(value['decoder_id'], 'D0')
            self.assertEqual(value['output_role'], 'reference')
            self.assertFalse(value['is_main_conclusion'])
        row = oldrow(method='D_U_QPSK', decoder_id='Dc', class_condition='U')
        value = runner.normalized(row, 'N512', replay.metadata(row, 'N512'))
        self.assertEqual(value['decoder_id'], 'Dc')
        self.assertTrue(value['is_main_conclusion'])

    def test_clean_oracle_seed0_and_no_paid_budget(self):
        row = oldrow(N='', snr_db='clean', noise_seed='0', method='', control='DIRECT',
                     phy_family='', stage='full_noiseless', projection='g8_c32')
        value = runner.normalized(row, 'M2_ORACLE', replay.metadata(row, 'M2_ORACLE'))
        self.assertEqual(value['snr_db'], 'clean')
        self.assertEqual(value['noise_seed'], '0')
        self.assertEqual(value['N'], '')
        self.assertEqual(value['method'], 'DIRECT')
        self.assertEqual(value['phy_family'], 'oracle')
        self.assertFalse(value['is_main_conclusion'])

    def test_historical_facets_remain_distinct(self):
        rows = [oldrow(phy_family=p, historical_reference='True') for p in ('QPSK', '16QAM')]
        values = [runner.normalized(x, 'M1', replay.metadata(x, 'M1')) for x in rows]
        self.assertNotEqual(analysis.group(values[0]), analysis.group(values[1]))
        self.assertNotEqual(values[0]['replay_row_id'], values[1]['replay_row_id'])
        self.assertTrue(all(x['replay_historical_reference'] for x in values))

    def test_rate_scope_is_source_only_reference(self):
        row = oldrow(N='', snr_db='clean', noise_seed='0', method='rate_curve_entropy',
            phy_family='source_only', stage='source_only_rate_curve', projection='m6_K32',
            m='6', q='32', order='entropy', wireless_claim='False')
        value = runner.normalized(row, 'M1_RATE', replay.metadata(row, 'M1_RATE'))
        self.assertEqual(value['scope'], 'source_only_rate_curve')
        self.assertEqual(value['output_role'], 'source_only_reference')
        self.assertEqual(value['N'], '')
        self.assertFalse(value['is_main_conclusion'])

    def test_checkpoint_binding_coverage_and_parity_rejections(self):
        value = dict(binding='ENGINEERING_BINDING', rows=[dict(replay_row_id='A', replay_parity_passed=True)])
        value['payload_sha256'] = runner.identity(value)
        self.assertIs(runner.checkpoint_valid(value, 'ENGINEERING_BINDING', ['A']), value)
        for binding, ids in (('WRONG', ['A']), ('ENGINEERING_BINDING', ['B'])):
            with self.assertRaises(RuntimeError):
                runner.checkpoint_valid(value, binding, ids)
        for rows in ([dict(replay_row_id='A', replay_parity_passed=False)],
                     [dict(replay_row_id='A', replay_parity_passed=True)] * 2):
            corrupt = dict(binding='ENGINEERING_BINDING', rows=rows)
            corrupt['payload_sha256'] = runner.identity(corrupt)
            with self.assertRaises(RuntimeError):
                runner.checkpoint_valid(corrupt, 'ENGINEERING_BINDING', ['A'])
        corrupt = dict(value, other='changed')
        with self.assertRaises(RuntimeError):
            runner.checkpoint_valid(corrupt, 'ENGINEERING_BINDING', ['A'])

    def test_full_main_and_resume_keep_rows_and_source_local_cache(self):
        with tempfile.TemporaryDirectory(prefix='ENGINEERING_SYNTHETIC_METRICS_') as temporary:
            root = Path(temporary); source = root / 'source'; source.mkdir()
            (source / 'engineering_fixture.py').write_text('# ENGINEERING/SYNTHETIC; not a scientific output\n')
            out, result, parent = root / 'out', root / 'result', root / 'parent'
            out.mkdir(); parent.mkdir()
            commit = '1' * 40
            publication = dict(status='PUSHED', commit=commit, remote_commit=commit, source_bindings={})
            runner.write(parent / 'completion.json', dict(status='AUTHORIZED_TWO_METHODS_COMPLETE', publication=publication))
            runner.write(parent / 'supervisor_status.json', dict(status='COMPLETE'))
            runner.write(out / 'source_publication.json', publication)
            runner.write(out / 'modelmanifest.json', {'ENGINEERING_SYNTHETIC': True})
            runner.write(out / 'models_qualification.json', dict(status='REAL_MODEL_WEIGHTS_QUALIFICATION_PASS',
                modelmanifest_sha256=runner.sha(out / 'modelmanifest.json'), source_bindings={},
                ENGINEERING_SYNTHETIC=True))
            engine = FakeEngine(root)
            FakeEvaluator.calls = FakeEvaluator.prepare_calls = 0
            fake_torch = SimpleNamespace(from_numpy=lambda x: ArrayTensor(x),
                tensor=lambda x, dtype=None: np.asarray(x), int64=np.int64,
                backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                    cudnn=SimpleNamespace(allow_tf32=False, benchmark=False, deterministic=True)),
                get_float32_matmul_precision=lambda: 'highest',
                are_deterministic_algorithms_enabled=lambda: True,
                get_num_threads=lambda: 2, get_num_interop_threads=lambda: 2)
            fake_replay = SimpleNamespace(create_engine=lambda _: engine, row_id=replay.row_id,
                rgb_fingerprint=replay.rgb_fingerprint, metric_cache_key=replay.metric_cache_key)
            modules = {'torch': fake_torch, 'replay': fake_replay,
                       'metric_models': SimpleNamespace(MetricEvaluator=FakeEvaluator), 'analysis': analysis}
            with mock.patch.multiple(runner, HERE=source, ROOT=root, OUT=out, RESULT=result, PARENT=parent), \
                 mock.patch.dict(sys.modules, modules), \
                 mock.patch.object(batch_speed,'qualify_and_select_batch',side_effect=fake_batch_qualification):
                runner.main()
                with (result / 'metrics_per_frame.csv').open(newline='') as h:
                    rows = list(csv.DictReader(h))
                self.assertEqual(len(rows), 600)
                self.assertEqual(len({x['replay_row_id'] for x in rows}), 600)
                self.assertEqual(FakeEvaluator.calls, 100)  # 6 equal RGB rows per source.
                self.assertEqual(FakeEvaluator.prepare_calls, 100)
                cfail = [x for x in rows if x['method'] == 'D_C_QPSK']
                self.assertEqual(len(cfail), 100)
                self.assertTrue(all(x['label_conditioned'] == 'True' and x['is_main_conclusion'] == 'False' for x in cfail))
                self.assertTrue(all(float(x['resnet50_top1_label']) == 1 for x in rows))
                self.assertTrue(all(float(x['dinov2_vitl14_cosine']) == .9 for x in rows))
                self.assertTrue(all(float(x['dino_cosine']) == .7 for x in rows))
                self.assertIn('dinov2_vitl14_cosine', runner.read(result / 'metrics_registration.json')['metric_availability'])
                clean = [x for x in rows if x['experiment'] == 'M2_ORACLE']
                self.assertEqual(len(clean), 100)
                self.assertTrue(all(x['noise_seed'] == '0' and x['snr_db'] == 'clean' and x['N'] == '' for x in clean))
                before = runner.sha(result / 'metrics_per_frame.csv')
                runner.main()
                self.assertEqual(FakeEvaluator.calls, 100)
                self.assertEqual(runner.sha(result / 'metrics_per_frame.csv'), before)
                self.assertEqual(runner.read(out / 'scoring_completion.json')['frames'], 600)
                registration=runner.read(result/'metrics_registration.json')
                completion=runner.read(out/'scoring_completion.json')
                batchsha=runner.sha(out/'metric_batch_qualification.json')
                self.assertEqual(registration['metric_batch_size'],1)
                self.assertEqual(registration['metric_batch_qualification_sha256'],batchsha)
                self.assertEqual(completion['metric_batch_qualification_sha256'],batchsha)
                self.assertEqual(completion['inputs'][str(out/'metric_batch_qualification.json')],batchsha)
                self.assertEqual(runner.sha(result/'metric_batch_qualification.json'),batchsha)
                class ChangesRuntime(FakeEvaluator):
                    def __init__(self, manifest, device):
                        super().__init__(manifest, device)
                        fake_torch.backends.cuda.matmul.allow_tf32 = True
                with mock.patch.dict(sys.modules, {'metric_models': SimpleNamespace(MetricEvaluator=ChangesRuntime)}):
                    with self.assertRaisesRegex(RuntimeError, 'runtime'):
                        runner.main()
                fake_torch.backends.cuda.matmul.allow_tf32 = False
                # A changed old input cannot silently reuse previous checkpoints.
                engine.layouts['N512'].rows.write_text('ENGINEERING_INPUT_MUTATED')
                with self.assertRaises(RuntimeError):
                    runner.main()


if __name__ == '__main__':
    unittest.main(verbosity=2)
