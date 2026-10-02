import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import sys
import cache_handoff as c
import final_runner
import concurrent_runner
from test_replay_compat import original as frozen_replay, locate
import ast


def fixture(directory):
    directory = Path(directory)
    bound = directory / 'frozen.txt'
    bound.write_text('frozen source / model / policy engineering fixture')
    bindings = {str(bound): c.sha(bound)}
    sources = []
    for i in range(100):
        inv = sorted([[s, f'{s}-{i}', f'original-full-row-{s}-{i}'] for s in c.STUDIES])
        sources.append(dict(source_index=i, source_id=f'image-{i}', preprocessing_id='frozen_preprocess',
            reference_sha256=f'reference-{i}', true_class_index=3, frames=4, row_inventory_sha256=c.identity(inv)))
    r = dict(studies=list(c.STUDIES), synthetic=False, training_updates=0, policy_selection_updates=0,
        metric_batch_size=1, sources=sources, metric_evaluator_identity='eval-id', modelmanifest_sha256='manifest',
        numerical_runtime={'precision': 'highest'}, source_bindings=bindings, frozen_inputs=bindings, asset_inputs=bindings)
    rows, scores = [], {}
    for i, study in enumerate(c.STUDIES):
        rgb = f'image-output-{i % 2}'
        key = c.key_from_hashes(rgb, 'reference-0', 'eval-id')
        values = {k: .25 for k in c.FLOAT_FIELDS}
        values.update(resnet50_prediction=3, resnet50_source_prediction=3, resnet50_top1_label=1,
                      resnet50_top1_source_prediction=1, resnet50_source_top1_label=1)
        scores[key] = dict(image_sha256=rgb, metrics=values)
        rows.append(dict(study=study, row_id=f'{study}-0', source_row_sha256=f'original-full-row-{study}-0',
            metric_cache_key=key, parity=dict(replay_parity_passed=True, synthetic=False,
                target_sha256='reference-0', original_row_sha256=f'{study}-0', rgb_sha256=rgb)))
    b = dict(source_id='image-0', preprocessing_id='frozen_preprocess', reference_sha256='reference-0',
             true_class_index=3, resnet50_source_prediction=3, resnet50_source_top1_label=1)
    payload = dict(status='SEALED_SOURCE_METRICS_COMPLETE', registration_sha256=c.identity(r),
        source_index=0, source=sources[0], synthetic=False, metric_batch_size=1, baseline=b, rows=rows, scores=scores)
    payload['payload_sha256'] = c.identity(payload)
    c.write(directory / 'concurrent_registration.json', r)
    c.write(directory / 'source_checkpoints/000.json', payload)
    c.write(directory / 'runtime_registration.json', {'source_bindings': bindings})
    c.write(directory / 'partial_launch.json', {'pid': 123, 'start_ticks': '9876'})
    c.snapshot_receipt(directory, r)
    return r, payload


def reseal(payload):
    payload['payload_sha256'] = c.identity({k: v for k, v in payload.items() if k != 'payload_sha256'})


def migration_fixture(directory):
    root = Path(directory)
    olddir, newdir = root/'R3', root/'R4'
    olddir.mkdir(); newdir.mkdir()
    old, first = fixture(olddir)
    common = dict(old['source_bindings'])
    runtime_bindings = []
    for folder in (olddir, newdir):
        runtime = folder/'runtime'; runtime.mkdir()
        compat, schedule = runtime/'replay_compat.py', runtime/'controller.py'
        compat.write_text('exact unchanged compatibility science')
        schedule.write_text('operational schedule ' + folder.name)
        runtime_bindings.append({str(compat): c.sha(compat), str(schedule): c.sha(schedule)})
    old.update(schema_version=1, original_m2_read_or_changed=False,
        replay_compatibility={'alias': 'same double precision latent error'},
        original_replay_inventory={'tolerances': frozen_replay.PARITY_TOLERANCES}, gpu_memory_fraction=.45,
        source_bindings=dict(common, **runtime_bindings[0]))
    assets = {}
    for name in ('assets_complete.json', 'modelmanifest.json', 'models_qualification.json'):
        path = root/name; c.write(path, {'unchanged':name}); assets[str(path)] = c.sha(path)
    old['asset_inputs'] = dict(assets)
    new = copy.deepcopy(old); new['source_bindings'] = dict(common, **runtime_bindings[1])
    for folder, reg in ((olddir, old), (newdir, new)):
        q = dict(status='CONCURRENT_SCALAR_QUALIFICATION_PASS', comparison={'passed': True},
            synthetic_images=True, scientific_result=False,
            binding={k:reg[k] for k in ('metric_evaluator_identity','modelmanifest_sha256','numerical_runtime',
                                       'source_bindings','gpu_memory_fraction')})
        q['payload_sha256'] = c.identity(q)
        path = folder/'concurrent_scalar_qualification.json'; c.write(path, q)
        reg['asset_inputs'][str(path)] = c.sha(path)
    c.write(olddir/'concurrent_registration.json',old)
    for index in range(6):
        payload = copy.deepcopy(first)
        payload.update(registration_sha256=c.identity(old), source_index=index, source=old['sources'][index])
        payload['baseline'].update(source_id=f'image-{index}',reference_sha256=f'reference-{index}')
        previous_scores=payload['scores']; payload['scores']={}
        for row in payload['rows']:
            study=row['study']; entry=previous_scores[row['metric_cache_key']]
            key=c.key_from_hashes(entry['image_sha256'],f'reference-{index}','eval-id')
            payload['scores'][key]=entry
            row.update(row_id=f'{study}-{index}',source_row_sha256=f'original-full-row-{study}-{index}',metric_cache_key=key)
            row['parity'].update(original_row_sha256=f'{study}-{index}',target_sha256=f'reference-{index}')
        reseal(payload); c.write(olddir/'source_checkpoints'/f'{index:03}.json',payload)
    c.snapshot_receipt(olddir,old)
    return olddir, newdir, new


class InheritanceTests(unittest.TestCase):
    def test_six_sources_exact_copy_then_resume_and_final_scalar_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            olddir,newdir,reg=migration_fixture(tmp)
            snapshot=c.CacheSnapshot(olddir)
            original=dict(snapshot.bindings)
            reg['inherited_cache']=c.inheritance_contract(snapshot,reg,newdir/'runtime')
            reg['asset_inputs'].update(snapshot.bindings)
            c.seal(newdir/'concurrent_registration.json',reg)
            self.assertEqual(c.import_snapshot(snapshot,newdir,reg),list(range(6)))
            copies={str(p):c.sha(p) for p in (newdir/'source_checkpoints').glob('*.json')}
            self.assertEqual(c.import_snapshot(snapshot,newdir,reg),list(range(6)))
            c.seal(newdir/'concurrent_registration.json',reg); c.verify(copies); c.verify(original)
            c.write(newdir/'runtime_registration.json',{'source_bindings':reg['source_bindings']})
            c.write(newdir/'partial_launch.json',{'pid':456,'start_ticks':'12345'})
            result=c.snapshot_receipt(newdir,reg)
            self.assertEqual(result['sources_complete'],6)
            adopted=c.CacheSnapshot(newdir)
            for index in range(6):
                scores,actual=adopted.scores_for(f'reference-{index}','eval-id',3,3,{'precision':'highest'})
                self.assertEqual(actual,index); self.assertEqual(len(scores),2)
                old=c.read(snapshot.paths[index]); new=c.read(adopted.paths[index])
                self.assertEqual(old['scores'],new['scores']); self.assertEqual(old['rows'],new['rows'])
                self.assertEqual(old['baseline'],new['baseline'])
            self.assertEqual(adopted.provenance()['inherited_cache']['source_indexes'],list(range(6)))
            self.assertEqual(adopted.scores_for('reference-6','eval-id',3,3,{'precision':'highest'}),({},6))
            c.verify(original)

    def test_changed_science_or_unqualified_model_cannot_inherit(self):
        with tempfile.TemporaryDirectory() as tmp:
            olddir,newdir,reg=migration_fixture(tmp); snapshot=c.CacheSnapshot(olddir)
            for field,value in [('metric_evaluator_identity','other'),('numerical_runtime',{'precision':'medium'}),
                                ('metric_batch_size',2),('replay_compatibility',{'alias':'different'}),
                                ('original_replay_inventory',{'tolerances':(1.,1.)})]:
                changed=copy.deepcopy(reg); changed[field]=value
                with self.assertRaisesRegex(RuntimeError,'scientific registration'):
                    c.inheritance_contract(snapshot,changed,newdir/'runtime')
            changed=copy.deepcopy(reg); changed['sources'][0]['true_class_index']=4
            with self.assertRaises(RuntimeError): c.inheritance_contract(snapshot,changed,newdir/'runtime')
            compat=newdir/'runtime/replay_compat.py'; compat.write_text('changed alias implementation')
            changed=copy.deepcopy(reg); changed['source_bindings'][str(compat)]=c.sha(compat)
            with self.assertRaisesRegex(RuntimeError,'compatibility source'):
                c.inheritance_contract(snapshot,changed,newdir/'runtime')

    def test_inherited_origin_and_exact_values_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            olddir,newdir,reg=migration_fixture(tmp); snapshot=c.CacheSnapshot(olddir)
            reg['inherited_cache']=c.inheritance_contract(snapshot,reg,newdir/'runtime')
            reg['asset_inputs'].update(snapshot.bindings)
            c.import_snapshot(snapshot,newdir,reg)
            original=c.read(newdir/'source_checkpoints/000.json')
            for mutation in (lambda p:p['scores'][next(iter(p['scores']))]['metrics'].update(dists=.5),
                             lambda p:p['rows'][0]['parity'].update(extra='changed'),
                             lambda p:p.pop('inherited_origin'),
                             lambda p:p['inherited_origin'].update(checkpoint_sha256='f'*64)):
                changed=copy.deepcopy(original); mutation(changed); reseal(changed)
                with self.assertRaises(RuntimeError): c.validate_checkpoint(changed,reg,0)
            path=snapshot.paths[0]; body=c.read(path)
            body['scores'][next(iter(body['scores']))]['metrics']['dists']=.5; reseal(body); c.write(path,body)
            with self.assertRaises(RuntimeError): c.validate_checkpoint(original,reg,0)

    def test_timed_replay_preserves_iterator_items_and_exception(self):
        clock=iter([0.,2.,5.,8.,10.,11.])
        timing={'replay_seconds':0.}
        with mock.patch.object(concurrent_runner.time,'perf_counter',side_effect=lambda:next(clock)):
            self.assertEqual(list(concurrent_runner.timed_replay(iter(['a','b']),timing)),['a','b'])
        self.assertEqual(timing['replay_seconds'],6.)
        def broken():
            yield 'first'
            raise RuntimeError('original scientific failure')
        with self.assertRaisesRegex(RuntimeError,'original scientific failure'):
            list(concurrent_runner.timed_replay(broken(),{'replay_seconds':0.}))


class CacheTests(unittest.TestCase):
    def test_actual_manifest_and_clip_tuples_resume_then_reuse_source_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, p = fixture(tmp)
            mock_engine = SimpleNamespace(synthetic=False, rows={'N512': []}, records=[],
                loaded={'identity': 'frozen-visual-model'}, artifacts={'input': 'digest'}, legacy_check=True)
            r['original_replay_inventory'] = frozen_replay.ReplayEngine.manifest(mock_engine)
            model_path = locate('experiments/unified-metrics-20261002/metric_models.py',
                                '.research/metrics_20261002/source/metric_models.py')
            tree = ast.parse(model_path.read_text())
            constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                         if isinstance(n, ast.Assign) and len(n.targets) == 1
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('CLIP_MEAN', 'CLIP_STD')}
            metadata = {'metrics': {'clip': {'preprocessing': {'mean': constants['CLIP_MEAN'], 'std': constants['CLIP_STD']}}}}
            self.assertIsInstance(r['original_replay_inventory']['parity_tolerances']['psnr_db'], tuple)
            self.assertIsInstance(metadata['metrics']['clip']['preprocessing']['mean'], tuple)
            regpath, metapath = Path(tmp)/'concurrent_registration.json', Path(tmp)/'model_metadata.json'
            p['registration_sha256'] = c.identity(r); reseal(p)
            c.write(regpath, r); c.write(Path(tmp)/'source_checkpoints/000.json', p); c.seal(metapath, metadata)
            initial = {str(path): c.sha(path) for path in (regpath, metapath, Path(tmp)/'source_checkpoints/000.json')}
            self.assertNotEqual(c.read(regpath), r)  # Exact original failure mechanism.
            c.seal(regpath, r); c.seal(metapath, metadata)
            c.verify(initial)
            c.validate_checkpoint(c.read(Path(tmp)/'source_checkpoints/000.json'), r, 0)
            self.assertEqual(c.identity(r), c.identity(c.read(regpath)))
            second = copy.deepcopy(p)
            second.update(source_index=1, source=r['sources'][1])
            second['baseline'].update(source_id='image-1', reference_sha256='reference-1')
            old_scores = second['scores']; second['scores'] = {}
            for row in second['rows']:
                study = row['study']; entry = old_scores[row['metric_cache_key']]
                key = c.key_from_hashes(entry['image_sha256'], 'reference-1', 'eval-id')
                second['scores'][key] = entry
                row.update(row_id=f'{study}-1', source_row_sha256=f'original-full-row-{study}-1', metric_cache_key=key)
                row['parity'].update(original_row_sha256=f'{study}-1', target_sha256='reference-1')
            reseal(second); c.validate_checkpoint(second, r, 1)
            c.write(Path(tmp)/'source_checkpoints/001.json', second)
            c.snapshot_receipt(tmp, r)
            snapshot = c.CacheSnapshot(tmp)
            self.assertEqual(snapshot.provenance()['source_indexes'], [0, 1])
            c.verify(initial)
            changed = copy.deepcopy(r)
            changed['original_replay_inventory']['parity_tolerances']['psnr_db'] = (1., 0.)
            with self.assertRaises(RuntimeError): c.seal(regpath, changed)
            changed_meta = copy.deepcopy(metadata)
            changed_meta['metrics']['clip']['preprocessing']['mean'] = (0., 0., 0.)
            with self.assertRaises(RuntimeError): c.seal(metapath, changed_meta)
            c.verify(initial)

    def test_only_original_dedicated_resource_exception_is_retryable(self):
        class ResourceBusy(RuntimeError): pass
        runtime = SimpleNamespace(ResourceBusy=ResourceBusy)
        with mock.patch.dict(sys.modules, {'latent_enhancement.runtime': runtime}):
            self.assertTrue(concurrent_runner.original_resource_pause(ResourceBusy('foreign GPU')))
            self.assertFalse(concurrent_runner.original_resource_pause(RuntimeError('foreign GPU ResourceBusy')))
            self.assertFalse(concurrent_runner.original_resource_pause(MemoryError('allocation failure')))
        with mock.patch.dict(sys.modules, {'latent_enhancement.runtime': SimpleNamespace()}):
            self.assertFalse(concurrent_runner.original_resource_pause(ResourceBusy('foreign GPU')))

    def test_snapshot_accepts_completed_subset_and_exact_duplicate_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            registration, payload = fixture(tmp)
            snapshot = c.CacheSnapshot(tmp)
            scores, index = snapshot.scores_for('reference-0', 'eval-id', 3, 3, {'precision': 'highest'})
            self.assertEqual(index, 0)
            self.assertEqual(len(scores), 2)
            self.assertEqual(snapshot.provenance()['source_indexes'], [0])
            self.assertEqual(snapshot.scores_for('reference-1', 'eval-id', 3, 3, {'precision': 'highest'}), ({}, 1))
            receipt = c.snapshot_receipt(tmp, registration)
            self.assertEqual(receipt['sources_complete'], 1)
            self.assertTrue(receipt['parity_passed'])
            c.verify(receipt['inputs']); c.verify(receipt['outputs'])

    def test_wrong_label_reference_prediction_model_and_runtime_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            snapshot = c.CacheSnapshot(tmp)
            for args in [('reference-0', 'eval-id', 4, 3, {'precision': 'highest'}),
                         ('reference-0', 'changed-model', 3, 3, {'precision': 'highest'}),
                         ('reference-0', 'eval-id', 3, 4, {'precision': 'highest'}),
                         ('reference-0', 'eval-id', 3, 3, {'precision': 'medium'}),
                         ('different-reference', 'eval-id', 3, 3, {'precision': 'highest'})]:
                with self.assertRaises(RuntimeError):
                    snapshot.scores_for(*args)

    def test_full_original_row_hash_and_parity_rejected_even_with_new_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, original = fixture(tmp)
            for mutate in (lambda p: p['rows'][0].update(source_row_sha256='changed-K-or-policy'),
                           lambda p: p['rows'][0]['parity'].update(replay_parity_passed=False),
                           lambda p: p['rows'][0]['parity'].update(target_sha256='changed'),
                           lambda p: p['baseline'].update(true_class_index=5)):
                p = copy.deepcopy(original)
                mutate(p); reseal(p)
                with self.assertRaises(RuntimeError):
                    c.validate_checkpoint(p, r, 0)

    def test_cache_values_fields_finiteness_and_classifier_denominators(self):
        values = dict.fromkeys(c.FLOAT_FIELDS, .1)
        values.update(resnet50_prediction=3, resnet50_source_prediction=3, resnet50_top1_label=1,
                      resnet50_top1_source_prediction=1, resnet50_source_top1_label=1)
        c.validate_scores(values, 3, 3)
        for delta in ({'dists': float('nan')}, {'resnet50_top1_label': 0}, {'resnet50_prediction': 1001},
                      {'resnet50_source_prediction': 4}, {'resnet50_top1_label': True}, {'unregistered': .3}):
            with self.assertRaises(RuntimeError):
                c.validate_scores(dict(values, **delta), 3, 3)

    def test_changed_bound_file_or_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, payload = fixture(tmp)
            snapshot = c.CacheSnapshot(tmp)
            path = Path(tmp) / 'source_checkpoints/000.json'
            payload['baseline']['resnet50_source_prediction'] = 1
            c.write(path, payload)
            with self.assertRaises(RuntimeError):
                snapshot.scores_for('reference-0', 'eval-id', 3, 3, {'precision': 'highest'})
            with self.assertRaises(RuntimeError):
                c.CacheSnapshot(tmp)
            fixture_path = Path(tmp) / 'frozen.txt'
            fixture_path.write_text('changed')
            with self.assertRaises(RuntimeError):
                c.verify(r['frozen_inputs'])

    def test_unsealed_or_added_checkpoint_cannot_enter_final_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            c.write(Path(tmp) / 'source_checkpoints/001.json', {'not': 'a sealed registered checkpoint'})
            with self.assertRaisesRegex(RuntimeError, 'inventory differs'):
                c.CacheSnapshot(tmp)

    def test_replay_iteration_order_does_not_change_row_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, p = fixture(tmp)
            p['rows'].reverse(); reseal(p)
            c.validate_checkpoint(p, r, 0)
            p['rows'][0] = p['rows'][1]; reseal(p)
            with self.assertRaises(RuntimeError):
                c.validate_checkpoint(p, r, 0)


class BridgeTests(unittest.TestCase):
    def fake_runner(self, directory):
        class Scorer:
            def __init__(self, *args): self.cache = {}; self.forward_count = 0
            @staticmethod
            def metric_values(values):
                assert values['label_conditioned'] is False
                return {k: v for k, v in values.items() if k != 'label_conditioned'}
            def add(self, key, rgb, row):
                if key in self.cache: row.update(self.cache[key])
                else: self.forward_count += 1; row['new_forward'] = True
        result, out = Path(directory) / 'result', Path(directory) / 'out'
        result.mkdir(); out.mkdir()
        return SimpleNamespace(PendingMetricScorer=Scorer, read=c.read, write=c.write,
            source_bindings=lambda: {'frozen': 'sha'}, RESULT=result, OUT=out,
            runtime_flags=lambda torch: {'precision': 'highest'})

    def test_cache_only_seeds_scalars_and_preserves_final_row_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            snapshot = c.CacheSnapshot(tmp)
            runner = self.fake_runner(tmp)
            old = runner.PendingMetricScorer
            replay = SimpleNamespace(rgb_fingerprint=lambda rgb: 'reference-0')
            evaluator = SimpleNamespace(identity=lambda: 'eval-id')
            with final_runner.bridge(runner, replay, None, snapshot) as reuse:
                scorer = runner.PendingMetricScorer(evaluator, None, ['rgb'], {'resnet50': [3]}, 3, 8, [1, 8])
                key = next(iter(scorer.cache))
                row = {'label_conditioned': True, 'is_main_conclusion': False, 'old_psnr': 20}
                scorer.add(key, 'rgb', row)
                self.assertEqual(scorer.forward_count, 0)
                self.assertTrue(row['label_conditioned'])
                self.assertFalse(row['is_main_conclusion'])
                self.assertEqual(row['old_psnr'], 20)
                self.assertIn('clip_image_cosine', row)
                self.assertEqual(reuse['cache_hit_frames'], 1)
                scorer.add('new-exact-pixel-key', 'new-rgb', {})
                self.assertEqual(scorer.forward_count, 1)
            self.assertIs(runner.PendingMetricScorer, old)

    def test_registration_mutation_and_resume_comparison_are_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, _ = fixture(tmp)
            snapshot, runner = c.CacheSnapshot(tmp), self.fake_runner(tmp)
            body = dict(source_ids=[s['source_id'] for s in r['sources']], modelmanifest_sha256='manifest',
                        metric_evaluator_identity='eval-id', numerical_runtime={'precision': 'highest'},
                        original_replay_inventory={'parity_tolerances': frozen_replay.PARITY_TOLERANCES})
            original = copy.deepcopy(body)
            path = runner.RESULT / 'metrics_registration.json'
            with final_runner.bridge(runner, None, None, snapshot):
                runner.write(path, body)
                self.assertIn('concurrent_cache', body)
                self.assertEqual(c.identity(c.read(path)), c.identity(body))
                self.assertEqual(runner.read(path), original)
                self.assertFalse(runner.read(path) != original)
                changed = copy.deepcopy(original)
                changed['original_replay_inventory']['parity_tolerances']['psnr_db'] = (1., 0.)
                self.assertTrue(runner.read(path) != changed)
            with final_runner.bridge(runner, None, None, snapshot):
                self.assertEqual(runner.read(path), original)
                second = copy.deepcopy(original)
                runner.write(path, second)
                self.assertEqual(body, second)

    def test_bridge_reverts_overrides_after_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            snapshot, runner = c.CacheSnapshot(tmp), self.fake_runner(tmp)
            original = (runner.PendingMetricScorer, runner.read, runner.write, runner.source_bindings)
            with self.assertRaisesRegex(RuntimeError, 'fixture-error'):
                with final_runner.bridge(runner, None, None, snapshot):
                    raise RuntimeError('fixture-error')
            self.assertEqual((runner.PendingMetricScorer, runner.read, runner.write, runner.source_bindings), original)

    def test_final_requires_checked_published_canonical_runtime_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / 'outputs/METRIC-CONCURRENT-R3-20261002/runtime'
            canonical = root / 'experiments/metric-concurrent-20261002'
            runtime.mkdir(parents=True); canonical.mkdir(parents=True)
            a, b = runtime / 'bridge.py', canonical / 'bridge.py'
            a.write_text('reviewed-source'); b.write_text('reviewed-source')
            own = {str(a): c.sha(a)}
            record = dict(status='PUSHED', checks='PASS', commit='a'*40, remote_commit='a'*40,
                          runtime_source_bindings=own, source_bindings={str(b): c.sha(b)})
            path = runtime.parent / 'extension_source_publication.json'
            c.write(path, record)
            proof = final_runner.published_extension(root, runtime.parent, own)
            self.assertIn(str(path), proof)
            record['status'] = 'COMMITTED'; c.write(path, record)
            with self.assertRaisesRegex(RuntimeError, 'not verified'):
                final_runner.published_extension(root, runtime.parent, own)
            record['status'] = 'PUSHED'; c.write(path, record)
            b.write_text('different-source')
            with self.assertRaises(RuntimeError):
                final_runner.published_extension(root, runtime.parent, own)


if __name__ == '__main__':
    unittest.main()
