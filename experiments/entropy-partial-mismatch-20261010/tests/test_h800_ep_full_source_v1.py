"""CPU-only guards for the unexecuted finite original900 source owner."""
import copy
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h800_ep_full_source_v1 as h
import ep_source_codec as partial


class Source900Tests(unittest.TestCase):
    def test_private_owner_preserves_older_caps_and_clones_identity(self):
        old = copy.deepcopy(h.source100.CAPS)
        runner = h.engine()
        self.assertEqual(h.source100.CAPS, old)
        self.assertEqual(h.core.SOURCE_CAPS, old)
        self.assertIsNot(runner.CAPS, h.source100.CAPS)
        self.assertEqual(runner.CAPS, dict(model_load=1, encoder=0, source_tx=900,
            source_rx=0, var_render=0, prior_scale=9000, decoder_forward=0))
        self.assertEqual(runner.worker_identity.__globals__['__file__'], h.__file__)
        self.assertEqual(runner.check_tools.__globals__['__file__'], h.__file__)

    def test_changed_source100_or_core_is_rejected(self):
        for name in ('SOURCE100_CODE_SHA', 'CORE_SHA'):
            with self.subTest(name=name), mock.patch.object(h, name, '0' * 64), self.assertRaises(RuntimeError):
                h.engine()

    def test_independent_finite_execution_bound(self):
        window = h.gate.source.execution(time.time() + 3600, 3600)
        self.assertEqual(window['max_seconds'], 3600)
        self.assertTrue(window['input_contract_is_not_execution_budget'])
        for seconds in (0, 3601):
            with self.assertRaises(RuntimeError):
                h.gate.source.execution(time.time() + 4000, seconds)

    def test_only_original_indices100_through999_can_encode(self):
        for index in (-1, 0, 32, 99, 1000, True):
            with self.subTest(index=index), self.assertRaisesRegex(RuntimeError, '0100..0999'):
                h.source900(dict(source_index=index), None, None, None, None, None, None)

    def input_fixture(self):
        records = [dict(source_index=i, source_id=f'source{i}',
            archive=dict(path=f'/first/{i}.npz', sha256='a' * 64),
            checkpoint=dict(path=f'/first/{i}.json', sha256='b' * 64)) for i in range(100)]
        restore = dict(schema='CAL900_ORIGINAL_ASSETS_RESTORED_V1', source_indices=list(range(100, 1000)),
            file_count=1800, source_manifest_sha256=h.MANIFEST_SHA,
            source_completion_sha256=h.SOURCE_COMPLETION_SHA,
            mapping=[dict(original_path=h.OLD_ASSET + f'{folder}/{i:04d}.{suffix}',
                actual_path=f'/restored/{folder}/{i}.{suffix}', sha256='c' * 64)
                for i in range(100, 1000) for folder, suffix in [('sources', 'npz'), ('source_checkpoints', 'json')]])
        return records, restore

    def test_all1000_mapping_preserves100_and_uses_only900_restored(self):
        first, restore = self.input_fixture()
        all_records = first + [dict(source_index=i) for i in range(100, 1000)]
        with mock.patch.object(h.source100, 'inputs', return_value=(first, {'prior': 'bound'})), \
             mock.patch.object(h.g, 'inside', side_effect=Path), \
             mock.patch.object(h.g, 'checked_json', return_value=restore), \
             mock.patch.object(h, 'full_input_records', return_value=all_records) as verify:
            result, pin = h.inputs()
        self.assertEqual(result, all_records)
        mapping = verify.call_args.args[2]
        self.assertEqual(len(mapping), 2000)
        self.assertEqual(mapping[h.OLD_ASSET + 'sources/0099.npz']['path'], '/first/99.npz')
        self.assertEqual(mapping[h.OLD_ASSET + 'sources/0100.npz']['path'], '/restored/sources/100.npz')
        self.assertEqual(mapping[h.OLD_ASSET + 'sources/0999.npz']['path'], '/restored/sources/999.npz')
        self.assertEqual(pin['sha256'], h.RESTORE_SHA)

    def test_restoration_cannot_duplicate_existing100(self):
        first, restore = self.input_fixture()
        restore['mapping'][0]['original_path'] = h.OLD_ASSET + 'sources/0099.npz'
        with mock.patch.object(h.source100, 'inputs', return_value=(first, {})), \
             mock.patch.object(h.g, 'inside', side_effect=Path), \
             mock.patch.object(h.g, 'checked_json', return_value=restore), \
             self.assertRaisesRegex(RuntimeError, 'duplicates'):
            h.inputs()

    def test_mapping_cannot_replace_previously_completed100(self):
        first, restore = self.input_fixture()
        altered = copy.deepcopy(first); altered[0]['source_id'] = 'new_source'
        with mock.patch.object(h.source100, 'inputs', return_value=(first, {})), \
             mock.patch.object(h.g, 'inside', side_effect=Path), \
             mock.patch.object(h.g, 'checked_json', return_value=restore), \
             mock.patch.object(h, 'full_input_records', return_value=altered), \
             self.assertRaisesRegex(RuntimeError, 'original100 input mapping changed'):
            h.inputs()

    def full_metadata_fixture(self):
        rows, outputs, relocation, checkpoints = [], {}, {}, {}
        for i in range(1000):
            archive = h.OLD_ASSET + f'sources/{i:04d}.npz'
            checkpoint = h.OLD_ASSET + f'source_checkpoints/{i:04d}.json'
            row = dict(source_index=i, source_id=f'original{i}', preprocessing_id='p' * 64,
                tokens_sha256='t' * 64, archive=archive, checkpoint=checkpoint, checkpoint_sha256='a' * 64)
            rows.append(row)
            outputs.update({archive: 'a' * 64, checkpoint: 'a' * 64})
            relocation[archive] = dict(path=f'/fixture/sources/{i:04d}.npz', sha256='a' * 64)
            relocation[checkpoint] = dict(path=f'/fixture/checkpoints/{i:04d}.json', sha256='a' * 64)
            checkpoints[relocation[checkpoint]['path']] = dict(row, original_calibration_index=i,
                outputs={archive: 'a' * 64}, evaluation_class_index=i)
        manifest = dict(source_count=1000, records=rows, source_ids=[r['source_id'] for r in rows])
        completion = dict(source_count=1000, outputs=outputs)
        values = dict(checkpoints, manifest=manifest, completion=completion)
        return values, relocation

    def verify_fixture(self, values, relocation):
        with mock.patch.object(h, 'checked', side_effect=lambda pin: values[pin['path']]), \
             mock.patch.object(h, 'sha', return_value='a' * 64), \
             mock.patch.object(Path, 'is_file', return_value=True), \
             mock.patch.object(Path, 'stat', return_value=types.SimpleNamespace(st_size=12)):
            return h.full_input_records(dict(path='manifest', sha256=h.MANIFEST_SHA),
                dict(path='completion', sha256=h.SOURCE_COMPLETION_SHA), relocation, Path)

    def test_full_original_manifest_and_checkpoint_mapping_preserves1000_order(self):
        values, relocation = self.full_metadata_fixture()
        records = self.verify_fixture(values, relocation)
        self.assertEqual([r['source_index'] for r in records], list(range(1000)))
        self.assertEqual(records[999]['source_id'], 'original999')

    def test_missing_original_asset_mapping_stops_full_input_closure(self):
        values, relocation = self.full_metadata_fixture()
        del relocation[h.OLD_ASSET + 'sources/0999.npz']
        with self.assertRaisesRegex(RuntimeError, 'Exact2000'):
            self.verify_fixture(values, relocation)

    def test_reordered_population_or_changed_checkpoint_is_refused(self):
        values, relocation = self.full_metadata_fixture()
        values['manifest']['source_ids'][999] = 'new_source'
        with self.assertRaisesRegex(RuntimeError, 'ordering'):
            self.verify_fixture(values, relocation)
        values, relocation = self.full_metadata_fixture()
        values['/fixture/checkpoints/0999.json']['source_id'] = 'new_source'
        with self.assertRaisesRegex(RuntimeError, 'checkpoint identity'):
            self.verify_fixture(values, relocation)

    def test_one_synthetic_source_records24_actual_streams_no_rx_render(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = h.g.Ledger(Path(td) / 'calls', lambda: None, h.CAPS)
            backend = types.SimpleNamespace(g=h.g, boundary=lambda: None)
            class Codec:
                def encode_endpoints(self, tokens, endpoints):
                    for scale in range(10):
                        ledger.call('prior_scale', lambda: None)
                        backend.traces.append(dict(source=999, role='TX', m=10, scale=scale, cdf_sha256='f' * 64))
                    return {(m, K): dict(m=m, K=K, arithmetic_bits=200+m+K,
                        bits=np.zeros(200+m+K, dtype=np.uint8)) for m, K in endpoints}
            read = mock.Mock(return_value=np.arange(680, dtype=np.int64))
            result = h.source900(dict(source_index=999, source_id='synthetic999'), backend,
                Codec(), partial, ledger, read, Path(td) / 'source')
            read.assert_called_once()
            loaded = h.core.link.load_streams(result['archive'], h.checked(result['metadata']), partial)
            self.assertEqual(len(loaded), 24)
            self.assertEqual(ledger.completed['source_tx'], 1)
            self.assertEqual(ledger.completed['prior_scale'], 10)
            for key in ('encoder', 'source_rx', 'var_render', 'decoder_forward'):
                self.assertEqual(ledger.completed[key], 0)

    def test_failed_tx_is_reserved_and_not_retried(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = h.g.Ledger(Path(td) / 'calls', lambda: None, h.CAPS)
            backend = types.SimpleNamespace(g=h.g, boundary=lambda: None)
            codec = mock.Mock(); codec.encode_endpoints.side_effect = RuntimeError('injected failure')
            with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                h.source900(dict(source_index=100, source_id='synthetic100'), backend,
                    codec, partial, ledger, lambda _: np.arange(680), Path(td) / 'source')
            codec.encode_endpoints.assert_called_once()
            self.assertEqual(ledger.reserved['source_tx'], 1)
            self.assertEqual(ledger.completed['source_tx'], 0)

    def test_owner_replay_stops_before_child_launch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / 'run').mkdir(); path = root / 'request.json'; path.write_text('{}')
            class Context:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            shared = mock.Mock(); shared.owner_lock.return_value = Context()
            runner = mock.Mock(); runner.device_from_request.return_value.index = 2
            args = mock.Mock(request=str(path), request_sha256='f' * 64)
            with mock.patch.object(h.g.sys, 'platform', 'linux'), \
                 mock.patch.object(h, 'registration', return_value=dict(spec={}, execution={})), \
                 mock.patch.object(h.g, 'inside', side_effect=Path), \
                 mock.patch.object(h.g, 'helper', return_value=shared), \
                 mock.patch.object(h.subprocess, 'Popen') as popen, \
                 self.assertRaisesRegex(RuntimeError, 'already claimed'):
                h.run(args, runner)
            popen.assert_not_called()


if __name__ == '__main__': unittest.main()
