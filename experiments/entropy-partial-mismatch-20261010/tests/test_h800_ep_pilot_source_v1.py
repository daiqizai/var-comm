"""CPU-only checks for source68 owner isolation, input closure and no replay."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h800_ep_pilot_source_v1 as h


class SourceOwnerTests(unittest.TestCase):
    def test_private_engine_keeps_older_caps_and_worker_source_unchanged(self):
        before = dict(h.gate.source.CAPS); runner = h.engine()
        self.assertEqual(h.gate.source.CAPS, before)
        self.assertIsNot(runner.CAPS, h.gate.source.CAPS)
        self.assertEqual(runner.worker_identity.__globals__['__file__'], h.__file__)
        self.assertEqual(runner.check_tools.__globals__['__file__'], h.__file__)
        self.assertEqual(runner.CAPS['source_tx'], 68)
        self.assertEqual(runner.CAPS['source_rx'], 0)

    def test_core_tamper_is_refused_before_prepare(self):
        with mock.patch.object(h, 'CORE_SHA', '0' * 64), self.assertRaisesRegex(RuntimeError, 'core changed'):
            h.engine()

    def test_deadline_is_finite_and_separate_from_900second_input_contract(self):
        value = h.gate.source.execution(time.time() + 3600, 3600)
        self.assertEqual(value['max_seconds'], 3600)
        self.assertTrue(value['input_contract_is_not_execution_budget'])
        with self.assertRaises(RuntimeError): h.gate.source.execution(time.time() + 9999, 3601)

    def input_fixtures(self):
        old = dict(records=[dict(source_index=i, archive=dict(path=f'/first/{i}.npz', sha256='a'*64),
                                checkpoint=dict(path=f'/first/{i}.json', sha256='b'*64)) for i in range(32)])
        restore = dict(schema='PILOT68_ORIGINAL_ASSETS_RESTORED_V1', source_indices=list(range(32, 100)),
            file_count=136, source_manifest_sha256=h.core.MANIFEST_SHA,
            source_completion_sha256=h.core.SOURCE_COMPLETION_SHA,
            mapping=[dict(original_path=h.core.OLD_ASSET+f'{folder}/{i:04d}.{suffix}',
                          actual_path=f'/restored/{folder}/{i}.{suffix}', sha256='c'*64)
                     for i in range(32,100) for folder,suffix in [('sources','npz'),('source_checkpoints','json')]])
        return old, restore

    def test_100_input_map_combines_existing32_and_only_missing68(self):
        old, restore = self.input_fixtures()
        with mock.patch.object(h.gate.source, 'input_map', return_value=old), \
             mock.patch.object(h.g, 'inside', side_effect=Path), \
             mock.patch.object(h.g, 'checked_json', return_value=restore), \
             mock.patch.object(h.core, 'input_records', return_value=['verified100']) as verify:
            records, pin = h.inputs()
        self.assertEqual(records, ['verified100'])
        mapping = verify.call_args.args[2]
        self.assertEqual(len(mapping), 200)
        self.assertEqual(mapping[h.core.OLD_ASSET+'sources/0000.npz']['path'], '/first/0.npz')
        self.assertEqual(mapping[h.core.OLD_ASSET+'sources/0099.npz']['path'], '/restored/sources/99.npz')
        self.assertEqual(pin['sha256'], h.RESTORE_SHA)

    def test_new_receipt_cannot_overwrite_or_reencode_original32(self):
        old, restore = self.input_fixtures()
        restore['mapping'][0]['original_path'] = h.core.OLD_ASSET+'sources/0000.npz'
        with mock.patch.object(h.gate.source, 'input_map', return_value=old), \
             mock.patch.object(h.g, 'inside', side_effect=Path), \
             mock.patch.object(h.g, 'checked_json', return_value=restore), \
             self.assertRaisesRegex(RuntimeError, 'duplicates'):
            h.inputs()

    def test_owner_replay_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'run').mkdir(); path=root/'request.json'; path.write_text('{}')
            request=dict(spec={},execution={})
            class Context:
                def __enter__(self):return self
                def __exit__(self,*a):pass
            shared=mock.Mock(); shared.owner_lock.return_value=Context()
            runner=mock.Mock(); runner.device_from_request.return_value.index=2
            args=mock.Mock(request=str(path),request_sha256='f'*64)
            with mock.patch.object(h.g.sys,'platform','linux'), mock.patch.object(h,'registration',return_value=request), \
                 mock.patch.object(h.g,'inside',side_effect=Path), mock.patch.object(h.g,'helper',return_value=shared), \
                 mock.patch.object(h.subprocess,'Popen') as popen, self.assertRaisesRegex(RuntimeError,'already claimed'):
                h.run(args,runner)
            popen.assert_not_called()


if __name__=='__main__':unittest.main()
