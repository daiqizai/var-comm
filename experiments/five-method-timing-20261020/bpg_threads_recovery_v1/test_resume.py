import os
import unittest
from unittest.mock import patch
import resume


class RecoveryTests(unittest.TestCase):
    def test_exact_scope(self):
        self.assertEqual(resume.stages(),[(m,r) for m in ('BPG','ARPC') for r in ('tx','phy','rx')])

    def test_only_encoder_environment(self):
        with patch.dict(os.environ,{},clear=True),patch.object(os,'sched_getaffinity',return_value=set(range(6))):
            env=resume.codec_environment([str(resume.T/'assets/bpgenc')])
            self.assertEqual(env['LD_PRELOAD'],str(resume.SHIM))
            self.assertIsNone(resume.codec_environment([str(resume.T/'assets/bpgdec')]))

    def test_wrong_affinity_rejected(self):
        with patch.dict(os.environ,{},clear=True),patch.object(os,'sched_getaffinity',return_value={0}):
            with self.assertRaises(AssertionError):
                resume.codec_environment([str(resume.T/'assets/bpgenc')])

    def test_existing_preload_rejected(self):
        with patch.dict(os.environ,{'LD_PRELOAD':'unexpected'}):
            with self.assertRaises(AssertionError):
                resume.codec_environment([str(resume.T/'assets/bpgenc')])

    def test_worker_transform_without_execution(self):
        w=resume.adapted_worker()
        self.assertTrue(callable(w.main))
        self.assertTrue(callable(w.firewall))

    def test_owner_transform_stops_before_actual_work(self):
        def stop(*args):raise RuntimeError('synthetic stop')
        with patch.object(resume,'preflight',stop):
            with self.assertRaisesRegex(RuntimeError,'synthetic stop'):
                resume.owner('synthetic')

    def test_dispatch_transform_stops_before_actual_work(self):
        def stop(*args):raise RuntimeError('synthetic stop')
        with patch.object(resume,'bind',stop):
            with self.assertRaisesRegex(RuntimeError,'synthetic stop'):
                resume.dispatch('synthetic')


if __name__=='__main__':
    unittest.main()
