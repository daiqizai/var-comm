"""CPU-only admission checks for the separate EP48 CPU and GPU owners."""
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep48_link_gate_v1 as h


class OwnerTests(unittest.TestCase):
    def test_private_cpu_environment_preserves_library_binding_but_changes_caches(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);old=dict(CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH='frozen-library-path',VIRTUAL_ENV='old',
                TMPDIR='old-cache',HOME='old-home',PATH='old-bin')
            shared=types.SimpleNamespace(controlled_environment=lambda base,out,threads:(None,
                dict(CUDA_VISIBLE_DEVICES='',TMPDIR=str(out/'tmp'),OMP_NUM_THREADS=str(threads))))
            with mock.patch.object(h.g,'helper',return_value=shared):env=h.controlled_cpu_environment(out,old)
            self.assertEqual(env['LD_LIBRARY_PATH'],'frozen-library-path');self.assertEqual(env['TMPDIR'],str(out/'tmp'))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(env['OMP_NUM_THREADS'],'2')
            self.assertEqual(env['HOME'],str(out/'home'));self.assertEqual(old['HOME'],'old-home')

    def test_cpu_environment_refuses_python_path_injection(self):
        with tempfile.TemporaryDirectory() as td:
            shared=types.SimpleNamespace(controlled_environment=lambda *args:(None,{}))
            with mock.patch.object(h.g,'helper',return_value=shared),self.assertRaisesRegex(RuntimeError,'path injection'):
                h.controlled_cpu_environment(Path(td),dict(PYTHONPATH='foreign'))

    def test_component_admission_requires_exact_completed_attempts(self):
        paths=dict(source_completion=h.SOURCE_ROOT/'registered/run/completion.json',source_wait=h.SOURCE_ROOT/'run_owner_actual_wait.json',
            phy_completion=h.PHY_ROOT/'completion.json',phy_wait=h.PHY_CONTROL/'qualify_actual_wait.json',phy_environment=h.PHY_CONTROL/'environment.json')
        hashes=dict(source_completion=h.SOURCE_DONE_SHA,source_wait=h.SOURCE_WAIT_SHA,phy_completion=h.PHY_SHA,
            phy_wait=h.PHY_WAIT_SHA,phy_environment=h.PHY_ENV_SHA)
        pins={k:dict(path=str(v),sha256=hashes[k]) for k,v in paths.items()}
        for key in pins:
            bad=copy.deepcopy(pins);bad[key]['sha256']='0'*64
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'Exact completed component attempt'):h.prerequisites(bad)
        bad=copy.deepcopy(pins);bad['phy_wait']['path']=str(h.PHY_CONTROL/'prepare_actual_wait.json')
        with self.assertRaisesRegex(RuntimeError,'receipt path'):h.prerequisites(bad)

    def test_original_owners_and_budgets_are_not_mutated(self):
        original=h.source.engine();fresh=h.engine()
        self.assertEqual(original.CAPS['prior_scale'],4928);self.assertEqual(fresh.CAPS['prior_scale'],960)
        self.assertEqual(fresh.CAPS['source_tx'],0);self.assertEqual(fresh.CAPS['source_rx'],48)
        self.assertEqual(fresh.CAPS['var_render'],48);self.assertEqual(fresh.CAPS['decoder_forward'],48)
        self.assertIs(original.adapt_backend.__code__,fresh.adapt_backend.__code__)
        self.assertEqual(fresh.check_tools.__globals__['__file__'],h.__file__)


if __name__=='__main__':unittest.main(verbosity=2)
