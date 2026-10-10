import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_new100_visual_v2 as v


class AlternateOwnerTests(unittest.TestCase):
    def test_engine_and_private_namespaces_resolve_all_dependencies(self):
        e=v.engine()
        self.assertIs(e.DeviceAdapter,v.resources.DeviceAdapter)
        self.assertEqual(e.CAPS,v.CAPS)
        self.assertIs(v.run.__code__,v.original.run.__code__)
        self.assertIs(v.worker.__code__,v.original.worker.__code__)
        self.assertIs(v.worker.__globals__['images'],v.images)
        self.assertIs(v.worker.__globals__['evaluation'],v.evaluation)
        self.assertEqual(v.evaluation.POPULATION,'independent_confirmation_new100')
        self.assertIs(v.evaluation.check_counts,v.evaluation.original.check_counts)
        self.assertIs(v.evaluation.pilot,v.core)
        self.assertIs(v.evaluation.confirmation,v.confirmation)
        self.assertTrue(callable(v.evaluation.raw.received_tokens))
        self.assertEqual(v.worker.__globals__['__file__'],v.__file__)

    def test_no_extra_source_or_image_budget_and_only_observed_devices(self):
        self.assertEqual(v.CAPS,dict(model_load=1,encoder=0,source_tx=0,source_rx=600,var_render=3600,prior_scale=42000,decoder_forward=3600))
        self.assertEqual(v.resources.DEVICE_SHA,'9e158c4b1634d0ebb4e19fcbdf456e35c60afd41c78fdb0395637652192abc9a')
        for i in (0,1,2,3):self.assertEqual(v.resources.policy(i)['target_index'],i)
        with self.assertRaises(RuntimeError):v.resources.policy(4)


if __name__=='__main__':unittest.main()
