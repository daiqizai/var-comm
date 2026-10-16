from metric_device_recovery_v2 import cuda_uuid_equal
import unittest
class DeviceTests(unittest.TestCase):
 def test_torch_registered_spelling(self):self.assertTrue(cuda_uuid_equal('ad85ce23-5025-3090-361b-27173fd1ac68','GPU-ad85ce23-5025-3090-361b-27173fd1ac68'))
 def test_same_spelling(self):self.assertTrue(cuda_uuid_equal('GPU-ad85ce23-5025-3090-361b-27173fd1ac68','GPU-ad85ce23-5025-3090-361b-27173fd1ac68'))
 def test_different_device_rejected(self):self.assertFalse(cuda_uuid_equal('70eb08db-3ea5-9394-c385-2125742754fd','GPU-ad85ce23-5025-3090-361b-27173fd1ac68'))
 def test_malformed_rejected(self):
  with self.assertRaises(ValueError):cuda_uuid_equal('GPU-GPU-ad85ce23-5025-3090-361b-27173fd1ac68','GPU-ad85ce23-5025-3090-361b-27173fd1ac68')
if __name__=='__main__':unittest.main()
