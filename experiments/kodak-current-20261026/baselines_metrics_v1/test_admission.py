"""Only newly requested admission guards; does not repeat the original16 tests."""
import unittest
import numpy as np
from common import *
class AdmissionTest(unittest.TestCase):
 def test_nest_precise_convention(self):
  x=np.zeros((3,256,256),np.float32);self.assertEqual(assert_array(x,ah(x),'sha256(str(dtype)+str(shape)+contiguous_bytes)'),ah(x))
 def test_arpc_precise_convention(self):
  x=np.full((3,256,256),.5,np.float32);self.assertEqual(assert_array(x,rawah(x),'sha256(array.tobytes(C)); float32 CHW [3,256,256]'),ah(x))
 def test_pci_nvml_driver_spelling(self):self.assertEqual(normalized_pci_bus('00000000:67:00.0'),normalized_pci_bus('0000:67:00.0'))
 def test_distinct_devices_not_collapsed(self):self.assertNotEqual(normalized_pci_bus('00000000:67:00.0'),normalized_pci_bus('0000:63:00.0'))
if __name__=='__main__':unittest.main()
