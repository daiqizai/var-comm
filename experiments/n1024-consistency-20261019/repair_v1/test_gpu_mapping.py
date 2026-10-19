"""Only the added identity/visibility semantics; no driver or model call."""
import unittest
from gpu_mapping import canonical_pci,select_mapping
class Tests(unittest.TestCase):
 def setUp(self):
  self.cuda=[dict(ordinal=0,pci='0000:1a:00.0',uuid='11111111-1111-1111-1111-111111111111'),dict(ordinal=1,pci='0000:6b:00.0',uuid='22222222-2222-2222-2222-222222222222')]
  self.nvml=dict(index=3,pci='00000000:6B:00.0',uuid='GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
 def test_different_uuid_namespaces_preserved(self):
  r=select_mapping(self.cuda,self.nvml);self.assertEqual(r['nvml_index'],3);self.assertEqual(r['nvml_uuid'],self.nvml['uuid']);self.assertEqual(r['cuda_uuid'],'GPU-'+self.cuda[1]['uuid'])
 def test_visibility_uses_cuda_ordinal(self):
  r=select_mapping(self.cuda,self.nvml);self.assertEqual(r['cuda_visible_devices'],'1');self.assertEqual(r['cuda_device_order'],'PCI_BUS_ID')
 def test_normalize_domain(self):self.assertEqual(canonical_pci('00000000:6B:00.0'),'0000:6b:00.0')
 def test_reject_missing_PCI(self):
  self.nvml['pci']='0000:99:00.0'
  with self.assertRaises(AssertionError):select_mapping(self.cuda,self.nvml)
 def test_reject_ambiguous_PCI(self):
  with self.assertRaises(AssertionError):select_mapping(self.cuda+self.cuda,self.nvml)
 def test_reject_uuid_not_identity(self):
  self.cuda[1]['uuid']='not-a-UUID'
  with self.assertRaises(ValueError):select_mapping(self.cuda,self.nvml)
if __name__=='__main__':unittest.main(verbosity=2)
