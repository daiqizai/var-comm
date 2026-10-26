"""Synthetic metadata checks; no real registration, model or CUDA calls."""
import ast,copy,unittest
from pathlib import Path
tree=ast.parse((Path(__file__).with_name('execution.py')).read_text())
fn=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='assert_mapping'][0]
ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'metadata','exec'),ns)
class Tests(unittest.TestCase):
 def setUp(self):
  self.m=dict(nvml_uuid='nvml-a',nvml_index=0,pci='0000:61:00.0',cuda_visible_devices='0',cuda_ordinal=0,cuda_device_order='PCI_BUS_ID',model_imports=0,context_creation_calls=0)
  self.g=dict(uuid='nvml-a',pci_bus_id='00000000:61:00.0')
 def test_join(self):ns['assert_mapping'](self.m,self.g)
 def test_wrong_pci(self):
  self.m['pci']='0000:62:00.0'
  with self.assertRaises(AssertionError):ns['assert_mapping'](self.m,self.g)
 def test_uuid_not_ordinal(self):
  self.m['cuda_visible_devices']='nvml-a'
  with self.assertRaises(AssertionError):ns['assert_mapping'](self.m,self.g)
 def test_no_context(self):
  self.m['context_creation_calls']=1
  with self.assertRaises(AssertionError):ns['assert_mapping'](self.m,self.g)
 def test_original_science_import_only(self):
  source=ast.unparse(tree)
  self.assertIn("importlib.import_module('kn_' + stage)",source)
  self.assertNotIn('encode_all',source);self.assertNotIn('render_received',source);self.assertNotIn('import torch',source)
 def test_exclusive_and_no_retry(self):
  source=ast.unparse(tree);self.assertIn('exist_ok=False',source);self.assertIn("OLD_L / 'execution.lock'",source)
  self.assertIn('no automatic retry',source);self.assertIn('child',source)
 def test_entry_names(self):
  source=ast.unparse(tree);self.assertIn('mod.run(PARENT)',source);self.assertNotIn('mod.main(PARENT)',source)
  self.assertIn("stage in ('phy', 'receiver', 'delivery')",source)
 def test_source_closure_has_exact_original_revision(self):
  source=ast.unparse(tree);self.assertIn("SOURCE_REVISION if stage == 'source' else expected",source)
if __name__=='__main__':unittest.main()
