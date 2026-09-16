import argparse, io, pathlib, pickle, sys, tempfile, unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import torch
from ruamel.yaml.comments import CommentedSeq
from ruamel.yaml.scalarfloat import ScalarFloat
import checkpoint_io as ck
class TestRestrictedCheckpoint(unittest.TestCase):
 def test_approved_metadata_and_tensors_only(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'tiny.pt';x=torch.arange(4)
   torch.save({'tensor':x,'args':argparse.Namespace(scales=CommentedSeq([1,1,2]),lr=ScalarFloat(.01))},p,pickle_protocol=2)
   loaded=torch.load(p,weights_only=False,pickle_module=ck._module(),map_location='cpu',mmap=True)
   self.assertTrue(torch.equal(x,loaded['tensor']));self.assertEqual(loaded['args'].scales,[1,1,2])
 def test_arbitrary_system_is_rejected_not_executed(self):
  with patch('os.system') as system:
   with self.assertRaises(pickle.UnpicklingError):ck._module().Unpickler(io.BytesIO(b"cos\nsystem\n(S'false'\ntR.")).load()
   system.assert_not_called()
 def test_eval_global_is_rejected(self):
  with self.assertRaises(pickle.UnpicklingError):ck._module().Unpickler(io.BytesIO(b'cbuiltins\neval\n.')).load()
 def test_wrong_size_rejected_before_unpickle(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'untrusted.pt';p.write_bytes(b'not a checkpoint')
   with self.assertRaisesRegex(RuntimeError,'two exact'):ck.load_xq_checkpoint(p)
 def test_framework_allowlist_is_version_pinned(self):
  with patch.object(torch,'__version__','other'):
   with self.assertRaisesRegex(RuntimeError,'audited torch'):ck._globals()
if __name__=='__main__':unittest.main()
