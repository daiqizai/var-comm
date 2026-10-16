"""Synthetic path boundaries and metadata-only Torch import; no model work."""
from runtime_boundary import *
import unittest,tempfile,json
class BoundaryTests(unittest.TestCase):
 def test_real_package_read(self):self.assertTrue(library_read(LIBRARIES[1]/'torch/__init__.py','r',os.O_RDONLY))
 def test_source_blocked(self):self.assertFalse(library_read(RT/'assets/fixed_existing32_seed_v1/data.npz','r',0))
 def test_library_write_denied(self):self.assertFalse(library_read(LIBRARIES[1]/'torch/__init__.py','w',os.O_WRONLY))
 def test_parent_escape_denied(self):self.assertFalse(library_read(LIBRARIES[1]/'../../../../source.npz','r',0))
 def test_sibling_denied(self):self.assertFalse(library_read(str(LIBRARIES[1])+'_other/torch.py','r',0))
 def test_rw_flags_denied(self):self.assertFalse(library_read(LIBRARIES[1]/'torch/__init__.py',None,os.O_RDWR))
if __name__=='__main__':
 if len(sys.argv)==2 and sys.argv[1]=='import':
  from tb_common import *
  out=B/'t1b_runtime_import_boundary_qualification_v1/import';out.mkdir(exist_ok=False);runtime_cache(out)
  firewall(out,[],[B,RT/'assets',RT/'qualification',R/'results',R/'outputs'])
  import torch
  assert not torch.cuda.is_initialized()
  torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
  assert not torch.cuda.is_initialized()
  save(out/'completion.json',dict(status='CPU_IMPORT_ONLY_PASS',torch=torch.__version__,CUDA_initialized=False,model_calls=0,source_reads=0,weight_reads=0,library_roots=[str(p)for p in LIBRARIES]))
 else:unittest.main()
