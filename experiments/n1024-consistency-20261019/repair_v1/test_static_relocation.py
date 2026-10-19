"""New relocation-only regression: test the exact prepare_v2 AST block on fixtures."""
import ast,copy,hashlib,tempfile,unittest
from pathlib import Path
class Tests(unittest.TestCase):
 def setUp(self):
  tree=ast.parse((Path(__file__).parent/'prepare_v2.py').read_bytes())
  loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple) and [getattr(x,'value',None) for x in n.iter.elts]==['completion','table']]
  assert len(loops)==1;self.code=compile(ast.Module(body=loops,type_ignores=[]),'exact_prepare_v2_static_loop','exec')
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.old={'static':{}}
  for key in ('completion','table'):
   p=self.root/key;p.write_bytes(('fixture-'+key).encode());sha=hashlib.sha256(p.read_bytes()).hexdigest()
   self.old['static'][key]={'path':str(p),'sha256':sha,'bytes':p.stat().st_size}
   self.old['static']['original_'+key]={'path':'/nonexistent/historical/'+key,'sha256':sha}
 def tearDown(self):self.tmp.cleanup()
 def runblock(self):
  def verify(d):
   p=Path(d['path']);assert p.is_file() and p.stat().st_size==d['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256']
  exec(self.code,{'old':self.old,'verify':verify})
 def test_relocated_exact_hash_with_unavailable_historical_path(self):self.runblock()
 def test_reject_changed_historical_hash(self):
  self.old['static']['original_table']['sha256']='0'*64
  with self.assertRaises(AssertionError):self.runblock()
 def test_reject_corrupt_current_bytes(self):
  Path(self.old['static']['table']['path']).write_bytes(b'wrong')
  with self.assertRaises(AssertionError):self.runblock()
 def test_no_extra_descriptor_whitelist(self):
  self.old['static']['unexpected']={'path':'/nonexistent/ignored_must_not_be_used','sha256':'0'*64};self.runblock()
if __name__=='__main__':unittest.main(verbosity=2)
