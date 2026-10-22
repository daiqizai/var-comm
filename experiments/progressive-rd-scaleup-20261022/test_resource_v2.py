import unittest,pathlib,hashlib,ast
from rdscale_resource_v2 import admitted
class ResourceTests(unittest.TestCase):
 def test_free_boundary(self):
  self.assertFalse(admitted(59999,0));self.assertTrue(admitted(60000,0))
 def test_shared_load_boundary(self):
  self.assertTrue(admitted(71354,57));self.assertTrue(admitted(60000,80));self.assertFalse(admitted(60000,81));self.assertFalse(admitted(60000,-1))
 def test_types(self):
  self.assertFalse(admitted(60000.,10));self.assertFalse(admitted(60000,True))
 def test_owner_exact_delta(self):
  e=pathlib.Path(__file__).resolve().parent
  s=(e/'owner.py').read_text().replace('import argparse,subprocess,signal,concurrent.futures,traceback,csv,collections','import argparse,subprocess,signal,concurrent.futures,traceback,csv,collections\nfrom rdscale_resource_v2 import admitted')
  s=s.replace("O/'qualification.json'","O/'qualification_resource_v2.json'").replace("fields[0]==resource['uuid']and int(fields[1])>=60000 and int(fields[2])<=10","fields[0]==resource['uuid']and admitted(int(fields[1]),int(fields[2]))")
  self.assertEqual(ast.dump(ast.parse(s)),ast.dump(ast.parse((e/'owner_resource_v2.py').read_text())))
 def test_supervisor_exact_delta(self):
  e=pathlib.Path(__file__).resolve().parent;s=(e/'supervisor.py').read_text().replace("str(E/'owner.py')","str(E/'owner_resource_v2.py')")
  self.assertEqual(ast.dump(ast.parse(s)),ast.dump(ast.parse((e/'supervisor_resource_v2.py').read_text())))
if __name__=='__main__':unittest.main()
