import unittest,ast,pathlib,tempfile,json,hashlib
src=pathlib.Path(__file__).with_name('recover_unlaunched.py');tree=ast.parse(src.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='eligible')
NAME='multiprefix_full_anchor_s2026102201'
class EligibilityTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.L=pathlib.Path(self.tmp.name);self.d=self.L/'dispatch_v1'/NAME;self.d.mkdir(parents=True)
  self.failure=dict(error="AssertionError(['gpu', '71465', '97'])",traceback='assert admitted(int(fields[1]),int(fields[2]))')
  self.write();ns=dict(L=self.L,NAMES=(NAME,),read=lambda p:json.loads(p.read_text()),pin=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns);self.eligible=ns['eligible'];self.reg=dict(jobs={NAME:{}})
 def tearDown(self):self.tmp.cleanup()
 def write(self):(self.d/'failure.json').write_text(json.dumps(self.failure))
 def test_zero_attempt_admitted(self):self.assertEqual(self.eligible(NAME,self.reg)['path'],str(self.d/'failure.json'))
 def test_worker_rejected(self):
  (self.L/'training_v1'/NAME).mkdir(parents=True)
  with self.assertRaises(AssertionError):self.eligible(NAME,self.reg)
 def test_launch_rejected(self):
  (self.d/'launch.json').write_text('{}')
  with self.assertRaises(AssertionError):self.eligible(NAME,self.reg)
 def test_wait_rejected(self):
  (self.d/'actual_wait.json').write_text('{}')
  with self.assertRaises(AssertionError):self.eligible(NAME,self.reg)
 def test_other_failure_rejected(self):
  self.failure['error']='model failed';self.write()
  with self.assertRaises(AssertionError):self.eligible(NAME,self.reg)
if __name__=='__main__':unittest.main()
