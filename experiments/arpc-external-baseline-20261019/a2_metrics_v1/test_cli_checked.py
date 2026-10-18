import json,pathlib,tempfile,unittest
from unittest import mock
import execute_checked as cli
class Receipt(unittest.TestCase):
    def check(self,**changes):
        d={'returncode':0,'timed_out':False,'terminated':False,'killed':False,'metadata_error':None,'process':{'pid':1,'start_ticks':1,'host':'synthetic'}};d.update(changes)
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'wait.json';p.write_text(json.dumps(d))
            with mock.patch.object(cli.owner,'gone',return_value=True):return cli.success(p)
    def test_import_real_launcher(self):self.assertIs(cli.execute.success,cli.success)
    def test_valid(self):self.assertEqual(self.check()['returncode'],0)
    def test_boolean_rejected(self):
        with self.assertRaises(AssertionError):self.check(returncode=False)
    def test_failed_rejected(self):
        with self.assertRaises(AssertionError):self.check(returncode=1)
    def test_timeout_rejected(self):
        with self.assertRaises(AssertionError):self.check(timed_out=True)
    def test_terminated_rejected(self):
        with self.assertRaises(AssertionError):self.check(terminated=True)
    def test_killed_rejected(self):
        with self.assertRaises(AssertionError):self.check(killed=True)
    def test_metadata_rejected(self):
        with self.assertRaises(AssertionError):self.check(metadata_error='error')
if __name__=='__main__':unittest.main(verbosity=2)
