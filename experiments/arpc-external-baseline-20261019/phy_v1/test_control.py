import pathlib,tempfile,json,unittest
from unittest.mock import patch
import owner,execute
class Control(unittest.TestCase):
    def test_wait_boolean_or_missing_closure_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'wait.json'
            with patch.object(owner,'gone',return_value=True):
                for rc,timed in [(False,False),(1,False),(0,True)]:
                    p.write_text(json.dumps({'returncode':rc,'timed_out':timed,'process':{}}))
                    with self.assertRaises(AssertionError):owner.success(p)
                p.write_text(json.dumps({'returncode':0,'timed_out':False,'process':{}}));owner.success(p)
            with patch.object(owner,'gone',return_value=False):
                with self.assertRaises(AssertionError):owner.success(p)
    def test_entry_imports_without_runtime(self):
        self.assertTrue(callable(execute.main));self.assertEqual(owner.CODE.name,'phy_v1')
if __name__=='__main__':unittest.main(verbosity=2)
