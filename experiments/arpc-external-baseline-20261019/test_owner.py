"""Linux CPU-only tests for process receipts, timeouts and source identity guards."""
import json,os,pathlib,sys,tempfile,unittest
import a1_owner as own
class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.p=pathlib.Path(self.temp.name);self.old=own.RUN;own.RUN=self.p
    def tearDown(self):own.RUN=self.old;self.temp.cleanup()
    def test_hash_change_rejected(self):
        p=self.p/'data';p.write_bytes(b'ab');d=own.desc(p);p.write_bytes(b'cd')
        with self.assertRaises(AssertionError):own.verify(d)
    def test_symlink_rejected(self):
        p=self.p/'data';p.write_bytes(b'ab');q=self.p/'alias';q.symlink_to(p)
        with self.assertRaises(AssertionError):own.desc(q)
    def test_live_identity_not_gone(self):self.assertFalse(own.gone(own.identity(os.getpid())))
    def test_success_wait_and_no_duplicate(self):
        own.run_stage('child',[sys.executable,'-c','print(123)'],dict(os.environ,CUDA_VISIBLE_DEVICES=''),10)
        w=json.loads((self.p/'child.actual_wait.json').read_text());self.assertEqual(type(w['returncode']),int);self.assertEqual(w['returncode'],0)
        self.assertTrue(own.gone(w['process']));self.assertEqual((self.p/'child.stdout').read_text().strip(),'123')
        with self.assertRaises(FileExistsError):own.run_stage('child',[sys.executable,'-c','raise Exception()'],os.environ,10)
    def test_failure_preserved(self):
        with self.assertRaises(AssertionError):own.run_stage('bad',[sys.executable,'-c','raise SystemExit(7)'],os.environ,10)
        w=json.loads((self.p/'bad.actual_wait.json').read_text());self.assertEqual(w['returncode'],7);self.assertTrue(own.gone(w['process']))
    def test_actual_timeout_preserved(self):
        with self.assertRaises(AssertionError):own.run_stage('slow',[sys.executable,'-c','import time;time.sleep(5)'],os.environ,0.05)
        w=json.loads((self.p/'slow.actual_wait.json').read_text());self.assertTrue(w['timed_out']);self.assertTrue(w['terminated']);self.assertTrue(own.gone(w['process']))
if __name__=='__main__':unittest.main()
