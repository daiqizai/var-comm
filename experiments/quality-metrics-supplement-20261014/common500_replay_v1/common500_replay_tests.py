import importlib.util,unittest,tempfile
from pathlib import Path
s=importlib.util.spec_from_file_location('replay',Path(__file__).with_name('common500_replay.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def fixture():return [dict(method=method,source_index=i,snr_db=snr,noise_seed=seed,N=1024) for method in m.METHODS for i in range(500) for snr in m.SNRS for seed in m.SEEDS]
class Tests(unittest.TestCase):
    def test_exact9000(self):self.assertEqual(len(m.roster(fixture())),9000)
    def test_missing_frame(self):
        with self.assertRaises(RuntimeError):m.roster(fixture()[:-1])
    def test_no_extra_snr(self):
        rows=fixture();rows[0]['snr_db']=13
        with self.assertRaises(RuntimeError):m.roster(rows)
    def test_wrong_noise_namespace(self):
        rows=fixture();rows[0]['noise_seed']=6201
        with self.assertRaises(RuntimeError):m.roster(rows)
    def test_wrong_checkpoint_method(self):
        rows=fixture();rows[0]['method']='P40000'
        with self.assertRaises(RuntimeError):m.roster(rows)
    def test_failed_attempt_counts_against_cap(self):
        with tempfile.TemporaryDirectory() as root:
            ledger=m.Ledger(Path(root)/'fixture.sqlite');ledger.caps={'P_receive':1}
            def failed():raise ValueError('fixture failure')
            with self.assertRaises(ValueError):ledger.call('P_receive','fixture',failed)
            with self.assertRaises(RuntimeError):ledger.call('P_receive','would be replay',lambda:1)
            self.assertEqual(ledger.summary()['rows'],[('P_receive','FAILED',1)]);ledger.db.close()
if __name__=='__main__':unittest.main()
