"""CPU admission checks only; these never invoke bootstrap or model inference."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('baseline_supplement_statistics',Path(__file__).with_name('statistics.py'))
stat=importlib.util.module_from_spec(spec);spec.loader.exec_module(stat)


class GridAdmission(unittest.TestCase):
    def setUp(self):
        self.ids=['source_a','source_b'];self.hashes=['a'*64,'b'*64]
        self.rows=[dict(method='P100000',source_index=str(i),source_id=self.ids[i],snr_db='4',noise_seed=str(n),
                       N='1024',checkpoint_step='100000',reference_sha256=self.hashes[i],reused='False',
                       header_accepted='False' if n==2002 else 'True',
                       **{m:'0' for m in stat.METRICS}) for i in range(2) for n in [2001,2002,2003]]

    def check(self, rows):
        return stat.validate_grid(rows,'P100000',self.ids,[4],[2001,2002,2003],self.hashes)

    def test_failures_and_zero_metrics_remain(self):
        self.assertEqual(len(self.check(self.rows)),6)

    def test_duplicate_rejected(self):
        with self.assertRaises(AssertionError):self.check(self.rows+[self.rows[0]])

    def test_missing_rejected(self):
        with self.assertRaises(AssertionError):self.check(self.rows[:-1])

    def test_wrong_noise_rejected(self):
        self.rows[0]['noise_seed']='6201'
        with self.assertRaises(AssertionError):self.check(self.rows)

    def test_wrong_pixels_rejected(self):
        self.rows[0]['reference_sha256']='c'*64
        with self.assertRaises(AssertionError):self.check(self.rows)

    def test_old_model_rejected(self):
        self.rows[0]['checkpoint_step']='40000'
        with self.assertRaises(AssertionError):self.check(self.rows)

    def test_nonfinite_rejected(self):
        self.rows[0]['lpips_alex']='nan'
        with self.assertRaises(AssertionError):self.check(self.rows)

    def test_nonbinary_agreement_rejected(self):
        self.rows[0]['convnext_top1_source_prediction']='0.5'
        with self.assertRaises(AssertionError):self.check(self.rows)


if __name__=='__main__':unittest.main()
