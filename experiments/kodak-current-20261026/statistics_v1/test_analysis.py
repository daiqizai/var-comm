import unittest
import numpy as np
import analysis as a

def grid():
    return [dict(method=m,snr_db=s,source_index=i,source_id=f'kodak/kodim{i+1:02d}',noise_seed=n,rx_status='CRC_REJECT' if i==0 else 'ACCEPT',**{k:float(i+ni/10+mi) for k in a.METRICS}) for mi,m in enumerate(a.METHODS) for s in a.SNRS for i in range(24) for ni,n in enumerate(a.NOISES)]

class SourceStatistics(unittest.TestCase):
    def test_noise_mean_and_failure_kept(self):
        arrays,long=a.source_arrays(grid())
        self.assertEqual(len(long),1728)
        self.assertAlmostEqual(arrays['NEST_PARTIAL',4,'dreamsim'][0],.1)
        self.assertAlmostEqual(arrays['NEST_PARTIAL',4,'dreamsim'][23],23.1)
    def test_missing_and_duplicate_rejected(self):
        rows=grid()
        with self.assertRaises(ValueError):a.source_arrays(rows[:-1])
        rows[-1]=rows[0]
        with self.assertRaises(ValueError):a.source_arrays(rows)
    def test_engineering_error_not_gray(self):
        rows=grid();rows[0]['engineering_error']='OOM'
        with self.assertRaises(ValueError):a.source_arrays(rows)
    def test_source_order_rejected(self):
        rows=grid();rows[0]['source_id']='kodak/kodim02'
        with self.assertRaises(ValueError):a.source_arrays(rows)
    def test_direct_paired_interval(self):
        x=np.arange(24,dtype=np.float64);y=x-2
        bank=np.tile(np.arange(24),(10000,1))
        got=a.interval(x-y,bank)
        self.assertEqual((got['mean'],got['ci_low'],got['ci_high']),(2.,2.,2.))
    def test_bank_identity_rejected(self):
        with self.assertRaises(ValueError):a.validate_bank(np.zeros((10000,24),dtype=np.int64))

if __name__=='__main__':unittest.main()
