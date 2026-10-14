import importlib.util,json,pathlib,sys,tempfile,unittest
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('m',HERE/'common500_metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources=[dict(source_index=i,source_id=str(i),reference_rgb_sha256=str(i)) for i in range(500)]
        cls.rows=[dict(method=a,source_index=i,source_id=str(i),reference_rgb_sha256=str(i),N=1024,snr_db=s,noise_seed=n,fixed_gray=True) for a in m.METHODS for i in range(500) for s in m.SNRS for n in m.SEEDS[a]]
    def test_full_grid_keeps_all_failed_frames(self):self.assertEqual(len(m.validate_grid(self.rows,self.sources)),27000)
    def test_missing_failure_rejected(self):
        with self.assertRaises(RuntimeError):m.validate_grid(self.rows[:-1],self.sources)
    def test_duplicate_frame_rejected(self):
        with self.assertRaises(RuntimeError):m.validate_grid(self.rows[:-1]+[self.rows[0]],self.sources)
    def test_foreign_noise_rejected(self):
        rows=list(self.rows);rows[0]=dict(rows[0],noise_seed=2001)
        with self.assertRaises(RuntimeError):m.validate_grid(rows,self.sources)
    def test_reference_identity_rejected(self):
        rows=list(self.rows);rows[0]=dict(rows[0],reference_rgb_sha256='other')
        with self.assertRaises(RuntimeError):m.validate_grid(rows,self.sources)
    def test_bad_snr_rejected(self):
        rows=list(self.rows);rows[0]=dict(rows[0],snr_db=13)
        with self.assertRaises(RuntimeError):m.validate_grid(rows,self.sources)
    def test_source_means_include_original_three_noises(self):
        rows=[dict(method=m.METHODS[0],snr_db=4,source_index=i,noise_seed=n,dists=i+(n-6201)*3) for i in range(500) for n in m.SEEDS[m.METHODS[0]]]
        a=m.source_values(rows,m.METHODS[0],4,'dists');np.testing.assert_array_equal(a,np.arange(500)+3)
        with self.assertRaises(RuntimeError):m.source_values(rows[:-1],m.METHODS[0],4,'dists')
    def test_pair_reuse_requires_reference_and_metric_registration(self):
        a=dict(reference_rgb_sha256='source1',reconstruction_sha256='gray');b=dict(a,reference_rgb_sha256='source2')
        self.assertNotEqual(m.pair_key(a,'reg'),m.pair_key(b,'reg'));self.assertNotEqual(m.pair_key(a,'reg'),m.pair_key(a,'other'))
    def test_failed_forward_consumes_cap(self):
        with tempfile.TemporaryDirectory() as d:
            l=m.Ledger(pathlib.Path(d)/'test.sqlite',{'load_dists':1})
            with self.assertRaises(ValueError):l.call('load_dists','x',lambda:(_ for _ in ()).throw(ValueError('synthetic')))
            with self.assertRaises(RuntimeError):l.call('load_dists','x',lambda:None)
            self.assertEqual(l.summary()['rows'],[('load_dists','FAILED',1)]);l.db.close()
    def test_boolean_failure_strings_are_explicit(self):
        self.assertFalse(m.boolean('False'));self.assertTrue(m.boolean(True))
        with self.assertRaises(RuntimeError):m.boolean('unknown')
    def test_no_pixel_clipping_or_shape_imputation(self):
        a=np.zeros((3,256,256),np.float32);self.assertEqual(m.pixels(a).shape,a.shape)
        a[0,0,0]=-0.01
        with self.assertRaises(RuntimeError):m.pixels(a)
    def test_direct_paired_bootstrap_not_marginal_subtraction(self):
        rows=[]
        for j,method in enumerate(m.METHODS):
            for snr in m.SNRS:
                for i in range(500):
                    for n in m.SEEDS[method]:rows.append(dict(method=method,snr_db=snr,source_index=i,noise_seed=n,dists=float(i+j),dreamsim=float(i+j),ms_ssim=float(i+j),fixed_gray=False))
        summary,paired,source,drawsha=m.statistics(rows);self.assertEqual((len(summary),len(paired),len(source)),(54,135,27000))
        delta=paired[0];self.assertEqual((delta['mean_difference'],delta['CI95_low'],delta['CI95_high']),(-1.,-1.,-1.));self.assertGreater(summary[0]['CI95_high']-summary[0]['CI95_low'],0)
        self.assertEqual(drawsha,m.ash(np.random.default_rng(2026100701).integers(0,500,(10000,500))))
    @classmethod
    def tearDownClass(cls):
        assert 'torch' not in sys.modules and 'torchvision' not in sys.modules,'Synthetic CPU tests cannot import scoring models'
if __name__=='__main__':unittest.main()
