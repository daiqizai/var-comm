"""Population, pairing and sealed-image rejection checks; no metric models."""
import unittest
import numpy as np
import score_n2048 as s

class ScoreContracts(unittest.TestCase):
    def test_float_hash_matches_published_domain(self):
        import hashlib
        a=np.zeros((3,256,256),np.float32)
        self.assertEqual(s.rgb_sha(a),hashlib.sha256(b'float32:3,256,256:RGB\0'+a.tobytes()).hexdigest())

    def test_uint8_reference_rejected(self):
        with self.assertRaises(RuntimeError):s.rgb_sha(np.zeros((3,256,256),np.uint8))

    def test_out_of_range_reconstruction_rejected(self):
        a=np.zeros((3,256,256),np.float32);a[0,0,0]=-1
        with self.assertRaises(RuntimeError):s.rgb_sha(a)

    def test_missing_population_rejected_before_summary(self):
        with self.assertRaises(RuntimeError):s.analyze([])

    def test_noise_averaging_precedes_source_resampling(self):
        rows=[]
        for i in range(100):
            for phy in s.PHYS:
                for snr in s.SNRS:
                    for seed in s.SEEDS:
                        for method in s.METHODS:
                            # The two methods have a constant source-paired difference
                            # despite large between-source and between-noise variation.
                            value=i*10+(seed-2001)*100+(3 if method=='entropy_policy' else 0)
                            r=dict(source_index=i,phy_family=phy,snr_db=snr,noise_seed=seed,method=method,
                                   header_ok=True,prefix_crc_ok=True,body_crc_ok=True,partial_used=False,prefix_false_accept=False)
                            r.update({m:value for m in s.METRICS});rows.append(r)
        _,paired=s.analyze(rows)
        selected=[x for x in paired if x['method_A']=='entropy_policy' and x['method_B']=='whole_policy']
        self.assertEqual(len(selected),len(s.PHYS)*len(s.SNRS)*len(s.METRICS))
        self.assertTrue(all(x['mean']==3 and x['ci_low']==3 and x['ci_high']==3 for x in selected))

    def test_bootstrap_rejects_missing_sources(self):
        with self.assertRaises(RuntimeError):s.interval(np.ones(99),np.zeros((1,100),int))

if __name__=='__main__':unittest.main()
