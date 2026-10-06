import unittest
import raw64_selection as m

class Selection(unittest.TestCase):
    def setup(self):
        profiles=[dict(candidate_id=k,K=0 if k in 'abc' else 1,groups=[dict(phy_key=k)]) for k in 'abcde']
        scores={s:{k:float(5-i) for i,k in enumerate('abcde')} for s in m.SNRS}
        return profiles,m.shortlist(profiles,scores)
    def test_global_across_modulations_and_exact_tie(self):
        p,short=self.setup()
        self.assertEqual(short['cells'][0]['candidate_ids'],['a','b','c'])
        self.assertEqual(short['cells'][1]['candidate_ids'],['a','d','e'])
        self.assertEqual(m.rank({'b':1.,'a':1.}),['a','b'])
    def test_full_source_noise_grid_and_no_development(self):
        p,short=self.setup();ids=[str(i) for i in range(1000)]
        points={(c['snr_db'],k) for c in short['cells'] for k in c['candidate_ids']}
        rows=[dict(population_role='calibration',source_index=i,source_id=ids[i],snr_db=s,candidate_id=k,noise_seed=n,
                   dinov2_vitl14_cosine=(1 if k=='b' else 0)+(.1 if n==4103 else 0))
              for s,k in points for i in range(1000) for n in m.NOISE]
        out=m.final_select(short,ids,rows)
        self.assertEqual(out['winners'][0]['candidate_id'],'b')
        self.assertAlmostEqual(out['winners'][0]['mean_DINO_L'],1+1/30)
        with self.assertRaises(ValueError):m.final_select(short,ids,rows[:-1])
        with self.assertRaises(ValueError):m.final_select(short,ids,rows+[rows[0]])
        rows[0]=dict(rows[0],population_role='development')
        with self.assertRaises(ValueError):m.final_select(short,ids,rows)
    def test_missing_proxy_candidate_is_not_silently_pruned(self):
        p,_=self.setup();scores={s:{k:1. for k in 'abcd'} for s in m.SNRS}
        with self.assertRaises(ValueError):m.shortlist(p,scores)

if __name__=='__main__':unittest.main()
