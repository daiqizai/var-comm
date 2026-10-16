import unittest
import tb_plan as p
class PlanTest(unittest.TestCase):
 def test_candidates_symmetric(self):
  cs=p.candidates();self.assertEqual(len(cs),24)
  self.assertEqual(len({c['candidate_id']for c in cs}),24)
  self.assertEqual({(c['q'],c['nominal_rate'],c['source_capacity_bits'])for c in cs if c['strategy_family']=='WHOLE'},{(c['q'],c['nominal_rate'],c['source_capacity_bits'])for c in cs if c['strategy_family']=='PARTIAL'})
 def test_fields_paid(self):
  self.assertTrue(all(c['source_capacity_bits']==c['k']-29 and c['n']==956*c['q']for c in p.candidates()))
 def test_m10_no_scale11(self):self.assertEqual(p.nt(10,0),680)
 def test_whole_fallback(self):self.assertEqual(p.fallback('WHOLE'),tuple((m,0)for m in range(10,3,-1)))
 def test_partial_zero_K_is_not_whole(self):
  points=p.fallback('PARTIAL');self.assertEqual(len(points),25);self.assertEqual(points[:2],((10,0),(9,192)))
  self.assertEqual([p.nt(*x)for x in points],sorted([p.nt(*x)for x in points],reverse=True))
 def test_finish_length_not_estimate(self):
  ls={e:5000 for e in p.ENDS};ls[10,0]=4752;ls[9,0]=4600;ls[9,192]=4700
  self.assertEqual(p.choose(ls,'WHOLE',4751),(9,0));self.assertEqual(p.choose(ls,'PARTIAL',4751),(9,192))
  ls[10,0]=4751;self.assertEqual(p.choose(ls,'WHOLE',4751),(10,0));self.assertEqual(p.choose(ls,'PARTIAL',4751),(10,0))
 def test_uint13_not_truncated(self):
  ls={e:100 for e in p.ENDS};ls[10,0]=8192;self.assertNotEqual(p.choose(ls,'WHOLE',9000),(10,0))
 def test_missing_stream_stops(self):
  with self.assertRaises(KeyError):p.choose({(10,0):9999},'PARTIAL',4751)
 def test_no_fit_stops(self):
  with self.assertRaises(ValueError):p.choose(dict.fromkeys((*p.ENDS,(10,0)),9999),'WHOLE',4751)
 def test_catalogue_preserves_old_ids(self):
  ps=[dict(profile_id=i,family='UNUSED',m=4,K=0,q=2,nominal_rate='1/2',token_count=30)for i in range(360)]
  for c in p.candidates()[:12]:
   i=p.candidates()[:12].index(c);ps[i].update(family='EC_VAR_WHOLE',m=9,q=c['q'],nominal_rate=c['nominal_rate'],k=c['k'],n=c['n'],source_capacity_bits=c['source_capacity_bits'],profile_key='legacy')
  old=dict(profiles=ps,catalogue_sha256='old',requested_profiles=360);new=p.extend_catalogue(old)
  self.assertEqual(new['profiles'][:360],old['profiles']);self.assertEqual(len(new['profiles']),372)
  self.assertEqual({x['token_count']for x in new['profiles'][360:]},{680})
  self.assertEqual({x['family']for x in new['profiles'][360:]},{'EC_VAR_WHOLE'})
 def test_old_illegal_crc_requires_replay(self):
  self.assertFalse(p.directory_unchanged_on_old_event(dict(header_crc_ok=True,header_fields_legal=False)))
  self.assertTrue(p.directory_unchanged_on_old_event(dict(header_crc_ok=False,header_fields_legal=False)))
 def test_legacy_exact_tie(self):
  self.assertEqual(p.select_winner({'m10_whole':.5,'m9':.5},{'m9'}),'m9')
  self.assertEqual(p.select_winner({'m10_whole':.500000001,'m9':.5},{'m9'}),'m10_whole')
if __name__=='__main__':unittest.main()
