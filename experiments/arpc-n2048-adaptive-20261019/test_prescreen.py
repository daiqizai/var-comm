import unittest
from prescreen import bounds,nominate
from adaptive_plan import layouts
class Tests(unittest.TestCase):
 def test_negative_gain_reverses_bounds(self):
  for actual,expected in zip(bounds(.5,.1,.5,.1,.9),(.3,.14,.46)):self.assertAlmostEqual(actual,expected)
 def test_no_fit_constant(self):self.assertEqual(bounds(.2,.2,.9,.1,1),(.2,.2,.2))
 def rows(self):
  return [dict(layout_id=g['layout_id'],policy_id=g['layout_id'],q=g['q'],nominal_rate=g['nominal_rate'],target=1,clean_dino_mean=.5,proxy_dino=.5,proxy_lower=.1,proxy_upper=.8)for g in layouts()]
 def test_more_than_two_not_hidden(self):
  d=nominate(self.rows());self.assertEqual(d['status'],'PENDING_MORE_THAN_TWO_PLAUSIBLE_POLICIES');self.assertEqual(len(d['uncovered']),10);self.assertEqual(d['additional_decoder_upper'],120000)
 def test_two_are_covered(self):
  r=self.rows()
  for i,x in enumerate(r):x.update(proxy_dino=.7 if i<2 else .1,proxy_lower=.6 if i<2 else 0,proxy_upper=.8 if i<2 else .2)
  self.assertEqual(nominate(r)['status'],'WITHIN_TWO_POLICY_CAP')
if __name__=='__main__':unittest.main(verbosity=2)
