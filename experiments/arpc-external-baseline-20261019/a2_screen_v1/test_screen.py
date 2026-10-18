import copy,unittest
import screen
def rows():
    out=[]
    for q in [2,4,6]:
        for rate in screen.RATES:
            for e in range(1,8):
                out.append({'q':q,'nominal_rate':rate,'target_endpoint':e,'strategy_id':f'{q}_{rate}_{e}',
                  'clean_dino_mean':.5,'expected_dino':.1,'lower_dino':.05,'upper_dino':.15})
    return out
class Nomination(unittest.TestCase):
    def test_negative_gain_interval_orders(self):
        for x,y in zip(screen.bounds(.5,.1,.5,.1,.9),[.3,.14,.46]):self.assertAlmostEqual(x,y)
    def test_zero_gain(self):self.assertEqual(screen.bounds(.2,.2,.5,.1,.9),(.2,.2,.2))
    def test_two_clear_leaders(self):
        a=rows()
        for r in a:
            if r['q']==2 and r['nominal_rate']=='1/2':r.update(expected_dino=.8,lower_dino=.7,upper_dino=.9)
            elif r['q']==2 and r['nominal_rate']=='2/3':r.update(expected_dino=.75,lower_dino=.65,upper_dino=.95)
        d=screen.nominate(a);self.assertEqual(d['status'],'CANDIDATES_WITHIN_ORIGINAL_CAP')
        self.assertEqual(d['proposed_candidates'],['2_1/2_7','2_2/3_7'])
    def test_three_uncertain_block(self):
        a=rows()
        for r in a:
            if r['q']==2 and r['nominal_rate'] in ['1/2','2/3','3/4']:r.update(expected_dino=.8,lower_dino=.7,upper_dino=.9)
        d=screen.nominate(a);self.assertEqual(d['status'],'BLOCKED_RELIABILITY_UNCERTAINTY_EXCEEDS_TWO')
        self.assertEqual(len(d['uncovered_possible_candidates']),1);self.assertFalse(d['new_channel_authorized_by_this_output'])
    def test_all_tied_stays_blocked(self):
        d=screen.nominate(rows());self.assertEqual(len(d['uncovered_possible_candidates']),10)
    def test_actual_quality_beats_target_number(self):
        a=rows()
        for r in a:
            if r['target_endpoint']==3:r['clean_dino_mean']=.7
        d=screen.nominate(a);self.assertTrue(all(s.endswith('_3') for s in d['within_MCS_leaders']))
    def test_no_input_mutation(self):
        a=rows();b=copy.deepcopy(a);screen.nominate(a);self.assertEqual(a,b)
    def test_missing_strategy_rejected(self):
        with self.assertRaises(AssertionError):screen.nominate(rows()[:-1])
if __name__=='__main__':unittest.main(verbosity=2)
