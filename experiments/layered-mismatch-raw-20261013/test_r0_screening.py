"""Synthetic CPU arithmetic only; no scientific backends or new probabilities."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import r0_screening as s


def row(key,snr,correct=48,n=64,undetected=0):
    result=dict(phy_key=key,snr_db=snr,n_blocks=n,n_correct=correct,
                n_reject=n-correct-undetected,n_undetected=undetected)
    for kind,count in [('correct',correct),('reject',n-correct-undetected),('undetected',undetected)]:
        p=count/n;z=1.959963984540054;d=1+z*z/n
        centre=(p+z*z/(2*n))/d;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
        result['p_'+kind]=p;result['p_'+kind+'_ci']=[max(0,centre-radius),min(1,centre+radius)]
    return result


def fixture():
    profiles=[]
    for i in range(4):
        profiles.append(dict(profile_id=i,candidate_id='S'+str(i),wire_key='S'+str(i),
            G=1,m=1,K=i,groups=[dict(phy_key='s'+str(i),modulation='64QAM' if i==0 else 'QPSK')]))
    for i in range(7):
        profiles.append(dict(profile_id=433+i,candidate_id='G'+str(i),wire_key='G'+str(i),
            G=2,j=1,m=2,K=i,groups=[dict(phy_key='base'),dict(phy_key='g'+str(i))],
            family_memberships=['U'] if i<4 else ['EEP']))
    ids=['source_B','source_A']
    q=[dict(source_id=source,m=p['m'],K=p['K'],dinov2_vitl14_cosine=.8)
       for source in ids for p in profiles]
    gray=[dict(source_id=source,dinov2_vitl14_cosine=.1) for source in ids]
    keys={g['phy_key'] for p in profiles for g in p['groups']}
    body=[row(key,snr) for key in sorted(keys) for snr in (4,7,10)]
    headers=[row('header',snr,20000,20000) for snr in (4,7,10)]
    anchors=[dict(hat_snr_db=hat,method=arm,profile_id=i,candidate_id='S'+str(i),wire_key='S'+str(i))
             for hat in (7,10) for arm,i in [('S_nom',3),('S_back3',2)]]
    return [profiles,ids,body,headers,q,[],gray,anchors]


class Tests(unittest.TestCase):
    def test_probability_uses_counts_and_keeps_undetected_separate(self):
        x=row('a',4,30,64,2)
        self.assertEqual(s.probability(x),30/64)
        self.assertEqual(x['n_undetected'],2)
        with self.assertRaises(ValueError):s.probability(x,expected_n=256)

    def test_nonpartitioned_or_nan_probability_rejected(self):
        x=row('a',4);x['n_reject']+=1
        with self.assertRaises(ValueError):s.probability(x)
        x=row('a',4);x['p_correct']=float('nan')
        with self.assertRaises(ValueError):s.probability(x)

    def test_exact_two_group_formula_and_single_limit(self):
        p=dict(G=2,j=1,m=2,K=0,groups=[dict(phy_key='a'),dict(phy_key='b')])
        quality={('s',1,0):.6,('s',2,0):.9};prob={('a',7):.8,('b',7):.7}
        self.assertAlmostEqual(s.drop_proxy(p,'s',7,prob,{7:.9},quality,{'s':.2}),.6392)
        p['G']=1
        self.assertAlmostEqual(s.drop_proxy(p,'s',7,prob,{7:.9},quality,{'s':.2}),.704)

    def test_zero_errors_is_point_estimate_not_claim_of_zero_true_bler(self):
        x=row('a',7,64)
        self.assertEqual(s.probability(x),1)
        self.assertLess(x['p_correct_ci'][0],1)
        self.assertGreater(x['p_reject_ci'][1],0)

    def test_per_hat_caps_ties_fixed_and_anchors_maintained(self):
        f=fixture();out=s.rank_and_nominate(*f)
        for hat in (7,10):
            rows=[x for x in out['rows'] if x['snr_hat']==hat]
            self.assertEqual({arm:sum(x['arm']==arm for x in rows) for arm in ('U','S','EEP','S_nom','S_back3')},
                             dict(U=3,S=3,EEP=2,S_nom=1,S_back3=1))
            self.assertEqual([x['candidate_id'] for x in rows if x['arm']=='S'],['S0','S1','S2'])
            self.assertEqual(next(x['candidate_id'] for x in rows if x['arm']=='S_nom'),'S3')
        self.assertFalse(out['global_best_claim']);self.assertFalse(out['KEEP_failed_quality_inferred'])
        self.assertEqual(out['GO_decision'],'NOT_RUN')

    def test_order_independence_and_q6_not_excluded(self):
        f=fixture();first=s.rank_and_nominate(*f)
        for i in (0,1,2,3,4,6,7):f[i]=list(reversed(f[i]))
        second=s.rank_and_nominate(*f)
        self.assertEqual(first['rows'],second['rows'])
        self.assertEqual(first['ranking'],second['ranking'])
        self.assertIn('S0',[r['candidate_id'] for r in first['rows']])

    def test_pressure_data_rejected_not_used(self):
        f=fixture();f[2].append(row('s0',13))
        with self.assertRaisesRegex(ValueError,'pressure'):s.rank_and_nominate(*f)

    def test_missing_bler_never_becomes_zero_or_one(self):
        f=fixture();f[2].pop()
        with self.assertRaisesRegex(ValueError,'Missing BLER'):s.rank_and_nominate(*f)

    def test_duplicate_clean_or_foreign_source_rejected(self):
        f=fixture();f[5].append(copy.deepcopy(f[4][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):s.rank_and_nominate(*f)
        f=fixture();f[4][0]['source_id']='holdout'
        with self.assertRaisesRegex(ValueError,'foreign'):s.rank_and_nominate(*f)

    def test_gray_is_source_specific_and_required(self):
        f=fixture();f[6].pop()
        with self.assertRaisesRegex(ValueError,'Missing per-source'):s.rank_and_nominate(*f)

    def test_same_physical_group_can_fill_two_families_without_new_id(self):
        f=fixture();f[0][4]['family_memberships']=['U','EEP']
        out=s.rank_and_nominate(*f)
        aliases=[r for r in out['rows'] if r['snr_hat']==7 and r['candidate_id']=='G0']
        self.assertEqual({r['arm'] for r in aliases},{'U','EEP'})
        self.assertEqual(len({r['physical_key'] for r in aliases}),1)

    def test_precision_not_smoothed_or_weighted_for_nomination(self):
        f=fixture();a=s.rank_and_nominate(*f)
        for i,x in enumerate(f[2]):
            if x['phy_key']=='s0':f[2][i]=row('s0',x['snr_db'],192,256)
        b=s.rank_and_nominate(*f)
        self.assertEqual(a['ranking'],b['ranking'])
        self.assertNotEqual(a['precision'],b['precision'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    raise SystemExit(0 if result.wasSuccessful() else 1)
