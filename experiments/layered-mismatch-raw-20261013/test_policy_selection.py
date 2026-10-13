import copy
import unittest
import policy_selection as p


def fixture():
    sources=[f's{i}' for i in range(200)]
    nominations=[];rows=[]
    for hat in (7,10):
        for arm in ('U','S','EEP','S_nom','S_back3'):
            rules=['KEEP','DROP'] if arm=='S' else (['KEEP'] if arm.startswith('S_') else ['DROP'])
            nominations.append(dict(snr_hat=hat,arm=arm,candidate_id=arm,
                                    physical_key='phy'+arm,receiver_variants=rules))
            for sid in sources:
                for true in (hat-3,hat,hat+3):
                    for noise in (4101,4102,4103):
                        for rule in rules:
                            dino=.65 if arm=='U' else .6
                            rows.append(dict(source_id=sid,snr_hat=hat,snr_true=true,noise_seed=noise,
                                candidate_id=arm,physical_key='phy'+arm,receiver_rule=rule,
                                psnr_db=20.,lpips_alex=.2,dinov2_vitl14_cosine=dino,
                                convnext_top1_source_prediction=0 if arm=='U' else 1,
                                execution_status='COMPLETE',source_truth_supplied_to_RX=False,
                                observation_key='observation',actual_rx_key='receive',reconstruction_sha256='image'))
    return sources,nominations,rows


class SelectionTests(unittest.TestCase):
    def test_complete_finite_grid_and_no_convnext_selection(self):
        s,n,r=fixture();out=p.select_from_complete_rows(r,s,n)
        self.assertTrue(out['gate']['r2_authorized'])
        self.assertEqual(out['logical_rows'],21600)
        self.assertTrue(all(not w['convnext_used_for_selection'] for w in out['winners']))
        self.assertEqual([w['receiver_rule'] for w in out['winners'] if w['method']=='S_robust'],['KEEP','KEEP'])

    def test_duplicate_missing_wrong_split_or_truth_rejected(self):
        s,n,r=fixture()
        for bad in (r+[r[0]],r[1:]):
            with self.assertRaises(ValueError):p.select_from_complete_rows(bad,s,n)
        altered=copy.deepcopy(r);altered[0]['source_truth_supplied_to_RX']=True
        with self.assertRaises(ValueError):p.select_from_complete_rows(altered,s,n)
        altered=copy.deepcopy(r);altered[0]['noise_seed']=2001
        with self.assertRaises(ValueError):p.select_from_complete_rows(altered,s,n)
        altered=copy.deepcopy(r)
        row=next(x for x in altered if x['candidate_id']=='S' and x['receiver_rule']=='DROP')
        row['observation_key']='different-noise'
        with self.assertRaises(ValueError):p.select_from_complete_rows(altered,s,n)

    def test_pressure_condition_cannot_change_winner(self):
        s,n,r=fixture()
        for row in r:
            if row['snr_true']==row['snr_hat']+3:
                row['dinov2_vitl14_cosine']=0. if row['candidate_id']=='U' else 1.
        self.assertTrue(p.select_from_complete_rows(r,s,n)['gate']['r2_authorized'])

    def test_strong_single_can_retain_a_mandatory_anchor(self):
        s,n,r=fixture()
        for row in r:
            if row['candidate_id']=='S_back3':row['dinov2_vitl14_cosine']=.7
        out=p.select_from_complete_rows(r,s,n)
        strong=[w for w in out['winners'] if w['method']=='S_robust']
        self.assertEqual([w['candidate_id'] for w in strong],['S_back3','S_back3'])
        self.assertFalse(out['gate']['r2_authorized'])
        self.assertEqual(out['logical_rows'],21600)

    def test_shortlist_caps_cannot_be_bypassed(self):
        s,n,r=fixture()
        for i in range(3):
            n.append(dict(n[0],candidate_id=f'extra{i}',physical_key=f'extra{i}'))
        with self.assertRaisesRegex(ValueError,'oversized U'):
            p.select_from_complete_rows(r,s,n)

    def test_aliases_unknown_arms_and_duplicate_nominations_rejected(self):
        s,n,r=fixture()
        for extra in (dict(n[0],candidate_id='alias'),dict(n[0],arm='FOREIGN'),n[0]):
            with self.assertRaises(ValueError):
                p.select_from_complete_rows(r,s,n+[extra])


if __name__=='__main__':unittest.main()
