"""Independent synthetic-data review; no real source/model/PHY execution."""
from pathlib import Path
import contextlib
import copy
import hashlib
import io
import json
import math
import sys
import time
import unittest

W=Path(__file__).resolve().parents[2]
E=W/'experiments/layered-mismatch-raw-20261013'
O=W/'results/layered_mismatch_raw_20261013/selection_independent_review_v1'
sys.path.insert(0,str(E))
import contract as c
import policy_selection as p
import test_policy_selection as fixture_module
import test_contract

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
names=('contract.py','policy_selection.py','test_policy_selection.py','test_contract.py')
bindings={name:sha(E/name) for name in names}

class IndependentReview(unittest.TestCase):
    def test_nominal_anchor_winner_must_strengthen_single(self):
        sources,nominations,rows=fixture_module.fixture()
        for row in rows:
            if row['candidate_id']=='S_nom': row[c.SELECTION_METRIC]=.9
        result=p.select_from_complete_rows(rows,sources,nominations)
        self.assertEqual([w['candidate_id'] for w in result['winners'] if w['method']=='S_robust'],['S_nom','S_nom'])
        self.assertFalse(result['gate']['r2_authorized'])
        self.assertEqual(result['logical_rows'],21600)

    def test_four_u_rejected_even_when_logical_total_below_cap(self):
        sources,nominations,rows=fixture_module.fixture()
        for hat in c.HATS:
            prototype=next(x for x in nominations if x['arm']=='U' and x['snr_hat']==hat)
            template=[x for x in rows if x['candidate_id']=='U' and x['snr_hat']==hat]
            for number in (2,3,4):
                cid='U'+str(number)
                nominations.append(dict(prototype,candidate_id=cid,physical_key='phy'+cid))
                rows.extend(dict(x,candidate_id=cid,physical_key='phy'+cid) for x in template)
        self.assertEqual(len(rows),32400)
        with self.assertRaisesRegex(ValueError,'oversized U'):p.select_from_complete_rows(rows,sources,nominations)

    def test_canonical_anchor_reuse_is_one_physical_and_one_logical_row(self):
        sources,nominations,rows=fixture_module.fixture()
        for nomination in nominations:
            if nomination['arm']=='S_nom':nomination.update(candidate_id='S',physical_key='phyS')
        rows=[r for r in rows if r['candidate_id']!='S_nom']
        result=p.select_from_complete_rows(rows,sources,nominations)
        self.assertEqual(result['logical_rows'],18000)
        self.assertEqual(c.validate_shortlist(nominations)['physical_candidates_per_hat'],{'7':4,'10':4})
        self.assertTrue(all(w['candidate_id']=='S' for w in result['winners'] if w['method']=='S_nom'))

    def test_physical_aliases_cannot_get_different_quality(self):
        sources,nominations,rows=fixture_module.fixture()
        alias=dict(nominations[0],candidate_id='false_alias')
        with self.assertRaisesRegex(ValueError,'canonical'):p.select_from_complete_rows(rows,sources,nominations+[alias])

    def test_real_three_noise_source_mixture_and_pressure_exclusion(self):
        sources,nominations,rows=fixture_module.fixture()
        for row in rows:
            if row['candidate_id']=='U':
                source_value=int(row['source_id'][1:])/10000
                noise_shift=(row['noise_seed']-4102)/1000
                base=.8 if row['snr_true']==row['snr_hat'] else (.4 if row['snr_true']==row['snr_hat']-3 else -.9)
                row[c.SELECTION_METRIC]=base+source_value+noise_shift
        result=p.select_from_complete_rows(rows,sources,nominations)
        expected=.75*.8+.25*.4+math.fsum(i/10000 for i in range(200))/200
        for winner in result['winners']:
            if winner['method']=='U_two': self.assertAlmostEqual(winner['cal200_DINO_J'],expected,places=14)

    def test_psnr_engineering_target_does_not_replace_dino_gate(self):
        sources,nominations,rows=fixture_module.fixture()
        for row in rows:
            if row['candidate_id']=='U':row['psnr_db']=0;row['lpips_alex']=1
        result=p.select_from_complete_rows(rows,sources,nominations)
        self.assertTrue(result['gate']['r2_authorized'])
        self.assertTrue(result['gate']['engineering_targets_are_separate'])

    def test_maximum_registered_shortlist_has_exact_frozen_caps(self):
        _,nominations,_=fixture_module.fixture()
        for hat in c.HATS:
            for arm,extra in (('U',2),('S',2),('EEP',1)):
                template=next(x for x in nominations if x['arm']==arm and x['snr_hat']==hat)
                for number in range(extra):
                    cid=arm+str(number+2)
                    nominations.append(dict(template,candidate_id=cid,physical_key='phy'+cid))
        result=c.validate_shortlist(nominations)
        self.assertEqual(result['physical_candidates_per_hat'],{'7':10,'10':10})
        self.assertEqual(result['physical_frame_upper_bound'],36000)
        self.assertEqual(result['logical_var_output_upper_bound'],46800)

suite=unittest.TestSuite()
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_contract))
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(fixture_module))
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentReview))
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
assert result.wasSuccessful(),stream.getvalue()
assert 'torch' not in sys.modules
assert bindings=={name:sha(E/name) for name in names},'Source changed during independent review'
O.mkdir()
receipt=dict(status='PASS_READ_ONLY_SELECTION_REVIEW_AFTER_FIXES', test_count=result.testsRun,
   real_model_calls=0, real_phy_calls=0, real_channel_calls=0, bootstrap_calls=0,
   synthetic_data_only=True, torch_imported=False, source_bindings=bindings,
   reviewer_source_sha256=sha(__file__), actual_exit_code=0,
   original_blockers=[
      dict(id='R1_CAP_BYPASS', evidence='Initial inspected selector did not invoke validate_shortlist; a four-U nomination could remain below its logical cap.',
           status='FIXED_SELECTOR_VALIDATES_SHORTLIST_BEFORE_REDUCING', verification='Synthetic complete 32400-row four-U fixture is rejected.'),
      dict(id='ANCHOR_EXCLUSION_FROM_STRONG_SINGLE', evidence="Initial inspected S_robust loop admitted only arm S, excluding mandatory fixed anchor outputs unless separately nominated as S.",
           status='FIXED_ANCHOR_KEEP_RESULTS_AUTOMATICALLY_INCLUDED', verification='Nominal-anchor DINO 0.9 becomes S_robust and prevents false GO against U 0.65, without new rows.'),
      dict(id='PHYSICAL_ALIAS_REPLAY', evidence='Separate candidate IDs for one physical key could create inconsistent logical reductions.',
           status='FIXED_ONE_CANONICAL_CANDIDATE_PER_PHYSICAL_KEY_WITHIN_HAT', verification='False alias rejected; canonical S/S_nom reuse reduces logical rows and keeps one observation.')],
   verified=['per-hat U3/S3/EEP2 plus two fixed anchors and total10 physical',
       'same actual receive identity required for KEEP/DROP; anchors reuse KEEP without additional DROP budget',
       'source-first three-noise average; .75 matched+.25 true=hat-3; true=hat+3 excluded from selection',
       'equal weighting of hats in both U-S_robust and U-S_back3; means>=.01 and both hats strictly positive',
       'ConvNeXt excluded; PSNR targets remain separate; no bootstrap during selection',
       'complete cal200 grid, duplicates/missing/foreign source/noise rejected'],
   readiness_limit='This pure-data function does not authenticate files by itself. Scientific owner must validate registered cal200/nomination hashes, frozen anchor/configuration identities, actual exit0 of PHY/visual/scoring, frame ledger and artifact SHAs before invoking it. Returned requires_authenticated_actual_stage_closure is a requirement, not a receipt.',
   r0_gap=dict(missing_exact_layout_snr_points=1852, no_new_bler_budget_assumed=True,
       no_unique_fair_shortlist_determined=True, note='Original R1 36000 covers nominated complete frames, not arbitrary scans of hundreds of missing body layouts. Updated clean-Q coverage belongs to r0_reuse final addendum; no missing success probabilities are filled with 0/1.'),
   reviewed_unix=time.time())
with (O/'review.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2)
(O/'tests.txt').write_text(stream.getvalue(),encoding='utf-8')
(O/'review_selection_final.py').write_text(Path(__file__).read_text(),encoding='utf-8')
print(json.dumps(receipt,sort_keys=True))
