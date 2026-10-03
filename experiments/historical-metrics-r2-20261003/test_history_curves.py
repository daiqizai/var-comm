"""Pure scientific point-selection regressions; no plotting/GPU dependency."""
from copy import deepcopy
import unittest
import history_curves as c


def original(experiment='N512',method='P512',N=512,phy='continuous',condition=False,decoder='Dc',**kw):
    row=dict(experiment=experiment,scope='historical_development',N=str(N),phy_family=phy,
        snr_db='13',method=method,projection='',control='',output_role='label_conditioned' if condition else 'main',
        decoder_id=decoder,label_conditioned=str(condition).lower(),group_id='original-'+method,
        metric='dinov2_vitl14_cosine',mean='.7',ci_low='.68',ci_high='.72',n_sources='100',
        n_frames='300',status='EVALUATED')
    row.update(kw);return row


def historical(study='FINAL_P2048_P3060',method='P2048',N=2048,phy='continuous',condition=False,decoder='Dc',**kw):
    row=dict(study=study,method=method,N=str(N),phy=phy,condition='C' if condition else 'U',
        snr_db='13',snr_definition='snr_db',decoder=decoder,label_conditioned=str(condition),
        oracle='False',reference_only='False',training_seed='2026092304',model_id='checkpoint-'+method,
        metric='new_dinov2_vitl14_cosine',mean='.8',ci_low='.78',ci_high='.82',sources='100',
        frames='300',projection='',group_id='historical-'+method)
    row.update(kw);return row


def select(historical_rows=(),original_rows=()):
    return c.select_points(list(historical_rows),list(original_rows),
        {'historical':'history.csv','six_study':'six.csv'},
        {'historical':'a'*64,'six_study':'b'*64})


class CurveScopeTests(unittest.TestCase):
    def test_same_primary_p_across_budgets_but_second_seed_separate(self):
        hist=[historical(),historical(method='P3060',N=3060),
            historical(study='FINAL_P4084_SELECTED_SEEDS',method='P4084_N4084_seed2026092304',N=4084),
            historical(study='FINAL_P4084_SELECTED_SEEDS',method='P4084_N4084_seed2026092404',N=4084,training_seed='2026092404')]
        points,_=select(hist,[original(),original(experiment='N1024',method='P1024',N=1024)])
        panel=[p for p in points if p['panel']=='U_QPSK']
        primary=[p for p in panel if p['base_series']=='P_SELECTED']
        self.assertEqual({p['N'] for p in primary},set(c.BUDGETS))
        self.assertEqual(len({p['series_id'] for p in primary}),1)
        self.assertEqual(len(c.contiguous_segments(primary)),1)
        other=[p for p in panel if p['base_series']=='P_SECOND_SEED']
        self.assertEqual(len(other),1)
        self.assertNotEqual(other[0]['series_id'],primary[0]['series_id'])
        self.assertEqual(primary[0]['training_seed'],'2026092304')
        self.assertEqual([p for p in primary if p['N']==512][0]['training_seed'],'')

    def test_no_lines_across_missing_registered_budget(self):
        points=[dict(N=n,connect=True) for n in (512,1024,3060,4084)]
        self.assertEqual([[p['N'] for p in s] for s in c.contiguous_segments(points)],[[512,1024],[3060,4084]])
        self.assertEqual(len(points),4)

    def test_adjacent_tick_labels_do_not_move_resource_points(self):
        points=[dict(N=3060),dict(N=3072),dict(N=6144)]
        original=deepcopy(points)
        ticks=c.resource_ticks(points)
        self.assertIn('3060\n3072',[label for _,label in ticks])
        self.assertIn('6144',[label for _,label in ticks])
        self.assertEqual(points,original)
        self.assertEqual(len(ticks),6)

    def test_paid_class_does_not_fill_unconditional_gap(self):
        hist=[historical(study='FINAL_DIGITAL_QPSK',method='final_raw_QPSK_N2048_Dc',N=2048,phy='QPSK',condition=True)]
        orig=[original(method='D_U_QPSK',phy='QPSK'),original(experiment='N1024',method='D_U_QPSK',N=1024,phy='QPSK')]
        points,_=select(hist,orig)
        self.assertEqual({p['N'] for p in points if p['panel']=='U_QPSK'},{512,1024})
        self.assertEqual({p['N'] for p in points if p['panel']=='C_QPSK'},{2048})

    def test_modulation_and_energy_annotation_remain_separate(self):
        hist=[historical(study='FINAL_DIGITAL_'+m,method='final_raw_'+m+'_N2048_Dc',N=2048,phy=m,condition=True) for m in ('QPSK','16QAM')]
        points,_=select(hist)
        self.assertEqual(len({p['series_id'] for p in points}),2)
        self.assertEqual({p['panel'] for p in points},{'C_QPSK','C_16QAM'})
        q=next(p for p in points if p['phy']=='16QAM')
        self.assertEqual(q['energy_protocol'],'original_actual_energy_not_assumed_2N')

    def test_d0_and_author_points_are_unconnected_references(self):
        hist=[historical(study='LEGACY_N3060_FINAL',method='raw_adaptive',N=3060,phy='QPSK',condition=True,decoder='D0'),
              historical(study='SELECTED_EXTERNAL_AUTHORS',method='swin_c48|author_assumption',N=6144,phy='author_original',decoder='SwinJSCC_author_decoder',condition='author_assumption')]
        # The condition above names a protocol; label access remains false.
        hist[1]['label_conditioned']='False'
        hist[1]['condition']='author_assumption'
        points,_=select(hist)
        self.assertEqual({p['panel'] for p in points},{'REFERENCE'})
        self.assertTrue(all(not p['connect'] for p in points))
        self.assertEqual(len({p['series_id'] for p in points}),2)

    def test_model_variants_not_averaged_if_same_series_budget(self):
        row=historical()
        with self.assertRaisesRegex(ValueError,'Duplicate resource point'):
            select([row,{**row,'model_id':'another_checkpoint'}])

    def test_oracle_clean_and_ablation_rows_do_not_enter_resource_curves(self):
        orig=[original(experiment='M2_ORACLE',method='VAR_GUIDED',N='',snr_db='clean',output_role='oracle'),
              original(experiment='M1',method='oracle_policy',phy='QPSK',output_role='paid_oracle_reference'),
              original(experiment='M1',method='raster_at_entropy',phy='QPSK')]
        points,excluded=select([],orig)
        self.assertEqual(points,[])
        self.assertEqual(len(excluded),3)

    def test_selection_independent_of_quality_values(self):
        rows=[original(method='D_U_QPSK',phy='QPSK'),original(experiment='M1',method='entropy_policy',phy='QPSK')]
        a,_=select([],rows)
        changed=[{**r,'mean':str(1-float(r['mean']))} for r in rows]
        b,_=select([],changed)
        self.assertEqual([(x['series_id'],x['N']) for x in a],[(x['series_id'],x['N']) for x in b])

    def test_equivalent_latent_snr_and_missing_metrics_excluded(self):
        hist=[historical(snr_definition='snr_equiv_db')]
        orig=[original(status='NOT_EVALUATED',mean='',ci_low='',ci_high='')]
        points,excluded=select(hist,orig)
        self.assertFalse(points)
        self.assertEqual({x['reason'] for x in excluded},{'different_snr_definition','metric_not_evaluated'})

    def test_legacy_phase2_not_joined_to_final_policy(self):
        hist=[historical(study='FINAL_PHASE2_N4084_DIGITAL',method='raw_adaptive_m789_Dc',N=4084,phy='QPSK',condition=True),
              historical(study='FINAL_DIGITAL_QPSK',method='final_raw_QPSK_N3060_Dc',N=3060,phy='QPSK',condition=True)]
        points,_=select(hist)
        self.assertEqual(len({p['series_id'] for p in points}),2)

    def test_original_summary_and_metric_name_are_retained(self):
        points,_=select([historical()])
        p=points[0]
        self.assertEqual(p['metric'],'dinov2_vitl14_cosine')
        self.assertIn('new_dinov2_vitl14_cosine',p['original_summary_json'])
        self.assertEqual(p['source_sha256'],'a'*64)
        self.assertEqual(p['original_group_id'],'historical-P2048')

    def test_real_m2_control_and_projection_spellings_remain_separate(self):
        # Names are from M2_CONTROLS and the frozen actual policy: g4_c16,
        # g6_c8 and g8_c8. Different projections are not one unregistered curve.
        orig=[original(experiment='M2_ACTUAL',method='VAR_GUIDED',control='VAR_GUIDED',
              phy='QPSK',projection='g4_c16'),
              original(experiment='M2_ACTUAL',method='VAR_GUIDED',control='VAR_GUIDED',
              phy='QPSK',N=1024,projection='g8_c8')]
        points,_=select([],orig)
        self.assertEqual(len(points),2)
        self.assertEqual(len({p['series_id'] for p in points}),2)


if __name__=='__main__':unittest.main()
