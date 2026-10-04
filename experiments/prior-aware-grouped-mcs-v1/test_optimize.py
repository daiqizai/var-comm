"""CPU scientific-contract tests; synthetic inputs cannot become real policies."""
import copy
import unittest
import numpy as np
import optimize as o


def profile(name,g=1,m=6,k=0,j=None,mods=None,rates=None,wire=None):
    mods=mods or ['QPSK']*g;rates=rates or ['1/2']*g
    return dict(stable_id=name,N=1024,G=g,m=m,K=k,j=j,header_phy_key='header',
        full_state_id=f'm{m}_K{k}',prefix_state_id=f'm{j}_K0' if g==2 else None,
        wire_key=wire or name,groups=[dict(modulation=mods[i],nominal_rate=rates[i],
            mcs_id=mods[i]+':r'+rates[i],phy_key=f'{name}_{i}') for i in range(g)])


def fixture():
    profiles=[profile('A',m=5),profile('B',g=2,j=4),
              profile('C',g=2,j=4,mods=['QPSK','16QAM']),profile('D',g=2,j=4,rates=['1/3','3/4'])]
    ids=['cal0','cal1'];rows=[]
    for source in ids:
        rows.append(dict(source_id=source,state_id='gray',receiver='gray',dinov2_vitl14_cosine=0.))
        for state,var,direct in [('m4_K0',.6,.2),('m5_K0',.5,.9),('m6_K0',.8,.4)]:
            for receiver,value in [('VAR',var),('direct',direct)]:
                rows.append(dict(source_id=source,state_id=state,receiver=receiver,dinov2_vitl14_cosine=value))
    quality=dict(population='calibration',source_ids=ids,rows=rows)
    success={'header':1.,'A_0':.9,'B_0':.9,'B_1':.7,'C_0':.99,'C_1':.99,'D_0':.98,'D_1':.9}
    probabilities=[]
    for key,p in success.items():
        probabilities.append(dict(phy_key=key,snr_db=7,source_id='*',applicability='source_independent_verified',
            p_correct=p,p_reject=1-p,p_undetected=0.,p_correct_ci=[max(0,p-.01),min(1,p+.01)]))
    return profiles,quality,dict(conditional_group_independence=True,rows=probabilities)


class ExpectationTests(unittest.TestCase):
    def test_contiguous_prefix_and_correct_header(self):
        out=o.expected_quality(np.array([.05]),[np.array([.4]),np.array([.8])],np.array([.8]),[np.array([.75]),np.array([.5])])
        self.assertAlmostEqual(out[0],.4*.05+.3*.4+.3*.8)

    def test_late_success_cannot_restore_first_failure(self):
        out=o.expected_quality(np.array([.03]),[np.array([.4]),np.array([.8])],np.array([1.]),[np.array([0.]),np.array([1.])])
        self.assertEqual(out.tolist(),[.03])

    def test_source_correlated_p_and_q_are_not_multiplied_after_averaging(self):
        out=o.expected_quality(np.zeros(2),[np.array([.1,.9])],np.ones(2),[np.array([.9,.1])])
        self.assertAlmostEqual(out.mean(),.09)
        self.assertNotAlmostEqual(out.mean(),.5*.5)

    def test_nonmonotonic_quality_probability_bound_uses_all_corners(self):
        q0=np.zeros(1);states=[np.array([.9]),np.array([.1])]
        lo,hi=o.expected_quality_bounds(q0,states,(np.ones(1),np.ones(1)),
            [(np.ones(1),np.ones(1)),(np.array([.2]),np.array([.8]))])
        self.assertAlmostEqual(lo[0],.26);self.assertAlmostEqual(hi[0],.74)

    def test_rejects_wrong_source_vector_shapes(self):
        with self.assertRaises(ValueError):o.expected_quality(np.zeros(2),[np.ones(2)],np.ones(2),[np.ones(1)])


class OptimizationTests(unittest.TestCase):
    def test_family_constraints_b4_and_matched_controls(self):
        p,q,b=fixture();result=o.optimize(p,q,b,snrs=[7],synthetic=True);cell=result['cells'][0]
        self.assertEqual({f:v['selected']['stable_id'] for f,v in cell['independent_optima'].items()},
                         {'B0':'A','B1':'B','B2':'D','B3':'C','B4':'A'})
        self.assertEqual(cell['strong_baseline']['stable_id'],'D')
        self.assertEqual(cell['frozen_distinct_validation_candidate']['selected']['stable_id'],'B')
        for family in ('B1','B2','B3'):
            chosen=cell['matched_to_B3'][family]['selected'];self.assertEqual((chosen['m'],chosen['K'],chosen['j']),(6,0,4))
        self.assertGreater(cell['prior_aware_delta_same_VAR_receiver'],0)
        self.assertFalse(cell['ready_for_actual_link'])

    def test_common_nominal_mcs_survives_different_effective_rates(self):
        candidate=profile('B',g=2,j=4)
        candidate['groups'][0]['actual_effective_rate']=.41;candidate['groups'][1]['actual_effective_rate']=.47
        self.assertIn('B1',o.memberships(candidate))

    def test_exact_score_tie_prefers_g1_then_stable_id(self):
        p=[profile('z',g=2,j=4),profile('b'),profile('a')]
        prediction={x['stable_id']:dict(mean=.5) for x in p}
        self.assertEqual([x['stable_id'] for x in o.ranked(p,prediction)],['a','b','z'])

    def test_positive_affine_quality_can_preserve_allocation(self):
        p,q,b=fixture()
        for row in q['rows']:
            if row['receiver']=='VAR':
                peer=next(v for v in q['rows'] if v['source_id']==row['source_id'] and v['state_id']==row['state_id'] and v['receiver']=='direct')
                row[o.OBJECTIVE]=peer[o.OBJECTIVE]*.5
        result=o.optimize(p,q,b,snrs=[7],synthetic=True);cell=result['cells'][0]
        self.assertEqual(cell['independent_optima']['B3']['selected']['stable_id'],cell['independent_optima']['B4']['selected']['stable_id'])
        self.assertEqual(cell['prior_aware_delta_same_VAR_receiver'],0)
        self.assertTrue(cell['B3_degenerated_to_single_group'])
        self.assertEqual(cell['matched_to_B3']['B1']['status'],'NOT_APPLICABLE')

    def test_independent_classifier_cannot_change_selected_policies(self):
        p,q,b=fixture();before=o.optimize(p,q,b,snrs=[7],synthetic=True)
        for i,row in enumerate(q['rows']):row[o.INDEPENDENT]=i%2;row['psnr_db']=1000-i
        after=o.optimize(p,q,b,snrs=[7],synthetic=True)
        self.assertEqual(before['cells'],after['cells'])
        self.assertEqual(before['refinement_requirements'],after['refinement_requirements'])

    def test_exact_source_probability_overrides_wildcard(self):
        p,q,b=fixture();row=dict(b['rows'][1],source_id='cal0',p_correct=.2,p_reject=.8,p_correct_ci=[.1,.3]);b['rows'].append(row)
        table=o.ProbabilityTable(b,q['source_ids']);vector=table.vector('A_0',7)[0]
        np.testing.assert_array_equal(vector,[.2,.9])

    def test_undeclared_wildcard_and_invalid_probability_fail(self):
        _,q,b=fixture();b['rows'][0].pop('applicability')
        with self.assertRaises(ValueError):o.ProbabilityTable(b,q['source_ids'])
        _,q,b=fixture();b['rows'][1]['p_undetected']=.3
        with self.assertRaises(ValueError):o.ProbabilityTable(b,q['source_ids'])

    def test_missing_q_or_probability_does_not_silently_drop_candidate(self):
        p,q,b=fixture();q['rows'].pop()
        with self.assertRaises(ValueError):o.optimize(p,q,b,snrs=[7],synthetic=True)
        p,q,b=fixture();b['rows'].pop()
        with self.assertRaises(ValueError):o.optimize(p,q,b,snrs=[7],synthetic=True)

    def test_development_and_small_real_calibration_are_rejected(self):
        p,q,b=fixture()
        with self.assertRaises(ValueError):o.optimize(p,q,b,snrs=[7])
        q['population']='development'
        with self.assertRaises(ValueError):o.optimize(p,q,b,snrs=[7],synthetic=True)

    def test_refinement_points_include_header_matched_and_extra_candidate(self):
        p,q,b=fixture();result=o.optimize(p,q,b,snrs=[7],synthetic=True)
        points={(row['phy_key'],row['snr_db']):row for row in result['refinement_requirements']}
        self.assertEqual(len(points),8);self.assertIn(('header',7),points)
        self.assertIn('1024/distinct_validation',points['B_0',7]['reasons'])
        self.assertTrue(all(not row['automatic_measurement_started'] for row in points.values()))

    def test_true_zero_bler_never_inferred_from_zero_observed_errors(self):
        p,q,b=fixture()
        for row in b['rows']:
            blocks=20000;correct=round(row['p_correct']*blocks)
            row.update(n_blocks=blocks,n_correct=correct,n_reject=blocks-correct,n_undetected=0)
        result=o.optimize(p,q,b,snrs=[7],synthetic=True)
        self.assertTrue(result['cells'][0]['probability_refinement_complete'])
        self.assertFalse(result['cells'][0]['ready_for_actual_link'])
        self.assertTrue(all(r['zero_observed_errors_is_not_zero_true_BLER'] for r in result['refinement_requirements']))


def gate_fixture():
    sources=[f'dev{i:03}' for i in range(100)]
    policies=dict(status='CALIBRATION_POLICIES_SELECTED',synthetic=False,source_count=1000,development_read=False,
        source_ids=[f'cal{i}' for i in range(1000)],cells=[])
    validations=[];rows=[]
    for snr in o.ACTUAL_SNRS:
        policies['cells'].append(dict(N=1024,snr_db=snr,ready_for_actual_link=True,independent_optima={
            f:dict(selected=dict(stable_id=f,G=1 if f=='B0' else 2)) for f in ('B0','B3')}))
        validations.append(dict(N=1024,snr_db=snr,status='PASS',synthetic=False,state_frequency_qualified=True,ranking_qualified=True,
            model_error_vs_gain_qualified=True,strong_baseline_validated=True,distinct_resource_candidate_validated=True))
        for source in sources:
            for seed in o.SEEDS:
                for family in ('B0','B3'):
                    rows.append(dict(N=1024,snr_db=snr,source_id=source,noise_seed=seed,family=family,stable_id=family,
                        synthetic=False,actual_bit_chain_executed=True,convnext_model_id='torchvision_ConvNeXt_Tiny_IMAGENET1K_V1',
                        dinov2_vitl14_cosine=.2 if family=='B0' else .3,
                        convnext_source_prediction_agreement=0 if family=='B0' else 1))
    return rows,sources,policies,validations


class DevelopmentGateTests(unittest.TestCase):
    def test_dual_metric_source_paired_gate_passes_full_grid(self):
        args=gate_fixture();result=o.development_extension_gate(*args)
        self.assertTrue(result['N2048_extension_allowed']);self.assertEqual(result['qualifying_snrs'],list(o.ACTUAL_SNRS))
        self.assertFalse(result['new_samples_to_seek_significance_allowed'])
        for point in result['working_points']:
            self.assertAlmostEqual(point['paired_B3_minus_B0'][o.OBJECTIVE]['mean'],.1)
            self.assertEqual(point['paired_B3_minus_B0'][o.INDEPENDENT]['ci_low'],1.)

    def test_dino_only_improvement_cannot_pass(self):
        rows,ids,policies,v=gate_fixture()
        for row in rows:row[o.INDEPENDENT]=1
        result=o.development_extension_gate(rows,ids,policies,v)
        self.assertFalse(result['N2048_extension_allowed']);self.assertEqual(result['qualifying_snrs'],[])

    def test_positive_metrics_at_different_snrs_do_not_count_as_joint(self):
        rows,ids,policies,v=gate_fixture()
        for row in rows:
            if row['snr_db'] in (4,7):row[o.INDEPENDENT]=1
            else:row[o.OBJECTIVE]=.2
        self.assertFalse(o.development_extension_gate(rows,ids,policies,v)['N2048_extension_allowed'])

    def test_single_group_degenerations_do_not_trigger_expansion(self):
        rows,ids,policies,v=gate_fixture()
        for cell in policies['cells']:cell['independent_optima']['B3']['selected']['G']=1
        self.assertFalse(o.development_extension_gate(rows,ids,policies,v)['N2048_extension_allowed'])

    def test_one_unqualified_model_working_point_blocks_extension(self):
        rows,ids,policies,v=gate_fixture();v[-1]['ranking_qualified']=False
        result=o.development_extension_gate(rows,ids,policies,v)
        self.assertFalse(result['N2048_extension_allowed']);self.assertEqual(len(result['qualifying_snrs']),3)

    def test_missing_failed_frame_or_duplicated_frame_is_rejected(self):
        rows,ids,policies,v=gate_fixture()
        with self.assertRaises(ValueError):o.development_extension_gate(rows[:-1],ids,policies,v)
        with self.assertRaises(ValueError):o.development_extension_gate(rows+[rows[0]],ids,policies,v)

    def test_noise_is_averaged_within_source_before_resampling(self):
        rows,ids,policies,v=gate_fixture()
        for row in rows:
            if row['family']=='B3':row[o.OBJECTIVE]={2001:.1,2002:.2,2003:.6}[row['noise_seed']]
        result=o.development_extension_gate(rows,ids,policies,v)
        metric=result['working_points'][0]['paired_B3_minus_B0'][o.OBJECTIVE]
        self.assertAlmostEqual(metric['mean'],.1);self.assertAlmostEqual(metric['ci_low'],.1)
        self.assertEqual(metric['n_sources'],100);self.assertEqual(metric['n_frames_per_method'],300)

    def test_unfrozen_or_fake_physical_results_rejected(self):
        rows,ids,policies,v=gate_fixture();rows[0]['actual_bit_chain_executed']=False
        with self.assertRaises(ValueError):o.development_extension_gate(rows,ids,policies,v)
        rows,ids,policies,v=gate_fixture();policies['status']='SCREEN_ONLY_NOT_DEPLOYABLE'
        with self.assertRaises(ValueError):o.development_extension_gate(rows,ids,policies,v)


if __name__=='__main__':unittest.main()
