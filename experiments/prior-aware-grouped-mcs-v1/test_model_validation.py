"""Synthetic matched-source fixtures; no development images or PHY simulation."""
import copy
import unittest
import numpy as np
import model_validation as m
import optimize as o


def fixture():
    ids=['synthetic_dev'+str(i) for i in range(100)];profiles={};choices={}
    for method,scale in zip(m.METHODS,(6,5,4)):
        profile=dict(stable_id=method,wire_key=method,N=1024,G=1,m=scale,K=0,j=None,full_state_id=f'm{scale}_K0',
            header_phy_key='header',groups=[dict(phy_key='body')])
        profiles[method]=profile;choices[method]=dict(profile=profile,predicted_VAR_mean={6:.8,5:.6,4:.4}[scale])
    policies=dict(status='CALIBRATION_POLICIES_SELECTED',source_count=1000,source_ids=['cal'+str(i) for i in range(1000)],synthetic=False,development_read=False,
        cells=[dict(N=1024,snr_db=s,ready_for_actual_link=True,independent_optima={'B3':dict(selected=choices['B3'])},
            strong_baseline=choices['strong_baseline'],frozen_distinct_validation_candidate=dict(status='FROZEN',selected=choices['validation_candidate'])) for s in o.ACTUAL_SNRS])
    bler=dict(status='REFINEMENT_COMPLETE',synthetic=False,conditional_group_independence=True,rows=[dict(phy_key=k,snr_db=s,source_id='*',
        applicability='iid_scrambled_payload_approximation',n_blocks=20000,n_correct=20000,n_reject=0,n_undetected=0,
        p_correct=1.,p_reject=0.,p_undetected=0.) for k in ('header','body') for s in o.ACTUAL_SNRS])
    clean=[];scores=[];frames={}
    for i,source in enumerate(ids):
        qs={'gray':.1,6:.8+i*.0001,5:.6+i*.0001,4:.4+i*.0001}
        for state,value in qs.items():clean.append(dict(source_id=source,state_id='gray' if state=='gray' else f'm{state}_K0',
            receiver='gray' if state=='gray' else 'VAR',dinov2_vitl14_cosine=value,diagnostic_only=True,development_Q_used_for_selection=False,policy_reselection_allowed=False))
        for snr in o.ACTUAL_SNRS:
            for seed in o.SEEDS:
                for method in m.METHODS:
                    profile=profiles[method];key=(source,snr,seed,method);fid=str(key)
                    frames[key]=dict(physical_frame_id=fid,synthetic=True,actual_bit_chain_executed=False,
                        receiver_event=dict(state=dict(kind='tokens',m=profile['m'],K=0,accepted_groups=1)),
                        offline_truth=dict(header_correct=True,groups=[dict(truth_correct_after_receiver=True)]))
                    scores.append(dict(N=1024,source_id=source,snr_db=snr,noise_seed=seed,method=method,stable_id=method,
                        physical_frame_id=fid,dinov2_vitl14_cosine=qs[profile['m']]))
    return policies,bler,ids,clean,scores,frames


class ValidationTests(unittest.TestCase):
    def test_well_resolved_correct_model_qualifies_but_remains_synthetic(self):
        result=m.validate(*fixture(),synthetic=True,bootstrap_replicates=300)
        self.assertEqual(result['status'],'SYNTHETIC_MODEL_VALIDATION_FIXTURE');self.assertTrue(result['all_four_points_qualified'])
        self.assertTrue(all(r['synthetic'] for r in result['model_validation']))
        for row in result['model_validation']:
            other=row['methods']['B3']['state_frequencies'][-1]
            self.assertGreater(other['envelope_high'],0)  # zero observed corruption is not zero true probability

    def test_reversed_actual_ranking_fails(self):
        args=list(fixture())
        for row in args[4]:
            if row['method']=='B3':
                row[o.OBJECTIVE]=.1;key=(row['source_id'],row['snr_db'],row['noise_seed'],'B3')
                args[5][key]['receiver_event']['state']=dict(kind='gray',m=0,K=0,accepted_groups=0)
        result=m.validate(*args,synthetic=True,bootstrap_replicates=300)
        self.assertFalse(result['all_four_points_qualified']);self.assertTrue(all(r['status']=='FAIL' for r in result['model_validation']))

    def test_error_same_order_as_gain_is_uncertain_without_adding_samples(self):
        args=list(fixture());dropped=set(args[2][:10])
        for row in args[4]:
            if row['method']=='B3' and row['source_id'] in dropped:
                row[o.OBJECTIVE]=.1;args[5][row['source_id'],row['snr_db'],row['noise_seed'],'B3']['receiver_event']['state']=dict(kind='gray',m=0,K=0,accepted_groups=0)
        result=m.validate(*args,synthetic=True,bootstrap_replicates=500)
        self.assertFalse(result['all_four_points_qualified']);self.assertFalse(result['additional_samples_allowed'])
        self.assertTrue(all(not r['model_error_vs_gain_qualified'] for r in result['model_validation']))

    def test_changed_correct_state_Q_is_rejected(self):
        args=list(fixture());args[3][1][o.OBJECTIVE]-=.02
        with self.assertRaisesRegex(RuntimeError,'Correct receive state'):m.validate(*args,synthetic=True,bootstrap_replicates=100)

    def test_development_quality_must_be_diagnostic_only(self):
        args=list(fixture());args[3][0]['development_Q_used_for_selection']=True
        with self.assertRaises(RuntimeError):m.validate(*args,synthetic=True,bootstrap_replicates=100)

    def test_missing_noise_repeat_and_unfrozen_neighbour_rejected(self):
        args=list(fixture());args[4].pop()
        with self.assertRaises(RuntimeError):m.validate(*args,synthetic=True,bootstrap_replicates=100)
        args=list(fixture());args[0]['cells'][0]['frozen_distinct_validation_candidate']['status']='CHOSEN_AFTER_DEV'
        with self.assertRaises(RuntimeError):m.validate(*args,synthetic=True,bootstrap_replicates=100)

    def test_false_accepted_payload_is_not_a_correct_full_state(self):
        args=fixture();frame=copy.deepcopy(next(iter(args[5].values())));profile=args[0]['cells'][0]['independent_optima']['B3']['selected']['profile']
        frame['offline_truth']['groups'][0]['truth_correct_after_receiver']=False
        self.assertEqual(m.observed_state(frame,profile),'other')
        frame['offline_truth']['header_correct']=False;self.assertEqual(m.observed_state(frame,profile),'other')

    def test_prediction_keeps_source_probability_quality_correlation(self):
        profile=dict(G=1,header_phy_key='h',groups=[dict(phy_key='b')],full_state_id='full')
        rows=[]
        for key in ('h','b'):
            for source,prob in [('a',1.),('b',1. if key=='h' else 0.)]:
                rows.append(dict(phy_key=key,snr_db=7,source_id=source,p_correct=prob,p_reject=1-prob,p_undetected=0.,
                    n_blocks=20000,n_correct=int(20000*prob),n_reject=int(20000*(1-prob)),n_undetected=0))
        quality={('a','gray','gray'):0.,('b','gray','gray'):0.,('a','full','VAR'):.9,('b','full','VAR'):.1}
        probability=o.ProbabilityTable(dict(conditional_group_independence=True,rows=rows),['a','b'])
        value=m.prediction(profile,7,['a','b'],quality,probability,2.)
        self.assertAlmostEqual(value['quality'].mean(),.45);self.assertNotAlmostEqual(value['quality'].mean(),.25)


if __name__=='__main__':unittest.main()
