"""CPU logic tests; fixtures are engineering-only, with no metric model load."""
import copy
import unittest
import external_history_confidence as h


def measured_row(error=1):
    return dict(selected=dict(group='P',N=2048,snr_db=7,noise_seed=2001,source_index=0,
        source_id='fixture',history_row_id='fixture_row',decoder='Dc',true_class_paid=False,
        training_seed=2026092304,semantic_error=error,true_label_error=error,psnr_db=20.,lpips_alex=.2,
        dinov2_vitl14_cosine=.8,clip_image_cosine=.9,dists=.2,dreamsim=.3),
        history_study='FINAL_P2048_P3060',image_sha256='a'*64,reference_sha256='b'*64,
        expected_prediction=1 if error else 2,expected_source_prediction=2,class_index=2)


def rows_fixture():
    result=[]
    groups=[('P',n) for n in (2048,3060,4084)]
    groups += [(g,n) for g in ('Digital raw / Dc','Digital arithmetic / Dc') for n in (2048,3060,4084)]
    groups += [(g,3060) for g in ('Legacy raw / D0','Legacy arithmetic / D0')]
    for group,N in groups:
        for snr in (7,13):
            for i in range(100):
                for seed in (2001,2002,2003):
                    wrong=int(i%2==0)
                    result.append(dict(group=group,N=N,snr_db=snr,source_index=i,noise_seed=seed,
                        history_row_id=f'{group}/{N}/{snr}/{i}/{seed}',confidently_wrong=wrong,
                        semantic_error=wrong,resnet50_top1_probability=.75,resnet50_source_top1_probability=.8,
                        decoder='D0' if group.startswith('Legacy') else 'Dc',true_class_paid=group!='P',
                        training_seed=2026092304 if group=='P' else None))
    return result


class ConfidenceTests(unittest.TestCase):
    def test_threshold_inclusive_and_original_prediction_is_reference(self):
        row=measured_row()
        value=h.enrich(row,{'prediction':1,'probability':.5},{'prediction':2,'probability':.3})
        self.assertEqual(value['confidently_wrong'],1)
        self.assertEqual(value['resnet50_source_top1_probability'],.3)
        self.assertEqual(h.enrich(row,{'prediction':1,'probability':.499},{'prediction':2,'probability':.3})['confidently_wrong'],0)
        correct=measured_row(0)
        self.assertEqual(h.enrich(correct,{'prediction':2,'probability':.99},{'prediction':2,'probability':.3})['confidently_wrong'],0)

    def test_any_old_argmax_mismatch_stops(self):
        row=measured_row()
        with self.assertRaises(RuntimeError): h.enrich(row,{'prediction':3,'probability':.9},{'prediction':2,'probability':.8})
        with self.assertRaises(RuntimeError): h.enrich(row,{'prediction':1,'probability':.9},{'prediction':4,'probability':.8})

    def test_probability_cache_requires_image_runtime_and_seal(self):
        value=dict(registration_sha256='runtime',image_sha256='a'*64,prediction=3,probability=.8)
        value['payload_sha256']=h.identity(value)
        h.check_probability(value,'runtime','a'*64)
        with self.assertRaises(RuntimeError): h.check_probability(value,'different','a'*64)
        with self.assertRaises(RuntimeError): h.check_probability(value,'runtime','b'*64)
        value['probability']=.9
        with self.assertRaises(RuntimeError): h.check_probability(value,'runtime','a'*64)

    def test_exact_scope_source_bootstrap_and_paired_differences(self):
        rows=rows_fixture();summary,contrasts=h.analyze(rows)
        self.assertEqual(len(summary),22*4)
        self.assertEqual(len(contrasts),16*4)
        errors=[r for r in summary if r['metric']=='confidently_wrong']
        self.assertTrue(all(r['mean']==.5 and r['sources']==100 and r['frames']==300 for r in errors))
        self.assertEqual(len({(r['ci_low'],r['ci_high']) for r in errors}),1)
        self.assertTrue(all(r['mean_difference']==r['ci_low']==r['ci_high']==0 for r in contrasts))
        self.assertTrue(all(not r['classification_main_eligible'] for r in contrasts))

    def test_no_missing_or_duplicate_noise_allowed(self):
        rows=rows_fixture()
        with self.assertRaises(RuntimeError): h.analyze(rows[:-1])
        rows[-1]['noise_seed']=2002
        with self.assertRaises(RuntimeError): h.analyze(rows)

    def test_csv_serialization_preserves_existing_values(self):
        self.assertEqual(h.serialized({'paid':True,'seed':None,'N':2048,'lpips':.1}),
            {'paid':'True','seed':'','N':'2048','lpips':'0.1'})


if __name__=='__main__': unittest.main()
