"""Synthetic admission/accounting checks. No real images or bootstrap draws."""
import copy
import unittest
import numpy as np
import analyze as a


def frames(ids):
    result=[]
    for i,sid in enumerate(ids):
        for snr in a.SNRS:
            for noise in a.NOISES:
                r=dict(method=a.NEW,N=1024,source_index=i,source_id=sid,snr_db=snr,noise_seed=noise,
                       physical_status='BODY_CRC_REJECT_KEEP' if noise==6202 else 'PAYLOAD_PARSED',
                       source_status='ACTUAL_RX_OUTPUT',source_parse_reason='')
                for j,metric in enumerate(a.METRICS):
                    r[metric]=int(noise==6203) if j==3 else float(10*i+noise-6201+j)
                result.append(r)
    return result


def streams(ids):
    result=[]
    for i,sid in enumerate(ids):
        for snr in a.SNRS:
            m,k,capacity=a.TARGETS[snr]
            result.append(dict(method=a.NEW,source_id=sid,source_index=i,snr_db=snr,target_m=m,target_K=k,
                               actual_m=m,actual_K=0,actual_T=100,arithmetic_bits=500,capacity_bits=capacity,
                               fallback_attempts=[dict(m=m,K=k,bits=capacity+1,fits=False),
                                                  dict(m=m,K=0,bits=500,fits=True)]))
    return result


class AnalysisAdmission(unittest.TestCase):
    def setUp(self):
        self.ids=['synthetic_source_0','synthetic_source_1']

    def test_source_mean_uses_all_three_noises_and_keeps_failure(self):
        data=frames(self.ids)
        keyed,vectors,means=a.normalize_frame_grid(data,self.ids)
        self.assertEqual(len(keyed),18)
        self.assertEqual(len(means),24)
        np.testing.assert_array_equal(vectors[a.NEW,4,a.METRICS[0]], [1,11])
        np.testing.assert_array_equal(vectors[a.NEW,4,a.METRICS[-1]], [1/3,1/3])
        failure=a.failure_table(data)
        self.assertEqual(sum(r['frame_count'] for r in failure if 'REJECT' in r['physical_status']),6)

    def test_incomplete_and_duplicate_frames_rejected(self):
        data=frames(self.ids)
        for wrong in (data[:-1], data+[data[0]]):
            with self.assertRaises(ValueError):a.normalize_frame_grid(wrong,self.ids)

    def test_wrong_source_order_or_non_binary_agreement_rejected(self):
        data=frames(self.ids);data[0]['source_id']=self.ids[1]
        with self.assertRaises(ValueError):a.normalize_frame_grid(data,self.ids)
        data=frames(self.ids);data[0][a.METRICS[-1]]=.5
        with self.assertRaises(ValueError):a.normalize_frame_grid(data,self.ids)

    def test_source_lengths_deduplicate_noise_but_not_inconsistent_tx(self):
        data=streams(self.ids)
        unique=a.source_accounting(data*3,self.ids)
        self.assertEqual(len(unique),6)
        self.assertEqual(unique[0]['source_saving_fraction'],1-500/1200)
        self.assertEqual(unique[0]['source_capacity_utilization'],500/1245)
        self.assertEqual(unique[0]['fallback_steps'],1)
        self.assertTrue(unique[0]['target_did_not_fit'])
        broken=copy.deepcopy(data);broken[0]['actual_T']=101
        with self.assertRaises(ValueError):a.source_accounting(data+broken,self.ids)

    def test_zero_token_stream_retains_failure_without_division(self):
        data=streams(self.ids)
        data[0].update(actual_T=0,arithmetic_bits=None,actual_m=0,actual_K=0,
                       fallback_attempts=[dict(m=7,K=25,bits=9999,fits=False)])
        unique=a.source_accounting(data,self.ids)
        self.assertFalse(unique[0]['final_stream_legal'])
        self.assertEqual(unique[0]['source_saving_fraction'],'')
        self.assertEqual(unique[0]['source_capacity_utilization'],'')
        prefix,summaries=a.length_summaries(unique)
        self.assertEqual(sum(r['source_count'] for r in prefix if r['snr_db']==4),2)
        self.assertEqual(summaries[0]['no_legal_stream_count'],1)

    def test_frozen_target_and_real_fit_capacity_checked(self):
        data=streams(self.ids);data[0]['target_K']=26
        with self.assertRaises(ValueError):a.source_accounting(data,self.ids)
        data=streams(self.ids);data[0]['fallback_attempts'][0]['fits']=True
        with self.assertRaises(ValueError):a.source_accounting(data,self.ids)

    def test_reference_vectors_require_exact_source_and_summary_grid(self):
        data=[];summary=[]
        for method in a.ALL_REFS:
            for snr in a.SNRS:
                for metric in a.METRICS:
                    for i,sid in enumerate(self.ids):
                        data.append(dict(method=method,snr_db=snr,metric=metric,source_index=i,source_id=sid,mean=i))
                    summary.append(dict(method=method,snr_db=snr,metric=metric,mean=.5,ci_low=0,ci_high=1,
                                        source_count=2,noise_count=3))
        vectors=a.reference_vectors(data,summary,self.ids)
        self.assertEqual(len(vectors),72)
        with self.assertRaises(ValueError):a.reference_vectors(data[:-1],summary,self.ids)
        summary[0]['mean']=.6
        with self.assertRaises(ValueError):a.reference_vectors(data,summary,self.ids)

    def test_lpips_not_sign_flipped_and_agreement_pp(self):
        row=a.display(dict(metric=a.METRICS[1],mean=-.01),paired=True)
        self.assertEqual(row['mean'],-.01);self.assertEqual(row['improvement_direction'],'negative')
        row=a.display(dict(metric=a.METRICS[-1],mean=.02),paired=True)
        self.assertEqual(row['mean'],.02);self.assertEqual(row['display_scale'],100)
        self.assertEqual(row['display_unit'],'percentage_points')


if __name__=='__main__':unittest.main()
