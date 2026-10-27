"""Synthetic joins only; no runtime science paths or model imports."""
import copy,unittest
from contracts import CHOICES
from plan import build_plan


def descriptor(n):return dict(path='/synthetic/'+n,bytes=1,sha256='a'*64)


def fixture():
    sources=[];refs=[];logical=[];outputs=[]
    for i in range(2):
        sid='source'+str(i);a=descriptor('source'+str(i))
        sources.append(dict(source_id=sid,source_index=i,archive=a,source_float_sha256=str(i)*64))
        refs.append(dict(source_id=sid,source_index=i,source=a,file=descriptor('reference'+str(i)),feature_array_sha256='b'*64))
        key='rx'+str(i)
        outputs.append(dict(key=key,image=descriptor(key),pixel_sha256='c'*64,source_truth_used=False,TX_CDF_used=False,status='ACCEPTED_SOURCE'))
        for snr,cids in CHOICES.items():
            for cid in cids:
                for noise in (4101,4102,4103):
                    gray=snr==4 and cid==cids[0]
                    logical.append(dict(event_key=f'{sid}_{snr}_{cid}_{noise}',source_id=sid,source_index=i,
                        snr_db=snr,candidate_id=cid,noise_seed=noise,frame=descriptor('frame'),wire_status='fixed',
                        RX_key=None if gray else key,origin='FIXED_GRAY'if gray else'NEW_UNIQUE_RX',
                        projection_status='FIXED_GRAY'if gray else'SOURCE_INPUT',reason='fixed'if gray else None))
    return dict(sources=sources,logical=logical,outputs=outputs,references=refs,reuse=[],gray_sha='d'*64,
                donor_metric_ids={'e'*64},expected_sources=2)


class PlanTests(unittest.TestCase):
    def test_logical_denominator_and_pairs(self):
        p=build_plan(**fixture());self.assertEqual(p['counts']['logical_frames'],36)
        self.assertEqual(p['counts']['source_pair_rows'],4)
        self.assertEqual([sum(len(p['frames'])for p in s['pairs'])for s in p['sources']],[18,18])
    def test_same_reconstruction_different_source_not_merged(self):
        p=build_plan(**fixture());self.assertEqual(len(p['sources']),2)
        self.assertNotEqual(p['sources'][0]['source_float_sha256'],p['sources'][1]['source_float_sha256'])
    def test_missing_output_is_not_gray(self):
        v=fixture();v['outputs'].pop()
        with self.assertRaises(AssertionError):build_plan(**v)
    def test_extra_output_rejected(self):
        v=fixture();x=copy.deepcopy(v['outputs'][0]);x['key']='other';v['outputs'].append(x)
        with self.assertRaises(AssertionError):build_plan(**v)
    def test_duplicate_event_rejected(self):
        v=fixture();v['logical'][1]['event_key']=v['logical'][0]['event_key']
        with self.assertRaises(AssertionError):build_plan(**v)
    def test_bad_reference_archive_rejected(self):
        v=fixture();v['references'][0]['source']=descriptor('wrong')
        with self.assertRaises(AssertionError):build_plan(**v)
    def test_old_score_reuse_requires_same_archive(self):
        v=fixture();v['reuse']=[dict(source_id='source0',archive=descriptor('wrong'),pixel_sha256='c'*64,
            metric_identity='e'*64,score=.2,file=descriptor('score'))]
        with self.assertRaises(AssertionError):build_plan(**v)
    def test_unaccepted_metric_identity_rejected(self):
        v=fixture();v['reuse']=[dict(source_id='source0',archive=v['sources'][0]['archive'],pixel_sha256='c'*64,
            metric_identity='f'*64,score=.2,file=descriptor('score'))]
        with self.assertRaises(AssertionError):build_plan(**v)
    def test_exact_reuse_only_changes_new_count(self):
        v=fixture();v['reuse']=[dict(source_id='source0',archive=v['sources'][0]['archive'],pixel_sha256='c'*64,
            metric_identity='e'*64,score=-.2,file=descriptor('score'))]
        p=build_plan(**v);self.assertEqual(p['counts']['logical_frames'],36)
        self.assertEqual(p['counts']['planned_old_pair_reuses'],1)
        self.assertEqual(p['counts']['planned_new_DINO_upper_bound'],3)
    def test_source_rejection_retains_all_frames(self):
        v=fixture();v['outputs'][0]['status']='FIXED_GRAY_SOURCE_REJECT';v['outputs'][0]['pixel_sha256']='d'*64
        p=build_plan(**v);self.assertEqual(p['counts']['source_pair_rows'],3)
        self.assertEqual(sum(len(x['frames'])for x in p['sources'][0]['pairs']),18)
        self.assertEqual(sum(f['source_RX_status']=='FIXED_GRAY_SOURCE_REJECT'for x in p['sources'][0]['pairs']for f in x['frames']),15)
    def test_false_source_rejection_gray_rejected(self):
        v=fixture();v['outputs'][0]['status']='FIXED_GRAY_SOURCE_REJECT'
        with self.assertRaises(AssertionError):build_plan(**v)
    def test_truth_injection_rejected(self):
        v=fixture();v['outputs'][0]['source_truth_used']=True
        with self.assertRaises(AssertionError):build_plan(**v)
    def test_unknown_status_not_imputed(self):
        v=fixture();v['outputs'][0]['status']='MISSING'
        with self.assertRaises(AssertionError):build_plan(**v)
    def test_negative_metric_not_filtered(self):
        v=fixture();v['reuse']=[dict(source_id='source0',archive=v['sources'][0]['archive'],pixel_sha256='d'*64,
            metric_identity='e'*64,score=-.7,file=descriptor('negative_score'))]
        p=build_plan(**v);self.assertEqual(p['counts']['planned_old_pair_reuses'],1)
    def test_missing_noise_rejected(self):
        v=fixture();v['logical'].pop()
        with self.assertRaises(AssertionError):build_plan(**v)


if __name__=='__main__':unittest.main()
