"""Synthetic only: identity isolation, complete denominators and frozen selection."""
from pathlib import Path
import copy,unittest
import numpy as np
from contracts import *


def fixture():
    population=[dict(source_id='synthetic_'+str(i),source_index=i)for i in range(3)]
    rows=[]
    for s in population:
        for snr,candidates in CHOICES.items():
            for cid in candidates:
                for noise in (4101,4102,4103):
                    rows.append(dict(s,snr_db=snr,candidate_id=cid,noise_seed=noise,**{METRIC:0.2+s['source_index']*0.1+(noise-4102)*0.01}))
    return population,rows


def metadata():
    d=dict(metric_selection_usage='old',metrics={'dinov2_vitl14':dict(weights='fixed',preprocessing='fixed')},numeric='FP32')
    rt={k:'fixed_'+k for k in RUNTIME_KEYS};return dict(DINO_L_metadata=d,**rt),copy.deepcopy(d),rt


class Contracts(unittest.TestCase):
    def closed_fixture(self):
        w=dict(actual_wait=True,returncode=0,process=dict(pid=1,start_ticks=10),parent_process=dict(pid=2,start_ticks=20))
        a=dict(status='PASS',violations=[],process=dict(pid=3,start_ticks=30),parent_process=dict(pid=4,start_ticks=40))
        ids=dict(worker=w['process'],owner=w['parent_process'],auditor=a['process'],audit_parent=a['parent_process'])
        o=dict(host=PRO_HOST,processes={k:dict(expected=v,same_generation_live=False)for k,v in ids.items()})
        return w,a,o
    def test_closed_receiver_correct_host(self):
        self.assertTrue(check_rx_closed(*self.closed_fixture()))
    def test_closed_receiver_wrong_host_rejected(self):
        w,a,o=self.closed_fixture();o['host']='MAIN'
        with self.assertRaises(AssertionError):check_rx_closed(w,a,o)
    def test_live_receiver_rejected(self):
        w,a,o=self.closed_fixture();o['processes']['worker']['same_generation_live']=True
        with self.assertRaises(AssertionError):check_rx_closed(w,a,o)
    def test_stale_receiver_identity_rejected(self):
        w,a,o=self.closed_fixture();o['processes']['worker']['expected']=dict(pid=1,start_ticks=11)
        with self.assertRaises(AssertionError):check_rx_closed(w,a,o)
    def test_native_image_preserves_negative_zero(self):
        x=np.zeros((3,256,256),np.float32);x[0,0,0]=-0.0
        self.assertIs(image(x),x);self.assertTrue(np.signbit(x[0,0,0]))
    def test_invalid_image_is_not_clipped(self):
        for val in (-.01,1.01,float('nan')):
            with self.assertRaises(AssertionError):image(np.full((3,256,256),val,np.float32))
    def test_attribution_only_difference_allowed(self):
        old,new,rt=metadata();new['metric_selection_usage']='new';self.assertTrue(check_identity(old,new,rt))
    def test_weights_change_rejected(self):
        old,new,rt=metadata();new['metrics']['dinov2_vitl14']['weights']='changed'
        with self.assertRaises(AssertionError):check_identity(old,new,rt)
    def test_transform_change_rejected(self):
        old,new,rt=metadata();new['metrics']['dinov2_vitl14']['preprocessing']='changed'
        with self.assertRaises(AssertionError):check_identity(old,new,rt)
    def test_runtime_change_rejected(self):
        old,new,rt=metadata();rt=copy.deepcopy(rt);rt['device']='other'
        with self.assertRaises(AssertionError):check_identity(old,new,rt)
    def test_other_metric_cannot_select(self):
        old,new,rt=metadata();old['DINO_L_metadata']['metrics']['clip']={};new['metrics']['clip']={}
        with self.assertRaises(AssertionError):check_identity(old,new,rt)
    def test_complete_grid(self):
        pop,rows=fixture();self.assertEqual(len(validate_grid(rows,pop,3)),3)
    def test_duplicate_frame_rejected(self):
        pop,rows=fixture();rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaises(AssertionError):validate_grid(rows,pop,3)
    def test_missing_frame_rejected(self):
        pop,rows=fixture()
        with self.assertRaises(AssertionError):validate_grid(rows[:-1],pop,3)
    def test_source_index_change_rejected(self):
        pop,rows=fixture();rows[0]['source_index']=2
        with self.assertRaises(AssertionError):validate_grid(rows,pop,3)
    def test_new_noise_rejected(self):
        pop,rows=fixture();rows[0]['noise_seed']=6201
        with self.assertRaises(AssertionError):validate_grid(rows,pop,3)
    def test_nonfinite_metric_rejected(self):
        pop,rows=fixture();rows[0][METRIC]=float('inf')
        with self.assertRaises(AssertionError):validate_grid(rows,pop,3)
    def test_original_aggregate_source_average_and_exact_tie(self):
        pop,rows=fixture();raw=Path(__file__).with_name('original_freeze.py').read_bytes()
        means,summary,policies=freeze_original(raw,rows,pop,3)
        self.assertEqual(len(means),18);self.assertEqual(len(summary),6)
        for row in means:self.assertAlmostEqual(row[METRIC],.2+int(row['source_id'][-1])*.1)
        for row in summary:self.assertAlmostEqual(row[METRIC],.3)
        for snr in CHOICES:self.assertEqual(policies[str(snr)],min(CHOICES[snr]))
        rows[0][METRIC]+=.09
        _,_,policies=freeze_original(raw,rows,pop,3)
        self.assertEqual(policies['4'],rows[0]['candidate_id'])
    def test_changed_original_function_rejected(self):
        pop,rows=fixture();raw=Path(__file__).with_name('original_freeze.py').read_bytes()+b'\n'
        with self.assertRaises(AssertionError):freeze_original(raw,rows,pop,3)


if __name__=='__main__':unittest.main()
