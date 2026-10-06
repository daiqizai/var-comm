import copy
import importlib.util
import math
import sys
import unittest
import tempfile
from unittest import mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import raw64_prescreen as p


def fixture():
    old = [dict(kind='header',phy_key='h',snr_db=s,n_blocks=20,n_correct=18,n_reject=2,n_undetected=0) for s in p.selection.SNRS]
    missing = [dict(phy_key='b',snr_db=s) for s in p.selection.SNRS]
    plan = dict(all_body_phy_keys=['b'],phase_caps={'proxy':1536},proxy=dict(reused_cells=[],reused_header_correct_cells=old,cells=missing))
    rows = [dict(point=dict(kind='body',**r),packet_calls=256,correct=128,rejected=120,undetected=8) for r in missing]
    return plan, dict(phase='proxy',packet_calls=1536,points=rows)


class Contracts(unittest.TestCase):
    def test_stable_identity_with_observation_fields(self):
        x=dict(pid=1310454,start_ticks=195576376,uid=1002,argv=['python','-B','owner'])
        self.assertTrue(p.same_process(x,dict(x,state='R',ppid=1310393)))
    def test_wrong_stable_identity_rejected(self):
        x=dict(pid=1,start_ticks=2,uid=1002,argv=['python','owner'])
        for key,value in [('pid',3),('start_ticks',4),('uid',0),('argv',['python','other'])]:
            y=dict(x);y[key]=value
            with self.subTest(key=key):self.assertFalse(p.same_process(x,y))
    def test_missing_stable_identity_rejected(self):
        x=dict(pid=1,start_ticks=2,uid=1002,argv=['python','owner'])
        for key in x:
            y=dict(x);del y[key]
            with self.subTest(key=key):self.assertFalse(p.same_process(y,y))
    def test_identity_comparison_does_not_modify_originals(self):
        x=dict(pid=1,start_ticks=2,uid=1002,argv=['python','owner']);y=dict(x,state='S',ppid=8)
        before=copy.deepcopy((x,y));self.assertTrue(p.same_process(x,y));self.assertEqual((x,y),before)
    def test_empirical_and_full_snr_grid(self):
        a,b=fixture();d=p.probability_map(a,b)
        self.assertEqual(len(d),12);self.assertEqual(d['header',13],.9);self.assertEqual(d['b',1],.5)
    def test_missing_fails(self):
        a,b=fixture();b['points'].pop()
        with self.assertRaises(ValueError):p.probability_map(a,b)
    def test_duplicate_fails(self):
        a,b=fixture();b['points'][-1]=b['points'][0]
        with self.assertRaises(ValueError):p.probability_map(a,b)
    def test_count_overlap_fails(self):
        a,b=fixture();b['points'][0]['undetected']=9
        with self.assertRaises(ValueError):p.probability_map(a,b)
    def test_reused_count_kept(self):
        a,b=fixture();a['proxy']['reused_header_correct_cells'][0].update(n_blocks=20000,n_correct=19997,n_reject=3)
        self.assertEqual(p.probability_map(a,b)['header',1],19997/20000)
    def test_no_interpolation(self):
        a,b=fixture();b['points'][0]['point']['snr_db']=2
        with self.assertRaises(ValueError):p.probability_map(a,b)
    def test_psnr_consistency(self):
        self.assertEqual(p.psnr(dict(mse=.01,psnr_db=20.)),20.)
        for row in [dict(mse=.01,psnr_db=21.),dict(mse=0,psnr_db=math.inf),dict(mse=.1,psnr_db=math.nan)]:
            with self.assertRaises(ValueError):p.psnr(row)
    def test_admission_exception_still_waits_and_seals_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            child=mock.Mock(pid=123);child.wait.return_value=0
            with mock.patch.object(p.subprocess,'Popen',return_value=child), mock.patch.object(p,'identity',side_effect=FileNotFoundError('leader exited')):
                with self.assertRaises(ValueError):p.waited_worker(tmp,['python','x.py'])
            child.wait.assert_called_once()
            ex=p.read(Path(tmp)/'worker_exit.json')
            self.assertTrue(ex['process_waited']);self.assertEqual(ex['exit_code'],0)
            self.assertIsNone(ex['identity']);self.assertIn('leader exited',ex['capture_error'])
            self.assertEqual(ex['log_sha256'],p.sha(Path(tmp)/'worker.log'))
    def test_normal_admission_retains_full_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            child=mock.Mock(pid=123);child.wait.return_value=0;argv=['python','x.py']
            ident=dict(pid=123,start_ticks=44,uid=1002,argv=argv)
            with mock.patch.object(p.subprocess,'Popen',return_value=child), mock.patch.object(p,'identity',return_value=ident):
                self.assertEqual(p.waited_worker(tmp,argv),ident)
            ex=p.read(Path(tmp)/'worker_exit.json');self.assertIsNone(ex['capture_error'])


if __name__=='__main__':unittest.main()
