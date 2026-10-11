"""CPU rejection tests for the successor's scientific admission gate."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import chain

class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patches=[patch.object(chain,'RESULT',self.root/'results'),patch.object(chain,'TMP',self.root/'tmp')]
        for p in self.patches:p.start()
        for branch,name,job,step in [('P','P_continue','P_train_gpu1_v2',100000),('Swin','Swin_1to19','Swin_train_v1',200000)]:
            d=chain.RESULT/name;d.mkdir(parents=True)
            (d/'model.pt').write_bytes(b'model');(d/'cal.csv').write_bytes(b'cal')
            chain.write(d/'registration.json',{'identity':branch})
            cp=chain.pin(d/'model.pt')
            selected=dict(step=step,completed_step=step,objective=1.,checkpoint=cp,csv=str(d/'cal.csv'),sha256=chain.sha(d/'cal.csv'),holdout_read=False,registration_sha256=chain.sha(d/'registration.json'))
            chain.write(d/'selected_checkpoint.json',selected)
            chain.write(d/'calibration_history.json',[dict(step=step-1000,objective=2.),dict(step=step,objective=1.)])
            chain.write(d/'completion.json',dict(status='TRAINING_AND_CALIBRATION_COMPLETE'))
            chain.write(d/'stopping_checks.json',[])
            chain.write(chain.TMP/'jobs'/job/'exit.json',dict(exit_code=0,actual_child_waited=True))
    def tearDown(self):
        for p in self.patches:p.stop()
        self.tmp.cleanup()
    def change(self,name,**values):
        p=chain.RESULT/'P_continue'/name;d=chain.read(p);d.update(values);chain.write(p,d)
    def test_completed_gate(self):self.assertEqual(chain.training_gate()[0]['P']['step'],100000)
    def test_unfinished(self):
        self.change('completion.json',status='TRAINING')
        with self.assertRaises(AssertionError):chain.training_gate()
    def test_holdout_selected(self):
        self.change('selected_checkpoint.json',holdout_read=True)
        with self.assertRaises(AssertionError):chain.training_gate()
    def test_wrong_winner(self):
        self.change('selected_checkpoint.json',step=99000)
        with self.assertRaises(AssertionError):chain.training_gate()
    def test_changed_weights(self):
        (chain.RESULT/'P_continue/model.pt').write_bytes(b'changed')
        with self.assertRaises(AssertionError):chain.training_gate()
    def test_unwaited_child(self):
        p=chain.TMP/'jobs/P_train_gpu1_v2/exit.json';chain.write(p,dict(exit_code=0,actual_child_waited=False))
        with self.assertRaises(AssertionError):chain.training_gate()
    def test_failed_child(self):
        p=chain.TMP/'jobs/P_train_gpu1_v2/exit.json';chain.write(p,dict(exit_code=1,actual_child_waited=True))
        with self.assertRaises(AssertionError):chain.training_gate()
    def test_tie_earliest(self):
        p=chain.RESULT/'P_continue/calibration_history.json';chain.write(p,[dict(step=99000,objective=1.),dict(step=100000,objective=1.)])
        with self.assertRaises(AssertionError):chain.training_gate()
if __name__=='__main__':unittest.main()
