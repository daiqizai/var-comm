"""Safety tests for the separate fixed-milestone admission; no model run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import fixed80k_adapter as a


class Admission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.p = a.paths(self.root)
        origin = self.p['origin']; origin.mkdir(parents=True)
        checkpoint = origin/'checkpoints/model_080000.pt'; checkpoint.parent.mkdir()
        checkpoint.write_bytes(b'engineering-test-checkpoint')
        self.patch = patch.object(a, 'CHECKPOINT_SHA', a.sha(checkpoint)); self.patch.start(); self.addCleanup(self.patch.stop)
        a.seal(origin/'registration.json', {'bindings':{}, 'config':{'seed':20261004}})
        a.seal(origin/'qualification.json', {'status':'PASS'})
        a.seal(origin/'status.json', {'status':'PAUSED','step':81551})
        a.seal(self.p['request'], {'text':'Use the exact 80k checkpoint','requested_checkpoint_step':80000,
            'training_must_not_resume_automatically':True,'scientific_convergence_claimed':False,
            'selection_rule':'exact_user_requested_checkpoint_not_calibration_best'})
        a.seal(self.p['revision']/'pause_verified.json', {'processes':{str(i):{'exists':False} for i in range(4)},
            'swin_training/status.json':{'status':'PAUSED','step':81551},'checkpoint_80000':{'sha256':a.CHECKPOINT_SHA}})
        config = {'version':'EXTERNAL-EVALUATION-20261004-R1','root':str(self.root),
            'training_output':str(origin),'output':str(self.p['base']/'evaluation'),
            'result':str(self.root/'results/external_comparison_20261004/evaluation'),
            'hifi_qualification_path':str(self.p['base']/'hifi_qualification/qualification.json'),
            'total_N':[1024,2048],'snrs':[1,7,13],'noise_seeds':[2001,2002,2003],
            'source_count':100,'sampling_seed':23,'schedule_mode':'actual_data_cbr','holdout_access':False}
        a.seal(self.p['runtime']/'external_eval_config.json',config)
        ids = ['source_'+str(i) for i in range(1000)]; cells = {}; bycell = {}
        for n in (1024,2048):
            for snr in (1,4,7,10,13):
                path = origin/'calibration/080000'/f'N{n}_snr{snr}_seed4101.json'
                rows = [dict(source_index=i,source_id=ids[i],mse=.125,header_accepted=True) for i in range(1000)]
                cell = dict(status='COMPLETE',synthetic=False,context=dict(step=80000,
                    registration_sha256=a.sha(origin/'registration.json'),checkpoint_sha256=a.CHECKPOINT_SHA,
                    seed=4101,N=n,snr=snr,source_ids=ids),rows=rows,mean_mse=.125,header_failures=0)
                cell['payload_sha256'] = a.identity(cell); a.seal(path,cell); cells[str(path)] = a.sha(path)
                bycell[f'N{n}_snr{snr}'] = .125
        self.cal = dict(step=80000,checkpoint=str(checkpoint),checkpoint_sha256=a.CHECKPOINT_SHA,
            source_count=1000,rows=10000,selection_uses_development=False,physical_metadata=True,
            cells=cells,mean_mse_by_N={'1024':.125,'2048':.125},mean_mse_by_cell=bycell,mean_mse=.125)
        a.seal(origin/'calibration/080000/completion.json',self.cal)

    def test_register_preserves_original_and_never_claims_training_complete(self):
        original = {str(f):a.sha(f) for f in self.p['origin'].rglob('*') if f.is_file()}
        done = a.register(self.root)
        self.assertEqual(done['status'],a.READY)
        self.assertFalse((self.p['origin']/'completion.json').exists())
        self.assertFalse(done['training_finished']); self.assertFalse(done['scientific_convergence_proven'])
        self.assertTrue(done['budget_truncated'])
        selected,_ = a.validate_selection(self.p['view'])
        self.assertEqual(selected['step'],80000); self.assertEqual(selected['actual_training_pause_step'],81551)
        self.assertEqual(selected['completed_step'],81551)
        a.verify(original)
        self.assertEqual(a.read(self.p['view']/'registration.json'),a.read(self.p['origin']/'registration.json'))
        a.register(self.root)

    def test_rejects_other_checkpoint_even_if_it_has_lower_mse(self):
        cal = copy.deepcopy(self.cal); cal['step'] = 72500; cal['mean_mse'] = .001
        with self.assertRaisesRegex(RuntimeError,'exact 80k'):
            a.validate_calibration(cal,self.p['origin'],a.sha(self.p['origin']/'registration.json'))

    def test_rejects_incomplete_calibration(self):
        cal = copy.deepcopy(self.cal); cal['cells'].pop(next(iter(cal['cells'])))
        with self.assertRaisesRegex(RuntimeError,'ten cells'):
            a.validate_calibration(cal,self.p['origin'],a.sha(self.p['origin']/'registration.json'))

    def test_rejects_running_origin(self):
        (self.p['origin']/'status.json').write_text(json.dumps({'status':'TRAINING','step':81551}))
        with self.assertRaisesRegex(RuntimeError,'safely paused'): a.register(self.root)

    def test_rejects_development_selected_checkpoint(self):
        a.register(self.root)
        path = self.p['view']/'selected_swin.json'; selected = a.read(path); selected['development_read'] = True
        path.write_text(json.dumps(selected))
        with self.assertRaisesRegex(RuntimeError,'Frozen binding changed'): a.validate_selection(self.p['view'])

    def test_rejects_checkpoint_bytes_change(self):
        a.register(self.root)
        (self.p['origin']/'checkpoints/model_080000.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'Frozen binding changed'): a.validate_selection(self.p['view'])

    def test_rejects_calibration_row_change(self):
        path = Path(next(iter(self.cal['cells']))); cell = a.read(path); cell['rows'][0]['mse'] = .0625
        path.write_text(json.dumps(cell))
        with self.assertRaisesRegex(RuntimeError,'Frozen binding changed'):
            a.validate_calibration(self.cal,self.p['origin'],a.sha(self.p['origin']/'registration.json'))

    def test_rejects_missing_process_exit_proof(self):
        (self.p['revision']/'pause_verified.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'process exit verification'): a.register(self.root)

    def test_rejects_request_to_select_best_instead_of_exact_step(self):
        path = self.p['request']; request = a.read(path); request['selection_rule'] = 'calibration_best'
        path.write_text(json.dumps(request))
        with self.assertRaisesRegex(RuntimeError,'exact 80k'): a.register(self.root)

    def test_rejects_live_original_pipeline(self):
        path = self.p['revision']/'pause_verified.json'; receipt = a.read(path); receipt['processes']['0']['exists'] = True
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(RuntimeError,'must have exited'): a.register(self.root)


if __name__ == '__main__': unittest.main(verbosity=2)
