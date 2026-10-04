"""CPU checks for revision admission, honest provenance and safe orchestration."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import fixed80k_delivery_common as c
import fixed80k_delivery as d
import fixed80k_report_adapter as r

class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.addCleanup(self.temp.cleanup)
        self.out=self.root/'evaluation';self.out.mkdir();self.eval=self.root/'eval.json'
        self.config=dict(root=str(self.root),external_eval_config=str(self.eval))
        c.write(self.eval,dict(output=str(self.out),hifi_qualification_path=str(self.root/'hifi.json'),training_output=str(self.root/'view')))
    def receipt(self,**changes):
        value=dict(status='EXTERNAL_RECONSTRUCTIONS_COMPLETE',sources=100,physical_frames=1800,rows=3600,
            synthetic=False,sampler_step_limit=None,bindings={},outputs={})
        value.update(changes);c.write(self.out/'reconstruction_completion.json',value);return value
    def test_actual_reconstruction_receipt(self):
        self.receipt();self.assertIn(str(self.out/'reconstruction_completion.json'),c.stage_proof(self.config,'reconstruct'))
    def test_reject_wrong_completion_spelling(self):
        self.receipt(status='EXTERNAL_RECONSTRUCTION_COMPLETE')
        with self.assertRaises(RuntimeError):c.stage_proof(self.config,'reconstruct')
    def test_reject_partial_coverage(self):
        for field,value in [('rows',3599),('sources',99),('synthetic',True),('sampler_step_limit',2)]:
            self.receipt(**{field:value})
            with self.assertRaises(RuntimeError):c.stage_proof(self.config,'reconstruct')
    def test_missing_receipt_is_waiting_only_when_requested(self):
        self.assertIsNone(c.stage_proof(self.config,'reconstruct',True))
        with self.assertRaises(RuntimeError):c.stage_proof(self.config,'reconstruct')
    def test_bound_output_changed(self):
        path=self.root/'image';path.write_bytes(b'actual');self.receipt(outputs={str(path):c.sha(path)})
        path.write_bytes(b'altered')
        with self.assertRaises(RuntimeError):c.stage_proof(self.config,'reconstruct')
    def test_immutable_record(self):
        path=self.root/'record';c.seal(path,dict(a=1));c.seal(path,dict(a=1))
        with self.assertRaises(RuntimeError):c.seal(path,dict(a=2))
    def test_exact_milestone_required(self):
        p=c.locations(self.root)
        value=dict(version=c.VERSION,root=str(self.root),output=str(p['delivery']),requested_step=80000,
            training_resume_allowed=False,full_plan_scope='original_five_method_comparison_with_user_fixed_80k_external_checkpoint',
            poll_seconds=15,bindings={})
        self.assertEqual(c.validate_config(value),value)
        for key,change in [('requested_step',81551),('training_resume_allowed',True)]:
            altered=dict(value,**{key:change})
            with self.assertRaises(RuntimeError):c.validate_config(altered)
    def test_runtime_choices(self):
        self.assertEqual(d.stage_kind('reconstruct'),('swin',True));self.assertEqual(d.stage_kind('cost_external'),('swin',True))
        self.assertEqual(d.stage_kind('external_score'),('metric',True));self.assertEqual(d.stage_kind('publish'),('report',False))
    def test_no_training_stage(self):
        self.assertNotIn('training',c.STAGES);self.assertEqual(c.STAGES[-1],'publish')
    def test_safe_pause_requires_same_process_and_token(self):
        status=self.out/'reconstruct_status.json';launch=dict(pid=222,launch_id='real',started_ns=0)
        for pid,token,expected in [(221,'real',False),(222,'wrong',False),(222,'real',True)]:
            c.write(status,dict(pid=pid,launch_id=token,safe_pause_handler_installed=True))
            self.assertEqual(d.safe_to_pause(self.config,'reconstruct',launch),expected)
    def test_publisher_never_signalled(self):
        self.assertFalse(d.safe_to_pause(self.config,'publish',dict(pid=222,launch_id='real',started_ns=0)))
    def test_manual_worker_exact_admission(self):
        p=c.locations(self.root);p['revision'].mkdir(parents=True);p['delivery'].mkdir()
        adapter=p['revision']/'fixed80k_adapter.py';adapter.write_text('frozen adapter',encoding='utf-8')
        config=dict(self.config,output=str(p['delivery']),swin_python='native-python',swin_environment={'CUDA_VISIBLE_DEVICES':'0'},bindings={})
        c.write(p['delivery']/'config.json',config)
        record=dict(pid=222,start_ticks='123',command=['native-python','-B','-u',str(adapter),'--root',str(self.root),
            '--stage','reconstruct','--launch-id','actual'],environment={'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1'},
            launch_id='actual',stage='reconstruct',started_unix=1.5,adapter_sha256=c.sha(adapter),
            evaluation_config_sha256=c.sha(self.eval),selected_checkpoint_step=80000)
        path=p['revision']/'reconstruct_launch.json';c.write(path,record);config['bindings'][str(path)]=c.sha(path)
        with patch.object(c,'process',return_value=None):
            result=d.manual_reconstruction(config);self.assertEqual(result['started_ns'],1500000000)
            record['selected_checkpoint_step']=81551;c.write(path,record);config['bindings'][str(path)]=c.sha(path)
            with self.assertRaises(RuntimeError):d.manual_reconstruction(config)
    def test_report_states_actual_and_evaluated_steps(self):
        for phrase in ('80,000','81,551','budget_truncated=True','user_requested_pause=True','unfinished','no convergence'):
            self.assertIn(phrase,r.DISCLOSURE)
    def test_historical_admission_unchanged(self):
        old=c.HERE.parent/'runtime/own_controls_report.py'
        if not old.exists():old=c.HERE/'frozen_dependencies/own_controls_report.py'
        self.assertTrue(old.exists(),'Frozen report source required; no skip')
        source=old.read_text(encoding='utf-8');new=Path(r.__file__).read_text(encoding='utf-8')
        begin=source.index('def protocol_metadata(');end=source.index("    folder=root/'outputs/EXTERNAL-COMPARISON-20261004/swin_training'",begin)
        self.assertIn(source[begin:end],new)
        resources=source[source.index('    resources=',end):source.index('\ndef policy_title(',end)]
        self.assertIn(resources,new)
    def test_revision_sources_parse(self):
        for file in c.HERE.glob('fixed80k_*.py'):ast.parse(file.read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main(verbosity=2)
