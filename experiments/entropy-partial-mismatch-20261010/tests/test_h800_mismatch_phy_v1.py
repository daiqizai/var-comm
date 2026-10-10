"""Metadata/fake-ledger checks only; no real packet, array, model or metric calls."""
from pathlib import Path
import json
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import h800_mismatch_phy_v1 as h
core=h.core


class MismatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).parent/'fixtures/common500_v1'
        cls.sources,cls.schedule=h.plan.validate_inputs(json.loads((root/'plan.json').read_bytes()),json.loads((root/'manifest.json').read_bytes()))

    def row(self,family='WHOLE'):
        source=self.sources[0]
        return dict(h.plan.frame(source,4,7,family,self.schedule),channel=h.plan.channel_identity(source,4,6201))

    def observation(self,row=None,**changes):
        value=core.observed_identity(row or self.row(),dict(waveform_sha256='w'*64),'p'*64,'n'*64,'r'*64,dict(runtime='frozen'))
        value.update(changes);return value

    def test_frozen_grid_and_downstream_bounds(self):
        rows=list(h.plan.logical_frames(self.sources,self.schedule));caps=h.inputs.future_caps()
        self.assertEqual(len(rows),5400);self.assertEqual(len({h.plan.physical_key(r) for r in rows}),4500)
        self.assertEqual(caps['actual_new_packet_decodes'],9000)
        self.assertEqual(caps['visual']['var_render'],4500);self.assertEqual(caps['visual']['prior_scale'],45000)
        self.assertEqual(caps['metrics']['image_scores'],4500)
        self.assertEqual(caps['qualification_calls'],0)
        self.assertEqual(caps['historical_PHY_reuse_admitted'],0)

    def test_original_counter_uses_true_SNR_and_stride500(self):
        row=self.row();self.assertEqual(row['channel']['public_counter'],1500)
        self.assertEqual(row['channel']['receiver_snr_db'],4);self.assertEqual(row['config_snr_db'],7)
        row['channel']['public_counter']=300
        with self.assertRaisesRegex(ValueError,'Original channel identity'):self.observation(row)

    def test_same_lookup7_families_only_share_after_actual_bytes_match(self):
        first=self.observation(self.row('WHOLE'));second=self.observation(self.row('PARTIAL'))
        self.assertEqual(first,second)
        for key in ('waveform_sha256','noise_sha256','received_sha256','payload_sha256'):
            altered=dict(first);altered[key]='changed'
            self.assertNotEqual(core.digest(altered),core.digest(second))

    def test_environment_identity_cannot_share_actual_cache(self):
        self.assertNotEqual(core.digest(self.observation()),core.digest(self.observation(runtime_identity={'runtime':'other'})))

    def actual(self,header=True,crc=False):
        return dict(header_ok=header,body_decode_complete=header,gray=not header,source_decode_complete=True,
            rule='KEEP',rx_profile_id=243 if header else None,source_status='KEEP_ACTUAL_HARD_TOKENS' if header else 'HEADER_REJECT_GRAY',
            body_crc_accept=crc,logical_packet_calls=2 if header else 1)

    def test_failed_body_keeps_tokens_and_received_profile_without_TX_filter(self):
        row=core.normalized(self.actual());self.assertEqual(row['rx_profile_id'],243)
        self.assertFalse(row['body_crc_accept']);self.assertFalse(row['gray']);self.assertTrue(row['body_attempted'])

    def test_header_failure_does_not_count_a_body_failure_attempt(self):
        row=core.normalized(self.actual(False));self.assertFalse(row['body_attempted']);self.assertTrue(row['gray'])
        wrong=self.actual(False);wrong['logical_packet_calls']=2
        with self.assertRaisesRegex(ValueError,'Header reject'):core.normalized(wrong)

    def test_unresolved_event_is_not_a_channel_failure(self):
        event=dict(event_id='one',kind='header',phy_key='h',phase='raw_holdout');req={};result={}
        entry=dict(event_id='one',kind='header',phy_key='h',request_sha256=core.digest(req),result_sha256=core.digest(result))
        ledger=mock.Mock();ledger.db.execute.return_value.fetchone.return_value=(json.dumps(event),'{}','{}','RESERVED')
        with self.assertRaisesRegex(ValueError,'Missing closed paid'):core.verify_paid(dict(packet_event_ids=['one'],packet_events=[entry]),ledger)

    def test_saved_pin_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.json';pin=core.save(p,{'a':1});p.write_text('{"a":2}')
            with self.assertRaisesRegex(ValueError,'Changed bound file'):core.readpin(pin)

    def test_private_owner_identity_does_not_modify_original_globals(self):
        before=h.base.engine().worker_identity.__globals__['__file__'];engine=h.engine()
        self.assertEqual(engine.worker_identity.__globals__['__file__'],h.__file__)
        self.assertEqual(h.base.engine().worker_identity.__globals__['__file__'],before)

    def test_exact_original_code_pin_cannot_be_changed(self):
        with mock.patch.dict(h.SOURCE_PINS,{'mismatch_plan.py':'0'*64}),self.assertRaisesRegex(RuntimeError,'Changed frozen'):
            h.engine()

    def test_no_GPU_metrics_historical_reuse_or_automatic_successor_admitted(self):
        request=dict(schema=h.SCHEMA,status='REGISTERED_NOT_EXECUTED',execution=h.base.execution(time.time()+600,600),
            complete_scientific_caps=h.inputs.future_caps(),records=[],frames=[],raw_runtime_binding={},input_closure={},CPU_identity={},
            CPU_python=str(h.gate.CPU_PYTHON),root=str(h.gate.R2),GPU_execution_admitted=False,metrics_execution_admitted=False,
            automatic_successor=False,historical_reuse_admitted=0,tool_bindings={})
        for key in ('GPU_execution_admitted','metrics_execution_admitted','automatic_successor','historical_reuse_admitted'):
            value=dict(request);value[key]=True
            with mock.patch.object(h.g,'inside',side_effect=Path),mock.patch.object(h.g,'checked_json',return_value=value),\
                mock.patch.object(h,'original_inputs',return_value=([],[],{},{})),mock.patch.object(h,'cpu_identity',return_value=({},{})),\
                self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'Only new finite CPU'):
                h.registration('request','a'*64)

    def test_existing_run_cannot_launch_twice(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'request.json';p.write_text('{}');(p.parent/'run').mkdir()
            class Lock:
                def __enter__(self):return self
                def __exit__(self,*args):pass
            helper=mock.Mock();helper.owner_lock.return_value=Lock()
            a=types.SimpleNamespace(request=str(p),request_sha256='a'*64)
            with mock.patch.object(h.g.sys,'platform','linux'),mock.patch.object(h.g,'inside',side_effect=Path),\
                mock.patch.object(h,'registration',return_value=(dict(execution={}),{})),mock.patch.object(h.g,'helper',return_value=helper),\
                mock.patch.object(h.subprocess,'Popen') as launch,self.assertRaisesRegex(RuntimeError,'already claimed'):
                h.run(a)
            launch.assert_not_called()

    def test_independent_atomic_cap_blocks_call_above_limit(self):
        path=Path(__file__).parents[2]/'wcl-evidence-closure-20261009/scripts'
        sys.path.insert(0,str(path));from t2_ledger import Ledger
        with tempfile.TemporaryDirectory() as td:
            ledger=Ledger(Path(td)/'ledger.sqlite','a'*64,1);ledger.call(dict(event_id='one'),{},lambda:{'ok':True})
            callback=mock.Mock()
            with self.assertRaisesRegex(RuntimeError,'cap exhausted'):ledger.call(dict(event_id='two'),{},callback)
            callback.assert_not_called();self.assertEqual(ledger.snapshot(),dict(total=1,unresolved=0,cap=1));ledger.close()


if __name__=='__main__':unittest.main()
