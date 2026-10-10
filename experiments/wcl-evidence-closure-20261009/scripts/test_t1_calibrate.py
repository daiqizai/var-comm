"""Synthetic receiver, provenance, budget and interruption contracts; no models."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
import t1_calibrate as c

def packet(family='EC_VAR_WHOLE',m=7):
    return dict(TX_profile_id=0,TX_family='EC_STATIC_WHOLE',TX_m=9,actual_RX=dict(status='PAYLOAD_PARSED',
        header=dict(header_ok=True,profile_id=9),rx_profile=dict(profile_id=9,family=family,m=m),
        body=dict(crc_accepted=True,parser_accepted=True,payload=[0,1,0,1])))

def source():
    return dict(schema='T1_SOURCE_ASSET_V1',streams={f:{'7':dict(bits_key='bits',received_tokens_key='rx')}for f in c.FAMILIES})

MODELS={'EC_STATIC_WHOLE':'s','EC_VAR_WHOLE':'v'}

class CalibrationContracts(unittest.TestCase):
    def test_actual_source_gate_status_and_exact_pairs(self):
        g=dict(status='T1_SOURCE32_REAL_RECOVERY_GATE_COMPLETE',source_count=32,whole_prefixes=[7,8,9],
            static_raw_image_pairs=96,exact_equal_image_pairs=96,historical_VAR_prefix_roundtrips_reused=96,holdout_used=False)
        c.check_source_gate(g);g['status']='T1_SOURCE32_GPU_GATE_PASS'
        with self.assertRaisesRegex(ValueError,'source32'):c.check_source_gate(g)

    def test_wrong_accepted_header_drives_actual_family_and_scale(self):
        a=dict(bits=np.array([0,1,0,1],np.uint8),rx=np.ones(c.OFFSETS[7],np.int64),tokens=np.full(680,4000,np.int64))
        codec=Mock();codec.decode.return_value=dict(received_tokens=np.zeros(c.OFFSETS[7],np.int64))
        with patch.object(c,'exact_received_cache',return_value=None)as cache:
            state,evidence=c.recover(source(),a,packet(),codec,MODELS)
        codec.decode.assert_called_once_with('EC_VAR_WHOLE',[0,1,0,1],7)
        self.assertEqual(cache.call_args.kwargs['family'],'EC_VAR_WHOLE');self.assertEqual(cache.call_args.kwargs['m'],7)
        self.assertEqual(set(cache.call_args.args[1]),{'bits','rx'})
        self.assertEqual(state['m'],7);self.assertTrue(all(x==0 for scale in state['prefix']for x in scale))
        self.assertEqual(evidence['new_VAR_source_decode'],1)

    def test_received_bit_cache_hit_never_decodes(self):
        codec=Mock();a=dict(bits=np.ones(4,np.uint8),rx=np.ones(c.OFFSETS[7],np.int64),tokens=np.ones(680,np.int64))
        with patch.object(c,'exact_received_cache',return_value=dict(received_tokens=a['rx'])):
            state,e=c.recover(source(),a,packet(),codec,MODELS)
        codec.decode.assert_not_called();self.assertEqual(e['new_VAR_source_decode'],0);self.assertEqual(state['m'],7)

    def test_header_crc_and_framing_failures_cannot_source_decode(self):
        for failure in('header','crc','parser'):
            p=packet();codec=Mock()
            if failure=='header':p['actual_RX']['header']['header_ok']=False
            elif failure=='crc':p['actual_RX']['body']['crc_accepted']=False
            else:p['actual_RX']['body']['parser_accepted']=False
            state,_=c.recover(source(),{},p,codec,MODELS)
            self.assertEqual(state['kind'],'gray');codec.decode.assert_not_called()

    def test_invalid_source_stream_is_gray_but_software_failure_is_not(self):
        codec=Mock();a=dict(bits=np.zeros(4,np.uint8),rx=np.ones(c.OFFSETS[7],np.int64))
        with patch.object(c,'exact_received_cache',return_value=None):
            codec.decode.side_effect=c.InvalidSourceStream('bad terminal')
            state,e=c.recover(source(),a,packet(),codec,MODELS);self.assertEqual(state['kind'],'gray');self.assertEqual(e['source_status'],'SOURCE_PARSE_REJECT')
            codec.decode.side_effect=RuntimeError('model failure')
            with self.assertRaisesRegex(RuntimeError,'model failure'):c.recover(source(),a,packet(),codec,MODELS)

    def test_completed_actual_decode_is_not_repeated_after_resume(self):
        with tempfile.TemporaryDirectory()as tmp:
            codec=Mock();codec.decode.return_value=dict(received_tokens=np.zeros(c.OFFSETS[7],np.int64))
            a=dict(bits=np.zeros(4,np.uint8),rx=np.zeros(c.OFFSETS[7],np.int64))
            with patch.object(c,'exact_received_cache',return_value=None):
                first=c.recover_once(tmp,'r',source(),a,packet(),codec,MODELS)
                second=c.recover_once(tmp,'r',source(),a,packet(),codec,MODELS)
            self.assertEqual(codec.decode.call_count,1);self.assertEqual(first,second)

    def test_interrupted_actual_decode_never_retries_automatically(self):
        with tempfile.TemporaryDirectory()as tmp:
            codec=Mock();codec.decode.side_effect=RuntimeError('interrupt')
            a=dict(bits=np.zeros(4,np.uint8),rx=np.zeros(c.OFFSETS[7],np.int64))
            with patch.object(c,'exact_received_cache',return_value=None):
                with self.assertRaisesRegex(RuntimeError,'interrupt'):c.recover_once(tmp,'r',source(),a,packet(),codec,MODELS)
                with self.assertRaisesRegex(ValueError,'no automatic repeat'):c.recover_once(tmp,'r',source(),a,packet(),codec,MODELS)
            self.assertEqual(codec.decode.call_count,1)

    def test_recovery_identity_uses_received_data_not_transmitter_choice(self):
        p=packet();a=c.recovery_input(p,MODELS);p.update(TX_profile_id=77,TX_family='EC_VAR_WHOLE',TX_m=4)
        self.assertEqual(a,c.recovery_input(p,MODELS))
        p['actual_RX']['rx_profile']['m']=8;self.assertNotEqual(a,c.recovery_input(p,MODELS))

    def test_exact_parent_decode_can_be_reused_without_another_VAR_call(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=packet();identity=c.recovery_input(p,MODELS);old=Path(tmp)/'old.json'
            state=c.receiver_state(np.zeros(c.OFFSETS[7],np.int64),7)
            c.save(old,dict(request_sha256='pilot',recovery_input=identity,state=state,evidence=dict(source_status='ARITHMETIC_SOURCE_DECODED',new_VAR_source_decode=1)))
            codec=Mock();a,b,proof=c.recover_once(Path(tmp)/'full','full',source(),{},p,codec,MODELS,{c.digest(identity):c.pin(old)})
            codec.decode.assert_not_called();self.assertEqual(a,state);self.assertEqual(b['new_VAR_source_decode'],0);self.assertTrue(b['source_decode_checkpoint_reused'])

    def test_frame_budget_rejects_expansion_and_duplicate_candidate(self):
        base=c.candidates();ids=[x['candidate_id']for x in base]
        r=dict(schema=c.SCHEMA,phase='pilot',snrs=c.SNRS,source_count=100,seeds=[4101],families=c.FAMILIES,candidates=base,
            workers=8,records=[dict(source_index=i,source_id=str(i))for i in range(100)],grid={f:{str(s):ids.copy()for s in c.SNRS}for f in c.FAMILIES},
            frame_count=21600,packet_cap=43200)
        c.validate_request(r);r['packet_cap']+=1
        with self.assertRaisesRegex(ValueError,'budget'):c.validate_request(r)
        r['packet_cap']-=1;r['grid'][c.FAMILIES[0]]['4'][1]=ids[0]
        with self.assertRaisesRegex(ValueError,'unique'):c.validate_request(r)

    def test_native_signal_handler_flag_is_honored(self):
        import time
        with patch.object(c.shared,'STOP',True):
            with self.assertRaisesRegex(ValueError,'Stopped'):c.guard(dict(deadline_unix=time.time()+60,max_seconds=60,stop_files=[]),time.monotonic())

    def test_close_cannot_recreate_a_missing_packet_ledger(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'missing.sqlite'
            with self.assertRaisesRegex(ValueError,'never recreate'):c.ledger_snapshot(p,'a'*64,2)
            self.assertFalse(p.exists())

    def test_readonly_close_checks_request_and_completed_packet_count(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'meter.sqlite';meter=c.Ledger(p,'a'*64,2)
            meter.call(dict(event_id='one'),dict(rx='actual'),lambda:dict(header_ok=True));meter.close()
            self.assertEqual(c.ledger_snapshot(p,'a'*64,2),dict(total=1,unresolved=0,cap=2))
            with self.assertRaisesRegex(ValueError,'request/cap'):c.ledger_snapshot(p,'b'*64,2)

if __name__=='__main__':unittest.main()
