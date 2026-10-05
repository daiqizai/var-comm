"""Public text and truthful completion-gate tests, without Git or science."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import prepare_initial_selection as p


class ClosedSelectionTests(unittest.TestCase):
    def fixture(self,root):
        op=root/'owner.json';rp=root/'reg.json';lp=root/'launch.json';cp=root/'config.json'
        entry=root/'stage_owner.py';entry.write_text('# fixture\n')
        execution=root/'execution';execution.mkdir();science=root/'science';science.mkdir()
        done=science/'completion.json';worker=dict(pid=11,uid=1002,start_ticks=111,argv=['python','select.py'])
        ident=dict(pid=10,uid=1002,start_ticks=110,argv=['python','-B',str(entry.resolve()),'--config',str(op)])
        owner=dict(registration=str(rp),out=str(root/'H'),owner_out=str(execution),stages=[dict(id='freeze',resource='cpu',jobs=[
            dict(id='initial_selection',argv=['python','select.py','--stage','run','--config',str(cp)],out=str(science),completion=str(done))])])
        p.save(op,owner);p.save(rp,dict(allowed_stage_ids=['freeze'],input_bindings={},source_bindings={}));p.save(cp,{})
        p.save(lp,dict(identity=ident,argv=ident['argv'],registration_sha256=p.sha(rp),owner_config_sha256=p.sha(op)))
        p.save(execution/'owner_identity.json',dict(ident,registration_sha256=p.sha(rp),config_sha256=p.sha(op)))
        p.save(done,dict(status='H_INITIAL_TRUE200_WHOLE_SELECTION_COMPLETE',registration_sha256=p.sha(rp),config_sha256=p.sha(cp),
            selected_count=8,source_count=200,measured_frames=9600,new_packet_decodes=0,new_visual_inference=0,GPU_used=False,
            original_render_owner_success=False,partial_reselected=False,development_used=False,holdout_used=False,
            full1000_calibration_complete=False,outputs={},source_bindings={},input_bindings={}))
        closed=dict(bindings={str(done):p.sha(done)},child_identities=[worker]);states={10:None,11:None}
        a=SimpleNamespace(__file__=str(entry),validate_config=lambda *args:None,
                          same_identity=lambda x,y:all(x[k]==y[k] for k in ('pid','uid','start_ticks','argv')))
        prior=SimpleNamespace(exited=lambda x,reader:reader(x['pid']) is None,verify_batch=lambda *args:copy.deepcopy(closed))
        return dict(a=a,prior=prior,owner_path=op,reg_path=rp,launch_path=lp,config_path=cp,state_reader=lambda pid:states[pid]),states,closed,execution,science

    def test_successful_and_exited_selector_required(self):
        with tempfile.TemporaryDirectory() as td:
            args,states,_,_,_=self.fixture(Path(td));result=p.verify_selection_closed(**args)
            self.assertEqual(result['done']['selected_count'],8)
            for pid in (10,11):
                states[pid]={'state':'Z'}
                with self.assertRaisesRegex(RuntimeError,'unreaped|fully exited'):p.verify_selection_closed(**args)
                states[pid]=None

    def test_worker_complete_cannot_override_failed_owner(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_,execution,_=self.fixture(Path(td));p.save(execution/'failure.json',{'error':'exit race'})
            with self.assertRaisesRegex(RuntimeError,'explicit future closeout'):p.verify_selection_closed(**args)

    def test_wrong_scientific_claim_and_changed_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,closed,_,science=self.fixture(Path(td));path=science/'completion.json'
            done=p.read(path);done['original_render_owner_success']=True;path.write_text(json.dumps(done));closed['bindings'][str(path)]=p.sha(path)
            with self.assertRaisesRegex(RuntimeError,'scientific receipt differs'):p.verify_selection_closed(**args)


class PublicTextTests(unittest.TestCase):
    def test_closed_archived_dynamic_json_is_structured_and_keeps_raw_sha(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'status.json';source.write_text('{"status":"FAILED","path":"/home/private/log"}\n')
            result=p.evidence_envelope(source,p.sha(source));self.assertEqual(result['original_raw_sha256'],p.sha(source))
            self.assertEqual(result['content']['status'],'FAILED');self.assertTrue(result['closed_evidence'])
            override=os.environ.get('H_PUBLICATION_TEST_SANITIZER')
            helper=Path(override) if override else Path(__file__).resolve().parents[1]/'delivery/publication/sanitize_registration.py'
            self.assertTrue(helper.is_file(),'Set H_PUBLICATION_TEST_SANITIZER to the verified sanitizer source when outside the local preparation tree')
            spec=importlib.util.spec_from_file_location('pub_sanitizer',helper);safe=importlib.util.module_from_spec(spec);spec.loader.exec_module(safe)
            sanitized=safe.Sanitizer('/repo').export(json.dumps(result).encode(),'.json').decode()
            self.assertNotIn('/home/private',sanitized);self.assertIn('EXTERNAL_ASSET/',sanitized)
            self.assertIn(p.sha(source),sanitized)

    def test_raw_payload_binary_and_changed_archive_refused(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'receipt.json';source.write_text('{"decoded_bits":[1,0]}')
            with self.assertRaisesRegex(RuntimeError,'Raw image'):p.evidence_envelope(source,p.sha(source))
            source.write_bytes(b'\0tensor')
            with self.assertRaisesRegex(RuntimeError,'binary'):p.evidence_envelope(source,p.sha(source))
            source.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'SHA changed'):p.evidence_envelope(source,'0'*64)

    def test_report_is_calibration_scoped_and_has_one_unique_marker(self):
        rows=[dict(snr_db=snr,arm=arm,target_m=7,q=4 if arm.startswith('H16') else 6,
                   nominal_rate='1/2',initial_true200_mean_psnr_db=21.125,
                   actual_m_histogram={'6':0,'7':200,'8':0,'9':0}) for snr in p.SNRS for arm in p.ARMS]
        text=p.report_section({'selected_candidates':rows},'bundle/README.md')
        self.assertEqual(text.count(p.MARKER),1);self.assertEqual(text.count('| 13 dB |'),4);self.assertEqual(text.count('| 19 dB |'),4)
        self.assertIn('不是development结果',text);self.assertIn('原owner仍没有成功完成凭证',text)
        self.assertIn('6个partial候选仅保留原预筛引用',text);self.assertNotIn('per_source_evidence',text)
        self.assertIn('实际源尺度（图数）',text);self.assertEqual(text.count('m7: 200'),8)
        self.assertFalse(any(line.endswith((' ','\t')) for line in text.splitlines()))

    def test_compact_selected_omits_full_per_frame_evidence_without_altering_source(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'selected_whole.json';result=dict(selected_candidates=[{'arm':'test'}],measured_frames=9600,
                per_source_evidence=[{'source':1,'noise_samples':[1,2,3]}]);p.save(source,result);before=p.sha(source)
            compact=p.compact_selected(result,source)
            self.assertNotIn('per_source_evidence',compact);self.assertEqual(compact['selected_candidates'],result['selected_candidates'])
            self.assertEqual(compact['public_export']['original_file_sha256'],before);self.assertEqual(p.sha(source),before)


if __name__=='__main__':unittest.main()
