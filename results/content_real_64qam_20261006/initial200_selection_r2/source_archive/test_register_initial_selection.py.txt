"""Closed-receipt and pure config tests; no physical call, GPU or model."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import register_initial_selection as r


class SelectionRegistrationTests(unittest.TestCase):
    def fixture(self,root):
        owner=root/'owner.json';reg=root/'reg.json';launch=root/'launch.json';render=root/'render.json'
        entry=root/'stage_owner.py';entry.write_text('# fixture\n')
        evidence=root/'owner';evidence.mkdir();out=root/'render';out.mkdir();done=out/'completion.json'
        cpu=root/'cpu.json';short=root/'short.json';r.save(cpu,{});r.save(short,{})
        cfg=dict(registration=str(reg),owner_out=str(evidence),out=str(root/'H'),stages=[dict(id='render',resource='gpu',jobs=[
            dict(argv=['python','-B','render.py','--config',str(render)],out=str(out),completion=str(done))])])
        r.save(owner,cfg);r.save(reg,dict(allowed_stage_ids=['render']))
        r.save(render,dict(out=str(out),registration=str(reg),render_owner_config=str(owner),cpu_completion=str(cpu),shortlist=str(short)))
        ident=dict(pid=100,uid=1002,start_ticks=1000,argv=['python','-B',str(entry.resolve()),'--config',str(owner)])
        child=dict(pid=101,uid=1002,start_ticks=1001,argv=['python','render.py'])
        r.save(launch,dict(identity=ident,argv=ident['argv'],registration_sha256=r.sha(reg),owner_config_sha256=r.sha(owner)))
        r.save(evidence/'owner_identity.json',dict(ident,registration_sha256=r.sha(reg),config_sha256=r.sha(owner)))
        completion=dict(status='H_INITIAL_TRUE200_RX_COMPLETE',registration_sha256=r.sha(reg),config_sha256=r.sha(render),
            source_count=200,frame_count=9600,images_scored=True,source_decode_complete=True,new_packet_decodes=0,
            budget_unchanged=True,policy_selection=False,development_used=False,holdout_used=False,
            outputs={},input_bindings={},source_bindings={},predecessor_closure_bindings={},visual_launch_bindings={},
            cpu_completion_sha256=r.sha(cpu),shortlist_sha256=r.sha(short))
        r.save(done,completion);closure=dict(bindings={str(done):r.sha(done)},child_identities=[child])
        states={100:None,101:None}
        a=SimpleNamespace(__file__=str(entry),validate_config=lambda *args:None)
        p=SimpleNamespace(exited=lambda ident,reader:reader(ident['pid']) is None,
                          verify_batch=lambda *args:copy.deepcopy(closure))
        args=dict(owner=a,prior=p,owner_path=owner,reg_path=reg,launch_path=launch,render_path=render,
                  state_reader=lambda pid:states[pid])
        return args,states,closure,done

    def test_exact_recovered_owner_child_and_scientific_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_,done=self.fixture(Path(td));value=r.verify_render_closed(**args)
            self.assertEqual(value['render_completion'],str(done));self.assertEqual(len(value['child_identities']),1)

    def test_owner_or_child_live_and_zombie_cannot_release_selection(self):
        with tempfile.TemporaryDirectory() as td:
            args,states,_,_=self.fixture(Path(td))
            for pid in (100,101):
                for state in ('R','Z'):
                    states[pid]={'state':state}
                    with self.assertRaisesRegex(RuntimeError,'live|unreaped'):r.verify_render_closed(**args)
                states[pid]=None

    def test_incomplete_scope_or_unbound_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,closure,done=self.fixture(Path(td));closure['bindings'][str(done)]='0'*64
            with self.assertRaisesRegex(RuntimeError,'unbound'):r.verify_render_closed(**args)
            value=r.read(done);value['frame_count']=9599;done.write_text(json.dumps(value));closure['bindings'][str(done)]=r.sha(done)
            with self.assertRaisesRegex(RuntimeError,'wrong-scope'):r.verify_render_closed(**args)

    def test_launch_substitution_and_stop_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_,_=self.fixture(Path(td));cfg=r.read(args['owner_path']);stop=Path(cfg['owner_out'])/'STOP';stop.write_text('stop')
            with self.assertRaisesRegex(RuntimeError,'STOP'):r.verify_render_closed(**args)
            stop.unlink();value=r.read(args['launch_path']);value['identity']['start_ticks']+=1
            value['argv']=['another'];args['launch_path'].write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,'launch identity'):r.verify_render_closed(**args)

    def test_original_budget_is_exact_and_reserve_not_free(self):
        budget=dict(created=True,charged=69960,phase_charged=dict(r.EXPECTED_CHARGES),unresolved=0,failed=0)
        r.check_budget(budget)
        for field,value in (('charged',69961),('unresolved',1),('failed',1)):
            wrong=copy.deepcopy(budget);wrong[field]=value
            with self.assertRaisesRegex(RuntimeError,'ledger differs'):r.check_budget(wrong)
        wrong=copy.deepcopy(budget);wrong['phase_charged']['engineering_reserve']=1
        with self.assertRaisesRegex(RuntimeError,'ledger differs'):r.check_budget(wrong)

    def test_only_cpu_freeze_preserves_owner_budget_and_no_hash_cycle(self):
        previous=dict(out='/H',cpu_affinities=[[0,1],[2,3]],gpu_affinity=[4,5],gpu_threads=2,
                      phase_limits=r.PHASES,S1_gate={'path':'s1'},prior_qualification_gate={'path':'q'},stages=[])
        request=dict(request_path='/request',selection_out='/H/selected',max_seconds=600,python='/python')
        cfg,owner=r.make_configs(request,previous,dict(root='/repo',stop_file='/H/STOP'),Path('/H/select_execution'))
        self.assertEqual([(s['id'],s['resource']) for s in owner['stages']],[('freeze','cpu')])
        job=owner['stages'][0]['jobs'][0];self.assertEqual(job['receipt_expect']['selected_count'],8)
        self.assertEqual(job['receipt_expect']['measured_frames'],9600)
        self.assertFalse(job['receipt_expect']['full1000_calibration_complete'])
        for key in ('cpu_affinities','gpu_affinity','phase_limits','S1_gate','prior_qualification_gate'):
            self.assertEqual(owner[key],previous[key])
        self.assertNotIn('registration_sha256',cfg);self.assertNotIn('owner_config_sha256',owner)
        self.assertEqual(previous['stages'],[])

    def test_nonlinux_register_stops_before_writes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'request.json';r.save(p,{'schema':'H_INITIAL_SELECTION_REGISTRATION_REQUEST_V1'})
            with patch.object(r.sys,'platform','win32'):
                with self.assertRaisesRegex(RuntimeError,'Linux process'):r.register(p)
            self.assertEqual([x.name for x in Path(td).iterdir()],['request.json'])

    def test_immutable_files_and_binding_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'receipt.json';r.save(p,{'a':1})
            with self.assertRaises(FileExistsError):r.save(p,{'a':2})
            with self.assertRaisesRegex(RuntimeError,'Pinned'):r.pin({'path':str(p),'sha256':'0'*64})
        with self.assertRaisesRegex(RuntimeError,'Conflicting'):r.merge({'a':'x'},{'a':'y'})

    def test_owner_precreated_empty_directory_allowed_but_attempt_never_reused(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'science';r.require_fresh_output(out)
            out.mkdir();r.require_fresh_output(out)  # Actual stage_owner.launch behavior.
            r.save(out/'attempt.json',{'status':'STARTED'})
            with self.assertRaisesRegex(RuntimeError,'preserve'):r.require_fresh_output(out)

    def test_owner_identity_receipt_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_,_=self.fixture(Path(td));cfg=r.read(args['owner_path'])
            p=Path(cfg['owner_out'])/'owner_identity.json';value=r.read(p);value['start_ticks']+=1;p.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,'Recorded render owner identity'):r.verify_render_closed(**args)


class SourceReceiptTests(unittest.TestCase):
    def fixture(self,root):
        out=root/'render';out.mkdir();[ (out/n).mkdir() for n in ('source_checkpoints','sources','images') ]
        ids=[f's{i}' for i in range(200)];outputs={};rows=[]
        for i,sid in enumerate(ids):
            archive=out/'images'/f'{i:04d}.npz';archive.write_bytes(b'synthetic sealed archive')
            local=[dict(source_index=i,source_id=sid,image_archive=str(archive),slot=k//3,noise_seed=6101+k%3) for k in range(48)]
            metrics=out/'sources'/f'{i:04d}.json';r.save(metrics,local)
            cp=out/'source_checkpoints'/f'{i:04d}.json';r.save(cp,dict(status='H_INITIAL_RX_SOURCE_COMPLETE',
                registration_sha256='a'*64,config_sha256='b'*64,source_id=sid,source_index=i,frame_count=48,
                images_scored=True,source_decode_complete=True,new_packet_decodes=0,
                input_bindings={},outputs=r.bind((archive,metrics))))
            outputs.update(r.bind((archive,metrics,cp)));rows.extend(local)
        return dict(out=str(out)),dict(outputs=outputs,registration_sha256='a'*64,config_sha256='b'*64),ids,rows

    def test_all200_real_source_receipt_layout_matches_flat_table(self):
        with tempfile.TemporaryDirectory() as td:
            cfg,done,ids,rows=self.fixture(Path(td));audit=r.audit_source_receipts(cfg,done,ids,rows)
            self.assertEqual(audit['frame_count'],9600);self.assertFalse(audit['RGB_arrays_redecoded'])
            rows[0]['noise_seed']=9999
            with self.assertRaisesRegex(RuntimeError,'Source metrics differ'):r.audit_source_receipts(cfg,done,ids,rows)

    def test_missing_or_changed_source_archive_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            cfg,done,ids,rows=self.fixture(Path(td));p=Path(rows[0]['image_archive']);p.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'unsealed'):r.audit_source_receipts(cfg,done,ids,rows)


if __name__=='__main__':unittest.main()
