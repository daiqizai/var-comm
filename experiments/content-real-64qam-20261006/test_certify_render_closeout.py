"""Pure closeout tests: no actual processes, GPU, ledger, or scientific reruns."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
import sys
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('closeout_tested',Path(__file__).with_name('certify_render_closeout.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class CloseoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def budget(self):
        return dict(created=True,charged=69960,phase_charged=dict(c.CHARGES),unresolved=0,failed=0,development_remaining=13200)

    def test_zombie_reused_or_live_pid_never_certified(self):
        identity={'pid':111}
        c.require_exited(identity,lambda _:None)
        for value in ({'state':'Z'},{'state':'S'},{'start_ticks':999,'state':'S'}):
            with self.assertRaisesRegex(RuntimeError,'still exists'):c.require_exited(identity,lambda _:value)

    def test_budget_cannot_refund_or_use_development(self):
        c.check_budget(self.budget())
        for key,value in (('charged',69959),('unresolved',1),('failed',1),('development_remaining',13199)):
            b=self.budget();b[key]=value
            with self.assertRaises(RuntimeError):c.check_budget(b)
        b=self.budget();b['phase_charged']['development']=1
        with self.assertRaises(RuntimeError):c.check_budget(b)

    def test_archive_22_keeps_originals_and_excludes_movable_stop(self):
        a={'status':'SECOND_FAILURE_PRESERVED_STOP_NOT_RELEASED','files':{},'original_files_unchanged':True,'STOP_released':False}
        stop=self.root/'STOP'
        for i in range(22):
            p=stop if i==0 else self.root/f'old_{i}.json';p.write_text(str(i))
            cp=self.root/f'copy_{i}.json';cp.write_bytes(p.read_bytes())
            a['files'][str(p)]={'copy':str(cp),'sha256':c.sha(p)}
        bound=c.audit_archive(a,stop,(stop,));self.assertNotIn(str(stop),bound)
        self.assertIn(a['files'][str(stop)]['copy'],bound)
        (self.root/'old_1.json').write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'Archived evidence'):c.audit_archive(a,stop,(stop,))

    def grid_fixture(self):
        out=self.root/'render';out.mkdir()
        for name in ('source_checkpoints','sources','images'):(out/name).mkdir()
        ids=[f'source{i}' for i in range(200)]
        candidates=[dict(slot=s,candidate_id=f'candidate{s}',arm='H16-R',target_m=8,q=4,nominal_rate='5/6',snr_db=13,K=0) for s in range(16)]
        allrows=[];outputs={}
        for i,sid in enumerate(ids):
            image=out/'images'/f'{i:04d}.npz';image.write_bytes(b'fixture archive bytes')
            rows=[]
            for candidate in candidates:
                for seed in (6101,6102,6103):
                    summary=dict(status='H_ACTUAL_RX_RECONSTRUCTION_COMPLETE',source_status='RAW_SOURCE_DECODED',
                        source_decode_complete=True,gray=False,new_packet_decodes=0,image_sha256='1'*64,receiver_view_sha256='2'*64,
                        target_image_used_for_reconstruction=False,truth_correction=False,cached_clean_image_used=False)
                    rows.append(dict(candidate,source_index=i,source_id=sid,noise_seed=seed,event_id=f'{i}/{candidate["slot"]}/{seed}',
                        rx_summary=summary,source_status='RAW_SOURCE_DECODED',gray=False,mse=.01,psnr_db=20.,
                        image_sha256='1'*64,receiver_view_sha256='2'*64,image_key='image_0000',image_archive=str(image)))
            sp=out/'sources'/f'{i:04d}.json';c.save(sp,rows)
            cp=out/'source_checkpoints'/f'{i:04d}.json';local=c.bind((image,sp))
            c.save(cp,dict(status='H_INITIAL_RX_SOURCE_COMPLETE',source_index=i,source_id=sid,frame_count=48,
                source_decode_complete=True,images_scored=True,new_packet_decodes=0,registration_sha256='reg',
                config_sha256='config',outputs=local,input_bindings={}))
            outputs.update(local);outputs[str(cp)]=c.sha(cp);allrows.extend(rows)
        path=out/'frame_metrics.json';c.save(path,allrows);outputs[str(path)]=c.sha(path)
        return {'out':str(out)},{'outputs':outputs,'registration_sha256':'reg','config_sha256':'config'},dict(source_ids=ids,whole_candidates=candidates),allrows,path

    def test_complete_200_by_16_by_3_grid_passes_missing_or_duplicate_fails(self):
        cfg,done,shortlist,rows,path=self.grid_fixture()
        result=c.audit_grid(cfg,done,shortlist);self.assertEqual(result['frame_count'],9600)
        rows[1]=copy.deepcopy(rows[0]);path.unlink();c.save(path,rows);done['outputs'][str(path)]=c.sha(path)
        with self.assertRaisesRegex(RuntimeError,'Duplicate frame'):c.audit_grid(cfg,done,shortlist)
        rows.pop();path.unlink();c.save(path,rows);done['outputs'][str(path)]=c.sha(path)
        with self.assertRaisesRegex(RuntimeError,'9600'):c.audit_grid(cfg,done,shortlist)

    def test_inconsistent_score_or_truth_substitute_fails_before_certification(self):
        cfg,done,shortlist,rows,path=self.grid_fixture()
        rows[0]['psnr_db']=21.;path.unlink();c.save(path,rows);done['outputs'][str(path)]=c.sha(path)
        with self.assertRaisesRegex(RuntimeError,'inconsistent image'):c.audit_grid(cfg,done,shortlist)
        rows[0]['psnr_db']=20.;rows[0]['rx_summary']['truth_correction']=True
        path.unlink();c.save(path,rows);done['outputs'][str(path)]=c.sha(path)
        with self.assertRaisesRegex(RuntimeError,'used truth'):c.audit_grid(cfg,done,shortlist)

    def test_stop_only_moves_matching_bytes_after_certificate_and_registered_exit(self):
        out=self.root/'closeout';out.mkdir();stop=self.root/'STOP';stop.write_text('original failure')
        local=self.root/'renderer_STOP';local.write_text('original failure')
        certificate=out/'completion.json';c.save(certificate,{'status':c.CERT_STATUS})
        ctx={'expected_stop':c.record(stop),'expected_stops':[c.record(stop),c.record(local)],'owner_identity':{'pid':1},'worker_identity':{'pid':2},
             'api':types.SimpleNamespace(raw_process_state=lambda _:None)}
        original=stop.read_bytes();release=c.release_stop(ctx,out,certificate,'reg')
        self.assertFalse(stop.exists());self.assertEqual((out/'released_STOP.original').read_bytes(),original)
        self.assertFalse(local.exists());self.assertEqual(len(release['released_markers']),2)
        self.assertEqual(release['certificate_sha256'],c.sha(certificate));self.assertFalse(release['original_owner_success'])
        with self.assertRaises((RuntimeError,FileNotFoundError)):c.release_stop(ctx,out,certificate,'reg')

    def test_mismatched_stop_never_released(self):
        out=self.root/'closeout';out.mkdir();stop=self.root/'STOP';stop.write_text('old')
        local=self.root/'renderer_STOP';local.write_text('old')
        cp=out/'completion.json';c.save(cp,{'status':c.CERT_STATUS})
        ctx={'expected_stop':c.record(stop),'expected_stops':[c.record(stop),c.record(local)],'owner_identity':{'pid':1},'worker_identity':{'pid':2},
             'api':types.SimpleNamespace(raw_process_state=lambda _:None)}
        stop.write_text('new independent stop')
        with self.assertRaisesRegex(RuntimeError,'STOP changed'):c.release_stop(ctx,out,cp,'reg')
        self.assertEqual(stop.read_text(),'new independent stop');self.assertFalse((out/'release_receipt.json').exists())

    def test_bad_second_stop_never_moves_first(self):
        out=self.root/'closeout';out.mkdir();stop=self.root/'STOP';stop.write_text('old')
        local=self.root/'renderer_STOP';local.write_text('old');cp=out/'completion.json';c.save(cp,{'status':c.CERT_STATUS})
        ctx={'expected_stop':c.record(stop),'expected_stops':[c.record(stop),c.record(local)],'owner_identity':{'pid':1},'worker_identity':{'pid':2},
             'api':types.SimpleNamespace(raw_process_state=lambda _:None)}
        local.write_text('new STOP')
        with self.assertRaisesRegex(RuntimeError,'STOP changed'):c.release_stop(ctx,out,cp,'reg')
        self.assertTrue(stop.exists());self.assertFalse((out/'released_STOP.original').exists())

    def test_second_move_exception_preserves_first_move_journal_without_success(self):
        out=self.root/'closeout';out.mkdir();stop=self.root/'STOP';stop.write_text('old')
        local=self.root/'renderer_STOP';local.write_text('old');cp=out/'completion.json';c.save(cp,{'status':c.CERT_STATUS})
        ctx={'expected_stop':c.record(stop),'expected_stops':[c.record(stop),c.record(local)],'owner_identity':{'pid':1},'worker_identity':{'pid':2},
             'api':types.SimpleNamespace(raw_process_state=lambda _:None)}
        rename=c.os.rename
        def fail_second(source,target):
            if Path(source)==local:raise OSError('injected second move failure')
            return rename(source,target)
        with patch.object(c.os,'rename',side_effect=fail_second),self.assertRaises(OSError):c.release_stop(ctx,out,cp,'reg')
        self.assertTrue((out/'released_STOP.original').exists());self.assertTrue((out/'stop_release_0.json').exists())
        self.assertTrue(local.exists());self.assertFalse((out/'release_receipt.json').exists())
        self.assertFalse((out/'closeout_completion.json').exists())

    def test_two_phase_closeout_never_manufactures_original_owner_success(self):
        h=self.root/'H';h.mkdir();old=h/'old_owner';old.mkdir();render=h/'render';render.mkdir()
        oc=old/'owner_config.json';c.save(oc,{'out':str(h)})
        rc=old/'render_config.json';c.save(rc,{'out':str(render)})
        request=self.root/'request.json';out=h/'closeout'
        c.save(request,dict(execution_dir=str(out),root=str(self.root),original_owner_config=c.record(oc),render_config=c.record(rc)))
        stop=h/'STOP';stop.write_text('diagnosed error');local=render/'STOP';local.write_text('diagnosed error');budget=self.budget()
        api=types.SimpleNamespace(raw_process_state=lambda _:None,budget_snapshot=lambda *args,**kwargs:budget)
        ctx=dict(original_registration_sha256='original-reg',owner_identity={'pid':11},worker_identity={'pid':12},
            render_completion={'path':'render','sha256':'0'*64},original_owner_config=c.record(oc),original_launch={'path':'launch','sha256':'0'*64},
            worker_exit_receipt={'path':'exit','sha256':'0'*64},original_failure={'path':'failure','sha256':'0'*64},
            bindings=c.bind((oc,rc)),outputs={},source_bindings={},budget=budget,grid_audit={'frame_count':9600},gpu_admission={},
            expected_stop=c.record(stop),expected_stops=[c.record(stop),c.record(local)],api=api,
            owner_config=dict(budget_path='ledger',budget_registration_sha256='budget'))
        fake_fcntl=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *args:None)
        with patch.dict(sys.modules,{'fcntl':fake_fcntl}),patch.object(c,'environment'),patch.object(c,'audit',return_value=ctx),patch.object(c,'gpu_empty',return_value={}):
            c.register(request);self.assertTrue(stop.exists());self.assertFalse((out/'completion.json').exists())
            done=c.run(request)
            self.assertEqual(done['status'],c.DONE_STATUS);self.assertFalse(done['original_owner_success'])
            self.assertFalse((old/'completion.json').exists());self.assertFalse(stop.exists())
            c.verify(done['outputs'])
            with self.assertRaisesRegex(RuntimeError,'attempt exists'):c.run(request)


if __name__=='__main__':unittest.main()
