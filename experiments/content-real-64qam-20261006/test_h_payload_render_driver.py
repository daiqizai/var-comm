"""Prepared GPU-stage contracts, exercised entirely with fake visual functions.

No model, actual PHY decoder, real source image or GPU is loaded here.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
if (HERE.parent/'phy_codec').exists():sys.path.insert(0,str(HERE.parent/'phy_codec'))
import test_h_payload_rx as fixtures
import h_payload_render_driver as d
import h_payload_rx as rx
import h64_phy as phy
import h64_source as codec
from test_h64_core import Provider


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.ReceiverTests.setUpClass()
    def setUp(self):
        self.f=fixtures.ReceiverTests('test_only_rx_fields_are_read_and_actual_raw_tokens_rendered');self.f.setUp()
        self.p=self.f.profile();self.bits=phy.raw_payload(self.f.scales,6)
        self.target=np.full((3,256,256),73,np.uint8)
        self.targetsha=hashlib.sha256(self.target.tobytes()).hexdigest()
        self.c=dict(slot=0,candidate_id='raw-candidate',arm='H64-R',target_m=6,K=0,q=6,nominal_rate='1/2',snr_db=13)
    def source(self):
        frames=[]
        for seed in d.SEEDS:
            t=self.f.trace(self.p,self.bits)
            t.update(candidate_slot=0,noise_seed=seed,source_id='synthetic0',source_index=0,
                     candidate_id=self.c['candidate_id'],arm=self.c['arm'],snr_db=13,event_id='event-'+str(seed),
                     evaluation_only=dict(header_correct=True,parsed_wire_matches_transmission=True,accepted_wire_mismatch=False))
            frames.append(t)
        return dict(source_id='synthetic0',source_index=0,frames=frames)
    def render(self,source=None,boundary=lambda:None):
        return d.render_source(source or self.source(),self.target,self.targetsha,[self.c],self.f.receiver,rx,
                               {'frozen_visual':'fixture'},boundary)
    def test_cache_uses_actual_payload_not_noise_score_or_tx_truth(self):
        cache=d.CachedReceiver(self.f.receiver,{'model':'frozen'})
        first=self.f.trace(self.p,self.bits);a=cache.reconstruct(rx.receiver_view(first),'a')
        second=copy.deepcopy(first);second['tx']={'truth':'other'};second['rx']['header']['score']=999
        b=cache.reconstruct(rx.receiver_view(second),'b')
        self.assertEqual((cache.hits,cache.misses),(1,1));self.assertEqual(len(self.f.rendered),1)
        np.testing.assert_array_equal(a['image'],b['image'])
        self.assertNotEqual(a['summary']['receiver_view_sha256'],b['summary']['receiver_view_sha256'])
        self.assertEqual(b['summary']['receiver_cache_origin_event'],'a')
        wrong=self.bits.copy();wrong[0]^=1
        c=cache.reconstruct(rx.receiver_view(self.f.trace(self.p,wrong)),'c')
        self.assertFalse(c['summary']['receiver_result_cache_hit']);self.assertEqual(len(self.f.rendered),2)
        self.assertNotEqual(c['summary']['image_sha256'],a['summary']['image_sha256'])
        b['image'][0,0,0]=0
        restored=cache.reconstruct(rx.receiver_view(first),'d')
        np.testing.assert_array_equal(restored['image'],a['image'])
    def test_actual_received_profile_also_changes_cache_identity(self):
        cache=d.CachedReceiver(self.f.receiver,{'model':'frozen'})
        cache.reconstruct(rx.receiver_view(self.f.trace(self.p,self.bits)),'a')
        q4=self.f.profile(q=4)
        b=cache.reconstruct(rx.receiver_view(self.f.trace(q4,self.bits)),'b')
        self.assertFalse(b['summary']['receiver_result_cache_hit']);self.assertEqual(cache.misses,2)
    def test_cached_arithmetic_receipt_still_requires_canonical_origin(self):
        encoded=codec.encode_prefixes(self.f.scales,lambda:Provider([]),self.f.primitives,(6,))
        p=self.f.profile('arithmetic');cache=d.CachedReceiver(self.f.receiver,{'model':'frozen'})
        view=rx.receiver_view(self.f.trace(p,encoded[6]['bits']))
        a=cache.reconstruct(view,'first');b=cache.reconstruct(view,'second')
        self.assertEqual(len(self.f.registry),1)
        self.assertTrue(a['summary']['arithmetic_canonical_executed_this_frame'])
        self.assertFalse(b['summary']['arithmetic_canonical_executed_this_frame'])
        self.assertTrue(b['summary']['arithmetic_canonical'])
        invalid=np.append(encoded[6]['bits'],0).astype(np.uint8)
        c=cache.reconstruct(rx.receiver_view(self.f.trace(p,invalid)),'invalid')
        self.assertEqual(c['summary']['source_status'],'ARITHMETIC_SOURCE_INVALID_GRAY')
        self.assertEqual(len(self.f.registry),2);self.assertEqual(len(self.f.rendered),1)
    def test_cache_hit_rechecks_actual_decoded_bits_before_reuse(self):
        cache=d.CachedReceiver(self.f.receiver,{'model':'frozen'});view=rx.receiver_view(self.f.trace(self.p,self.bits))
        cache.reconstruct(view,'a');view['rx']['body']['decoded_bits'][14]^=1
        with self.assertRaisesRegex(rx.TraceIntegrityError,'parse differs'):cache.reconstruct(view,'b')
        self.assertEqual(cache.hits,0)
    def test_source_coverage_psnr_and_image_dedup_are_real(self):
        calls=[];rows,images,stats=self.render(boundary=lambda:calls.append(1))
        self.assertEqual(len(calls),3);self.assertEqual(len(rows),3);self.assertEqual(len(images),1)
        self.assertEqual(stats['receiver_cache_hits'],2);self.assertEqual(stats['suffix_renderer_calls'],1)
        self.assertEqual({r['noise_seed'] for r in rows},set(d.SEEDS))
        for r in rows:
            self.assertEqual(r['source_status'],'RAW_SOURCE_DECODED');self.assertFalse(r['gray'])
            self.assertAlmostEqual(r['psnr_db'],-10*np.log10(r['mse']))
            self.assertFalse(r['rx_summary']['target_image_used_for_reconstruction'])
            self.assertFalse(r['rx_summary']['cached_clean_image_used'])
    def test_missing_duplicate_and_wrong_candidate_are_rejected(self):
        source=self.source();source['frames'].pop()
        with self.assertRaisesRegex(RuntimeError,'Missing actual'):self.render(source)
        source=self.source();source['frames'].append(copy.deepcopy(source['frames'][0]))
        with self.assertRaisesRegex(RuntimeError,'duplicate'):self.render(source)
        source=self.source();source['frames'][0]['arm']='H16-A'
        with self.assertRaisesRegex(RuntimeError,'identity'):self.render(source)
    def test_software_failure_and_stop_propagate_not_gray(self):
        self.f.receiver.render_tokens=lambda *_:(_ for _ in ()).throw(RuntimeError('model failure'))
        with self.assertRaisesRegex(RuntimeError,'model failure'):self.render()
        with self.assertRaisesRegex(RuntimeError,'STOP'):self.render(boundary=lambda:(_ for _ in ()).throw(RuntimeError('STOP')))
    def test_sealed_archive_detects_changed_rgb_and_missing_keys(self):
        rows,images,_=self.render()
        with tempfile.TemporaryDirectory() as td:
            archive=Path(td)/'images.npz';np.savez(archive,**images)
            d.validate_source_archive(rows,archive,rx)
            changed={k:v.copy() for k,v in images.items()};changed['image_0000'][0,0,0]=.01;np.savez(archive,**changed)
            with self.assertRaisesRegex(RuntimeError,'RGB/RX'):d.validate_source_archive(rows,archive,rx)
            np.savez(archive,other=images['image_0000'])
            with self.assertRaisesRegex(RuntimeError,'coverage'):d.validate_source_archive(rows,archive,rx)

    def test_real_render_row_schema_roundtrips_into_independent_selector(self):
        import h_payload_select as selector
        ids=[f'synthetic_source_{i:04d}' for i in range(200)];candidates=[];source=self.source();source['source_id']=ids[0]
        original=source['frames'];source['frames']=[]
        for slot,(snr,arm) in enumerate((snr,arm) for snr in (13,19) for arm in selector.ARMS):
            wire=dict(arm=arm,q=4 if arm.startswith('H16') else 6,nominal_rate='1/2',target_m=6,K=0)
            key=selector.digest(wire)
            c=dict(wire,candidate_id='HWHOLE:'+key,policy_key=key,snr_db=snr,slot=slot,
                   mean_attempted_tx_probability_scales=float(6 if arm.endswith('-A') else 0))
            candidates.append(c)
            for frame in original:
                f=copy.deepcopy(frame);f.update(candidate_slot=slot,candidate_id=c['candidate_id'],arm=arm,snr_db=snr,
                    source_id=ids[0],event_id=f'fixture:{slot}:{frame["noise_seed"]}')
                # An actual legal received raw profile is deliberately shared,
                # even for other TX families: accepted mismatch is not repaired.
                f['evaluation_only']=dict(header_correct=False,parsed_wire_matches_transmission=False,accepted_wire_mismatch=True)
                source['frames'].append(f)
        sample,images,_=d.render_source(source,self.target,self.targetsha,candidates,self.f.receiver,rx,{'model':'fixture'},lambda:None)
        with tempfile.TemporaryDirectory() as td:
            archive=Path(td)/'synthetic_images.npz';np.savez(archive,**images)
            for row in sample:row['image_archive']=str(archive)
            d.validate_source_archive(sample,archive,rx)
            rows=[]
            for i,sid in enumerate(ids):
                for row in sample:
                    r=copy.deepcopy(row);r.update(source_index=i,source_id=sid,event_id=f'{i}:{r["slot"]}:{r["noise_seed"]}')
                    rows.append(r)
            shortlist=dict(status='H_EXPECTED_PSNR_SHORTLIST_FROZEN',ready_for_real_calibration=True,
                           source_ids=ids,whole_candidates=candidates)
            sb=d.canonical(shortlist).encode();mb=d.canonical(rows).encode();mp=str(Path(td)/'frame_metrics.json')
            completion=dict(status='H_INITIAL_TRUE200_RX_COMPLETE',images_scored=True,source_decode_complete=True,
                source_count=200,frame_count=len(rows),new_packet_decodes=0,development_used=False,
                registration_sha256='a'*64,shortlist_sha256=hashlib.sha256(sb).hexdigest(),
                outputs={mp:hashlib.sha256(mb).hexdigest(),str(archive):d.sha(archive)})
            protocol=dict(schema='H_CODEC_PROTOCOL_V1',status='FROZEN_BEFORE_DATA',main_snrs_db=[13,19],
                population={'initial_and_full_cal_noise_seeds':list(d.SEEDS)},calibration={'no_development_selection':True})
            interpretation=dict(schema='H_PRESCREEN_INTERPRETATION_V1',status='FROZEN_BEFORE_COARSE_DATA',original_protocol_changed=False)
            result=selector.select_initial(sb,mb,frame_metrics_path=mp,render_completion=completion,protocol=protocol,interpretation=interpretation)
            self.assertEqual(result['selected_count'],8);self.assertEqual(result['measured_frames'],4800)
            broken=d.canonical(rows[:-1]).encode();completion['outputs'][mp]=hashlib.sha256(broken).hexdigest()
            with self.assertRaisesRegex(ValueError,'frame count'):
                selector.select_initial(sb,broken,frame_metrics_path=mp,render_completion=completion,protocol=protocol,interpretation=interpretation)


class AdmissionTests(unittest.TestCase):
    def test_all_native_modules_must_be_bound_even_when_models_match(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);names=('m1_common.py','m1_native.py','m1_phy.py','m1_performance.py','m1_extra.py')
            for name in names:(root/name).write_text('# frozen\n')
            bindings={str(root/n):d.sha(root/n) for n in names}
            d.require_native_bindings(root,bindings)
            bindings.pop(str(root/'m1_extra.py'))
            with self.assertRaisesRegex(RuntimeError,'unbound'):d.require_native_bindings(root,bindings)
            bindings[str(root/'m1_extra.py')]=d.sha(root/'m1_extra.py')
            (root/'m1_native.py').write_text('# changed\n')
            with self.assertRaisesRegex(RuntimeError,'unbound'):d.require_native_bindings(root,bindings)

    def fixture(self,td):
        base=Path(td);reg=base/'reg.json';d.atomic(reg,{'registered':True})
        config=base/'config.json';d.atomic(config,{'prepared':True})
        parent=dict(pid=100,start_ticks=10,uid=1002,argv=['python','owner.py'])
        argv=['python','-B',str(Path(d.__file__).resolve()),'--config',str(config)]
        child=dict(pid=101,start_ticks=11,uid=1002,argv=argv)
        out=base/'out';owner_out=base/'owner';job=dict(id='gpu',argv=argv,out=str(out),completion=str(out/'completion.json'))
        owner_cfg=dict(registration=str(reg),owner_out=str(owner_out),stages=[dict(id='render',resource='gpu',jobs=[job])],
                       gpu_threads=6,gpu_affinity=[24,25,26,27,28,29],gpu_device=0)
        owner_config=base/'owner_config.json';d.atomic(owner_config,owner_cfg)
        d.atomic(owner_out/'owner_identity.json',dict(parent,registration_sha256=d.sha(reg)))
        launch=owner_out/'stages/render/workers/gpu/launch.json'
        d.atomic(launch,dict(identity=child,registration_sha256=d.sha(reg),resource='gpu',argv=argv,threads=6,affinity=owner_cfg['gpu_affinity']))
        admission=owner_out/'stages/render/gpu_admission.json'
        d.atomic(admission,dict(status='GPU_IDLE_CONFIRMED',registration_sha256=d.sha(reg)))
        api=SimpleNamespace(validate_config=lambda *_:None,command_entry=lambda args:Path(args[2]).resolve(),
            identity=lambda pid:copy.deepcopy(parent if pid==100 else child),
            same_identity=lambda a,b:all(a[k]==b[k] for k in ('pid','start_ticks','uid','argv')))
        ctx=dict(cfg=dict(registration=str(reg),render_owner_config=str(owner_config),out=str(out)),reg={},owner=api)
        return ctx,config,launch,admission,parent,child
    def call(self,ctx,config):
        with patch.object(d.os,'getppid',return_value=100),patch.object(d.os,'getpid',return_value=101),\
             patch.object(d.os,'sched_getaffinity',return_value=set(range(24,30)),create=True),\
             patch.dict(d.os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            return d.verify_live_visual_owner(ctx,str(config))
    def test_exact_live_gpu_owner_and_launched_worker_required(self):
        with tempfile.TemporaryDirectory() as td:
            ctx,cfg,launch,admission,parent,child=self.fixture(td)
            result=self.call(ctx,cfg);self.assertEqual(result['worker_identity'],child);self.assertEqual(len(result['bindings']),3)
            data=d.read(launch);data['identity']['start_ticks']+=1;d.atomic(launch,data)
            with self.assertRaisesRegex(RuntimeError,'runtime identity'):self.call(ctx,cfg)
    def test_foreign_parent_and_missing_or_bad_admission_block(self):
        with tempfile.TemporaryDirectory() as td:
            ctx,cfg,launch,admission,parent,child=self.fixture(td)
            original=ctx['owner'].identity
            ctx['owner'].identity=lambda pid:dict(original(pid),uid=2000)
            with self.assertRaisesRegex(RuntimeError,'live parent'):self.call(ctx,cfg)
            ctx['owner'].identity=original
            d.atomic(admission,dict(status='GPU_BUSY',registration_sha256=d.sha(ctx['cfg']['registration'])))
            with self.assertRaisesRegex(RuntimeError,'exclusive GPU'):self.call(ctx,cfg)
            admission.unlink()
            with self.assertRaises(FileNotFoundError):self.call(ctx,cfg)


if __name__=='__main__':unittest.main()
