"""CPU-only output, exact-population and live-owner contract tests."""
import copy
import importlib.util
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
for p in (HERE,HERE.parent,HERE.parent/'payload',HERE.parent/'phy_codec'):
    sys.path.insert(0,str(p))
import h_full_payload_render_driver as d
import test_h_full_payload_rx as fixtures


class FullRenderTests(unittest.TestCase):
    def test_launch_rechecks_same_eight_gib_storage_reserve_as_registration(self):
        cap=36*1024**3
        with patch.object(d.shutil,'disk_usage',return_value=SimpleNamespace(free=cap+8*1024**3)):
            d.check_storage('/registered/output',cap)
        for free in (cap+1024**3,cap+8*1024**3-1):
            with patch.object(d.shutil,'disk_usage',return_value=SimpleNamespace(free=free)):
                with self.assertRaisesRegex(RuntimeError,'headroom'):d.check_storage('/registered/output',cap)

    def test_owner_precreated_empty_output_allowed_but_no_second_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'render';out.mkdir();d.claim_output(out,'a'*64)
            original=(out/'attempt.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError,'Prior visual attempt'):d.claim_output(out,'a'*64)
            self.assertEqual((out/'attempt.json').read_bytes(),original)
            other=Path(td)/'other';other.mkdir();(other/'unrelated.txt').write_text('keep')
            with self.assertRaisesRegex(RuntimeError,'Prior visual attempt'):d.claim_output(other,'a'*64)
            self.assertFalse((other/'attempt.json').exists())

    def test_actual_rgb_writer_seals_all42_rows_and_rejects_changed_image(self):
        fixtures.FullRXTests.setUpClass();f=fixtures.FullRXTests();f.setUp()
        rows,images,cost=f.render()
        with tempfile.TemporaryDirectory() as td:
            outputs,size=d.write_source(Path(td)/'valid',0,'synthetic0',rows,images,cost,{},'a'*64,fixtures.cache_api,fixtures.rx)
            self.assertEqual(len(outputs),3);self.assertGreater(size,0)
            for path,value in outputs.items():self.assertEqual(d.sha(path),value)
            cp=d.read(Path(td)/'valid/source_checkpoints/0000.json')
            self.assertEqual(cp['phase_frame_counts'],{'whole_calibration':24,'partial_calibration':18})
            with np.load(Path(td)/'valid/images/0000.npz',allow_pickle=False) as z:
                self.assertTrue(all(z[k].dtype==np.float32 for k in z.files))
            damaged={k:v.copy() for k,v in images.items()};damaged[next(iter(damaged))][0,0,0]+=.01
            with self.assertRaisesRegex(RuntimeError,'RGB/RX evidence'):
                d.write_source(Path(td)/'bad',0,'synthetic0',copy.deepcopy(rows),damaged,cost,{},'a'*64,fixtures.cache_api,fixtures.rx)
            self.assertFalse((Path(td)/'bad/source_checkpoints/0000.json').exists())

    def test_exact42000_population_rejects_missing_duplicate_or_phase_swap(self):
        ids=[f'source{i}' for i in range(1000)]
        rows=[dict(source_index=i,source_id=sid,full_slot=s,noise_seed=n,
                   phase='whole_calibration' if s<8 else 'partial_calibration',
                   rx_summary={'source_decode_complete':True})
              for i,sid in enumerate(ids) for s in range(14) for n in (6101,6102,6103)]
        self.assertEqual(d.validate_complete_rows(rows,ids),{'whole_calibration':24000,'partial_calibration':18000})
        with self.assertRaisesRegex(RuntimeError,'coverage incomplete'):d.validate_complete_rows(rows[:-1],ids)
        bad=rows.copy();bad[-1]=rows[0]
        with self.assertRaisesRegex(RuntimeError,'duplicate'):d.validate_complete_rows(bad,ids)
        bad=rows.copy();bad[0]=dict(rows[0],phase='partial_calibration')
        with self.assertRaisesRegex(RuntimeError,'state not complete'):d.validate_complete_rows(bad,ids)
        bad=rows.copy();bad[0]=dict(rows[0],rx_summary={'source_decode_complete':False})
        with self.assertRaisesRegex(RuntimeError,'state not complete'):d.validate_complete_rows(bad,ids)

    def test_original_numeric_flags_model_and_complete_source_closure_required(self):
        ctx=dict(calibration={'identity':{'frozen':'model'}},expected_flags={'threads':6,'fp32':True},
                 static_bindings={'/original.py':'sha'},bound={'/original.py':'sha'})
        native=SimpleNamespace(loaded={'identity':{'frozen':'model'}},flags={'threads':6,'fp32':True},driver_bindings={'/original.py':'sha'})
        d.validate_native(native,ctx)
        for field,value in [('flags',{'threads':2,'fp32':True}),('loaded',{'identity':{'frozen':'other'}}),
                            ('driver_bindings',{'/original.py':'sha','/new.py':'new'})]:
            bad=copy.deepcopy(native);setattr(bad,field,value)
            with self.assertRaisesRegex(RuntimeError,'differ'):d.validate_native(bad,ctx)
        with self.assertRaisesRegex(RuntimeError,'closure'):d.validate_native(native,dict(ctx,bound={}))

    def make_owner(self,td):
        root=Path(td);reg=root/'reg.json';cfg=root/'config.json';ocp=root/'owner_config.json';base=root/'owner';out=root/'render'
        reg.write_text('{}');cfg.write_text('{}')
        argv=[str(root/'python'),'-B',str(Path(d.__file__).resolve()),'--config',str(cfg.resolve())]
        ownerid=dict(pid=45,start_ticks=100,uid=1002,argv=['owner'])
        workerid=dict(pid=46,start_ticks=200,uid=1002,argv=argv)
        oc=dict(registration=str(reg),out=str(root),owner_out=str(base),gpu_threads=6,gpu_affinity=[24,25,26,27,28,29],gpu_device=0,
                stages=[dict(id='render',resource='gpu',jobs=[dict(id='full1000_render',argv=argv,out=str(out),completion=str(out/'completion.json'))])])
        d.save(ocp,oc);oid=dict(ownerid,registration_sha256=d.sha(reg),config_sha256=d.sha(ocp))
        d.save(base/'owner_identity.json',oid)
        lp=base/'stages/render/workers/full1000_render/launch.json';gp=base/'stages/render/gpu_admission.json'
        d.save(lp,dict(registration_sha256=d.sha(reg),resource='gpu',argv=argv,identity=workerid,threads=6,affinity=oc['gpu_affinity']))
        d.save(gp,dict(status='GPU_IDLE_CONFIRMED',registration_sha256=d.sha(reg)))
        def same(a,b):return all(a[k]==b[k] for k in ('pid','start_ticks','uid','argv'))
        owner=SimpleNamespace(validate_config=lambda *_:None,command_entry=lambda x:Path(x[2]),same_identity=same,
                              identity=lambda pid: ownerid if pid==45 else workerid)
        ctx=dict(cfg=dict(visual_owner_config=str(ocp),registration=str(reg),H_out=str(root),out=str(out)),owner=owner,reg={})
        return ctx,cfg,base,lp,gp

    def test_visual_admission_requires_live_bound_parent_worker_and_gpu_lock_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            ctx,cfg,base,lp,gp=self.make_owner(td)
            with patch.object(d.os,'getppid',return_value=45),patch.object(d.os,'getpid',return_value=46), \
                 patch.object(d.os,'sched_getaffinity',return_value=set(range(24,30)),create=True),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                result=d.visual_admission(ctx,cfg);self.assertEqual(len(result['bindings']),3)
                oid=d.read(base/'owner_identity.json');oid['config_sha256']='0'*64
                (base/'owner_identity.json').write_text(json.dumps(oid))
                with self.assertRaisesRegex(RuntimeError,'live parent'):d.visual_admission(ctx,cfg)
                oid['config_sha256']=d.sha(ctx['cfg']['visual_owner_config']);(base/'owner_identity.json').write_text(json.dumps(oid))
                launch=d.read(lp);launch['identity']['start_ticks']+=1;lp.write_text(json.dumps(launch))
                with self.assertRaisesRegex(RuntimeError,'process identity'):d.visual_admission(ctx,cfg)
                launch['identity']['start_ticks']-=1;lp.write_text(json.dumps(launch))
                gp.write_text(json.dumps({'status':'UNREGISTERED','registration_sha256':d.sha(ctx['cfg']['registration'])}))
                with self.assertRaisesRegex(RuntimeError,'GPU admission'):d.visual_admission(ctx,cfg)


if __name__=='__main__':unittest.main()
