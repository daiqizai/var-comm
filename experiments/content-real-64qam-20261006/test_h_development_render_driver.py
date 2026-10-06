"""Synthetic CPU tests for new orchestration; frozen receiver fixtures stay unchanged."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).absolute().parent
for p in (HERE,HERE.parent/'payload',HERE.parent/'phy_codec',HERE.parent/'full_calibration',HERE.parent/'runner'):
    sys.path.insert(0,str(p))
import h_development_render_driver as d
import h_development_rx as adapter
import test_h_development_rx as fixtures


class RenderDriverTests(unittest.TestCase):
    def test_owner_precreated_empty_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rx';p.mkdir();d.claim_output(p,'a'*64);original=(p/'attempt.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError,'Prior visual attempt'):d.claim_output(p,'a'*64)
            self.assertEqual((p/'attempt.json').read_bytes(),original)

    def test_budget_is_exact_paid_H18_and_preserves_MAIN(self):
        before=dict(charged=164760,phase_charged={'development':10800,'coarse':49152},development_remaining=2400,failed=0,unresolved=0)
        d.assert_budget(before,before,{'before':before})
        for field,value in [('charged',164761),('development_remaining',2399),('unresolved',1)]:
            with self.assertRaisesRegex(RuntimeError,'budget changed'):
                d.assert_budget(dict(before,**{field:value}),before,{'before':before})
        changed=copy.deepcopy(before);changed['phase_charged']['coarse']+=1
        with self.assertRaises(RuntimeError):d.assert_budget(changed,before,{'before':before})

    def test_complete_visual_identity_flags_and_static_graph_required(self):
        ctx=dict(expected_identity={'visual':'frozen'},expected_flags={'threads':6,'fp32':True},
                 static_bindings={'/source.py':'a'},bound={'/source.py':'a'})
        native=SimpleNamespace(loaded={'identity':{'visual':'frozen'}},flags={'threads':6,'fp32':True},driver_bindings={'/source.py':'a'})
        d.validate_native(native,ctx)
        for field,value in [('flags',{'threads':2}),('loaded',{'identity':{'visual':'other'}}),('driver_bindings',{'/source.py':'a','/new.py':'b'})]:
            bad=copy.deepcopy(native);setattr(bad,field,value)
            with self.assertRaises(RuntimeError):d.validate_native(bad,ctx)

    def test_real_adapter_writer_keeps_float_rgb_and_all54_rows(self):
        fixtures.DevelopmentRXTests.setUpClass();f=fixtures.DevelopmentRXTests();f.setUp();rows,images,cost=f.render()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);outputs,size=adapter.write_source(out,0,'dev0',rows,images,cost,{},'a'*64,fixtures.cache_api,fixtures.rx)
            self.assertGreater(size,0);d.verify(outputs)
            stored=d.read(out/'sources/0000.json');self.assertEqual(len(stored),54)
            self.assertTrue(all(r['image_archive']==str(out/'images/0000.npz') for r in stored))
            self.assertTrue(all('image_archive' not in r for r in rows))
            with np.load(out/'images/0000.npz',allow_pickle=False) as z:
                self.assertTrue(all(z[k].dtype==np.float32 for k in z.files))
            with self.assertRaisesRegex(RuntimeError,'no retry'):adapter.write_source(out,0,'dev0',rows,images,cost,{},'a'*64,fixtures.cache_api,fixtures.rx)

    def test_no_main_missing_or_duplicate_row_can_be_completed(self):
        ids=[f'dev{i}' for i in range(100)]
        rows=[dict(source_index=i,source_id=s,development_slot=slot,noise_seed=n,phase='development',source_status='RAW_SOURCE_DECODED',
                   rx_summary={'source_decode_complete':True}) for i,s in enumerate(ids) for slot in range(18) for n in (6201,6202,6203)]
        self.assertEqual(adapter.validate_complete_rows(rows,ids)['frame_count'],5400)
        for bad in (rows[:-1],rows[:-1]+[dict(rows[-1],development_slot=18)],rows[:-1]+[rows[0]]):
            with self.assertRaises(RuntimeError):adapter.validate_complete_rows(bad,ids)

    def owner_fixture(self,p):
        reg,cp,op=p/'reg.json',p/'cfg.json',p/'owner.json';d.save(reg,{});d.save(cp,{})
        base,out=p/'execution',p/'rx';argv=['python','-B',str(Path(d.__file__).absolute()),'--config',str(cp)]
        oid=dict(pid=45,start_ticks=100,uid=1002,argv=['owner']);wid=dict(pid=46,start_ticks=200,uid=1002,argv=argv)
        owner=dict(registration=str(reg),out=str(p),owner_out=str(base),gpu_threads=6,gpu_affinity=list(range(6)),gpu_device=0,
                   stages=[dict(id='render',resource='gpu',jobs=[dict(id='development_render',argv=argv,out=str(out),completion=str(out/'completion.json'))])])
        d.save(op,owner);d.save(base/'owner_identity.json',dict(oid,registration_sha256=d.sha(reg),config_sha256=d.sha(op)))
        lp=base/'stages/render/workers/development_render/launch.json';gp=base/'stages/render/gpu_admission.json'
        d.save(lp,dict(registration_sha256=d.sha(reg),resource='gpu',argv=argv,identity=wid,threads=6,affinity=list(range(6))))
        d.save(gp,dict(status='GPU_IDLE_CONFIRMED',registration_sha256=d.sha(reg)))
        api=SimpleNamespace(validate_config=lambda *a:None,command_entry=lambda a:Path(a[2]),
            same_identity=lambda a,b:all(a[k]==b[k] for k in ('pid','start_ticks','uid','argv')),
            identity=lambda pid:oid if pid==45 else wid)
        ctx=dict(cfg=dict(visual_owner_config=str(op),registration=str(reg),H_out=str(p),out=str(out)),reg={},owner=api)
        return ctx,cp,base,lp,gp

    def test_exclusive_gpu_parent_worker_and_idle_admission_required(self):
        with tempfile.TemporaryDirectory() as td:
            ctx,cp,base,lp,gp=self.owner_fixture(Path(td))
            with patch.object(d.os,'getppid',return_value=45),patch.object(d.os,'getpid',return_value=46), \
                 patch.object(d.os,'sched_getaffinity',return_value=set(range(6)),create=True),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                self.assertEqual(len(d.visual_admission(ctx,cp)['bindings']),3)
                old=d.read(lp);bad=copy.deepcopy(old);bad['identity']['start_ticks']+=1;lp.write_text(json.dumps(bad))
                with self.assertRaisesRegex(RuntimeError,'identity/resources'):d.visual_admission(ctx,cp)
                lp.write_text(json.dumps(old));gp.write_text(json.dumps({'status':'UNKNOWN','registration_sha256':d.sha(ctx['cfg']['registration'])}))
                with self.assertRaisesRegex(RuntimeError,'admission absent'):d.visual_admission(ctx,cp)

    def test_unsealed_registration_stops_before_module_import(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);reg=p/'reg.json';cfg=p/'cfg.json'
            d.save(reg,dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',allowed_stage_ids=['render'],source_stage_scope=d.SCOPE,
                           source_bindings={},input_bindings={}))
            d.save(cfg,dict(schema='H_DEVELOPMENT_RX_CONFIG_V1',registration=str(reg)))
            with patch.object(d,'module') as loader:
                with self.assertRaises((RuntimeError,KeyError)):d.load_registered(cfg)
                loader.assert_not_called()


if __name__=='__main__':unittest.main()
