"""CPU-only persistence and admission tests; no Torch/models/PHY/registration."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import p600_received_replay_core as core
import p600_received_replay_driver as d
from test_p600_received_replay_core import fixture, FakeClassifier


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.base = fixture()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.ctx,self.pixels,self.image,self.z,self.docs = copy.deepcopy(self.base)

    def tearDown(self): self.tmp.cleanup()

    def result(self,index=0):
        return core.replay_source(self.ctx,index,self.pixels,
            latents=SimpleNamespace(batch=lambda i,s:np.stack([self.z]*3)),
            decoder=lambda batch:np.stack([self.image]*3),classifier=FakeClassifier())

    def test_real_core_to_writer_and_archive_reader_preserves_scalar_grid(self):
        result = self.result(); out = self.root/'replay'
        sourcepin = self.root/'source.txt'; sourcepin.write_text('sealed')
        rows,outputs,nbytes = d.write_source(out,result,d.bind((sourcepin,)),'a'*64,core)
        reread = d.read_source_output(out,0,self.ctx['source_ids'][0],outputs,'a'*64,core)
        self.assertEqual(rows,reread); self.assertGreater(nbytes,6*3*256*256*4)
        self.assertEqual(len(outputs),3)
        for r in reread:
            original = self.ctx['scalar'][core.row_key(r)]
            self.assertTrue(all(r[k] == v for k,v in original.items()))
            self.assertEqual(r['p_replay_image_archive'],str(out/'images/0000.npz'))
        with np.load(out/'images/0000.npz',allow_pickle=False) as z:
            self.assertEqual(len(z.files),6)
            self.assertTrue(all(z[k].dtype == np.float32 for k in z.files))

    def test_unchanged_hash_requires_real_RGB_not_cosmetic_new_manifest(self):
        out = self.root/'replay'; rows,outputs,_ = d.write_source(out,self.result(),{},'a'*64,core)
        path = out/'images/0000.npz'
        with np.load(path,allow_pickle=False) as z: arrays = {k:z[k].copy() for k in z.files}
        arrays[next(iter(arrays))][0,0,0] += np.float32(.125)
        with path.open('wb') as f: np.savez(f,**arrays)
        # Even updating file-level receipts cannot pass the original per-RGB SHA.
        outputs[str(path)] = d.sha(path)
        cp = out/'source_checkpoints/0000.json'; cpvalue = d.read(cp)
        cpvalue['outputs'][str(path)] = d.sha(path); cp.write_text(json.dumps(cpvalue))
        outputs[str(cp)] = d.sha(cp)
        with self.assertRaisesRegex(RuntimeError,'Archived RGB parity'):
            d.read_source_output(out,0,self.ctx['source_ids'][0],outputs,'a'*64,core)

    def test_source_input_change_or_wrong_registration_is_rejected(self):
        out = self.root/'replay'; p = self.root/'input'; p.write_text('old')
        _,outputs,_ = d.write_source(out,self.result(),d.bind((p,)),'a'*64,core)
        with self.assertRaisesRegex(RuntimeError,'checkpoint differs'):
            d.read_source_output(out,0,self.ctx['source_ids'][0],outputs,'b'*64,core)
        p.write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'Changed bound file'):
            d.read_source_output(out,0,self.ctx['source_ids'][0],outputs,'a'*64,core)

    def test_owner_precreated_empty_output_only_and_no_retry(self):
        out = self.root/'job'; out.mkdir()
        d.claim_output(out,'a'*64)
        self.assertEqual(d.read(out/'attempt.json')['status'],'P600_RECEIVED_REPLAY_ATTEMPT')
        with self.assertRaisesRegex(RuntimeError,'no retry'): d.claim_output(out,'a'*64)
        with self.assertRaises(FileExistsError): d.save(out/'attempt.json',{})

    def source_assets(self):
        pop = dict(stage='m1_development',calibration_or_development='m1_development',
            source_ids=self.ctx['source_ids'],preprocessing_ids=self.ctx['preprocessing_ids'],
            data_bindings=[dict(index=i,rgb_sha256=self.ctx['preprocessing_ids'][i]) for i in range(100)])
        ap = self.root/'pixels.npz'
        with ap.open('xb') as f: np.savez(f,pixels=self.pixels,tokens=np.zeros(680,np.int64))
        outputs = d.bind((ap,)); records = []
        for i in range(100):
            cp = self.root/'assets'/f'{i:04d}.json'
            row = dict(source_index=i,source_id=pop['source_ids'][i],preprocessing_id=pop['preprocessing_ids'][i],
                evaluation_class_index=1,archive=str(ap),original_development_data_binding=pop['data_bindings'][i])
            d.save(cp,dict(row,outputs=d.bind((ap,))))
            outputs[str(cp)] = d.sha(cp); records.append(dict(row,checkpoint=str(cp),checkpoint_sha256=d.sha(cp)))
        manifest = dict(status='H_DEVELOPMENT100_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE',
            source_count=100,source_ids=pop['source_ids'],records=records)
        done = dict(status=manifest['status'],source_count=100,outputs=outputs)
        return dict(population=pop,manifest=manifest,assets=done)

    def test_pre_model_source100_identity_and_actual_original_label_key(self):
        assetctx = self.source_assets()
        d.original_assets(assetctx,self.ctx)
        pixels,inputs = d.load_pixels(assetctx,99)
        np.testing.assert_array_equal(pixels,self.pixels); self.assertEqual(len(inputs),2)
        assetctx['manifest']['records'][99]['checkpoint_sha256'] = '0'*64
        with self.assertRaisesRegex(RuntimeError,'checkpoint missing/unsealed'):
            d.original_assets(assetctx,self.ctx)

    def test_source_pixels_byte_mismatch_is_not_repaired(self):
        assetctx = self.source_assets(); path = Path(assetctx['manifest']['records'][0]['archive'])
        wrong = self.pixels.copy(); wrong[0,0,0] = 1
        with path.open('wb') as f: np.savez(f,pixels=wrong)
        cp = Path(assetctx['manifest']['records'][0]['checkpoint']); value = d.read(cp)
        value['outputs'] = d.bind((path,)); cp.write_text(json.dumps(value))
        assetctx['manifest']['records'][0]['checkpoint_sha256'] = d.sha(cp)
        assetctx['assets']['outputs'].update(d.bind((cp,path)))
        with self.assertRaisesRegex(RuntimeError,'Original pixel bytes'):
            d.load_pixels(assetctx,0)

    def test_normal_predecessor_requires_original_closed_batch_not_just_absentPID(self):
        config = self.root/'metricconfig.json'; reg = self.root/'reg.json'; owner = self.root/'owner.json'
        launch = self.root/'launch.json'; receipt = self.root/'completion.json'; entry = self.root/'metric.py'
        entry.write_text('# bound fixture\n'); owner.write_text('{}'); launch.write_text('{}'); receipt.write_text('{}')
        d.save(config,dict(metric_driver_module=str(entry),registration=str(reg),visual_owner_config=str(owner),owner_module='oldowner'))
        d.save(reg,dict(source_bindings=d.bind((entry,)),input_bindings={}))
        old = dict(stages=[dict(id='development',resource='gpu',jobs=[dict(id='development_metrics',
            argv=['/UM/python','-B',str(entry),'--config',str(config)])])])
        before = dict(charged=164760)
        done = dict(source_ids=self.ctx['source_ids'],budget_before=before,budget_after=before,outputs={})
        close = dict(owner=old,done=done,owner_done=dict(budget=before),bindings={})
        calls = []
        def closed_batch(*args): calls.append(args); return close
        ctx = dict(cpu=SimpleNamespace(closed_batch=closed_batch),owner=SimpleNamespace(raw_process_state=lambda pid:None),
            wait=object(),source_ids=self.ctx['source_ids'],before=before,bindings={})
        md = SimpleNamespace(DONE='H_DEVELOPMENT_METRICS_COMPLETE',load_registered=lambda path:ctx)
        spec = dict(config=str(config),registration=str(reg),owner_config=str(owner),launch=str(launch),completion=str(receipt))
        with patch.object(d,'module',return_value=md):
            result = d.original_metrics_closed(spec,str(entry))
        self.assertEqual(len(calls),1); self.assertEqual(calls[0][4]['frame_count'],5400)
        self.assertEqual(calls[0][4]['unified_neural_metrics_run'],True)
        self.assertIs(result['closed'],close)
        md.load_registered = lambda path: (_ for _ in ()).throw(RuntimeError('live metric worker'))
        with patch.object(d,'module',return_value=md), self.assertRaisesRegex(RuntimeError,'live metric worker'):
            d.original_metrics_closed(spec,str(entry))

    def test_preflight_and_run_have_no_channel_noise_or_ledger_mutation_symbols(self):
        import ast
        tree = ast.parse(Path(d.__file__).read_text(encoding='utf-8'))
        forbidden = {'decode_once','receive_frame','receive_body','transmit','awgn','randn','randn_like','BudgetLedger'}
        calls = {n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
            for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
        self.assertFalse(calls & forbidden)
        funcs = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        preflight = ast.unparse(funcs['inspect_inputs'])
        self.assertNotIn('build_backends(',preflight); self.assertNotIn('save(',preflight)

    def test_pre_admission_rejects_mutated_binding_before_importing_any_model(self):
        p = self.root/'bound_input.json'; p.write_text('{}'); bound = d.bind((p,)); p.write_text('{"changed":true}')
        with patch.object(d,'module') as loaded, self.assertRaisesRegex(RuntimeError,'Changed bound file'):
            d.inspect_inputs(dict(schema='P600_RECEIVED_REPLAY_CONFIG_V1'),bound)
        loaded.assert_not_called()
        p.write_text('{}'); q = self.root/'same_identity_different_scalar.json'; q.write_text('{"psnr":999}')
        config = dict(scalar_inventory=str(p),metrics_registration=str(p),numerical_reference=str(p))
        audit = dict(input_bindings=d.bind((p,)))
        d.audited_input_paths(config,audit)
        config['scalar_inventory'] = str(q)
        with self.assertRaisesRegex(RuntimeError,'exact latent-audited file'):
            d.audited_input_paths(config,audit)


if __name__ == '__main__': unittest.main()
