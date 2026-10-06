"""Registration construction and fail-closed qualification tests; no real inputs."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).absolute().parent
for p in (HERE,HERE.parent/'runner'):sys.path.insert(0,str(p))
import register_development_render as r
import h_development_render_driver as d
import stage_owner
import test_register_development_source as source_fixtures


class RegistrationTests(unittest.TestCase):
    def test_single_gpu_owner_uses_original_flags_and_real_validator(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);old,visual,source,unused=source_fixtures.Registration().fixture(root)
            paths=dict(cpu_driver_module=HERE/'h_development_driver.py',rx_adapter_module=HERE/'h_development_rx.py',
                       visual_driver_module=HERE/'h_development_render_driver.py')
            req=dict(execution_dir=str(root/'H/devrx_execution'),render_out=str(root/'H/devrx'),python=sys.executable,
                     max_seconds=7200,max_archive_bytes=6*1024**3)
            ctx=dict(old=old,source_cfg=source,spec={'config':'actual_CPU_driver'})
            cfg,owner=r.make_configs(req,ctx,paths);op=root/'owner.json';d.save(op,owner)
            reg=dict(branch='H',status='H_EXECUTION_REVISION_REGISTERED',owner_config_sha256=d.sha(op),allowed_stage_ids=['render'],
                     phase_limits=old['phase_limits'],source_bindings=d.bind((paths['visual_driver_module'],)))
            stage_owner.validate_config(owner,reg,d.sha(op))
            self.assertEqual(owner['gpu_threads'],6);self.assertEqual(len(owner['stages']),1)
            job=owner['stages'][0]['jobs'][0];self.assertEqual(job['id'],'development_render')
            self.assertEqual(job['receipt_expect']['frame_count'],5400)
            self.assertFalse(job['receipt_expect']['overall_development_complete'])
            self.assertFalse(job['receipt_expect']['MAIN_complete'])
            self.assertEqual(cfg['native_runtime'],source['native_runtime'])
            self.assertEqual(cfg['cpu_batch'],ctx['spec'])

    def test_qualification_requires_exact_new_entries_and_closed_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);entry=p/'entry.py';entry.write_text('# fixture');log=p/'tests.log';log.write_text('PASS')
            q=dict(status='H_DEVELOPMENT_RENDER_CPU_QUALIFICATION_PASS',GPU_used=False,new_packet_decodes=0,
                   results=[{'exit_code':0}],source_bindings=d.bind((entry,)),input_bindings={},outputs=d.bind((log,)))
            r.qualification(q,(entry,))
            for key,value in [('GPU_used',True),('new_packet_decodes',1),('results',[{'exit_code':1}]),('source_bindings',{})]:
                with self.assertRaises(RuntimeError):r.qualification(dict(q,**{key:value}),(entry,))
            log.write_text('changed')
            with self.assertRaises(RuntimeError):r.qualification(q,(entry,))

    def test_original_numeric_identity_requires_the_development_population(self):
        flags=dict(threads=6,deterministic=True,matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,precision='highest')
        n=dict(status='REAL_NATIVE_QUALIFICATION_PASS',synthetic=False,frozen_identity={'id':'shared'},numerical_runtime=flags)
        done=dict(frozen_visual_identity={'id':'shared'},numerical_runtime=flags)
        self.assertEqual(d.visual_identity({},done,{'identity':{'id':'shared'}},n,{'identity':{'id':'shared'}}),flags)
        with self.assertRaisesRegex(RuntimeError,'identities differ'):
            d.visual_identity({},done,{'identity':{'id':'other'}},n,{'identity':{'id':'shared'}})
        with self.assertRaisesRegex(RuntimeError,'flags differ'):
            d.visual_identity({},dict(done,numerical_runtime=dict(flags,threads=2)),{'identity':{'id':'shared'}},n,{'identity':{'id':'shared'}})

    def test_no_registration_or_launch_after_missing_pins(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'request.json';d.save(p,dict(schema='H_DEVELOPMENT_RENDER_REGISTRATION_REQUEST_V1'))
            with patch.object(r.sys,'platform','linux'),patch.object(r.time,'time',return_value=0),patch.object(r,'inspect_inputs') as inspect:
                with self.assertRaises(KeyError):r.register(p)
                inspect.assert_not_called()


if __name__=='__main__':unittest.main()
