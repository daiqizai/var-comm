"""Metadata-only registrar tests; no model, GPU, packet, launch or real asset reads."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).absolute().parent
for p in (HERE,HERE.parent/'runner'):sys.path.insert(0,str(p))
import register_development_metrics as r
import h_development_metrics_driver as d
import stage_owner
import test_register_development_source as source_fixtures


class MetricRegistrationTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);old,visual,source,_=source_fixtures.Registration().fixture(root)
        prior={k:source[k] for k in ('root','H_out','owner_module','wait_module','ledger','budget_registration',
                                     'phase_limits','stop_file','protocol')}
        prior.update(cpu_driver_module=str(HERE/'h_development_driver.py'),
                     visual_driver_module=str(HERE/'h_development_render_driver.py'))
        req=dict(execution_dir=str(root/'H/metrics_execution'),metrics_out=str(root/'H/metrics'),
                 python=sys.executable,max_seconds=7200)
        paths=dict(metric_driver_module=HERE/'h_development_metrics_driver.py',
                   metric_adapter_module=HERE/'h_development_metric_adapter.py',metric_assets=root/'metric_assets.json')
        ma=dict(paths={k:str(root/(k+'.json')) for k in d.METRIC_PATHS})
        spec={k:str(root/(k+'.json')) for k in r.BATCH_KEYS}
        ctx=dict(old=old,render_cfg=prior,spec=spec,metric_assets=ma)
        return req,paths,ctx

    def test_real_owner_validator_admits_only_one_existing_development_GPU_stage(self):
        with tempfile.TemporaryDirectory() as td:
            req,paths,ctx=self.fixture(td);cfg,owner=r.make_configs(req,ctx,paths)
            op=Path(td)/'owner.json';d.save(op,owner)
            reg=dict(branch='H',status='H_EXECUTION_REVISION_REGISTERED',owner_config_sha256=d.sha(op),
                     allowed_stage_ids=['development'],phase_limits=owner['phase_limits'],
                     source_bindings=d.bind((paths['metric_driver_module'],)))
            stage_owner.validate_config(owner,reg,d.sha(op))
            self.assertNotIn('metrics',stage_owner.STAGE_IDS)
            self.assertEqual(len(owner['stages']),1);stage=owner['stages'][0]
            self.assertEqual((stage['id'],stage['resource']),('development','gpu'))
            self.assertEqual(owner['gpu_threads'],6);self.assertEqual(owner['gpu_affinity'],ctx['old']['gpu_affinity'])
            job=stage['jobs'][0];self.assertEqual(job['id'],'development_metrics')
            self.assertEqual(job['argv'],[req['python'],'-B',str(paths['metric_driver_module']),'--config',str(Path(req['execution_dir'])/'metrics_config.json')])
            self.assertEqual(job['receipt_expect']['frame_count'],5400)
            for key in ('MAIN_complete','P_complete','overall_development_complete','H_full_delivery_claimed',
                        'statistical_aggregation_run','online_timing_measured'):
                self.assertIs(job['receipt_expect'][key],False)
            self.assertEqual(set(d.REQUIRED)-set(cfg),set())
            self.assertEqual(cfg['render_batch'],ctx['spec'])

    def test_clone_preserves_source_owner_and_asset_paths_without_graph_reenumeration(self):
        with tempfile.TemporaryDirectory() as td:
            req,paths,ctx=self.fixture(td);old=copy.deepcopy(ctx)
            cfg,owner=r.make_configs(req,ctx,paths)
            self.assertEqual(ctx,old)
            for key in d.METRIC_PATHS:self.assertEqual(cfg[key],ctx['metric_assets']['paths'][key])
            self.assertEqual(cfg['render_driver_module'],ctx['render_cfg']['visual_driver_module'])
            self.assertNotIn('visual_source_closure',cfg)
            text=Path(r.__file__).read_text(encoding='utf-8');tree=ast.parse(text)
            calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            self.assertNotIn('collect_bindings',calls)
            self.assertFalse(set(calls)&{'Popen','load_suite','score','launch','run'})

    def test_qualification_requires_actual_interpreter_all_entries_and_zero_science(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);entry=p/'entry.py';entry.write_text('# fixture');log=p/'tests.log';log.write_text('PASS')
            q=dict(status='H_DEVELOPMENT_METRICS_CPU_QUALIFICATION_PASS',GPU_used=False,new_packet_decodes=0,
                   real_metric_calls=0,python=sys.executable,results=[{'exit_code':0}],source_bindings=d.bind((entry,)),
                   input_bindings={},outputs=d.bind((log,)))
            r.qualification(q,(entry,),sys.executable)
            for key,value in [('GPU_used',True),('new_packet_decodes',1),('real_metric_calls',1),
                              ('python','/wrong/ldpc/bin/python'),('source_bindings',{}),('results',[{'exit_code':1}])]:
                with self.assertRaises(RuntimeError):r.qualification(dict(q,**{key:value}),(entry,),sys.executable)
            log.write_text('changed')
            with self.assertRaises(RuntimeError):r.qualification(q,(entry,),sys.executable)

    def test_sha_pin_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'actual.json';p.write_text('{}');v=dict(path=str(p),sha256=d.sha(p))
            self.assertEqual(r.pin(v),p);p.write_text('{"changed":1}')
            with self.assertRaises(RuntimeError):r.pin(v)

    def inspect_fixture(self,root):
        root=Path(root);req,paths,ctx=self.fixture(root)
        spec={}
        for key in r.BATCH_KEYS:
            p=root/(key+'.json');d.save(p,{'role':key});spec[key]=dict(path=str(p),sha256=d.sha(p))
        req['render_batch']=spec
        ctx.update(before=dict(charged=164760,phase_charged={'development':10800},development_remaining=2400,unresolved=0,failed=0),
                   source_ids=[f's{i}' for i in range(100)],schedule=list(range(18)))
        ctx['old']['stages']=[dict(id='render',resource='gpu',jobs=[dict(argv=[sys.executable,'-B','frozen_render.py'])])]
        paths['prepared_qualification']=root/'q.json';d.save(paths['metric_assets'],{'prepared':True})
        q=dict(status='H_DEVELOPMENT_METRICS_CPU_QUALIFICATION_PASS',GPU_used=False,new_packet_decodes=0,
               real_metric_calls=0,python=sys.executable,results=[{'exit_code':0}],outputs={},
               source_bindings=d.bind((Path(r.__file__).absolute(),paths['metric_driver_module'],paths['metric_adapter_module'])),
               input_bindings=d.bind((paths['metric_assets'],)))
        d.save(paths['prepared_qualification'],q)
        return req,paths,ctx

    def test_closed_render_budget_reserved_MAIN_and_original_python_are_hard_gates(self):
        with tempfile.TemporaryDirectory() as td:
            req,paths,ctx=self.inspect_fixture(td)
            with patch.object(d,'normal_render_closed',return_value=ctx),patch.object(d,'metric_assets',
                     return_value=(ctx['metric_assets'],{},dict(threads=6,interop_threads=2))):
                got=r.inspect_inputs(req,paths);self.assertEqual(got['python'],sys.executable)
                with self.assertRaisesRegex(RuntimeError,'interpreter'):
                    r.inspect_inputs(dict(req,python='/other/bin/python'),paths)
                for key,value in [('charged',164761),('development_remaining',2399),('failed',1),('unresolved',1)]:
                    old=ctx['before'][key];ctx['before'][key]=value
                    with self.assertRaisesRegex(RuntimeError,'budget'):r.inspect_inputs(req,paths)
                    ctx['before'][key]=old
                ctx['before']['phase_charged']['development']=10801
                with self.assertRaisesRegex(RuntimeError,'budget'):r.inspect_inputs(req,paths)

    def test_metric_manifest_must_be_bound_by_qualification(self):
        with tempfile.TemporaryDirectory() as td:
            req,paths,ctx=self.inspect_fixture(td);q=d.read(paths['prepared_qualification']);q['input_bindings']={}
            paths['prepared_qualification'].write_text(__import__('json').dumps(q))
            with patch.object(d,'normal_render_closed',return_value=ctx),patch.object(d,'metric_assets',
                     return_value=(ctx['metric_assets'],{},dict(threads=6,interop_threads=2))):
                with self.assertRaisesRegex(RuntimeError,'asset manifest'):r.inspect_inputs(req,paths)

    def test_wrong_or_missing_request_is_rejected_before_directory_or_any_launch(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'request.json';d.save(p,dict(schema='wrong'))
            with patch.object(r.sys,'platform','linux'),patch.object(r.time,'time',return_value=0),patch.object(r,'inspect_inputs') as inspect:
                with self.assertRaisesRegex(RuntimeError,'request'):r.register(p)
                inspect.assert_not_called()
            self.assertEqual(set(Path(td).iterdir()),{p})


if __name__=='__main__':unittest.main()
