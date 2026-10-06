"""Metadata/closure and persistence tests; no model, real image or packet calls."""
import copy
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h_development_metrics_driver as m


def budget():
    return dict(charged=164760,failed=0,unresolved=0,phase_charged={'development':10800},development_remaining=2400)


def assets_fixture(root):
    paths={name:str(root/(name+'.json')) for name in m.METRIC_PATHS}
    for p in paths.values():Path(p).write_text('{}')
    flags=dict(threads=6,interop_threads=2,deterministic=True,matmul_tf32=False,cudnn_tf32=False,
        precision='highest',cudnn_deterministic=True,cudnn_benchmark=False)
    Path(paths['metrics_registration']).unlink();m.save(paths['metrics_registration'],dict(synthetic=False,
        modelmanifest_sha256=m.sha(paths['modelmanifest']),numerical_runtime=flags))
    Path(paths['models_qualification']).unlink();m.save(paths['models_qualification'],dict(status='REAL_MODEL_WEIGHTS_QUALIFICATION_PASS',
        modelmanifest_sha256=m.sha(paths['modelmanifest'])))
    Path(paths['independent_binding']).unlink();m.save(paths['independent_binding'],dict(
        status='FROZEN_IDENTITY_CLARIFICATION_BEFORE_ANY_H_C_VALIDATION',used_for_selection=False,
        reference_implementation_sha256=m.sha(paths['validation_module']),weights_sha256=m.sha(paths['convnext_weights'])))
    value=dict(status='H_ORIGINAL_METRIC_ASSETS_BOUND',paths=paths,source_bindings={},input_bindings=m.bind(paths.values()))
    p=root/'metric_assets.json';m.save(p,value);return p,value,flags


def closure_fixture(root):
    p=lambda n:str(root/n)
    for n in ('owner.py','wait.py','cpu.py','rx.py','renderer.py','old_source.py','owner_config.json','launch.json','images.bin','budget.json'):
        Path(p(n)).write_text('fixture')
    graph=dict(status='EXACT_SOURCE_CLOSURE_MATCH',source_bindings=m.bind((p('old_source.py'),)))
    m.save(p('graph.json'),graph)
    cfg=dict(registration=p('registration.json'),visual_owner_config=p('owner_config.json'),owner_module=p('owner.py'),
        wait_module=p('wait.py'),cpu_driver_module=p('cpu.py'),rx_adapter_module=p('rx.py'),visual_driver_module=p('renderer.py'),
        visual_source_closure=p('graph.json'),ledger=p('ledger.sqlite'),budget_registration=p('budget.json'),phase_limits={},
        cpu_batch={'actual':'CPU'},out=p('render'))
    m.save(p('render_config.json'),cfg)
    reg=dict(source_bindings=m.bind((p('owner.py'),p('wait.py'),p('cpu.py'),p('rx.py'),p('renderer.py'),p('old_source.py'))),
        input_bindings=m.bind((p('owner_config.json'),p('render_config.json'),p('graph.json'))))
    m.save(p('registration.json'),reg)
    done=dict(visual_source_bindings=graph['source_bindings'],outputs=m.bind((p('images.bin'),)),source_ids=['i'+str(i) for i in range(100)],
        budget_before=budget(),budget_after=budget())
    m.save(p('science_completion.json'),done)
    old=dict(stages=[dict(id='render',jobs=[dict(id='development_render',argv=['ORIGINAL_UM','-B',p('renderer.py'),'--config',p('render_config.json')])])])
    closed=dict(done=done,owner=old,owner_done={'budget':budget()},bindings={})
    group=dict(before=budget(),source_ids=done['source_ids'],bindings={},population={},schedule=[],context={'assets':{},'manifest':{}})
    calls=[]
    def closed_batch(*args):calls.append(args);return closed
    owner=SimpleNamespace(raw_process_state=lambda pid:None,budget_snapshot=lambda *a,**k:budget())
    modules={p('owner.py'):owner,p('wait.py'):object(),p('cpu.py'):SimpleNamespace(closed_batch=closed_batch),
             p('rx.py'):SimpleNamespace(verify_cpu_closed=lambda *a:group)}
    spec=dict(config=p('render_config.json'),owner_config=p('owner_config.json'),registration=p('registration.json'),
        launch=p('launch.json'),completion=p('science_completion.json'))
    return spec,modules,closed,calls


class MetricDriverTests(unittest.TestCase):
    def test_original_asset_manifest_flags_and_all_eight_paths(self):
        with tempfile.TemporaryDirectory() as td:
            p,value,flags=assets_fixture(Path(td));self.assertEqual(m.metric_assets(p)[2],flags)
            changed=copy.deepcopy(value);changed['paths'].pop('convnext_weights')
            cp=Path(td)/'missing.json';m.save(cp,changed)
            with self.assertRaisesRegex(RuntimeError,'asset manifest'):m.metric_assets(cp)
            Path(value['paths']['convnext_weights']).write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'Changed bound file'):m.metric_assets(p)

    def test_normal_closure_uses_exact_historical_graph_and_original_receipt_api(self):
        with tempfile.TemporaryDirectory() as td:
            spec,mods,closed,calls=closure_fixture(Path(td))
            # A later source is irrelevant to the already sealed visual graph.
            (Path(td)/'new_metrics.py').write_text('new CPU implementation')
            with patch.object(m,'module',side_effect=lambda p,n:mods[p]):ctx=m.normal_render_closed(spec)
            self.assertEqual(ctx['before'],budget());self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][4]['frame_count'],5400)
            self.assertIs(calls[0][4]['unified_neural_metrics_run'],False)
            self.assertEqual(calls[0][2]['config'],spec['owner_config'])
            closed['done']['visual_source_bindings']={}
            with patch.object(m,'module',side_effect=lambda p,n:mods[p]),self.assertRaisesRegex(RuntimeError,'Historical executed'):
                m.normal_render_closed(spec)

    def test_failed_or_live_predecessor_is_not_replaced_by_science_completion(self):
        with tempfile.TemporaryDirectory() as td:
            spec,mods,closed,calls=closure_fixture(Path(td))
            def blocked(*args):raise RuntimeError('Predecessor child is failed/live/unreaped')
            mods[str(Path(td)/'cpu.py')].closed_batch=blocked
            with patch.object(m,'module',side_effect=lambda p,n:mods[p]),self.assertRaisesRegex(RuntimeError,'failed/live/unreaped'):
                m.normal_render_closed(spec)

    def test_foreign_gpu_and_sustained_heat_are_not_ignored(self):
        health=m.GPUHealth()
        with patch.object(m.subprocess,'check_output',return_value=str(os.getpid()+1)),self.assertRaisesRegex(RuntimeError,'Foreign'):
            health.check(force=True)
        health=m.GPUHealth()
        with patch.object(m.subprocess,'check_output',side_effect=[str(os.getpid()),'86, Not Active']*3):
            health.check(force=True);health.check(force=True)
            with self.assertRaisesRegex(RuntimeError,'thermal'):health.check(force=True)

    def test_reference_checks_source_boundary_and_scoring_keeps_each_input(self):
        calls=[];backend=SimpleNamespace(identity='frozen',prepare=lambda x:calls.append(('prepare',x)),score=lambda *x:calls.append(('score',x)))
        health=SimpleNamespace(check=lambda:calls.append('health'))
        g=m.GuardedBackend(backend,health,lambda:calls.append('boundary'))
        g.prepare('original');g.score('original','actualRGB')
        self.assertEqual(calls,['boundary','health',('prepare','original'),'health',('score',('original','actualRGB'))])

    def test_checkpoint_seals_54_rows_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            rows=[dict(source_index=4,source_id='frozen',noise_seed=6201+i%3,slot=i//3) for i in range(54)]
            outputs=m.write_source(td,4,'frozen',rows,{'unique_metric_pairs':10},{},'a'*64,'frozen-evaluator')
            m.verify(outputs);cp=m.read(Path(td)/'source_checkpoints/0004.json')
            self.assertEqual(cp['frame_count'],54);self.assertEqual(cp['new_packet_decodes'],0)
            with self.assertRaises(FileExistsError):m.write_source(td,4,'frozen',rows,{}, {},'a'*64,'frozen-evaluator')

    def test_unregistered_entry_stops_before_models(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);m.save(p/'reg.json',dict(status='not admitted'))
            m.save(p/'config.json',dict(registration=str(p/'reg.json'),schema='wrong'))
            with patch.object(m,'build_backend',side_effect=AssertionError('must not load')):
                with self.assertRaises(RuntimeError):m.load_registered(p/'config.json')

    def test_actual_loader_constructor_arguments_and_returned_asset_closure(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);policy=p/'frozen_policy.json';policy.write_text('{}');weight=p/'weight';weight.write_text('fixed')
            flags={'threads':6};bindings=m.bind((weight,));suite=(1,2,3,4,5,bindings,flags);calls=[]
            loader=SimpleNamespace(load_suite=lambda root,device:suite,numeric_flags=lambda torch:flags)
            validation=SimpleNamespace(ConvNeXtValidation=lambda path,digest,**kwargs:calls.append((path,digest,kwargs)) or 'classifier')
            adapter=SimpleNamespace(CONVNEXT_SHA=m.sha(weight),SuiteBackend=lambda *args:args)
            cfg=dict(root=td,suite_module=str(p/'suite.py'),replay_module=str(p/'replay.py'),
                validation_module=str(p/'validation.py'),convnext_weights=str(weight))
            ctx=dict(cfg=cfg,adapter=adapter,bound=m.merge(bindings,m.bind((policy,))),metric_flags=flags,
                group={'cfg':{'finalized':str(policy)}})
            def mod(path,name):return loader if path==cfg['suite_module'] else validation
            with patch.object(m,'module',side_effect=mod),patch.dict(sys.modules,{'torch':SimpleNamespace()}):
                backend=m.build_backend(ctx)
                self.assertEqual(backend[0],suite);self.assertEqual(backend[1],'classifier')
                self.assertEqual(calls[0][2],dict(device='cuda:0',stage='development',policy_path=str(policy),policy_sha256=m.sha(policy)))
                ctx['bound'].pop(str(weight))
                with self.assertRaisesRegex(RuntimeError,'not preregistered'):m.build_backend(ctx)


if __name__=='__main__':unittest.main()
