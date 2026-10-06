"""Pure metadata/config registration tests; no source images or model calls."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).absolute().parent
for p in (HERE,HERE.parent/'phy_codec',HERE.parent/'payload',HERE.parent/'full_calibration',HERE.parent/'runner'):
    sys.path.insert(0,str(p))
import register_development_source as r
import stage_owner as owner


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value),encoding='utf-8')


class Registration(unittest.TestCase):
    def fixture(self,root):
        h=root/'H';h.mkdir();scope=root/'repo';scope.mkdir()
        old=dict(root=str(scope),out=str(h),branch='H',registration=str(h/'old.json'),owner_out=str(h/'oldowner'),
            budget_path=str(h/'ledger.sqlite'),budget_registration=str(h/'budget.json'),phase_limits=r.c.d.PHASES,
            qualification_started_unix=owner.QUALIFICATION_START,qualification_deadline_unix=owner.QUALIFICATION_DEADLINE,
            overall_deadline_unix=r.c.DEADLINE,cpu_affinities=[[0,1],[2,3]],gpu_threads=6,gpu_affinity=[4,5,6,7,8,9])
        visual=dict(root=old['root'],H_out=old['out'],ledger=old['budget_path'],budget_registration=old['budget_registration'],
            protocol=str(h/'protocol.json'),calibration_registration=str(h/'cal.json'),owner_module=str(root/'owner.py'),
            wait_module=str(root/'wait.py'),runtime_dir=str(root/'runtime'),uep_runtime=str(root/'uep'),native_runtime=str(root/'native'),
            var_source=str(root/'var'),dino_source=str(root/'dino'),static_closure_module=str(root/'static.py'),
            numerical_reference=str(root/'numeric.json'),source_driver_module=str(root/'original_source.py'))
        draft=dict(visual,schema='H_DEVELOPMENT100_PREPARATION_CONFIG_V1',kind='assets',
            phase_limits=old['phase_limits'],overall_deadline_unix=r.c.DEADLINE,stop_file=str(h/'STOP'),cpu_affinity=[0,1],
            numerical_reference_field=['numerical_runtime'])
        req=dict(execution_dir=str(h/'new_execution'),source_out=str(h/'new_source'),python=sys.executable,max_seconds=3600)
        return old,visual,draft,req

    def test_single_cpu_assets_and_gpu_codec_owner_configs_real_validator(self):
        with tempfile.TemporaryDirectory() as td:
            old,visual,draft,req=self.fixture(Path(td))
            for kind,name,resource,status in [('assets','development100_assets.py','cpu',r.c.ASSETS_DONE),
                                             ('source','development100_source_driver.py','gpu',r.c.SOURCE_DONE)]:
                cfg,oc=r.make_configs(req,dict(draft,kind=kind),old,kind,HERE/name)
                op=Path(td)/f'{kind}_owner.json';write(op,oc)
                reg=dict(branch='H',status='H_EXECUTION_REVISION_REGISTERED',owner_config_sha256=r.c.sha(op),
                    allowed_stage_ids=['source'],phase_limits=owner.read(op)['phase_limits'],source_bindings={str((HERE/name).resolve()):r.c.sha(HERE/name)})
                owner.validate_config(oc,reg,r.c.sha(op))
                self.assertEqual(len(oc['stages']),1);s=oc['stages'][0]
                self.assertEqual((s['id'],s['resource'],len(s['jobs'])),('source',resource,1))
                self.assertEqual(s['jobs'][0]['accepted_statuses'],[status])
                self.assertEqual(s['jobs'][0]['receipt_expect']['new_packet_decodes'],0)
                self.assertEqual(s['jobs'][0]['receipt_expect']['development_used'],True)
                self.assertEqual(cfg['owner_config'],str(Path(req['execution_dir'])/'owner_config.json'))
                self.assertEqual(cfg['out'],req['source_out'])

    def test_exact_original_resource_and_visual_settings_required(self):
        with tempfile.TemporaryDirectory() as td:
            old,v,cfg,_=self.fixture(Path(td));r.validate_resources(cfg,'assets',old,v)
            cfg['kind']='source';r.validate_resources(cfg,'source',old,v)
            for key,value in [('native_runtime','/changed'),('cpu_affinity',[7,8]),('overall_deadline_unix',r.c.DEADLINE+1)]:
                bad=copy.deepcopy(cfg);bad[key]=value
                with self.assertRaises(RuntimeError):r.validate_resources(bad,'source',old,v)

    def test_sources_require_normally_closed_assets_and_selector(self):
        with tempfile.TemporaryDirectory() as td:
            old,v,cfg,_=self.fixture(Path(td));cfg.update(selection_completion='selection_done',owner_module='original_owner')
            cfg['prerequisites']={name:dict(config=name+'_owner',registration=name+'_reg',launch=name+'_launch',completion=name+'_done')
                                  for name in ('selection','assets')}
            api=SimpleNamespace(raw_process_state=lambda _:None);calls=[]
            def closed(a,w,spec,module,expect,state):
                calls.append(expect);return dict(owner=old)
            with patch.object(r.c.d,'closed_batch',side_effect=closed):
                self.assertEqual(set(r.close_predecessors(cfg,'source',api,object())),{'selection','assets'})
            self.assertEqual(calls[0]['whole_policy_reselected'],False)
            self.assertEqual(calls[1]['source_codec_complete'],False)
            bad=copy.deepcopy(cfg);del bad['prerequisites']['assets']
            with self.assertRaisesRegex(RuntimeError,'Exact normal'):
                r.close_predecessors(bad,'source',api,object())

    def test_failed_or_live_predecessor_is_not_caught_as_normal(self):
        cfg=dict(prerequisites={'selection':dict(config='a',registration='b',launch='c',completion='d')},owner_module='e')
        with patch.object(r.c.d,'closed_batch',side_effect=RuntimeError('owner live')):
            with self.assertRaisesRegex(RuntimeError,'owner live'):
                r.close_predecessors(cfg,'assets',SimpleNamespace(raw_process_state=None),None)

    def test_original_visual_is_derived_from_bound_real_selector_request(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);vp=p/'render.json';req=p/'selector_request.json';cp=p/'selection_config.json'
            write(vp,{'root':'ORIGINAL'});write(req,{'render_config':{'path':str(vp),'sha256':r.c.sha(vp)}})
            write(cp,{'request_path':str(req)})
            closed=dict(owner={'stages':[{'jobs':[{'id':'full1000_selection','argv':['python','-B','selector.py','--config',str(cp)]}]}]},
                registration=dict(source_stage_scope='FULL1000_CALIBRATION_SELECTION_ONLY',allowed_stage_ids=['freeze'],
                    source_bindings={},input_bindings=r.c.bind((vp,req,cp))))
            visual,pins=r.original_visual(closed);self.assertEqual(visual,{'root':'ORIGINAL'});self.assertEqual(len(pins),3)
            write(vp,{'root':'CHANGED'})
            with self.assertRaisesRegex(RuntimeError,'Pinned file'):
                r.original_visual(closed)

    def test_request_pin_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.json';write(p,{'status':'frozen'})
            record={'path':str(p),'sha256':r.c.sha(p)};self.assertEqual(r.pin(record),p)
            write(p,{'status':'changed'})
            with self.assertRaises(RuntimeError):r.pin(record)


if __name__=='__main__':unittest.main()
