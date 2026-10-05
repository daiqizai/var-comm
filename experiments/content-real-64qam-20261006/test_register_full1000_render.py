"""Pure CPU visual-registration tests; no network, model or PHY calls."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import register_full1000_render as r
import register_partial1000_cpu as partial
import register_full1000_source as source
import register_full1000_assets as t
import test_register_full1000_assets as at
import test_register_partial1000_cpu as pt
import test_register_full1000_source as st


def final_fixture(whole=48000,part=36000):
    wb,ws=pt.budgets(whole);counts=copy.deepcopy(wb['phase_charged']);counts['partial_calibration']=part
    rows=copy.deepcopy(ws['ledger']['counts'])+[dict(phase='partial_calibration',kind='header',status='COMPLETE',count=18000)]
    if part>18000:rows.append(dict(phase='partial_calibration',kind='body',status='COMPLETE',count=part-18000))
    ledger=dict(ws['ledger'],counts=rows,charged=sum(counts.values()),phase_charged=counts,
        phase_remaining={p:t.PHASES[p]-n for p,n in counts.items()})
    done=dict(ledger=ledger,logical_packet_events=part,ledger_audit=dict(status='FULL_CPU_LEDGER_TRACE_MATCH',all_complete=True,paid_events=part))
    pb=dict(created=True,charged=sum(counts.values()),phase_charged=counts,unresolved=0,failed=0,development_remaining=13200)
    return dict(t=t,partial_helper=partial,whole_reg={'budget_before':at.budget()},whole_owner={'budget':wb},whole_done=ws,
        partial_reg={'budget_before':wb},partial_owner={'budget':pb},partial_done=done)


class VisualRegistrationTests(unittest.TestCase):
    def test_exact_two_closed_phase_budgets_and_failed_headers_no_free_calls(self):
        for whole,part in ((48000,36000),(47999,35999),(24000,18000)):
            args=final_fixture(whole,part);v=r.final_budget(**args)
            self.assertEqual(v['charged'],69960+whole+part);self.assertEqual(v['unresolved'],0)
            self.assertEqual(v['phase_charged']['development'],0)

    def test_partial_registration_must_start_at_exact_whole_close(self):
        args=final_fixture();args['partial_reg']=copy.deepcopy(args['partial_reg']);args['partial_reg']['budget_before']['charged']-=1
        with self.assertRaisesRegex(RuntimeError,'closed whole budget'):r.final_budget(**args)

    def test_all_headers_required_pending_or_other_phase_advance_blocks_visuals(self):
        for operation in ('header','pending','reserve','owner','count','duplicates'):
            args=final_fixture();ledger=args['partial_done']['ledger']
            if operation=='header':ledger['counts'][-2]['count']-=1
            elif operation=='pending':ledger['counts'][-1]['status']='RESERVED'
            elif operation=='reserve':ledger['phase_charged']['engineering_reserve']=1
            elif operation=='owner':args['partial_owner']['budget']['failed']=1
            elif operation=='count':args['partial_done']['ledger_audit']['paid_events']-=1
            else:ledger['counts'].append(copy.deepcopy(ledger['counts'][0]))
            with self.assertRaises(RuntimeError,msg=operation):r.final_budget(**args)

    def test_one_gpu_stage_original_six_threads_all_42000_frames_no_followup(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,_,_=pt.PartialRegistrationTests().fixture(root)
            whole=partial.verify_whole_closed(**args);old=whole['old'];oldpaths=args['oldpaths']
            e=root/'H/render_exec';out=root/'H/render';driver=Path(__file__).with_name('h_full_payload_render_driver.py').resolve()
            paths=dict(visual_driver_module=driver,rx_adapter_module=Path(__file__).with_name('h_full_payload_rx.py').resolve())
            sourcecfg={k:'/frozen/'+k for k in ('runtime_dir','uep_runtime','native_runtime','var_source','dino_source',
                'calibration_registration','numerical_reference','source_driver_module','static_closure_module','ledger','budget_registration','stop_file')}
            sourcecfg.update(root=str(root),H_out=old['out'],phase_limits=t.PHASES)
            spec={k:'/closed/'+k for k in ('config','registration','owner_config','launch','completion')}
            ctx=dict(old=old,cfg={'protocol':'/frozen/protocol'},oldpaths=oldpaths,cpu_batches={p:dict(spec) for p in ('whole_calibration','partial_calibration')})
            request=dict(execution_dir=str(e),render_out=str(out),python=sys.executable,max_seconds=7200)
            cfg,oc=r.make_configs(request,ctx,paths,{'cfg':sourcecfg})
            self.assertEqual(len(oc['stages']),1);stage=oc['stages'][0];self.assertEqual((stage['id'],stage['resource'],len(stage['jobs'])),('render','gpu',1))
            job=stage['jobs'][0];self.assertEqual(job['id'],'full1000_render');self.assertEqual(job['accepted_statuses'],['H_FULL1000_RX_COMPLETE'])
            self.assertEqual(job['receipt_expect']['phase_frame_counts'],{'whole_calibration':24000,'partial_calibration':18000})
            self.assertTrue(job['receipt_expect']['arithmetic_source_decode_complete']);self.assertFalse(job['receipt_expect']['policy_selection'])
            self.assertEqual(cfg['max_archive_bytes'],36*1024**3);self.assertEqual(cfg['cpu_batches'],ctx['cpu_batches'])
            for k in ('gpu_affinity','gpu_threads','cpu_affinities','phase_limits'):self.assertEqual(oc[k],old[k])
            reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',owner_config_sha256='test',allowed_stage_ids=['render'],phase_limits=t.PHASES,source_bindings=r.bind((driver,)))
            args['a'].validate_config(oc,reg,'test')
            self.assertNotIn('execution_registration_sha256',cfg)

    def visual_fixture(self,root):
        helper,base,reference,bound,actual,*_=st.SourceRegistrationTests().graph_fixture(root)
        flags=dict(threads=6,interop_threads=2,deterministic=True,matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,precision='highest',torch='original')
        cfg=dict(base,root=str(root),var_source=reference['inputs']['var_source'],dino_source=reference['inputs']['dino_source'])
        for key in ('static_closure_module','source_driver_module'):
            p=root/(key+'.py');p.write_text('# frozen\n');cfg[key]=str(p);bound[str(p)]=r.sha(p)
        for key,value in [('numerical_reference',dict(status='REAL_NATIVE_QUALIFICATION_PASS',synthetic=False,frozen_identity={'VAR':'same'},numerical_runtime=flags)),
                          ('calibration_registration',dict(identity={'VAR':'same'}))]:
            p=root/(key+'.json');at.rewrite(p,value);cfg[key]=str(p);bound[str(p)]=r.sha(p)
        ref=root/'visual_graph.json';at.rewrite(ref,reference)
        paths={'source_registrar_module':Path(source.__file__).resolve(),'source_visual_closure':ref}
        ctx=dict(source={'source_cfg':cfg,'source_done':dict(frozen_visual_identity={'VAR':'same'},numerical_runtime=flags)},bound=bound,old={'root':str(root),'gpu_threads':6})
        return ctx,paths,helper,actual

    def test_complete_visual_closure_new_files_added_without_changing_old(self):
        with tempfile.TemporaryDirectory() as td:
            ctx,paths,helper,actual=self.visual_fixture(Path(td))
            with patch.object(r,'module',side_effect=[source,helper]),patch.object(helper,'collect_bindings',return_value=actual):v=r.visual_inputs(ctx,paths)
            self.assertEqual(v['actual'],actual);self.assertTrue(v['delta']['missing']);self.assertFalse(v['delta']['changed'])

    def test_old_numeric_flags_identity_or_source_sha_change_is_not_waived(self):
        with tempfile.TemporaryDirectory() as td:
            ctx,paths,helper,actual=self.visual_fixture(Path(td))
            bad=copy.deepcopy(ctx);bad['source']['source_done']['numerical_runtime']['matmul_tf32']=True
            with self.assertRaisesRegex(RuntimeError,'FP32'):r.visual_inputs(bad,paths)
            bad=copy.deepcopy(ctx);bad['source']['source_done']['frozen_visual_identity']={'VAR':'different'}
            with self.assertRaisesRegex(RuntimeError,'identities'):r.visual_inputs(bad,paths)
            changed=dict(actual);changed[next(k for k in actual if k in ctx['bound'])]='0'*64
            with patch.object(r,'module',side_effect=[source,helper]),patch.object(helper,'collect_bindings',return_value=changed):
                with self.assertRaisesRegex(RuntimeError,'existing source changed'):r.visual_inputs(ctx,paths)

    def test_registration_refuses_nonlinux_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'request.json';at.rewrite(p,{'schema':'H_FULL1000_RENDER_REGISTRATION_REQUEST_V1'})
            with patch.object(r.sys,'platform','win32'):
                with self.assertRaisesRegex(RuntimeError,'Linux process'):r.register(p)
            self.assertEqual([x.name for x in Path(td).iterdir()],['request.json'])

    def test_registrar_never_calls_models_physical_decoder_or_scheduler(self):
        tree=ast.parse(Path(r.__file__).read_text());forbidden={'run_frame','receive_frame','decode_once','build_runtime','build_native','Popen','system','render_source'}
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,forbidden)


if __name__=='__main__':unittest.main()
