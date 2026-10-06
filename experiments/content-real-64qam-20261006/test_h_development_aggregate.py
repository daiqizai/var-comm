"""Only synthetic statistics, JSON seals and rejection gates; no scientific data."""
from collections import defaultdict
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock

import h_development_aggregate as a


def fixture():
    schedule=[];ids=[f's{i}' for i in range(100)]
    entries=[(a.ROLES[0],arm,snr) for arm in a.ARMS for snr in (13,19)]
    entries += [(a.ROLES[1],'H64-RAW-COMPLETE-STATE-CONTROL',snr) for snr in (13,19)]
    entries += [(a.ROLES[2],arm,snr) for arm in a.ARMS for snr in (13,19)]
    for slot,(role,arm,snr) in enumerate(entries):
        schedule.append(dict(development_slot=slot,role=role,phase='development',status='FROZEN_POLICY_READY',
            candidate=dict(candidate_id=f'c{slot}',arm=arm,snr_db=snr)))
    plan=a.comparison_plan(schedule);rows=[]
    for i,sid in enumerate(ids):
        for slot,s in enumerate(schedule):
            for n in a.statistics.SEEDS:
                good=n!=6203;fixed=s['role']==a.ROLES[2]
                row=dict(point_id=f'H18_SLOT_{slot:02d}',development_slot=slot,source_index=i,source_id=sid,
                    noise_seed=n,N=1024,role=s['role'],**s['candidate'],used_for_selection=False,holdout_used=False,
                    psnr_db=20+slot*.1+i*.001,lpips_alex=.2-slot*.001,nominal_rate='1/2',
                    tx_target_m=7 if fixed else 9,tx_actual_m=7 if fixed else 8,tx_actual_K=0,tx_source_mode='raw',
                    tx_fell_back=not fixed,tx_profile_id=slot,source_status='RAW_SOURCE_DECODED' if good else 'WIRE_REJECT_GRAY',
                    rx_summary={'source_decode_complete':True},rx_accepted_m=7 if good else None,rx_accepted_K=0 if good else None,
                    rx_declared_profile_id=slot if good else None,rx_canonical_accepted=good,gray=not good,true_class=1,
                    resnet50_source_prediction=1,convnext_source_prediction=1,resnet50_prediction=1 if good else 2,
                    convnext_prediction=1 if good else 2,evaluation_only=dict(header_correct=good,
                    parsed_wire_matches_transmission=good,accepted_wire_mismatch=False,canonical_decode_invalid=False,
                    reconstructed_after_accepted_wire_mismatch=False))
                rows.append(row)
    return schedule,ids,rows,plan


class AggregationTests(unittest.TestCase):
    def test_predetermined_catalog_has26_sameSNR_contrasts_and_preserves_roles(self):
        schedule,ids,rows,plan=fixture()
        self.assertEqual(len(plan['comparisons']),26)
        self.assertEqual([len([c for c in plan['catalog'] if c['role']==r]) for r in a.ROLES],[8,2,8])
        self.assertTrue(all(plan['points'][c['method']]['snr_db']==plan['points'][c['reference']]['snr_db'] for c in plan['comparisons']))
        changed=copy.deepcopy(rows);changed[0]['psnr_db']=-100000
        self.assertEqual(a.comparison_plan(schedule),plan)  # Scores never enter selection.
        self.assertEqual(sum(c['kind']=='PARTIAL_MINUS_RAW64_WHOLE' for c in plan['comparisons']),2)

    def test_source_means_original_bootstrap_exact_constant_contrast_no_seed_relabel(self):
        schedule,ids,rows,plan=fixture()
        result=a.summarize(rows,ids,plan,('psnr_db','lpips_alex'))
        self.assertEqual(len(result['summary']),36);self.assertEqual(len(result['paired']),52)
        first=result['paired'][0]
        self.assertAlmostEqual(first['mean'],.2);self.assertAlmostEqual(first['ci_low'],.2);self.assertAlmostEqual(first['ci_high'],.2)
        self.assertEqual(first['bootstrap_seed'],2026100605);self.assertEqual(first['bootstrap_replicates'],10000)
        self.assertEqual(first['bootstrap_unit'],'source after mean of three registered noises')
        self.assertEqual(len(result['source_means']),3600);self.assertFalse(result['policy_selection'])

    def test_TX100_RX300_denominators_keep_gray_and_wrong_accepted(self):
        _,ids,rows,plan=fixture()
        rows[0]['evaluation_only'].update(accepted_wire_mismatch=True,reconstructed_after_accepted_wire_mismatch=True)
        values=a.fallback_tables(rows,ids,plan)
        for name,denom in (('tx_source_scale',100),('rx_received_scale',300),('receiver_status',300)):
            counts=defaultdict(int)
            for r in values[name]:counts[r['point_id']]+=r['count'];self.assertEqual(r['denominator'],denom)
            self.assertEqual(set(counts.values()),{denom});self.assertEqual(len(counts),18)
        wire=[x for x in values['wire_diagnostics'] if x['point_id']=='H18_SLOT_00' and x['diagnostic']=='accepted_wire_mismatch'][0]
        self.assertEqual(wire['count'],1);self.assertEqual(wire['denominator'],300)
        conditional=[x for x in values['classifier_transition'] if x['point_id']=='H18_SLOT_00']
        self.assertEqual([x['denominator'] for x in conditional],[300,0,300,0])
        self.assertTrue(all(x['fraction'] is None for x in conditional if x['denominator']==0))

    def test_missing_duplicate_wrongnoise_and_mixed_TX_rejected(self):
        _,ids,rows,plan=fixture()
        for changed in (rows[:-1],rows+[rows[-1]]):
            with self.assertRaises(ValueError):a.fallback_tables(changed,ids,plan)
        changed=copy.deepcopy(rows);changed[0]['noise_seed']=2001
        with self.assertRaises(ValueError):a.fallback_tables(changed,ids,plan)
        changed=copy.deepcopy(rows);changed[0]['tx_actual_m']=7
        with self.assertRaisesRegex(ValueError,'paired noises'):a.fallback_tables(changed,ids,plan)

    def test_unresolved_receivers_fixed_control_change_and_fake_RX_rejected(self):
        _,ids,rows,plan=fixture()
        for field,value in (('source_status','ARITHMETIC_PENDING'),('rx_accepted_m',8)):
            changed=copy.deepcopy(rows);changed[2][field]=value
            with self.assertRaises(ValueError):a.fallback_tables(changed,ids,plan)
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['development_slot']==10:r['nominal_rate']='2/3'
        with self.assertRaisesRegex(ValueError,'Fixedm7'):a.fallback_tables(changed,ids,plan)

    def test_prediction_indices_are_never_numeric_metrics_and_fullmetric_scope_kept(self):
        self.assertNotIn('resnet50_prediction',a.METRICS);self.assertNotIn('convnext_prediction',a.METRICS)
        self.assertIn('convnext_source_prediction_agreement',a.METRICS)
        self.assertIn('dino_specificity',a.METRICS);self.assertIn('psnr_db',a.METRICS)
        self.assertNotIn('F_recovery_error',a.METRICS)
        self.assertEqual(a.sha(a.statistics.__file__),a.STATISTICS_SHA)

    def test_only_declared_roles_and_ready_scope_can_enter_catalog(self):
        schedule,_,_,_=fixture()
        for field,value in (('role','BEST_H'),('status','NOT_FEASIBLE')):
            changed=copy.deepcopy(schedule);changed[0][field]=value
            with self.assertRaises(ValueError):a.comparison_plan(changed)
        changed=copy.deepcopy(schedule);changed[0]['candidate']['arm']='P1024'
        with self.assertRaises(ValueError):a.comparison_plan(changed)

    def test_predecessor_failure_does_not_write_attempt_or_read_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);request=base/'request.json';out=base/'output'
            a.save(request,dict(schema='H18_DESCRIPTIVE_AGGREGATION_REQUEST_V1',comparisons='FROZEN_26_FACTOR_CONTRASTS',
                P_admitted=False,MAIN_admitted=False,out=str(out)))
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}),\
                patch.object(a,'normal_metrics_closed',side_effect=ValueError('normal owner absent')),patch.object(a,'load_rows') as load:
                with self.assertRaisesRegex(ValueError,'owner absent'):a.run(request)
                load.assert_not_called();self.assertFalse(out.exists())

    def test_comparison_plan_is_saved_before_any_metric_row_read(self):
        schedule,ids,_,_=fixture()
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);request=base/'request.json';out=base/'out';policy=base/'policy.json';a.save(policy,{})
            a.save(request,dict(schema='H18_DESCRIPTIVE_AGGREGATION_REQUEST_V1',comparisons='FROZEN_26_FACTOR_CONTRASTS',
                P_admitted=False,MAIN_admitted=False,out=str(out)))
            ctx=dict(schedule=schedule,source_ids=ids,cfg={'out':str(base/'metrics')},render_cfg={'out':str(base/'render')},
                group={'cfg':{'out':str(base/'cpu'),'finalized':str(policy)}})
            def no_scores(*args):
                self.assertTrue((out/'comparison_plan.json').exists());self.assertTrue((out/'frozen_schedule.json').exists())
                raise ValueError('synthetic stop before score access')
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}),\
                 patch.object(a,'normal_metrics_closed',return_value=(ctx,{})),patch.object(a,'load_rows',side_effect=no_scores):
                with self.assertRaisesRegex(ValueError,'before score access'):a.run(request)
            self.assertTrue((out/'failure.json').exists());self.assertFalse((out/'completion.json').exists())
            with self.assertRaisesRegex(ValueError,'prior attempts'):a.run(request)

    def test_CSV_is_LF_and_missing_values_stay_blank(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rows.csv';a.write_csv(p,[dict(value=None,count=0,status='MISSING')])
            self.assertNotIn(b'\r\n',p.read_bytes());self.assertIn(b',0,MISSING\n',p.read_bytes())

    def test_normal_metric_gate_uses_original_closed_batch_and_rejects_changed_budget(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);spec={k:str(base/(k+'.json')) for k in ('config','owner_config','registration','launch','completion')}
            for p in spec.values():a.save(p,{})
            driverpath=str(Path(a.__file__).with_name('h_development_metrics_driver.py'))
            request=dict(metric_batch=spec,metric_driver=driverpath,input_bindings=a.bind(spec.values()),
                source_bindings=a.bind((a.__file__,a.statistics.__file__,driverpath)))
            before=dict(charged=164760,failed=0,unresolved=0,phase_charged={'development':10800},development_remaining=2400)
            cfg=dict(registration=spec['registration'],visual_owner_config=spec['owner_config'],owner_module='original-owner',out=str(base/'metric-out'))
            old=dict(stages=[dict(id='development',resource='gpu',jobs=[dict(id='development_metrics',
                argv=['original-UM-python','-B',driverpath,'--config',spec['config']],out=cfg['out'])])])
            done=dict(registration_sha256=a.sha(spec['registration']),config_sha256=a.sha(spec['config']),
                source_ids=['s'],noise_seeds=[6201,6202,6203],budget_before=before,budget_after=before,
                outputs={},input_bindings={},source_bindings={})
            closed=dict(owner=old,done=done,owner_done={'budget':before})
            cpu=SimpleNamespace(closed_batch=Mock(return_value=closed));owner=SimpleNamespace(raw_process_state='REAL_STATE_READER')
            ctx=dict(cfg=cfg,source_ids=['s'],cpu=cpu,owner=owner,wait='ORIGINAL_WAIT',before=before,group={})
            driver=SimpleNamespace(load_registered=Mock(return_value=ctx),DONE='H_DEVELOPMENT_METRICS_COMPLETE',u=SimpleNamespace(assert_budget=Mock()))
            with patch.object(a,'module',return_value=driver):
                self.assertIs(a.normal_metrics_closed(request)[1],closed)
                args=cpu.closed_batch.call_args.args
                self.assertEqual(args[2],dict(config=spec['owner_config'],registration=spec['registration'],launch=spec['launch'],completion=spec['completion']))
                self.assertEqual(args[4]['frame_count'],5400);self.assertFalse(args[4]['H_full_delivery_claimed'])
                done['budget_after']=dict(before,charged=164761)
                with self.assertRaisesRegex(ValueError,'budget changed'):a.normal_metrics_closed(request)
                cpu.closed_batch.side_effect=ValueError('owner still alive')
                with self.assertRaisesRegex(ValueError,'still alive'):a.normal_metrics_closed(request)

    def test_all100_metric_checkpoints_and_flattened_rows_must_agree(self):
        schedule,ids,rows,plan=fixture()
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);reg=base/'reg.json';a.save(reg,{})
            out=base/'metrics';cpu=base/'cpu';outputs={};traceoutputs={};flat=[]
            for i,sid in enumerate(ids):
                values=[dict(r,metric_evaluator_identity='frozen',metric_batch_size=1) for r in rows if r['source_index']==i]
                rp=out/'sources'/f'{i:04d}.json';cp=out/'source_checkpoints'/f'{i:04d}.json'
                tp=cpu/f'worker_{i%2}'/'traces'/f'{i:04d}.json';a.save(tp,{'fixture':i});a.save(rp,values)
                a.save(cp,dict(status='H_DEVELOPMENT_METRIC_SOURCE_COMPLETE',source_index=i,source_id=sid,frame_count=54,
                    registration_sha256=a.sha(reg),metric_evaluator_identity='frozen',new_packet_decodes=0,
                    outputs=a.bind((rp,)),input_bindings=a.bind((tp,))))
                outputs.update(a.bind((rp,cp)));traceoutputs.update(a.bind((tp,)));flat.extend(values)
            fp=out/'frame_metrics.json';a.save(fp,flat);outputs.update(a.bind((fp,)))
            adapter=SimpleNamespace(validate_source_rows=Mock(),attach_wire_accounting=lambda values,trace:values)
            ctx=dict(cfg={'out':str(out),'registration':str(reg)},source_ids=ids,schedule=schedule,adapter=adapter,
                group={'cfg':{'out':str(cpu)},'done':{'outputs':traceoutputs}})
            closed={'done':{'outputs':outputs,'metric_evaluator_identity':'frozen'}}
            loaded,bound=a.load_rows(ctx,closed);self.assertEqual(len(loaded),5400);self.assertEqual(adapter.validate_source_rows.call_count,100)
            self.assertEqual(len(bound),301)
            # Even a newly SHA-pinned inconsistent flat file cannot replace the source-sealed rows.
            fp.write_text('[]',encoding='utf-8');outputs[str(fp)]=a.sha(fp)
            with self.assertRaisesRegex(ValueError,'Flattened frame'):a.load_rows(ctx,closed)


if __name__=='__main__':unittest.main()
