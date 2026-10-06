"""Synthetic CPU driver qualification. No actual quality data or model calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from unittest.mock import Mock
import h_p_statistics_driver as d
import h_p_result_normalization as n
from test_h_p_result_normalization import fixture


def metadata_fixture(base):
    proof=base/'implementation.json';d.save(proof,{'synthetic':True});binding=d.bind([proof])
    metrics={}
    for m in n.METRICS:
        if m in n.P_MISSING:
            metrics[m]=dict(status='MISSING_IN_P',reason='Original scalar absent');continue
        ident=dict(definition='fixture:'+m,models={} if m in ('psnr_db','ms_ssim') else {'weight':'a'*64},preprocessing='RGB01')
        branches={b:dict(identity=copy.deepcopy(ident),precision='float32',batch_execution='B1',evidence=binding.copy()) for b in ('H','P')}
        if m=='psnr_db':
            branches['H']['precision']='H_float64_MSE_then_minus10_log10'
            branches['P']['precision']='P_retained_original_float32_Torch_PSNR_scalar'
        if m=='ms_ssim':
            for v in branches.values():v['implementation']=binding.copy()
        metrics[m]=dict(status='UNRESOLVED_IDENTITY',metadata_audit_status='MATCHED_FROZEN_METADATA_WAIT_ACTUAL_GATES',branches=branches,
            definition_reviewed_before_comparison=True)
    meta={'metrics':{'fixture':'identity'}};model=base/'models.json';d.save(model,meta)
    paths=dict(P_metadata=str(model),P_manifest=str(model),H_completion=str(base/'H.json'),P_completion=str(base/'P.json'))
    classifier=dict(classification_batch_size=1,used_for_selection=False,policy_sha256='H',weights_sha256='a'*64,preprocessing='frozen')
    h=dict(metric_metadata=dict(independent=classifier,metric_batch_size=1,metric_evaluator=meta),numerical_runtime={'fp32':True},
        input_bindings=binding.copy(),source_bindings={})
    p=dict(classifier_identity=dict(classifier,policy_sha256='P'),numerical_runtime={'fp32':True},input_bindings=binding.copy(),source_bindings={},
        legacy_scalars_preserved=True,RGB_exact_parity=True,ConvNeXt_scored=True)
    reg=dict(numerical_runtime={'fp32':True},metric_batch_size=16,old_metric_values_retained=True,new_metrics_used_for_selection=False,modelmanifest_sha256=d.sha(model))
    d.save(paths['H_completion'],h);d.save(paths['P_completion'],p)
    template=dict(status='H_P_METRIC_METADATA_AUDIT_PREPARED_NOT_FINAL_ADMISSION',current_admitted_count=0,metrics=metrics)
    return template,h,p,reg,paths


def plan_fixture():
    roles=['H_WHOLE_SYSTEM']*8+['H_RAW_PARTIAL_SYSTEM']*2+['H_FIXED_M7_ATTRIBUTION']*8
    return {'catalog':[dict(development_slot=i,point_id=f'H18_SLOT_{i:02d}',snr_db=(13,19)[i%2],
        role=roles[i],arm='fixturearm',candidate_id=f'candidate{i}') for i in range(18)]}


class DriverTests(unittest.TestCase):
    def test_metadata_only_admits18_and_preserves_three_missing(self):
        with tempfile.TemporaryDirectory() as td:
            args=metadata_fixture(Path(td));v=d.metadata_admission(*args)
            self.assertEqual(sum(x['status']=='ADMITTED' for x in v['metrics'].values()),18)
            self.assertEqual({k for k,x in v['metrics'].items() if x['status']=='MISSING_IN_P'},n.P_MISSING)
            self.assertFalse(v['quality_rows_used_for_admission']);self.assertTrue(v['generated_from_identity_only'])
            self.assertNotEqual(v['actual_classifier_identities']['H']['policy_sha256'],v['actual_classifier_identities']['P']['policy_sha256'])

    def test_classifier_changes_beyond_policy_identity_block(self):
        with tempfile.TemporaryDirectory() as td:
            args=metadata_fixture(Path(td));args[2]['classifier_identity']['weights_sha256']='b'*64
            with self.assertRaisesRegex(ValueError,'ConvNeXt'):d.metadata_admission(*args)

    def test_historical_batch_or_runtime_not_silently_equated(self):
        with tempfile.TemporaryDirectory() as td:
            args=metadata_fixture(Path(td));args[3]['metric_batch_size']=1
            with self.assertRaisesRegex(ValueError,'batch/value'):d.metadata_admission(*args)
            args[3]['metric_batch_size']=16;args[2]['numerical_runtime']={'fp32':False}
            with self.assertRaisesRegex(ValueError,'numerical'):d.metadata_admission(*args)

    def test_missing_rgb_proof_and_unbound_identity_block(self):
        with tempfile.TemporaryDirectory() as td:
            args=metadata_fixture(Path(td));args[2]['RGB_exact_parity']=False
            with self.assertRaisesRegex(ValueError,'replay proof'):d.metadata_admission(*args)
            args[2]['RGB_exact_parity']=True;args[2]['input_bindings']={}
            with self.assertRaisesRegex(ValueError,'not sealed'):d.metadata_admission(*args)

    def test_comparisons_are_all18_same_snr_keep_whole_partial_fixed(self):
        v=d.comparison_plan(plan_fixture());self.assertEqual(len(v['comparisons']),18)
        self.assertEqual([r['role'] for r in v['comparisons']].count('H_RAW_PARTIAL_SYSTEM'),2)
        self.assertEqual({r['reference'] for r in v['comparisons']},{'P1024_SNR_13','P1024_SNR_19'})
        self.assertEqual(v['noise_seeds'],{'H':[6201,6202,6203],'P':[2001,2002,2003]})
        self.assertFalse(v['selection']);self.assertFalse(v['H_success_evaluated'])

    def test_incomplete_or_relabelled_catalog_rejected(self):
        p=plan_fixture();p['catalog'].pop()
        with self.assertRaises(ValueError):d.comparison_plan(p)
        p=plan_fixture();p['catalog'][0]['point_id']='selected_best'
        with self.assertRaisesRegex(ValueError,'identity'):d.comparison_plan(p)

    def test_full_synthetic_grid_real_bootstrap_uses_source_mean_original_noises(self):
        with tempfile.TemporaryDirectory() as td:
            h,p,targets,admission=fixture(Path(td));before=(n.identity(h['rows']),n.identity(p['rows']))
            with patch.object(n,'bridge_targets',return_value=targets):v,r=d.calculate(h,p,admission,d.comparison_plan(plan_fixture()))
            self.assertEqual((len(r['summary']),len(r['paired']),len(r['source_means'])),(360,324,36000))
            self.assertEqual(before,(n.identity(h['rows']),n.identity(p['rows'])))
            self.assertEqual(r['paired'][0]['bootstrap_seed'],2026100605)
            self.assertEqual(r['paired'][0]['method_noise_seeds'],[6201,6202,6203])
            self.assertEqual(r['paired'][0]['reference_noise_seeds'],[2001,2002,2003])
            self.assertFalse(v['MAIN_complete'])

    def test_missing_frame_cannot_turn_into_pairwise_intersection(self):
        with tempfile.TemporaryDirectory() as td:
            h,p,targets,admission=fixture(Path(td));p['rows'].pop()
            with patch.object(n,'bridge_targets',return_value=targets),self.assertRaises(ValueError):
                d.calculate(h,p,admission,d.comparison_plan(plan_fixture()))

    def test_owner_precreated_empty_directory_allowed_prior_evidence_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'job';out.mkdir();d.claim_output(out,'a'*64);first=(out/'attempt.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'Prior attempt'):d.claim_output(out,'b'*64)
            self.assertEqual(first,(out/'attempt.json').read_bytes())

    def test_budget_no_unresolved_and_reserved_MAIN_never_consumed(self):
        b=dict(charged=164760,phase_charged={'development':10800},development_remaining=2400,failed=0,unresolved=0)
        d.check_budget(b)
        for k in ('charged','development_remaining','failed','unresolved'):
            bad=copy.deepcopy(b);bad[k]+=1
            with self.assertRaises(ValueError):d.check_budget(bad)

    def test_csv_is_lf_and_json_claim_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);d.csv_file(out/'x.csv',[{'a':'one','b':2}]);self.assertNotIn(b'\r',(out/'x.csv').read_bytes())
            d.save(out/'x.json',{'proof':True})
            with self.assertRaises(FileExistsError):d.save(out/'x.json',{})

    def test_historical_graph_all_original_bytes_checked_later_files_not_retroactive(self):
        with tempfile.TemporaryDirectory() as td:
            b=Path(td);old=b/'old.py';old.write_text('x=1\n');new=b/'later.py';new.write_text('x=2\n')
            graph=b/'graph.json';sources=d.bind([old]);d.save(graph,dict(status='EXACT_SOURCE_CLOSURE_MATCH',source_bindings=sources))
            reg=dict(source_bindings=sources,input_bindings=d.bind([graph]));done={'visual_source_bindings':sources}
            result=d.historical_graph({'visual_source_closure':str(graph)},reg,done)
            self.assertNotIn(str(new),result['source_bindings'])
            old.write_text('x=3\n')
            with self.assertRaisesRegex(ValueError,'Changed bound'):d.historical_graph({'visual_source_closure':str(graph)},reg,done)

    def test_closed_p_loader_requires_all100_cp_six_rows_and_exact_flat_copy(self):
        with tempfile.TemporaryDirectory() as td:
            b=Path(td);ids=[f's{i}' for i in range(100)];outputs={};rows=[]
            for i,sid in enumerate(ids):
                rp=b/'sources'/f'{i:04d}.json';part=[dict(source_index=i,source_id=sid,slot=j) for j in range(6)]
                d.save(rp,part);rows.extend(part)
                cp=b/'source_checkpoints'/f'{i:04d}.json'
                d.save(cp,dict(status='P600_RECEIVED_REPLAY_SOURCE_COMPLETE',source_index=i,source_id=sid,
                    frame_count=6,RGB_parity_frames=6,ConvNeXt_frames=6,registration_sha256='a'*64,
                    outputs=d.bind([rp]),input_bindings={}))
                outputs.update(d.bind([rp,cp]))
            fp=b/'frame_metrics.json';d.save(fp,rows);outputs.update(d.bind([fp]))
            config={'out':str(b)}
            for name in ('core_module','scalar_inventory','metrics_registration','numerical_reference','latent_inventory','selected_P_policy'):
                path=b/(name+'.json');d.save(path,{});config[name]=str(path)
            cctx={'selected_P_sha256':d.sha(config['selected_P_policy']),'source_ids':ids}
            core=SimpleNamespace(validate_complete=Mock(),context=Mock(return_value=cctx))
            ctx=dict(cfg=config,driver=SimpleNamespace(original_assets=Mock(return_value=({},{},{}))),H_context={},audit={},inventory={},bound={})
            done=dict(source_ids=ids,registration_sha256='a'*64,outputs=outputs);closed=dict(done=done,bindings={})
            with patch.object(n,'module',return_value=core):value=d.load_closed_p(ctx,closed)
            self.assertEqual(len(value['rows']),600);core.validate_complete.assert_called_once();core.context.assert_called_once()
            outputs.pop(str(b/'source_checkpoints/0099.json'))
            with patch.object(n,'module',return_value=core),self.assertRaisesRegex(ValueError,'not sealed'):d.load_closed_p(ctx,closed)


if __name__=='__main__':unittest.main()
