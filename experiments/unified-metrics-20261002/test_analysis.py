"""Synthetic CPU fixtures only; no images/models/GPU or experimental selection."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('_unified_metrics_analysis',Path(__file__).with_name('analysis.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def fixture(n=4):
    baseline=[];rows=[];keys=[]
    descriptors=[dict(experiment='N512',scope='dev',N=512,phy_family='continuous',snr_db=7,method='P512',
        projection='',control='',output_role='main',decoder_id='Dc',label_conditioned=False),
        dict(experiment='N512',scope='dev',N=512,phy_family='continuous',snr_db=7,method='A1_policy',
        projection='',control='',output_role='main',decoder_id='Dc',label_conditioned=False),
        dict(experiment='M2',scope='oracle_clean',N='',phy_family='ORACLE',snr_db='clean',method='UNGUIDED',
        projection='g8_c32',control='UNGUIDED',output_role='oracle',decoder_id='Dc',label_conditioned=False)]
    reg=dict(source_ids=[f's{i}' for i in range(n)],modelmanifest_sha256='9'*64,replay_parity_passed=True,
        expected_groups=descriptors,metric_availability={m:dict(status='READY') for m in a.NEW_METRICS})
    for i in range(n):
        prediction=i if i%2==0 else 999
        baseline.append(dict(source_id=f's{i}',source_index=i,preprocessing_id=f'p{i}',reference_sha256=f'{i:064x}',
            true_class_index=i,resnet50_source_prediction=prediction,resnet50_source_top1_label=int(prediction==i)))
    for desc in descriptors:
        for i in range(n):
            original=baseline[i]['resnet50_source_prediction']
            for seed in [0] if desc['snr_db']=='clean' else a.NOISE_SEEDS:
                offset=0 if seed==0 else seed-2001
                pred=original if offset<2 else 998
                gain=.01 if desc['method']=='A1_policy' else 0
                rows.append(dict(**desc,source_id=f's{i}',source_index=i,preprocessing_id=f'p{i}',
                    reference_sha256=f'{i:064x}',image_sha256=f'{100+i+offset:064x}',noise_seed=seed,
                    replay_parity_passed=True,modelmanifest_sha256='9'*64,is_main_conclusion=desc['scope']=='dev',
                    psnr_db=30+i+offset,lpips_alex=.2+offset*.01,dino_cosine=.8+i*.01,dino_mismatched=.3,
                    clip_image_cosine=.7+i*.01+offset*.001+gain,dinov2_vitl14_cosine=.72+i*.01+offset*.001+gain,
                    dists=.2+i*.01+offset*.001-gain,
                    dreamsim=.1+i*.01+offset*.001-gain,ms_ssim=.8+i*.01+offset*.001+gain,
                    resnet50_prediction=pred,resnet50_source_prediction=original,
                    resnet50_top1_label=int(pred==i),resnet50_top1_source_prediction=int(pred==original),
                    resnet50_source_top1_label=int(original==i),latent_sq_err_final=''))
    return rows,baseline,reg


class AnalysisTests(unittest.TestCase):
    def test_complete_noise_and_clean_scope_inventory(self):
        rows,baseline,reg=fixture()
        groups,sources,identities,availability,flags=a.validate(rows,baseline,reg,4,(7,))
        self.assertEqual(len(groups),3);self.assertEqual(sources,['s0','s1','s2','s3'])
        clean=next(k for k in groups if k[4]=='clean');self.assertEqual(len(groups[clean]),4)
        noisy=next(k for k in groups if k[5]=='P512');self.assertEqual(len(groups[noisy]),12)
        for bad in [rows[:-1],rows+[rows[0]]]:
            with self.assertRaises(ValueError):a.validate(bad,baseline,reg,4,(7,))
        wrong=copy.deepcopy(rows);wrong[-1]['noise_seed']=2001
        with self.assertRaises(ValueError):a.validate(wrong,baseline,reg,4,(7,))

    def test_gate_registered_partial_grid_and_oracle_are_supplementary(self):
        rows,baseline,reg=fixture()
        actual_rows=[copy.deepcopy(row) for row in rows if row['method']=='P512']
        for row in actual_rows:
            row.update(experiment='M2_ACTUAL',scope='actual_link',method='GUIDED',projection='g8_c32')
        desc={name:actual_rows[0][name] for name in a.GROUP_FIELDS}
        reg['expected_groups']=[desc]
        a.validate(actual_rows,baseline,reg,4)
        bad=copy.deepcopy(actual_rows);bad[0]['snr_db']=4
        with self.assertRaises(ValueError):a.validate(bad,baseline,reg,4)
        for row in actual_rows:
            row.update(experiment='M2_ORACLE',scope='oracle_noisy',output_role='oracle')
        reg['expected_groups']=[{name:actual_rows[0][name] for name in a.GROUP_FIELDS}]
        with self.assertRaises(ValueError):a.validate(actual_rows,baseline,reg,4)
        for row in actual_rows:row['is_main_conclusion']=False
        a.validate(actual_rows,baseline,reg,4)

    def test_source_means_before_bootstrap_and_shared_resamples(self):
        rows,baseline,reg=fixture();groups,sources,_,availability,flags=a.validate(rows,baseline,reg,4,(7,))
        bootstrap=a.Bootstrap(200,20261002)
        summary,per_source,means=a.summarize(groups,sources,availability,flags,bootstrap)
        key=next(k for k in groups if k[5]=='P512')
        np.testing.assert_allclose(means[key,'psnr_db'],[31,32,33,34])
        item=next(r for r in summary if r['group_id']==a.group_id(key) and r['metric']=='psnr_db')
        self.assertEqual(item['n_sources'],4);self.assertEqual(item['n_frames'],12)
        expected=np.random.default_rng(20261002).integers(0,4,(200,4))
        np.testing.assert_array_equal(bootstrap.indices[4],expected)
        ci=bootstrap.interval([31,32,33,34])
        lo,hi=np.percentile(np.asarray([31,32,33,34])[expected].mean(1),[2.5,97.5])
        self.assertEqual(ci['ci_low'],lo);self.assertEqual(ci['ci_high'],hi)
        table=a.paired(groups,sources,reg,means,flags,bootstrap)
        r=next(r for r in table if r['method_A']=='A1_policy' and r['method_B']=='P512' and r['metric']=='clip_image_cosine')
        self.assertAlmostEqual(r['mean'],.01,12);self.assertAlmostEqual(r['ci_low'],.01,12)
        self.assertAlmostEqual(r['ci_high'],.01,12)

    def test_source_only_rate_curve_is_one_clean_frame_and_same_projection_pairs(self):
        original,baseline,reg=fixture()
        clean=[row for row in original if row['snr_db']=='clean'];rows=[];descriptors=[]
        for projection in ('m4_K12','m5_K16'):
            for order in ('entropy','raster','random','oracle'):
                desc=dict(experiment='M1_RATE',scope='source_only_rate_curve',N='',phy_family='source_only',
                    snr_db='clean',method='rate_curve_'+order,projection=projection,control=order,
                    output_role='reference',decoder_id='Dc',label_conditioned=False)
                descriptors.append(desc)
                for source in clean:
                    value=dict(source);value.update(desc,is_main_conclusion=False);rows.append(value)
        reg['expected_groups']=descriptors
        groups,sources,_,availability,flags=a.validate(rows,baseline,reg,4)
        self.assertEqual(len(groups),8);self.assertTrue(all(len(frames)==4 for frames in groups.values()))
        summary,means_table,means=a.summarize(groups,sources,availability,flags,a.Bootstrap(20))
        self.assertTrue(all(row['n_noise']==1 for row in means_table))
        self.assertTrue(all(row['n_expected_frames']==4 and row['aggregation']=='one_clean_output_per_source' for row in summary))
        pairs=a.contrasts(groups,reg);self.assertEqual(len(pairs),6)
        self.assertTrue(all(x[0][6]==x[1][6] and x[0][5]=='rate_curve_entropy' for x in pairs))
        table=a.paired(groups,sources,reg,means,flags,a.Bootstrap(20))
        self.assertTrue(all(not row['is_main_conclusion'] for row in table))
        for field,value in [('is_main_conclusion',True),('N',512),('snr_db',7),('noise_seed',2001)]:
            wrong=copy.deepcopy(rows);wrong[0][field]=value
            with self.assertRaises(ValueError):a.validate(wrong,baseline,reg,4)
        with self.assertRaises(ValueError):a.validate(rows[:-1],baseline,reg,4)

    def test_dinov2_large_is_required_separate_from_retained_small_metric(self):
        rows,baseline,reg=fixture();before=copy.deepcopy(rows)
        groups,sources,_,availability,flags=a.validate(rows,baseline,reg,4,(7,))
        self.assertEqual(a.NEW_METRICS['dinov2_vitl14_cosine'],'higher')
        summary,source_means,means=a.summarize(groups,sources,availability,flags,a.Bootstrap(100))
        key=next(k for k in groups if k[5]=='P512')
        np.testing.assert_allclose(means[key,'dino_cosine'],[.8,.81,.82,.83])
        np.testing.assert_allclose(means[key,'dinov2_vitl14_cosine'],[.721,.731,.741,.751])
        self.assertTrue(any(row['metric']=='dinov2_vitl14_cosine' for row in summary))
        self.assertTrue(any(row['metric']=='dinov2_vitl14_cosine' for row in source_means))
        comparison=a.paired(groups,sources,reg,means,flags,a.Bootstrap(100))
        item=next(row for row in comparison if row['metric']=='dinov2_vitl14_cosine')
        self.assertAlmostEqual(item['mean'],.01,12);self.assertAlmostEqual(item['ci_low'],.01,12)
        self.assertEqual(rows,before)
        for missing in ('dino_cosine','dinov2_vitl14_cosine'):
            wrong=copy.deepcopy(rows);wrong[0].pop(missing)
            with self.assertRaises((ValueError,KeyError)):a.validate(wrong,baseline,reg,4,(7,))
        unregistered=copy.deepcopy(reg);unregistered['metric_availability'].pop('dinov2_vitl14_cosine')
        with self.assertRaises(ValueError):a.validate(rows,baseline,unregistered,4,(7,))

    def test_source_classification_baseline_once_and_flags_coherent(self):
        rows,baseline,reg=fixture();sources,identities=a.original_sources(baseline,4)
        self.assertEqual(np.mean([identities[s]['correct'] for s in sources]),.5)
        with self.assertRaises(ValueError):a.original_sources(baseline+[baseline[0]],4)
        wrong=copy.deepcopy(rows);wrong[0]['resnet50_top1_label']=1-wrong[0]['resnet50_top1_label']
        with self.assertRaises(ValueError):a.validate(wrong,baseline,reg,4,(7,))
        wrong=copy.deepcopy(rows);wrong[0]['resnet50_source_prediction']=998
        with self.assertRaises(ValueError):a.validate(wrong,baseline,reg,4,(7,))

    def test_conditioned_D0_and_parity_identity_do_not_silently_enter_main(self):
        rows,baseline,reg=fixture()
        for field,value in [('label_conditioned',True),('method','D_U_QPSK_D0'),('decoder_id','D0'),
                            ('replay_parity_passed',False),('modelmanifest_sha256','8'*64)]:
            wrong=copy.deepcopy(rows);wrong[0][field]=value
            with self.assertRaises(ValueError):a.validate(wrong,baseline,reg,4,(7,))
        wrong=copy.deepcopy(rows);wrong[0]['method']='D_C_QPSK';wrong[0]['is_main_conclusion']=False
        with self.assertRaises(ValueError):a.validate(wrong,baseline,reg,4,(7,))

    def test_metric_unavailability_explicit_and_grey_frames_still_required(self):
        rows,baseline,reg=fixture();reg['metric_availability']['dreamsim']=dict(status='UNAVAILABLE_NOT_REGISTERED',reason='synthetic fixture')
        for row in rows:row['dreamsim']=''
        groups,sources,_,availability,flags=a.validate(rows,baseline,reg,4,(7,))
        summary,_,_=a.summarize(groups,sources,availability,flags,a.Bootstrap(20))
        self.assertTrue(all(r['status']=='NOT_EVALUATED' and r['n_sources']==0 for r in summary if r['metric']=='dreamsim'))
        rows[0]['dreamsim']=.2
        with self.assertRaises(ValueError):a.validate(rows,baseline,reg,4,(7,))
        rows,baseline,reg=fixture();rows[0]['header_ok']=False;rows[0]['clip_image_cosine']=''
        with self.assertRaises(ValueError):a.validate(rows,baseline,reg,4,(7,))

    def test_registered_contrasts_never_rank_by_metric_or_cross_scope_implicitly(self):
        rows,baseline,reg=fixture();groups,sources,_,availability,flags=a.validate(rows,baseline,reg,4,(7,))
        pairs=a.contrasts(groups,reg)
        self.assertTrue(all(x[0][1]==x[1][1] for x in pairs))
        for row in rows:row['clip_image_cosine']=.01
        self.assertEqual(pairs,a.contrasts(groups,reg))
        keys=list(groups);reg['contrasts']=[dict(name='explicit',group_A=a.context(keys[1]),group_B=a.context(keys[0]))]
        self.assertEqual(a.contrasts(groups,reg)[0][2],'explicit')

    def test_immutable_registration_manifest_policy_and_synthetic_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inp=root/'original.csv';policy=root/'policy.json';stop=root/'stop.json';manifest=root/'models.json'
            inp.write_text('frozen rows');policy.write_text('frozen policy');stop.write_text('{}');manifest.write_text('{}')
            reg=dict(synthetic=False,training_updates=0,original_pipeline_complete=True,
                input_tables={str(inp):a.sha(inp)},frozen_policy_bindings={str(policy):a.sha(policy)},
                pipeline_completion_bindings={str(stop):a.sha(stop)},modelmanifest_path=str(manifest),modelmanifest_sha256=a.sha(manifest))
            a.verify_registration(reg)
            synthetic=dict(reg,synthetic=True)
            with self.assertRaises(ValueError):a.verify_registration(synthetic)
            a.verify_registration(synthetic,True)
            policy.write_text('new choice')
            with self.assertRaises(ValueError):a.verify_registration(reg)

    def test_analysis_outputs_are_sealed_and_empty_contrasts_have_csv_header(self):
        with tempfile.TemporaryDirectory() as folder:
            result=Path(folder);rows,baseline,reg=fixture(100)
            manifest=result/'modelmanifest.json';a.write_json(manifest,{'synthetic_fixture':True})
            old=result/'old.csv';old.write_text('frozen synthetic original table')
            policy=result/'policy.json';a.write_json(policy,{'synthetic_fixture':True})
            stop=result/'pipeline_complete.json';a.write_json(stop,{'status':'COMPLETE','synthetic':True})
            reg.update(synthetic=True,training_updates=0,original_pipeline_complete=True,contrasts=[],
                modelmanifest_path=str(manifest),modelmanifest_sha256=a.sha(manifest),
                input_tables={str(old):a.sha(old)},frozen_policy_bindings={str(policy):a.sha(policy)},
                pipeline_completion_bindings={str(stop):a.sha(stop)})
            for row in rows:row['modelmanifest_sha256']=a.sha(manifest)
            a.write_csv(result/'metrics_per_frame.csv',rows)
            a.write_csv(result/'source_baseline.csv',baseline)
            a.write_json(result/'metrics_registration.json',reg)
            with self.assertRaises(ValueError):a.analyze(result,replicates=50)
            receipt=a.analyze(result,replicates=50,allow_synthetic=True)
            self.assertEqual(receipt['status'],'COMPLETE');self.assertTrue(receipt['synthetic'])
            self.assertEqual(receipt['frames'],700);self.assertEqual(receipt['sources'],100)
            self.assertEqual(a.read_csv(result/'metrics_paired_intervals.csv'),[])
            self.assertTrue((result/'metrics_paired_intervals.csv').read_text().startswith('contrast,group_A,'))
            for p,h in receipt['inputs'].items():self.assertEqual(a.sha(p),h)
            for p,h in receipt['outputs'].items():self.assertEqual(a.sha(p),h)
            report=(result/'METRICS_REPORT.md').read_text(encoding='utf-8')
            for label in ('DINOv2 ViT-S/14','DINOv2 ViT-L/14','dinov2_vitl14_cosine','1024维'):
                self.assertIn(label,report)


if __name__=='__main__':unittest.main()
