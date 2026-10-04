"""CPU report contracts; synthetic fixtures never enter scientific outputs."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import own_controls_report as r


def fixtures():
    records=[dict(source_index=i,image_id=f'source{i}',preprocessing_id=f'prep{i}',class_index=i) for i in range(100)]
    own=[];ext=[]
    for i in range(100):
        for n in r.c.BUDGETS:
            for snr in r.c.SNRS:
                for seed in r.c.SEEDS:
                    for j,m in enumerate(r.METHODS[n]):
                        row=dict(source_index=i,source_id=f'source{i}',preprocessing_id=f'prep{i}',true_class_index=i,
                            N=n,snr_db=snr,noise_seed=seed,method=m,synthetic=False,label_conditioned=False,
                            E=2*n,actual_energy=2*n,modelmanifest_sha256='weights',reference_sha256=f'target{i}',
                            image_sha256=f'image{i}_{n}_{snr}_{seed}_{m}',resnet50_prediction=i,resnet50_source_prediction=i,
                            resnet50_top1_probability=.9,resnet50_top1_label=1,resnet50_top1_source_prediction=1,
                            semantic_error=0,confidently_wrong=0,dino_mismatched=.2,decoder_id='Dc' if j<3 else 'Swin',
                            **{k:.8 for k in r.METRICS if k not in ('resnet50_top1_label','resnet50_top1_source_prediction','semantic_error','confidently_wrong')})
                        row['dino_specificity']=.6
                        if j>=3:
                            row.update(audit_used_to_control_receiver=False,observed_sha256=f'wave{i}_{n}_{snr}_{seed}',
                                header_accepted=True,complete_posterior_schedule=j==4,NFE=250 if j==4 else 0,t_start=250,fallback='',
                                TX_seconds=.01,RX_seconds=10. if j==4 else .1,offline_false_accept=False)
                            ext.append(row)
                        else:own.append(row)
    return own,ext,records


def policy_fixtures():
    own,ext,records=fixtures();indexed=r.validate_rows(own,ext,records,'weights')
    old_du=dict(development_read=False,levels={str(s):{'QPSK':{'U':{'action_m':6}}} for s in r.c.SNRS})
    old_m1=dict(development_read=False,cells=[]);new=dict(development_read=False,cells=[])
    for s in r.c.SNRS:
        old_m1['cells'].append(dict(N=1024,phy_family='QPSK',snr_db=s,method='entropy',action=dict(m=6,q=0 if s==1 else 48,order='whole' if s==1 else 'entropy')))
        for m in r.METHODS[2048][1:3]:
            new['cells'].append(dict(snr_db=s,method=m,action=dict(m=7,q=0,order='whole')))
    for (i,n,s,seed,m),row in indexed.items():
        if m not in ('D_U','M1'):continue
        native=(dict(action_m=6) if m=='D_U' else dict(old_m1['cells'][list(r.c.SNRS).index(s)]['action'])) if n==1024 else dict(m=7,q=0,order='whole')
        original=dict(original_scientific_row=native) if n==1024 else native
        row['original_row_json']=json.dumps(original);row['original_row_sha256']=r.c.identity(original)
        if n==2048:row['image_sha256']=f'shared{i}_{s}_{seed}'
    return indexed,old_du,old_m1,new


class FiveMethodTests(unittest.TestCase):
    def test_cost_gate_requires_all_five_and_labels_dev_population(self):
        import own_controls_cost_common as costs
        measured=[dict(N=n,snr_db=s,method=m,frames=4,n_frames=4,source_population='original_calibration_indices_0_1_2_3')
                  for n in (1024,2048) for s in (1,7,13) for m in r.METHODS[n]]
        old=[dict(N=1024,snr_db=1,method=r.c.METHODS[0],n_frames=300)]
        with patch.object(costs,'verify_report',return_value=(measured,{'proof':'hash'})):
            rows,bindings,done=r.cost_gate(Path('receipt'),old)
        self.assertTrue(done);self.assertEqual(len(rows),31)
        self.assertEqual(rows[-1]['source_population'],'development100_x3_noise')
        self.assertFalse(rows[-1]['same_population_all5']);self.assertEqual(rows[-1]['frames'],300)
        with patch.object(costs,'verify_report',return_value=(measured[:18],{})),self.assertRaises(RuntimeError):
            r.cost_gate(Path('receipt'),old)

    def test_policies_are_read_from_calibration_and_K0_is_whole_scale(self):
        indexed,*policies=policy_fixtures();rows=r.policy_table(indexed,*policies)
        self.assertEqual(len(rows),12)
        self.assertTrue(all(x['all_300_actions_match'] and x['comparison_frames']==300 for x in rows))
        self.assertTrue(all(x['same_rgb_as_D_U_frames']==300 for x in rows if x['N']==2048 and x['comparison_family']=='M1'))
        self.assertIn('whole-scale K=0',r.policy_title(2048,1,'M1',rows))
        self.assertIn('K=48',r.policy_title(1024,7,'M1',rows))
        legacy=next(x for x in rows if (x['N'],x['snr_db'],x['comparison_family'])==(1024,1,'D_U'))
        self.assertEqual(legacy['paid_class_bits'],10);self.assertFalse(legacy['class_condition_used'])

    def test_policy_rejects_development_choice_and_measured_action_drift(self):
        indexed,*policies=policy_fixtures()
        policies[0]['development_read']=True
        with self.assertRaisesRegex(RuntimeError,'development'):r.policy_table(indexed,*policies)
        policies[0]['development_read']=False
        row=indexed[0,1024,1,2001,'D_U'];original=dict(original_scientific_row=dict(action_m=5))
        row['original_row_json']=json.dumps(original);row['original_row_sha256']=r.c.identity(original)
        with self.assertRaisesRegex(RuntimeError,'action differs'):r.policy_table(indexed,*policies)

    def test_policy_rejects_unsealed_original_provenance_and_K0_pixel_drift(self):
        indexed,*policies=policy_fixtures();row=indexed[0,2048,1,2001,'M1']
        row['original_row_sha256']='changed'
        with self.assertRaisesRegex(RuntimeError,'provenance'):r.policy_table(indexed,*policies)
        row['original_row_sha256']=r.c.identity(json.loads(row['original_row_json']))
        row['image_sha256']='different'
        with self.assertRaisesRegex(RuntimeError,'Identical N2048'):r.policy_table(indexed,*policies)

    def test_metadata_requires_completed_scorer_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'policy.json';p.write_text('{"development_read": false}',encoding='utf-8')
            self.assertEqual(r.admitted_json(p,{str(p):r.c.sha(p)}),dict(development_read=False))
            with self.assertRaisesRegex(RuntimeError,'authenticated'):r.admitted_json(p,{})

    def test_full_same_source_target_inventory(self):
        own,ext,records=fixtures();index=r.validate_rows(own,ext,records,'weights')
        self.assertEqual(len(index),9000)
        with self.assertRaises(RuntimeError):r.validate_rows(own[:-1],ext,records,'weights')
        for field,value in [('reference_sha256','another target'),('preprocessing_id','another preprocessing'),
                            ('true_class_index',999),('modelmanifest_sha256','other'),('label_conditioned',True),('E',2049)]:
            bad=copy.deepcopy(own);bad[0][field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):r.validate_rows(bad,ext,records,'weights')

    def test_paid_class_legacy_cannot_be_relabeled_new_N2048_control(self):
        own,ext,records=fixtures()
        row=next(x for x in own if x['N']==2048 and x['method']=='D_U_whole_N2048_v1')
        row['method']='legacy_paid_class_N2048'
        with self.assertRaises(RuntimeError):r.validate_rows(own,ext,records,'weights')

    def test_semantic_probability_and_specificity_consistency(self):
        own,ext,records=fixtures()
        for field,value in [('semantic_error',1),('confidently_wrong',1),('resnet50_top1_label',0),('resnet50_top1_probability',1.1),('dino_specificity',.3)]:
            bad=copy.deepcopy(own);bad[0][field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):r.validate_rows(bad,ext,records,'weights')

    def test_bootstrap_uses_source_means_and_all_declared_pairs(self):
        own,ext,records=fixtures();indexed=r.validate_rows(own,ext,records,'weights')
        for row in indexed.values():
            row['psnr_db']=20+int(row['source_index'])/100+int(row['noise_seed'])-2001
            if r.family(row['N'],row['method'])=='M1':row['psnr_db']+=2
        class EngineeringBootstrap:
            def interval(self,values):
                a=np.asarray(values);self_values.append(a.copy())
                return dict(mean=float(a.mean()),ci_low=float(a.min()),ci_high=float(a.max()),n_sources=len(a))
        self_values=[]
        summary,per_source,paired=r.aggregate(indexed,types.SimpleNamespace(Bootstrap=EngineeringBootstrap))
        self.assertEqual((len(summary),len(per_source),len(paired)),(390,39000,780))
        self.assertTrue(all(len(x)==100 for x in self_values))
        point=next(x for x in per_source if x['N']==1024 and x['snr_db']==1 and x['comparison_family']=='P' and x['metric']=='psnr_db' and x['source_index']==0)
        self.assertEqual(point['value'],21.)
        pair=next(x for x in paired if x['family_A']=='M1' and x['family_B']=='P' and x['metric']=='psnr_db')
        self.assertAlmostEqual(pair['mean'],2.)
        scopes={x['comparison_scope'] for x in paired if (x['family_A'],x['family_B'])!=('HiFiDiffCom','SwinJSCC')}
        self.assertEqual(scopes,{'same_source_means; physical_noise_and_waveform_identity_not_assumed'})

    def test_original_bootstrap_implementation_remains_source_unit(self):
        source=Path(os.environ.get('OWN_CONTROLS_TEST_ROOT',str(Path(__file__).resolve().parents[2]/'historical_eval_20261003/source')))
        path=source/'experiments/unified-metrics-20261002/analysis.py'
        if not path.exists():self.skipTest('Original analysis source unavailable in this engineering checkout')
        tree=ast.parse(path.read_text(encoding='utf-8'));klass=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Bootstrap')
        ns=dict(np=np,REPLICATES=10000,SEED=20261002);exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),ns)
        boot=ns['Bootstrap']();values=np.linspace(0,1,100);interval=boot.interval(values)
        expected=np.random.default_rng(20261002).integers(0,100,(10000,100));means=values[expected].mean(1)
        self.assertEqual(interval['n_sources'],100);self.assertEqual(interval['mean'],float(values.mean()))
        np.testing.assert_allclose([interval['ci_low'],interval['ci_high']],np.percentile(means,[2.5,97.5]),rtol=0,atol=0)

    def test_full_plan_cannot_hide_missing_receiver_cost(self):
        complete,missing=r.completion_claim(9000,480,True,False)
        self.assertFalse(complete);self.assertEqual(missing,['matched_own_receiver_cost'])
        self.assertEqual(r.completion_claim(9000,480,True,True),(True,[]))
        with self.assertRaises(RuntimeError):r.completion_claim(8999,480,True,True)
        with self.assertRaises(RuntimeError):r.completion_claim(9000,479,True,True)
        rows,bindings,valid=r.cost_gate(None,[])
        self.assertEqual(len(rows),18);self.assertFalse(valid);self.assertEqual(bindings,{})
        self.assertTrue(all(x['status']=='NOT_MEASURED_COMPARABLY' for x in rows))

    def test_fixed_float_identity_rejects_missing_or_swapped_images(self):
        image=np.zeros((3,256,256),np.float32);fingerprint=r.c.rgb_sha(image)
        indexed={};population={}
        for i in r.external.FIXED:
            images={}
            for n in r.c.BUDGETS:
                for s in r.c.SNRS:
                    for m in r.FAMILIES:
                        indexed[i,n,s,2001,m]=dict(reference_sha256=fingerprint,image_sha256=fingerprint)
                        images[n,s,2001,m]=image
            population[i]=(image,images)
        r.verify_figure_population(indexed,population)
        altered=image.copy();altered[0,0,0]=1
        population[0][1][1024,1,2001,'D_U']=altered
        with self.assertRaises(RuntimeError):r.verify_figure_population(indexed,population)


if __name__=='__main__':unittest.main()
