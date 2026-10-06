"""Synthetic H/P source/hash/metric admission tests; no actual quality rows."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

import h_p_result_normalization as n
import h_p_main_source_statistics as statistics


def fixture(base):
    proof=base/'identity.json';proof.write_text('{"fixture":true}\n',encoding='utf-8')
    binding={str(proof):n.sha(proof)};ids=[f'source{i}' for i in range(100)]
    pix=np.zeros((3,256,256),dtype=np.uint8);hs=n.target_hashes(pix);targets=[]
    for i,sid in enumerate(ids):
        r=n.target_record(i,sid,pix,pix,hs['preprocessing_sha256'],hs['preprocessing_sha256'],hs['H'],hs['P'])
        r['true_class']=1;targets.append(r)
    admission=dict(status='FROZEN_H_P_COMMON_METRIC_ADMISSION',used_for_selection=False,input_bindings=binding,metrics={})
    for m in n.METRICS:
        if m in n.P_MISSING:
            admission['metrics'][m]=dict(status='MISSING_IN_P',reason='Not present in old scalar record');continue
        ident=dict(definition='fixture:'+m,models={} if m in ('psnr_db','ms_ssim') else {'fixture-weight':'a'*64},preprocessing='fixture RGB01')
        branches={b:dict(identity=copy.deepcopy(ident),precision='fixture_float32',batch_execution='fixture_B1',evidence=binding)
            for b in ('H','P')}
        if m=='ms_ssim':
            for proof in branches.values():proof['implementation']=binding.copy()
        if m=='psnr_db':
            branches['H']['precision']='H_float64_MSE_then_minus10_log10'
            branches['P']['precision']='P_retained_original_float32_Torch_PSNR_scalar'
        admission['metrics'][m]=dict(status='ADMITTED',branches=branches,numerical_difference_admitted=True,
            numerical_difference_note='Explicit synthetic fixture precision provenance',definition_reviewed_before_comparison=True)
    data={}
    for b in ('H','P'):
        rows=[]
        for i,sid in enumerate(ids):
            slots=range(18) if b=='H' else range(2)
            for slot in slots:
                snr=(13,19)[slot%2]
                for seed in n.NOISE[b]:
                    r=dict(source_index=i,source_id=sid,N=1024,snr_db=snr,noise_seed=seed,reference_sha256=hs[b],
                        resnet50_prediction=2,resnet50_source_prediction=2,convnext_source_prediction=3)
                    r.update({m:(False if m in n.BOOL_METRICS else .2) for m in n.METRICS})
                    r['dino_specificity']=0.;r['semantic_error']=0
                    if b=='H':
                        r.update(development_slot=slot,point_id=f'H18_SLOT_{slot:02d}',true_class=1,
                            used_for_selection=False,holdout_used=False,mismatch_source_id=ids[(i+1)%100])
                    else:
                        for m in n.P_MISSING|set(n.P_DERIVED):r.pop(m)
                        for k in ('source_index','N','snr_db','noise_seed','psnr_db','lpips_alex'):r[k]=str(r[k])
                        r.update(method='P1024',true_class_index='1',p_replay_exact_RGB=True,p_replay_new_noise_draws=0,
                            p_replay_new_packet_decodes=0,replay_mismatch_source_id=ids[(i+1)%100])
                    rows.append(r)
        data[b]=dict(branch=b,normal_owner_success=True,rows=rows,source_ids=ids,policy_sha256=('b' if b=='H' else 'c')*64,bindings=binding)
    return data['H'],data['P'],targets,admission


class NormalizationTests(unittest.TestCase):
    def test_reference_hashes_differ_but_exact_pixels_have_one_explicit_common_domain(self):
        p=np.zeros((3,256,256),np.uint8);p[0,0,0]=42;v=n.target_hashes(p)
        self.assertNotEqual(v['H'],v['P'])
        result=n.target_record(0,'s',p,p,v['preprocessing_sha256'],v['preprocessing_sha256'],v['H'],v['P'])
        self.assertEqual(result['reference_sha256'],v['common']);self.assertEqual(result['original_reference_sha256'],{'H':v['H'],'P':v['P']})
        changed=p.copy();changed[0,0,1]=1
        with self.assertRaisesRegex(ValueError,'pixels differ'):n.target_record(0,'s',p,changed,v['preprocessing_sha256'],v['preprocessing_sha256'],v['H'],v['P'])
        with self.assertRaisesRegex(ValueError,'hash domain'):n.target_record(0,'s',p,p,v['preprocessing_sha256'],v['preprocessing_sha256'],v['P'],v['H'])

    def test_full_view_preserves_old_rows_and_branch_noise_and_is_accepted_by_pure_statistics(self):
        with tempfile.TemporaryDirectory() as td:
            h,p,targets,c=fixture(Path(td));before=(n.identity(h['rows']),n.identity(p['rows']))
            out=n.normalize(h,p,targets,c)
            self.assertEqual(len(out['rows']),6000);self.assertEqual(len(out['points']),20);self.assertEqual(len(out['metrics']),18)
            self.assertEqual(before,(n.identity(h['rows']),n.identity(p['rows'])))
            self.assertEqual(out['points']['H18_SLOT_00']['noise_seeds'],[6201,6202,6203])
            self.assertEqual(out['points']['P1024_SNR_13']['noise_seeds'],[2001,2002,2003])
            means=statistics.source_means(out['rows'],out['source_ids'],out['points'],out['metrics'])
            self.assertEqual(means['P1024_SNR_13']['psnr_db'].shape,(100,))
            self.assertFalse(out['MAIN_complete']);self.assertFalse(out['comparisons_selected'])
            self.assertFalse(out['frame_level_noise_pairing_claimed'])

    def test_three_missing_P_metrics_never_backfilled_and_two_derivations_are_declared(self):
        with tempfile.TemporaryDirectory() as td:
            h,p,targets,c=fixture(Path(td));out=n.normalize(h,p,targets,c)
            missing={r['metric'] for r in out['availability'] if r['status']=='MISSING_IN_P'}
            self.assertEqual(missing,n.P_MISSING)
            derived={r['metric'] for r in out['availability'] if r.get('value_origin_P')=='DERIVED_FROM_RETAINED_SCALARS'}
            self.assertEqual(derived,set(n.P_DERIVED))
            for m in n.P_MISSING:
                with self.assertRaisesRegex(ValueError,'unavailable'):n.value(p['rows'][0],m,'P')
            self.assertEqual(n.value(dict(resnet50_prediction=3,resnet50_source_prediction=2),'semantic_error','P'),1)
            self.assertAlmostEqual(n.value(dict(dino_cosine='.8',dino_mismatched='.3'),'dino_specificity','P'),.5)

    def test_wrong_model_or_unbound_identity_evidence_blocks_admission(self):
        with tempfile.TemporaryDirectory() as td:
            h,p,targets,c=fixture(Path(td));bad=copy.deepcopy(c)
            bad['metrics']['lpips_alex']['branches']['P']['identity']['models']['fixture-weight']='d'*64
            with self.assertRaisesRegex(ValueError,'model/preprocessing'):n.admit_metrics(bad,{'H':h,'P':p})
            bad=copy.deepcopy(p);bad['bindings']={}
            with self.assertRaisesRegex(ValueError,'that admitted branch'):n.admit_metrics(c,{'H':h,'P':bad})
            bad=copy.deepcopy(c);bad['metrics']['mse']=copy.deepcopy(c['metrics']['psnr_db'])
            with self.assertRaisesRegex(ValueError,'cannot be inferred'):n.admit_metrics(bad,{'H':h,'P':p})

    def test_float64_H_and_retained_P_float32_precision_are_not_implicitly_equated(self):
        with tempfile.TemporaryDirectory() as td:
            h,p,targets,c=fixture(Path(td));bad=copy.deepcopy(c)
            bad['metrics']['psnr_db']['numerical_difference_admitted']=False
            with self.assertRaisesRegex(ValueError,'PSNR precision'):n.admit_metrics(bad,{'H':h,'P':p})
            bad=copy.deepcopy(c);bad['metrics']['lpips_alex']['branches']['P']['batch_execution']='old mixed B1/B16'
            bad['metrics']['lpips_alex']['numerical_difference_admitted']=False
            with self.assertRaisesRegex(ValueError,'numerical/batch'):n.admit_metrics(bad,{'H':h,'P':p})

    def test_MS_SSIM_has_no_learned_weights_but_requires_same_bound_implementation(self):
        with tempfile.TemporaryDirectory() as td:
            h,p,targets,c=fixture(Path(td));n.admit_metrics(c,{'H':h,'P':p})
            bad=copy.deepcopy(c);bad['metrics']['ms_ssim']['branches']['H']['implementation']={}
            with self.assertRaisesRegex(ValueError,'implementation must'):n.admit_metrics(bad,{'H':h,'P':p})
            bad=copy.deepcopy(c);bad['metrics']['ms_ssim']['branches']['H']['identity']['models']={'fake-weight':'a'*64}
            with self.assertRaisesRegex(ValueError,'no learned'):n.admit_metrics(bad,{'H':h,'P':p})

    def test_pending_P_missing_frame_seed_coercion_or_source_prediction_difference_refused(self):
        with tempfile.TemporaryDirectory() as td:
            h,p,targets,c=fixture(Path(td));bad=copy.deepcopy(p);bad['normal_owner_success']=False
            with self.assertRaisesRegex(ValueError,'pending P'):n.normalize(h,bad,targets,c)
            bad=copy.deepcopy(p);bad['rows'].pop()
            with self.assertRaisesRegex(ValueError,'Full H18/P2'):n.normalize(h,bad,targets,c)
            for key,value in (('noise_seed','6201'),('resnet50_source_prediction',4),('reference_sha256','f'*64)):
                bad=copy.deepcopy(p);bad['rows'][0][key]=value
                with self.assertRaises(ValueError):n.normalize(h,bad,targets,c)

    def test_DINO_mismatch_pair_must_equal_original_across_branches(self):
        with tempfile.TemporaryDirectory() as td:
            h,p,targets,c=fixture(Path(td));p['rows'][0]['replay_mismatch_source_id']='source2'
            with self.assertRaisesRegex(ValueError,'permutation differs'):n.normalize(h,p,targets,c)

    def test_loader_reads_only_exact_bound_source_pixels_and_rejects_archive_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);archive=base/'pixels.npz';cp=base/'checkpoint.json';pix=np.zeros((3,256,256),np.uint8)
            np.savez(archive,pixels=pix,tokens=np.array([123]))
            pre=n.target_hashes(pix)['preprocessing_sha256'];ids=[f's{i}' for i in range(100)]
            entry=dict(source_index=0,source_id=ids[0],preprocessing_id=pre,original_development_data_binding='bound',
                archive=str(archive),evaluation_class_index=1,outputs={str(archive):n.sha(archive)})
            cp.write_text(json.dumps(entry),encoding='utf-8')
            record=dict(entry,checkpoint=str(cp),checkpoint_sha256=n.sha(cp))
            status='H_DEVELOPMENT100_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE'
            manifest=dict(status=status,source_ids=ids,records=[record]+[{}]*99)
            completion=dict(status=status,outputs={str(cp):n.sha(cp),str(archive):n.sha(archive)})
            population=dict(source_ids=ids,stage='m1_development',calibration_or_development='m1_development',
                preprocessing_ids=[pre]*100,data_bindings=['bound']*100)
            actual,info=n.load_target(manifest,completion,population,0)
            self.assertTrue(np.array_equal(actual,pix));self.assertEqual(info['true_class'],1)
            archive.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed bound'):n.load_target(manifest,completion,population,0)


if __name__=='__main__':unittest.main()
