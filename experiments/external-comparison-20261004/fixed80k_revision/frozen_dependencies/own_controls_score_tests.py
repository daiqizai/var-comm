"""CPU contract tests using actual lossless cache files, never model stubs as results."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import own_controls_score_common as c
import own_controls_score as scorer
from external_eval_common import write,sha,identity,rgb_sha,atomic_npz


def sealed(path,value):
    value = dict(value); value['payload_sha256'] = identity(value); write(path,value); return value


def fixture(folder):
    """The native target has equal uint8 identity but genuinely different float RGB."""
    native = np.full((3,256,256),np.float32(128/255),dtype=np.float32)
    target = np.nextafter(native,np.float32(1))
    image = np.full_like(target,.375)
    record = dict(source_index=0,image_id='fixed-source',class_index=7,
        preprocessing_id=hashlib.sha256(np.rint(target*255).astype(np.uint8).tobytes()).hexdigest())
    policy = dict(cells=[dict(snr_db=s,method=m,action=dict(N=2048,phy='QPSK',m=6,q=0,order='entropy'))
                        for s in c.SNRS for m in c.METHODS[2048][1:]])
    reg = dict(cache_inventory=[{} for _ in range(100)],policy=policy,policy_sha256='policy-sha')
    entry = dict(source_index=0,record=record,reference_sha256=rgb_sha(target),stages={})
    for stage in ('export-n1024','development'):
        directory = folder/stage; directory.mkdir()
        registration = dict(stage=stage); rp = directory/'registration.json'; write(rp,registration)
        rows = []
        for n,s,seed,m in c.expected_keys():
            if stage=='export-n1024' and n!=1024: continue
            if stage=='development' and (n!=2048 or m=='P2048'):continue
            row = dict(source_index=0,source_id=record['image_id'],source_class_index=7,
                preprocessing_id=record['preprocessing_id'],N=n,snr_db=s,noise_seed=seed,method=m,
                image_slot=0,image_sha256=rgb_sha(image),reference_sha256=rgb_sha(target),
                native_reference_sha256=rgb_sha(native),policy_selection_updates=0)
            if stage=='export-n1024':
                original = dict(method='entropy_policy' if m.startswith('M1') else m,snr_db=s,noise_seed=seed,
                    psnr_db=17.25,lpips_alex=.6,dino_cosine=.7)
                row.update(original_scientific_row=original,original_scientific_row_sha256=identity(original),
                    original_scalar_parity=dict(status='PASS'),original_completed_metrics=dict(image_sha256=rgb_sha(image),
                        reference_sha256=rgb_sha(native),clip_image_cosine=.9))
            else:
                row.update(E=4096,phy_family='QPSK',m=6,q=0,order='entropy',policy_sha256='policy-sha',
                    policy_selected_on_development=False,semantic_side_information=False,receiver_class_embedding=1000,
                    native_psnr_db=18.5,native_lpips_alex=.5,native_dino_cosine=.75)
            rows.append(row)
        cache = directory/'cache.npz'; proof = c.c.save_float_cache(cache,[image],native,target)
        cp_path = directory/'checkpoint.json'
        sealed(cp_path,dict(binding=identity(registration),source_index=0,rows=rows,float_reconstructions=proof,synthetic=False))
        entry['stages'][stage] = dict(checkpoint=str(cp_path),checkpoint_sha256=sha(cp_path),
            registration=str(rp),registration_sha256=sha(rp),archive=str(cache),archive_sha256=sha(cache))
    directory = folder/'P2048'; directory.mkdir(); cache = directory/'cache.npz'; rp = directory/'registration.json'
    oldreg = dict(source_identity=[record],modelmanifest_sha256='m',evaluator_identity='e',numerical_runtime={},
        metric_batch_size=1,batch_qualification_sha256='b',metric_qualified_batch_sizes=[1])
    write(rp,oldreg); rows,parities = [],[]
    for s in c.SNRS:
        for seed in c.SEEDS:
            original = dict(method='P2048',N=2048,snr_db=s,noise_seed=seed,psnr_db=20.,lpips_alex=.3,dino_cosine=.8)
            rid = f'p-{s}-{seed}'
            meta = dict(method_id='P2048',N=2048,decoder_id='Dc',label_conditioned=False,training_seed=2026092304)
            row = dict(original,history_row_id=rid,history_study=c.STUDY,history_source_index=0,
                history_source_id=record['image_id'],history_preprocessing_id=record['preprocessing_id'],
                history_true_class_index=7,history_image_sha256=rgb_sha(image),history_reference_sha256=rgb_sha(target),
                history_original_row_sha256=identity(original),history_replay_parity_passed=True,
                history_metadata_json=json.dumps(meta))
            rows.append(row);parities.append(dict(replay_parity_passed=True,synthetic=False,history_row_id=rid))
    ids=[r['history_row_id'] for r in rows];slots=[0]*9
    atomic_npz(cache,images=image[None],source_rgb=target,row_ids=np.asarray(ids),image_slots=np.asarray(slots))
    proof=dict(path=str(cache),sha256=sha(cache),dtype='float32',layout='CHW',rows=9,unique_images=1,
        row_ids=ids,image_slots=slots,image_sha256=[hashlib.sha256(image.tobytes()).hexdigest()],
        reference_sha256=hashlib.sha256(target.tobytes()).hexdigest())
    evaluation={k:oldreg[k] for k in ('modelmanifest_sha256','evaluator_identity','numerical_runtime',
                                    'metric_batch_size','batch_qualification_sha256','metric_qualified_batch_sizes')}
    cp_path=directory/'checkpoint.json'
    sealed(cp_path,dict(binding=identity(oldreg),source_index=0,rows=rows,parity=parities,float_reconstructions=proof,
        evaluation_identity=evaluation,metric_batch_sizes_used=[1],unique_images=1))
    entry['stages']['P2048']=dict(checkpoint=str(cp_path),checkpoint_sha256=sha(cp_path),registration=str(rp),
        registration_sha256=sha(rp),archive=str(cache),archive_sha256=sha(cache))
    reg['cache_inventory'][0]=entry
    return reg,target,native,image


class OwnScoreTests(unittest.TestCase):
    def test_expected_scope_is_exact(self):
        self.assertEqual(len(c.expected_keys()),54)
        ids=[rid for i in range(100) for rid in c.expected_ids(i)]
        self.assertEqual(len(ids),5400);self.assertEqual(len(set(ids)),5400)
        self.assertEqual(set(k[0] for k in c.expected_keys()),{1024,2048})

    def test_p2048_selection_rejects_wrong_seed_or_class(self):
        base=dict(history_metadata_json=json.dumps(dict(method='P2048',N=2048,decoder='Dc',
            label_conditioned=False,training_seed=2026092304)),snr_db=7,noise_seed=2001)
        self.assertEqual(c.selected_p2048(base)[0],(2048,7,2001,'P2048'))
        for change in (dict(training_seed=999),dict(label_conditioned=True),dict(decoder='D0'),dict(N=3060)):
            other=dict(base);other['history_metadata_json']=json.dumps(dict(json.loads(base['history_metadata_json']),**change))
            with self.assertRaises(RuntimeError):c.selected_p2048(other)
        self.assertIsNone(c.selected_p2048(dict(base,snr_db=19)))

    def test_real_float_cache_reference_and_native_scores_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,target,native,image=fixture(Path(tmp))
            result,rows,mapping,bindings=c.load_source_payload(tmp,0,reg)
            np.testing.assert_array_equal(result,target);self.assertFalse(np.array_equal(native,target))
            self.assertEqual(len(rows),54);self.assertEqual(len(mapping),54);self.assertEqual(len(bindings),9)
            self.assertEqual(rows[0]['native_psnr_db'],17.25);self.assertNotIn('psnr_db',rows[0])
            self.assertEqual(rows[-3]['native_psnr_db'],20.)
            self.assertEqual(rows[0]['native_reference_sha256'],rgb_sha(native))
            self.assertEqual(rows[0]['reference_sha256'],rgb_sha(target))
            self.assertEqual(json.loads(rows[0]['original_row_json'])['original_completed_metrics']['clip_image_cosine'],.9)
            images,slots=c.deduplicate(rows,mapping)
            self.assertEqual(images.shape,(1,3,256,256));self.assertEqual(slots,[0]*54)
            np.testing.assert_array_equal(images[0],image)
            a,b,d=c.load_source_images(tmp,0,reg);np.testing.assert_array_equal(a,target);self.assertEqual(set(b),set(mapping))

    def test_modified_container_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,*_=fixture(Path(tmp));p=Path(reg['cache_inventory'][0]['stages']['development']['archive'])
            with p.open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(RuntimeError,'changed'):c.load_source_payload(tmp,0,reg)

    def test_duplicate_frames_are_rejected_even_if_resealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,*_=fixture(Path(tmp));spec=reg['cache_inventory'][0]['stages']['export-n1024']
            path=Path(spec['checkpoint']);cp=c.read(path);cp.pop('payload_sha256');cp['rows'][1]=copy.deepcopy(cp['rows'][0])
            sealed(path,cp);spec['checkpoint_sha256']=sha(path)
            with self.assertRaisesRegex(RuntimeError,'Duplicate'):c.load_source_payload(tmp,0,reg)

    def test_resealed_different_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,*_=fixture(Path(tmp));reg['cache_inventory'][0]['reference_sha256']='f'*64
            with self.assertRaisesRegex(RuntimeError,'common reference'):c.load_source_payload(tmp,0,reg)

    def test_native_metric_cannot_be_silently_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg,*_=fixture(Path(tmp));spec=reg['cache_inventory'][0]['stages']['export-n1024'];path=Path(spec['checkpoint'])
            cp=c.read(path);cp.pop('payload_sha256');cp['rows'][0]['original_scientific_row']['psnr_db']=99
            sealed(path,cp);spec['checkpoint_sha256']=sha(path)
            with self.assertRaisesRegex(RuntimeError,'native/metric'):c.load_source_payload(tmp,0,reg)

    def test_confidence_means_agreement_with_original_prediction(self):
        row={'resnet50_prediction':8}
        scorer.classification(row,.5,8,7);self.assertEqual((row['semantic_error'],row['confidently_wrong']),(1,1))
        scorer.classification(row,.499,8,7);self.assertEqual(row['confidently_wrong'],0)
        scorer.classification(row,.99,8,8);self.assertEqual(row['semantic_error'],0)
        with self.assertRaises(RuntimeError):scorer.classification(row,.9,9,8)

    def test_score_resume_binds_inputs_batches_and_classifier(self):
        baseline=dict(source_id='s',preprocessing_id='p',reference_sha256='r',true_class_index=7,resnet50_source_prediction=7)
        rows=[dict(replay_row_id=rid,replay_parity_passed=True,source_index=0,source_id='s',preprocessing_id='p',
            reference_sha256='r',resnet50_source_prediction=7,resnet50_prediction=7,resnet50_top1_label=1,
            resnet50_top1_source_prediction=1,semantic_error=0,confidently_wrong=0,resnet50_top1_probability=.9)
            for rid in c.expected_ids(0)]
        numerical=dict(qualified_batch_sizes=[1]);inputs={'one':'hash'}
        value=dict(binding='b',source_index=0,rows=rows,baseline=baseline,input_bindings=inputs,
            evaluation_identity=numerical,unique_images=1,metric_batch_sizes_used=[1])
        value['payload_sha256']=identity(value)
        c.validate_scored_checkpoint(value,'b',0,inputs,numerical)
        with self.assertRaises(RuntimeError):c.validate_scored_checkpoint(value,'b',0,{'one':'other'},numerical)
        with self.assertRaises(RuntimeError):c.validate_scored_checkpoint(value,'b',0,inputs,dict(qualified_batch_sizes=[2]))
        other=copy.deepcopy(value);other['rows'][0]['confidently_wrong']=1;other.pop('payload_sha256');other['payload_sha256']=identity(other)
        with self.assertRaisesRegex(RuntimeError,'classification'):c.validate_scored_checkpoint(other,'b',0,inputs,numerical)

    def test_group_and_contrast_scope_uses_frozen_analysis(self):
        candidates=[]
        for ancestor in Path(__file__).resolve().parents:
            candidates += [ancestor/'experiments/unified-metrics-20261002/analysis.py',
                           ancestor/'.research/metrics_20261002/source/analysis.py']
        path=next((p for p in candidates if p.is_file()),None)
        self.assertIsNotNone(path,'Original CPU analysis module needed for scope check')
        spec=importlib.util.spec_from_file_location('_test_own_analysis',path);analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
        with tempfile.TemporaryDirectory() as tmp:
            reg,*_=fixture(Path(tmp));_,rows,_,_=c.load_source_payload(tmp,0,reg)
            groups,contrasts=scorer.expected_groups(analysis,rows)
        self.assertEqual(len(groups),18);self.assertEqual(len(contrasts),18)
        self.assertTrue(all(x['group_A']['N']==x['group_B']['N'] and x['group_A']['snr_db']==x['group_B']['snr_db'] for x in contrasts))
        pairs=list(analysis.contrasts(groups,dict(contrasts=contrasts)))
        self.assertEqual(len(pairs),18)


if __name__=='__main__':
    unittest.main()
