"""Synthetic CPU-only adapter tests; no Torch, model, GPU or packet decoder."""
import ast
from contextlib import nullcontext
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import h_development_metric_adapter as m

HERE = Path(__file__).absolute().parent


def fixture():
    target = np.zeros((3,256,256), np.float32); image = np.full_like(target,.5)
    ids = [f'source{i}' for i in range(100)]; schedule=[]; rows=[]
    for slot in range(18):
        role='H_WHOLE_SYSTEM' if slot<8 else 'H_RAW_PARTIAL_SYSTEM' if slot<10 else 'H_FIXED_M7_ATTRIBUTION'
        candidate=dict(candidate_id=f'c{slot}',snr_db=13 if slot%2==0 else 19,arm=f'arm{slot}')
        schedule.append(dict(development_slot=slot,role=role,candidate=candidate,status='FROZEN_POLICY_READY',phase='development'))
        for seed in m.SEEDS:
            rows.append(dict(development_slot=slot,noise_seed=seed,source_index=0,source_id=ids[0],phase='development',
                role=role,**candidate,source_status='WIRE_REJECT_GRAY',rx_summary={'source_decode_complete':True,'image_sha256':m.rgb_sha(image)},
                image_key='gray',image_sha256=m.rgb_sha(image),mse=.25,psnr_db=float(-10*np.log10(.25))))
    source=dict(source_index=0,source_id=ids[0],class_index=1,target=target,reference_sha256=m.rgb_sha(target),
        target_preprocessing_sha256=hashlib.sha256(np.zeros(target.shape,np.uint8).tobytes()).hexdigest(),rows=rows,images={'gray':image})
    mismatch=dict(source_ids=ids,permutation=list(range(1,100))+[0],replay_source_sha256=m.REPLAY_SHA)
    backend=FakeBackend(); refs=[]
    for sid in ids:
        p=backend.prepare(target); f=p['dino_feature']
        refs.append(dict(source_id=sid,reference_sha256=m.rgb_sha(target),prepared=p,
            feature_sha256=hashlib.sha256(f.tobytes()).hexdigest(),evaluator_identity=backend.identity))
    return source,schedule,mismatch,backend,refs


class FakeBackend:
    identity='fake-frozen-model-and-flags'
    def __init__(self):self.calls=0
    def prepare(self,reference):return {'dino_feature':np.array([1,0],np.float32)}
    def score(self,reference,image,label,prepared,negative):
        self.calls+=1
        d={k:.2 for k in m.REQUIRED}
        for k in ('resnet50_top1_label','resnet50_top1_source_prediction','resnet50_source_top1_label',
                  'convnext_top1_label','convnext_top1_source_prediction','convnext_source_correct_to_wrong','convnext_source_wrong_to_correct'):
            d[k]=False
        d.update(label_conditioned=False,convnext_used_for_selection=False,legacy_quality_psnr_db=6.020600318,
            convnext_source_prediction_agreement=0,resnet50_prediction=2,resnet50_source_prediction=1,
            convnext_prediction=2,convnext_source_prediction=1)
        return d


class Tensor:
    def __init__(self,array):self.a=np.asarray(array)
    def to(self,_):return self
    def float(self):return Tensor(self.a.astype(np.float32))
    def cpu(self):return self
    def numpy(self):return self.a
    def tolist(self):return self.a.tolist()
    def __getitem__(self,k):return Tensor(self.a[k])
    def __int__(self):return int(self.a)
    def softmax(self,axis):
        x=np.exp(self.a-self.a.max(axis=axis,keepdims=True));return Tensor(x/x.sum(axis=axis,keepdims=True))
    def max(self,axis):return Tensor(self.a.max(axis=axis)),Tensor(self.a.argmax(axis=axis))


class AdapterTests(unittest.TestCase):
    def test_load_actual_archive_schema_label_and_hashes_without_tokens(self):
        s,sch,mis,b,refs=fixture()
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); assetdir=base/'assets'; assetdir.mkdir(); out=base/'render'
            for name in ('images','sources','source_checkpoints'):(out/name).mkdir(parents=True)
            archive=assetdir/'0000.npz'; target=np.zeros((3,256,256),np.uint8)
            # Access to this intentionally invalid token object would fail under allow_pickle=False.
            np.savez(archive,pixels=target,tokens=np.array([object()],dtype=object))
            pre=hashlib.sha256(target.tobytes()).hexdigest();binding={'index':0,'rgb_sha256':pre,'source_npz_sha256':'a'*64}
            cp=assetdir/'0000.json';asset=dict(source_index=0,source_id='source0',preprocessing_id=pre,
                original_development_data_binding=binding,evaluation_class_index=7,archive=str(archive),outputs={str(archive):m.sha(archive)})
            cp.write_text(json.dumps(asset));entry=dict(asset,checkpoint=str(cp),checkpoint_sha256=m.sha(cp))
            pop=dict(stage='m1_development',calibration_or_development='m1_development',source_ids=mis['source_ids'],
                preprocessing_ids=[pre]*100,data_bindings=[binding]*100)
            status='H_DEVELOPMENT100_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE'
            manifest=dict(status=status,source_count=100,source_ids=mis['source_ids'],records=[entry]*100)
            assets=dict(status=status,source_count=100,outputs={str(cp):m.sha(cp),str(archive):m.sha(archive)})
            imagepath=out/'images/0000.npz';np.savez(imagepath,**s['images'])
            rows=copy.deepcopy(s['rows'])
            for row in rows:row.update(image_archive=str(imagepath),target_preprocessing_sha256=pre)
            rp=out/'sources/0000.json';rp.write_text(json.dumps(rows));rcp=out/'source_checkpoints/0000.json'
            doc=dict(status='H_DEVELOPMENT_RX_SOURCE_COMPLETE',source_index=0,source_id='source0',frame_count=54,
                registration_sha256='b'*64,source_decode_complete=True,outputs={str(rp):m.sha(rp),str(imagepath):m.sha(imagepath)})
            rcp.write_text(json.dumps(doc))
            render=dict(status='H_DEVELOPMENT_RX_COMPLETE',source_count=100,frame_count=5400,source_ids=mis['source_ids'],
                source_decode_complete=True,MAIN_frames=0,registration_sha256='b'*64,
                outputs={str(x):m.sha(x) for x in (rcp,rp,imagepath)})
            got=m.load_source(0,pop,manifest,assets,render,out,sch)
            self.assertEqual(got['class_index'],7);self.assertEqual(len(got['rows']),54);self.assertTrue(np.array_equal(got['target'],s['target']))
            self.assertNotIn('tokens',got);self.assertEqual(got['reference_sha256'],m.rgb_sha(s['target']))
            manifest['records'][0]['evaluation_class_index']=8
            with self.assertRaisesRegex(RuntimeError,'label/archive'):m.load_source(0,pop,manifest,assets,render,out,sch)

    def test_all54_failures_retained_and_exact_image_reused(self):
        s,sch,mis,b,refs=fixture(); rows,cost=m.score_source(s,b,refs,mis,sch)
        self.assertEqual(len(rows),54);self.assertEqual(b.calls,1);self.assertEqual(cost['exact_duplicate_metric_reuses'],53)
        self.assertTrue(all(r['source_status']=='WIRE_REJECT_GRAY' for r in rows))
        self.assertTrue(all(r['mse']==.25 and r['psnr_db']==s['rows'][0]['psnr_db'] for r in rows))
        self.assertTrue(all(r['F_recovery_error'] is None and r['FID_status']=='DEFERRED_HOLDOUT' for r in rows))
        self.assertIs(rows[0]['convnext_top1_label'],False)

    def test_changed_rgb_and_reference_cannot_reuse_scores(self):
        s,sch,mis,b,refs=fixture();s['images']['gray'][0,0,0]=.6
        with self.assertRaisesRegex(RuntimeError,'RGB changed'):m.score_source(s,b,refs,mis,sch)
        s,sch,mis,b,refs=fixture();s['target'][0,0,0]=.3
        with self.assertRaisesRegex(RuntimeError,'provenance changed'):m.score_source(s,b,refs,mis,sch)

    def test_no_cross_source_cache_even_if_rgb_is_identical(self):
        s,sch,mis,b,refs=fixture();m.score_source(s,b,refs,mis,sch)
        s['source_index']=1;s['source_id']='source1'
        for r in s['rows']:r.update(source_index=1,source_id='source1')
        m.score_source(s,b,refs,mis,sch);self.assertEqual(b.calls,2)

    def test_old_noise_MAIN_and_pending_frames_rejected(self):
        for key,value in (('noise_seed',2001),('development_slot',18),('source_status','ARITHMETIC_CANONICAL_PENDING')):
            s,sch,mis,b,refs=fixture();s['rows'][0][key]=value
            with self.assertRaises(RuntimeError):m.score_source(s,b,refs,mis,sch)

    def test_unknown_quality_and_software_failure_never_become_zero_or_gray(self):
        s,sch,mis,b,refs=fixture();original=b.score
        def missing(*a):
            d=original(*a);d['clip_image_cosine']=None;return d
        b.score=missing
        with self.assertRaisesRegex(RuntimeError,'Incomplete actual metrics'):m.score_source(s,b,refs,mis,sch)
        def broken(*a):raise ValueError('software problem')
        b.score=broken
        with self.assertRaisesRegex(ValueError,'software problem'):m.score_source(s,b,refs,mis,sch)

    def test_mismatch_feature_and_model_identity_bound(self):
        s,sch,mis,b,refs=fixture();refs[1]['prepared']['dino_feature'][0]=.9
        with self.assertRaisesRegex(RuntimeError,'feature bytes'):m.score_source(s,b,refs,mis,sch)
        s,sch,mis,b,refs=fixture();b.identity='different'
        with self.assertRaisesRegex(RuntimeError,'provenance changed'):m.score_source(s,b,refs,mis,sch)

    def test_original_derangement_code_exact_source_and100population(self):
        path=HERE.parents[1]/'metrics_20261002/source/replay.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ReplayEngine')
        fn=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_derangement'));fn.decorator_list=[]
        namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
        replay=SimpleNamespace(__file__=str(path),ReplayEngine=SimpleNamespace(_derangement=namespace['_derangement']))
        ids=[f's{i}' for i in range(100)];got=m.frozen_mismatch(replay,ids)
        self.assertEqual(got['permutation'],namespace['_derangement']());self.assertEqual(got['seed'],20260930)
        with self.assertRaises(RuntimeError):m.frozen_mismatch(replay,ids[:99])

    def test_legacy_float32_psnr_is_diagnostic_not_H_replacement(self):
        s,sch,mis,b,refs=fixture();rows,_=m.score_source(s,b,refs,mis,sch)
        self.assertNotEqual(rows[0]['legacy_quality_psnr_db'],rows[0]['psnr_db'])
        s['rows'][0]['mse']+=1e-12
        with self.assertRaisesRegex(RuntimeError,'MSE/PSNR changed'):m.score_source(s,b,refs,mis,sch)

    def test_batch_one_actual_suite_and_independent_API_bridge_with_fakes(self):
        calls=[];flags={'threads':6,'matmul_tf32':False}
        def cosine(a,b,dim):
            return Tensor((a.a*b.a).sum(axis=dim)/(np.linalg.norm(a.a,axis=dim)*np.linalg.norm(b.a,axis=dim)))
        torch=SimpleNamespace(from_numpy=Tensor,inference_mode=nullcontext,autocast=lambda **k:nullcontext(),
            nn=SimpleNamespace(functional=SimpleNamespace(cosine_similarity=cosine)))
        metric=SimpleNamespace(no_network=nullcontext,preprocess_resnet50=lambda x:x)
        class Eval:
            device=SimpleNamespace(type='cpu');metadata={'metrics':{'clip':{'status':'READY'}}}
            models={'resnet50':lambda image:Tensor([[0,1,2]])}
            def identity(self):return m.identity(self.metadata)
            def prepare_reference(self,x):calls.append(('prepare',x.a.shape));return {'resnet50':Tensor([1])}
            def score(self,ref,image,labels,label_conditioned,prepared):
                calls.append(('score',image.a.shape,labels,label_conditioned))
                d={k:.4 for k in ('clip_image_cosine','dists','dreamsim','dinov2_vitl14_cosine','ms_ssim')}
                d.update(resnet50_prediction=2,resnet50_source_prediction=1,resnet50_top1_label=False,
                    resnet50_top1_source_prediction=False,resnet50_source_top1_label=True,label_conditioned=False);return [d]
        def quality(ref,images,*args):
            self.assertEqual(len(images),1);calls.append(('legacy',1));return [dict(psnr_db=6.02,lpips_alex=.2,dino_cosine=0.)],np.array([1,0],np.float32),np.array([[0,1]],np.float32)
        native=SimpleNamespace(dino_features=lambda d,x:Tensor(np.array([[1,0]],np.float32)),quality_metrics=quality)
        validation=HERE.parents[1]/'prior_aware_uep_20261004/runtime/validation_metrics.py'
        sp=importlib.util.spec_from_file_location('h_test_validation',validation);v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
        class Conv:
            identity=dict(weights_sha256=m.CONVNEXT_SHA,classification_batch_size=1,used_for_selection=False)
            def predict(self,rgb):calls.append(('conv-predict',rgb.shape));return 1
            def score(self,rgb,*,true_label,source_prediction):
                calls.append(('conv-score',rgb.shape));return dict(v.diagnostics(true_label,source_prediction,2),convnext_used_for_selection=False)
        b=m.SuiteBackend((Eval(),metric,native,None,None,{},flags),Conv(),torch,lambda _:dict(flags))
        ref=np.zeros((3,256,256),np.float32);prepared=b.prepare(ref)
        values=b.score(ref,np.full_like(ref,.5),1,prepared,np.array([0,1],np.float32))
        self.assertEqual(values['dino_mismatched'],1.);self.assertTrue(values['convnext_source_correct_to_wrong'])
        self.assertIn(('score',(1,3,256,256),[1],False),calls);self.assertIs(values['label_conditioned'],False)
        b.evaluator.metadata['changed']=True
        with self.assertRaisesRegex(RuntimeError,'identity changed'):b.check()

    def test_no_constructor_statistics_or_execution_entry(self):
        text=Path(m.__file__).read_text();tree=ast.parse(text)
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(calls)&{'paired','summarize','ConvNeXtValidation','MetricEvaluator','Popen'})
        self.assertNotIn("__name__ == '__main__'",text)

    def test_scored_rows_fit_new_H_statistics_without_old_defaults(self):
        import h_development_statistics as statistics
        s,sch,mis,b,refs=fixture();rows,_=m.score_source(s,b,refs,mis,sch)
        full=[dict(row,source_index=i,source_id=mis['source_ids'][i]) for i in range(100) for row in rows]
        means=statistics.validate(full,mis['source_ids'],m.declared_points(sch),['psnr_db','clip_image_cosine','convnext_source_prediction_agreement'])
        self.assertEqual(len(means),18);self.assertTrue(all(v['psnr_db'].shape==(100,) for v in means.values()))
        self.assertEqual(statistics.SEEDS,m.SEEDS);self.assertEqual(statistics.BOOTSTRAP_SEED,2026100605)

    def test_TX_fallback_and_actual_RX_scale_are_separate_postscore(self):
        s,sch,mis,b,refs=fixture();rows=copy.deepcopy(s['rows']);frames=[]
        for j,row in enumerate(rows):
            row.update(event_id=f'e{j}',gray=True,received_m=None,received_K=None,received_profile_id=None,nominal_rate='1/2')
            fixed=row['role']=='H_FIXED_M7_ATTRIBUTION'
            tx=dict(profile_id=1,target_m=7 if fixed else 9,actual_m=7 if fixed else 8,K=0,mode='raw',fell_back=not fixed)
            fields={k:row[k] for k in ('event_id','source_id','source_index','development_slot','noise_seed','snr_db','candidate_id','role','arm')}
            frames.append(dict(fields,status='H_DEVELOPMENT_CPU_FRAME_TRACE',stage='development',tx=tx,
                total_symbols=1024,frame_normalized=False,header_energy=136.,body_energy=1912.,total_energy=2048.))
        rows[0].update(gray=False,source_status='RAW_SOURCE_DECODED',received_m=6,received_K=0,received_profile_id=3)
        source=dict(status='H_DEVELOPMENT_CPU_SOURCE_TRACES',frames=frames)
        got=m.attach_wire_accounting(rows,source)
        self.assertEqual((got[0]['tx_target_m'],got[0]['tx_actual_m'],got[0]['rx_accepted_m']),(9,8,6))
        self.assertIsNone(got[1]['rx_accepted_m']);self.assertTrue(got[0]['rx_canonical_accepted'])
        self.assertNotIn('tx_actual_m',rows[0]);self.assertEqual(len(got),54)
        source['frames'][1]['tx']['actual_m']=7
        with self.assertRaisesRegex(RuntimeError,'depend on receiver noise'):m.attach_wire_accounting(rows,source)


if __name__=='__main__':unittest.main()
