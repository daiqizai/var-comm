"""Synthetic CPU integration: exact old rows, six studies, and durable resume."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import cache_assembly as a
import final_runner as final


def locate(relative, fallback):
    for root in (Path.cwd(),*Path(__file__).resolve().parents):
        for name in (relative,fallback):
            path=root/name
            if path.is_file():return path
    raise FileNotFoundError(relative)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


support=load('assembly_original_runner_fixture',locate('experiments/unified-metrics-20261002/test_runner.py',
              '.research/metrics_20261002/source/test_runner.py'))
c=load('assembly_original_cache',locate('outputs/METRIC-CONCURRENT-R4-20261002/runtime/cache_handoff.py',
       '.research/metrics_concurrent_r4_20261002/cache_handoff.py'))
r,p=support.runner,support.replay


class Engine(support.FakeEngine):
    def __init__(self,root):
        super().__init__(root);self.replayed=[]
    def target(self,index):return np.full((3,256,256),index/101,np.float32)
    def manifest(self):
        return dict(super().manifest(),model_identity='engineering-original-model',
            source_ids=[x['image_id'] for x in self.records],
            preprocessing_ids=[x['preprocessing_id'] for x in self.records],parity_tolerances=p.PARITY_TOLERANCES)
    def iterate_source(self,study,index):
        self.replayed.append((study,index))
        for row in self.by_source[study].get(index,[]):
            rgb=np.full((3,256,256),.6 if study.startswith('M2') else .5,np.float32)
            parity=dict(replay_parity_passed=True,synthetic=False,metric_deltas={'psnr_db':0.},
                rgb_sha256=p.rgb_fingerprint(rgb),target_sha256=p.rgb_fingerprint(self.target(index)),
                original_row_sha256=p.row_id(study,row))
            yield dict(row,**self.metadata(row,study)),rgb,parity


def make_snapshot(root,engine,manifest_sha,flags):
    directory=root/'cache';directory.mkdir()
    sources=[]
    for index,record in enumerate(engine.records):
        inventory=c.row_inventory(engine,p,index)
        sources.append(dict(source_index=index,source_id=record['image_id'],preprocessing_id=record['preprocessing_id'],
            reference_sha256=p.rgb_fingerprint(engine.target(index)),true_class_index=0,
            frames=len(inventory),row_inventory_sha256=c.identity(inventory)))
    bound=dict(engine.bindings)
    reg=dict(studies=list(c.STUDIES),synthetic=False,training_updates=0,policy_selection_updates=0,
        metric_batch_size=1,sources=sources,metric_evaluator_identity='a'*64,modelmanifest_sha256=manifest_sha,
        numerical_runtime=flags,source_bindings=bound,frozen_inputs=bound,asset_inputs=bound,
        original_replay_inventory=engine.manifest())
    c.write(directory/'concurrent_registration.json',reg)
    for index,source in enumerate(sources):
        rows,scores=[],{}
        for study in c.STUDIES:
            originals={p.row_id(study,row):row for row in engine.by_source[study].get(index,[])}
            for row,rgb,parity in engine.iterate_source(study,index):
                rowid=p.row_id(study,row);key=p.metric_cache_key(rgb,engine.target(index),'a'*64)
                values=dict(clip_image_cosine=.8,dinov2_vitl14_cosine=.9,dists=.1,dreamsim=.2,ms_ssim=.6,
                    resnet50_prediction=0,resnet50_source_prediction=0,resnet50_top1_label=1,
                    resnet50_top1_source_prediction=1,resnet50_source_top1_label=1)
                scores[key]=dict(image_sha256=parity['rgb_sha256'],metrics=values)
                rows.append(dict(study=study,row_id=rowid,source_row_sha256=p.source_row_hash(originals[rowid]),
                    metric_cache_key=key,parity=parity))
        payload=dict(status='SEALED_SOURCE_METRICS_COMPLETE',synthetic=False,metric_batch_size=1,
            registration_sha256=c.identity(reg),source_index=index,source=source,rows=rows,scores=scores,
            baseline=dict(source_id=source['source_id'],preprocessing_id=source['preprocessing_id'],
                reference_sha256=source['reference_sha256'],true_class_index=0,
                resnet50_source_prediction=0,resnet50_source_top1_label=1))
        payload['payload_sha256']=c.identity(payload);c.validate_checkpoint(payload,reg,index)
        c.write(directory/'source_checkpoints'/f'{index:03}.json',payload)
    c.write(directory/'runtime_registration.json',{'source_bindings':bound})
    c.write(directory/'partial_launch.json',{'pid':123,'start_ticks':456})
    c.snapshot_receipt(directory,reg)
    return c.CacheSnapshot(directory)


def fake_torch():
    return SimpleNamespace(from_numpy=support.ArrayTensor,tensor=lambda x,dtype=None:np.asarray(x),int64=np.int64,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False,benchmark=False,deterministic=True)),
        get_float32_matmul_precision=lambda:'highest',are_deterministic_algorithms_enabled=lambda:True,
        get_num_threads=lambda:2,get_num_interop_threads=lambda:2)


def source_assembly(snapshot,engine,index=0,**changes):
    params=dict(reference_sha=snapshot.registration['sources'][index]['reference_sha256'],
        evaluator_identity='a'*64,numerical_runtime=snapshot.registration['numerical_runtime'],
        truth=0,prediction=0,manifest_sha=snapshot.registration['modelmanifest_sha256'])
    params.update(changes)
    return a.SourceAssembly(snapshot,c,p,r,engine,index,**params)


class AssemblyTests(unittest.TestCase):
    def test_real_qualification_binds_exact_implementation_and_checked_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'000.json';c.write(checkpoint,{'engineering':True})
            tables={}
            for index in range(4):
                path=root/f'original-{index}.csv';path.write_text('engineering original fields')
                tables[str(path)]=c.sha(path)
            sources={str(Path(module.__file__).resolve()):c.sha(module.__file__) for module in (a,c,p,r)}
            snapshot=SimpleNamespace(paths={0:checkpoint},bindings={str(checkpoint):c.sha(checkpoint)},
                registration={'sources':[{'frames':3}],'frozen_inputs':tables,'source_bindings':sources})
            proof=dict(status='REAL_CACHED_ROW_ASSEMBLY_PASS',fresh_reconstruction=False,fresh_metric_inference=False,
                source_indices=[0],rows=3,prior_actual_replay_parity_preserved=True,scalar_values_exact=True,
                original_fields_exact=True,classification_boundaries_preserved=True,torch_imported=False,
                assembler_sha256=c.sha(a.__file__),source_bindings=sources,original_tables=tables,
                checked_checkpoint_bindings=snapshot.bindings)
            path=root/'real_cached_row_qualification.json';c.write(path,proof)
            bindings,result=final.verify_real_qualification(root,snapshot,c)
            self.assertIn(str(path),bindings);self.assertEqual(result,proof)
            for field,value in [('assembler_sha256','f'*64),('original_fields_exact',False),('source_indices',[0,0]),
                                ('checked_checkpoint_bindings',{}),('torch_imported',True)]:
                c.write(path,dict(proof,**{field:value}))
                with self.assertRaises(RuntimeError):final.verify_real_qualification(root,snapshot,c)

    def test_full_final_matches_original_main_then_resumes_without_forward_or_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();(source/'original.py').write_text('engineering fixture')
            parent=root/'parent';parent.mkdir();engine=Engine(root);torch=fake_torch()
            pub=dict(status='PUSHED',commit='1'*40,remote_commit='1'*40,source_bindings={})
            r.write(parent/'completion.json',dict(status='AUTHORIZED_TWO_METHODS_COMPLETE',publication=pub))
            r.write(parent/'supervisor_status.json',dict(status='COMPLETE'))
            outputs=[]
            for name in ('baseline','assembled'):
                out=root/name;out.mkdir();result=root/(name+'_result');outputs.append((out,result))
                r.write(out/'source_publication.json',pub);r.write(out/'modelmanifest.json',{'ENGINEERING_SYNTHETIC':True})
                r.write(out/'models_qualification.json',dict(status='REAL_MODEL_WEIGHTS_QUALIFICATION_PASS',
                    modelmanifest_sha256=r.sha(out/'modelmanifest.json'),source_bindings={}))
            snapshot=make_snapshot(root,engine,r.sha(outputs[0][0]/'modelmanifest.json'),r.runtime_flags(torch))
            fake_replay=SimpleNamespace(create_engine=lambda _:engine,row_id=p.row_id,source_row_hash=p.source_row_hash,
                rgb_fingerprint=p.rgb_fingerprint,metric_cache_key=p.metric_cache_key)
            modules={'torch':torch,'replay':fake_replay,'metric_models':SimpleNamespace(MetricEvaluator=support.FakeEvaluator),
                     'analysis':support.analysis}
            def run(number,new):
                out,result=outputs[number]
                with mock.patch.multiple(r,HERE=source,ROOT=root,OUT=out,RESULT=result,PARENT=parent), \
                     mock.patch.dict(sys.modules,modules), \
                     mock.patch.object(support.batch_speed,'qualify_and_select_batch',side_effect=support.fake_batch_qualification), \
                     mock.patch.object(c,'source_bindings',return_value=engine.bindings):
                    if new:final.run_final(snapshot,r,fake_replay,c,engine.bindings)
                    else:r.main()
            run(0,False)
            baseline=[row for path in sorted((outputs[0][0]/'source_checkpoints').glob('*.json')) for row in c.read(path)['rows']]
            engine.replayed=[];support.FakeEvaluator.calls=0
            run(1,True)
            measured=[row for path in sorted((outputs[1][0]/'source_checkpoints').glob('*.json')) for row in c.read(path)['rows']]
            self.assertEqual([{k:v for k,v in row.items() if k!='parity_origin'} for row in measured],baseline)
            self.assertEqual(len(measured),600)
            self.assertTrue(all(study.startswith('M2') for study,index in engine.replayed))
            self.assertEqual(support.FakeEvaluator.calls,100)
            conditioned=[row for row in measured if row['method']=='D_C_QPSK']
            self.assertTrue(all(row['label_conditioned'] and not row['is_main_conclusion'] for row in conditioned))
            self.assertTrue(all(row['classification_interpretation']=='label_conditioned_descriptive_only' for row in conditioned))
            completion=c.read(outputs[1][0]/'scoring_completion.json')
            self.assertEqual(completion['parity_origin_counts'],{a.HISTORICAL:500,a.FRESH:100})
            c.verify(completion['inputs']);c.verify(completion['outputs'])
            before=c.sha(outputs[1][1]/'metrics_per_frame.csv');engine.replayed=[]
            run(1,True)
            self.assertEqual(support.FakeEvaluator.calls,100);self.assertEqual(engine.replayed,[])
            self.assertEqual(c.sha(outputs[1][1]/'metrics_per_frame.csv'),before)
            self.assertEqual(c.read(outputs[1][0]/'scoring_completion.json')['parity_origin_counts'],completion['parity_origin_counts'])

    def test_science_identity_full_fields_and_coverage_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);engine=Engine(root);snapshot=make_snapshot(root,engine,'manifest',{'precision':'highest'})
            for change in ({'truth':1},{'prediction':1},{'evaluator_identity':'other'},
                           {'numerical_runtime':{'precision':'medium'}},{'manifest_sha':'other'}):
                with self.assertRaises(RuntimeError):source_assembly(snapshot,engine,**change)
            changed=engine.by_source['N512'][0][0]
            before=changed['psnr_db'];changed['psnr_db']='21'
            with self.assertRaisesRegex(RuntimeError,'inventory'):source_assembly(snapshot,engine)
            changed['psnr_db']=before
            engine.by_source['N512'][0].append(copy.deepcopy(changed))
            with self.assertRaises(RuntimeError):source_assembly(snapshot,engine)

    def test_no_rgb_or_model_call_needed_and_non_cached_studies_cannot_be_assembled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);engine=Engine(root);snapshot=make_snapshot(root,engine,'manifest',{'precision':'highest'})
            with mock.patch.object(engine,'iterate_source',side_effect=AssertionError('must not replay')), \
                 mock.patch.object(p,'rgb_array',side_effect=AssertionError('must not fabricate RGB')):
                source=source_assembly(snapshot,engine)
                rows=list(source.assemble('N512'))
                self.assertEqual(len(rows),3)
                self.assertEqual(a.origin_counts([x[0] for x in rows],[x[1] for x in rows]),{a.HISTORICAL:3,a.FRESH:0})
                self.assertFalse(source.available('M2_ORACLE'))
                with self.assertRaises(RuntimeError):list(source.assemble('M2_ORACLE'))

    def test_origin_cannot_claim_cached_rows_were_fresh(self):
        row=dict(replay_row_id='r',image_sha256='image',reference_sha256='source',parity_origin=a.HISTORICAL)
        proof=dict(replay_parity_passed=True,synthetic=False,original_row_sha256='r',rgb_sha256='image',
                   target_sha256='source',parity_origin=a.FRESH)
        with self.assertRaises(RuntimeError):a.origin_counts([row],[proof])


if __name__=='__main__':unittest.main()
