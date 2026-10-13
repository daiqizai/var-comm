"""Synthetic CPU tests only; never import torch or execute a scientific call."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

P = Path(__file__).with_name('score_selection.py')
spec = importlib.util.spec_from_file_location('r1_score_tested', P)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def nominations():
    rows = []
    for hat in (7, 10):
        for arm, cid, physical, variants in [('U', 'group', 'g', ('DROP',)),
            ('EEP', 'group', 'g', ('DROP',)), ('S', 'single', 's', ('KEEP', 'DROP')),
            ('S_nom', 'single', 's', ('KEEP',)), ('S_back3', 'single', 's', ('KEEP',))]:
            rows.append(dict(snr_hat=hat, arm=arm, candidate_id=cid, physical_key=physical,
                             receiver_variants=variants))
    return rows


def sources():
    return [dict(source_id=f's{i}', source_index=1000-i, selection_slot=i) for i in range(200)]


def full_rows():
    rows = []; byid = {r['source_id']: r for r in sources()}
    for key, physical in m.expected_grid(sources(), nominations()).items():
        sid, hat, true, noise, cid, rule = key
        row = dict(byid[sid], snr_hat=hat, snr_true=true, noise_seed=noise,
            candidate_id=cid, physical_key=physical, receiver_rule=rule, view_id=m.digest(key),
            status='COMPLETE', source_truth_supplied_to_RX=False,
            observation_key=m.digest([sid, hat, true, noise, physical]),
            actual_rx_key=m.digest(['rx', sid, hat, true, noise, physical]),
            receiver_state_sha256='b'*64, model_identity='c'*64,
            reconstruction=dict(image_sha256='d'*64, actual_input_key='e'*64))
        rows.append(row)
    return rows


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def source(self):
        pixels = np.arange(3*256*256, dtype=np.uint8).reshape(3,256,256)
        path = self.directory/'source.npz'
        np.savez(path, pixels=pixels, tokens=np.array([999]))
        return dict(source_id='s', source_index=12, selection_slot=0,
            archive=m.pin(path), preprocessing_id=hashlib.sha256(pixels.tobytes()).hexdigest(),
            evaluation_class_index=9), pixels

    def row(self, source, rule='KEEP'):
        path = self.directory/'image.npz'
        image = np.full((3,256,256), .5, dtype=np.float32)
        if not path.exists(): np.savez(path, image=image)
        return dict(source_id=source['source_id'], source_index=source['source_index'], selection_slot=0,
            snr_hat=7, snr_true=4, noise_seed=4101, candidate_id='s', physical_key='f'*64,
            receiver_rule=rule, view_id=m.digest([rule]), observation_key='a'*64, actual_rx_key='b'*64,
            receiver_state_sha256='c'*64, model_identity='d'*64,
            reconstruction=dict(image_archive=m.pin(path), image_sha256=m.array_sha(image), actual_input_key='e'*64),
            actual_receiver_outcome=dict(fixed_gray=True, CRC=False))

    def test_source_pixels_preserve_original_uint8_crop_and_scale(self):
        source, pixels = self.source()
        np.testing.assert_array_equal(m.source_pixels(source), pixels.astype(np.float32)/255)

    def test_source_hash_mismatch_stops(self):
        source, _ = self.source(); source['preprocessing_id'] = 'a'*64
        with self.assertRaises(ValueError): m.source_pixels(source)

    def test_metric_source_loader_does_not_read_tokens(self):
        node = next(n for n in ast.parse(P.read_text()).body if isinstance(n, ast.FunctionDef) and n.name=='source_pixels')
        self.assertFalse(any(isinstance(n, ast.Constant) and n.value=='tokens' for n in ast.walk(node)))

    def test_real_reconstruction_hash_checked_even_for_gray(self):
        source, _ = self.source(); row = self.row(source)
        np.testing.assert_array_equal(m.reconstructed_pixels(row), np.full((3,256,256), .5, np.float32))
        row['reconstruction']['image_sha256'] = 'z'*64
        with self.assertRaises(ValueError): m.reconstructed_pixels(row)

    def test_pair_cache_omits_rule_but_not_actual_state_or_noise(self):
        source, _ = self.source(); keep = self.row(source); drop = self.row(source, 'DROP')
        self.assertEqual(m.pair_key(keep, source, 'metric'), m.pair_key(drop, source, 'metric'))
        # Different raw gray metadata (e.g. order None/raster) must not defeat
        # exact normalized received input and actual image/event reuse.
        drop['receiver_state_sha256']='q'*64
        self.assertEqual(m.pair_key(keep, source, 'metric'), m.pair_key(drop, source, 'metric'))
        changed=copy.deepcopy(drop);changed['reconstruction']['actual_input_key']='i'*64
        self.assertNotEqual(m.pair_key(keep,source,'metric'),m.pair_key(changed,source,'metric'))
        for field, value in [('noise_seed', 4102), ('actual_rx_key', 'x'*64),
                             ('snr_true', 7), ('model_identity', 'w'*64)]:
            changed = dict(drop, **{field: value})
            self.assertNotEqual(m.pair_key(keep, source, 'metric'), m.pair_key(changed, source, 'metric'))

    def test_score_exact_event_pair_once_keep_all_logical_failures(self):
        source, _ = self.source(); rows = [self.row(source), self.row(source, 'DROP')]; calls=[]
        rows[1]['receiver_state_sha256']='q'*64
        class Fake:
            identity = 'm'*64
            def prepare(self, target, sid): calls.append(('reference',sid)); return {'p':1}
            def score(self, target, image, label, prepared, sid, key):
                calls.append(('pair',sid,label,key)); return dict(psnr_db=10., lpips_alex=.7, dinov2_vitl14_cosine=.1)
        output, refs, count = m.score_rows(rows, [source], Fake(), self.directory/'out', lambda:None)
        self.assertEqual(count, 1); self.assertEqual(len(output),2); self.assertEqual(len(refs),1)
        self.assertEqual([c[0] for c in calls], ['reference','pair'])
        self.assertEqual(output[1]['metric_reuse'], 'EXACT_ACTUAL_EVENT_PAIR_REUSE')
        self.assertTrue(all(r['failed_logical_observation_retained'] for r in output))
        self.assertTrue(all(r['actual_receiver_outcome']['fixed_gray'] for r in output))
        self.assertEqual(m.authenticate_scored_rows(output, rows, [source], Fake.identity),1)
        output[1]['lpips_alex'] = .6
        with self.assertRaises(ValueError): m.authenticate_scored_rows(output,rows,[source],Fake.identity)

    def closed_stage_fixture(self, stage='visual'):
        root=self.directory/'owner';root.mkdir(exist_ok=True)
        output=self.directory/stage;output.mkdir(exist_ok=True)
        registration={'path':'master.json','sha256':'m'*64}
        visual={'path':'visual.json','sha256':'v'*64}
        reg=dict(runtime={'root':str(root),stage+'_directory':str(output)},visual_registration=visual)
        owner=dict(process=dict(pid=123,start_ticks=456),registration=registration)
        value=visual if stage=='visual' else registration
        script='score_selection.py' if stage=='score' else 'receiver_visual.py'
        action='score' if stage=='score' else ('visual' if stage=='visual' else 'project')
        argv=['python',str(m.HERE/script),action,'--registration',value['path'],
              '--registration-sha256' if stage=='score' else '--expected-registration',value['sha256'],
              '--parent' if stage=='score' else '--parent-pid','123']
        if stage!='score':argv+=['--parent-ticks','456']
        launch=dict(stage=stage,process=dict(pid=567,start_ticks=890,state='R'),argv=argv,registration=registration)
        wait=dict(actual_wait=True,returncode=0,pid=567,stage=stage,timed_out=False,terminated=False,killed=False,stopped=False)
        for p,v in [(root/'owner_launch.json',owner),(root/(stage+'_launch.json'),launch),
                    (output/'actual_wait.json',wait),(root/(stage+'_actual_wait.json'),wait)]:
            p.write_text(json.dumps(v))
        return reg,registration,launch,wait

    def test_wait_requires_actual_exit_and_exact_dispatch_identity(self):
        reg,rp,launch,wait=self.closed_stage_fixture()
        path=Path(reg['runtime']['visual_directory'])/'actual_wait.json'
        self.assertEqual(m.successful_wait(reg,rp,'visual',lambda _:None),m.pin(path))
        for field,value in [('returncode',1),('timed_out',True),('stopped',True),('actual_wait',False),('pid',568)]:
            changed=dict(wait,**{field:value});path.write_text(json.dumps(changed))
            with self.subTest(field=field),self.assertRaises(ValueError):m.successful_wait(reg,rp,'visual',lambda _:None)
        path.write_text(json.dumps(wait))
        for identity in (dict(pid=567,start_ticks=890,state='R'),dict(pid=567,start_ticks=890,state='S')):
            with self.assertRaises(ValueError):m.successful_wait(reg,rp,'visual',lambda _:identity)
        self.assertEqual(m.successful_wait(reg,rp,'visual',lambda _:dict(pid=567,start_ticks=891,state='R')),m.pin(path))
        launch['process']['start_ticks']=None
        (Path(reg['runtime']['root'])/'visual_launch.json').write_text(json.dumps(launch))
        with self.assertRaises(ValueError):m.successful_wait(reg,rp,'visual',lambda _:None)

    def test_wait_rejects_wrong_registration_program_and_owner(self):
        reg,rp,launch,wait=self.closed_stage_fixture('score')
        p=Path(reg['runtime']['root'])/'score_launch.json'
        for variant in ('registration','program','parent','sha'):
            bad=copy.deepcopy(launch)
            if variant=='registration':bad['registration']['sha256']='w'*64
            elif variant=='program':bad['argv'][1]='wrong.py'
            elif variant=='parent':bad['argv'][-1]='999'
            else:bad['argv'][bad['argv'].index('--registration-sha256')+1]='x'*64
            p.write_text(json.dumps(bad))
            with self.subTest(variant=variant),self.assertRaises(ValueError):m.successful_wait(reg,rp,'score',lambda _:None)

    def test_visual_preflight_checks_projection_before_metric_construction(self):
        reg,rp,_,_=self.closed_stage_fixture('visual')
        pr,_,_,_=self.closed_stage_fixture('projection');reg['runtime'].update(pr['runtime'])
        projection=Path(reg['runtime']['projection_directory']);visual=Path(reg['runtime']['visual_directory'])
        def write(p,value):p.write_text(json.dumps(value));return m.pin(p)
        mp=write(projection/'model_inputs.json',[dict(view_id='one')])
        jp=write(projection/'logical_join.json',[dict(view_id='one',source_id='s')])
        ap=write(projection/'audit.json',dict(status='PASS'))
        vp=write(projection/'visual_input_completion.json',dict(status='COMPLETE_RX_ONLY_PROJECTION',
            model_inputs=mp,logical_count=1,source_truth_supplied_to_RX=False,
            projection_digest=m.digest([mp['sha256'],jp['sha256'],ap['sha256']])))
        project_done=dict(status='COMPLETE_R1_RX_PROJECTION',registration=rp,model_inputs=mp,logical_join=jp,
            audit=ap,visual_input_completion=vp,logical_count=1)
        write(projection/'completion.json',project_done)
        ip=write(visual/'image_rows.json',[dict(view_id='one',status='COMPLETE')])
        done=dict(status='COMPLETE_R1_VISUAL',registration=reg['visual_registration'],source_truth_supplied_to_RX=False,
            image_rows=ip,model_inputs=mp,projection_completion=vp,
            projection_actual_wait=m.pin(projection/'actual_wait.json'),logical_rows=1)
        write(visual/'completion.json',done)
        reg['pins']=dict(cal200_sources=write(self.directory/'sources.json',dict(records=[])),
                         nominations=write(self.directory/'nominations.json',dict(rows=[])))
        reg['expected_logical_var_images']=1
        import receiver_visual
        with patch.object(m,'successful_wait',side_effect=lambda r,p,s:m.pin(Path(r['runtime'][s+'_directory'])/'actual_wait.json')),\
             patch.object(m,'validate_visual_rows',return_value=dict(logical_rows=1)),patch.object(m,'ThreeMetrics') as construct,\
             patch.object(receiver_visual,'audit_visual',side_effect=lambda r:dict(status='PASS_CPU_ACTUAL_VISUAL_CLOSURE',
                 completion=m.pin(visual/'completion.json'),model_inputs=mp)):
            rows,*_=m.closed_visual(reg,rp);self.assertEqual(rows[0]['source_id'],'s')
            for field,value in [('registration',dict(path='wrong',sha256='z'*64)),('model_inputs',dict(path='wrong',sha256='z'*64))]:
                write(projection/'completion.json',dict(project_done,**{field:value}))
                with self.assertRaises(ValueError):m.closed_visual(reg,rp)
            construct.assert_not_called()

    def test_actual_nested_ledger_sequences_are_audited(self):
        p=self.directory/'calls.sqlite';db=sqlite3.connect(p)
        db.execute('CREATE TABLE calls(id INTEGER PRIMARY KEY,kind TEXT,metadata TEXT,state TEXT)')
        calls=[('model_constructions',dict(model='LPIPS_ALEX')),('model_constructions',dict(model='DINOv2_ViTL14')),
            ('reference_preparations',dict(source_id='s')),('dinov2_vitl14_reference',dict(source_id='s')),
            ('image_scores',dict(pair_key='pair',source_id='s')),('dinov2_vitl14_reconstruction',dict(pair_key='pair',source_id='s')),
            ('lpips_pair',dict(pair_key='pair')),
            ('lpips_alexnet_backbone_forward',dict(pair_key='pair')),('lpips_alexnet_backbone_forward',dict(pair_key='pair'))]
        for kind,meta in calls:db.execute('INSERT INTO calls(kind,metadata,state) VALUES(?,?,?)',(kind,json.dumps(meta),'COMPLETE'))
        db.commit();counts=dict(caps=m.score_caps(1),completed={k:sum(kind==k for kind,_ in calls) for k in m.score_caps(1)})
        rows=[dict(metric_reuse='ACTUAL_NEW_PAIR',metric_pair_key='pair',source_id='s')];refs=[dict(source_id='s')]
        m.audit_score_ledger(m.pin(p),counts,rows,refs)
        db.execute("UPDATE calls SET metadata=? WHERE kind='dinov2_vitl14_reconstruction'",(json.dumps(dict(pair_key='wrong')),));db.commit();db.close()
        with self.assertRaises(ValueError):m.audit_score_ledger(m.pin(p),counts,rows,refs)


class Grid(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.rows = full_rows()

    def test_import_model_free_and_only_original_r1_three_metrics(self):
        self.assertNotIn('torch',sys.modules); self.assertNotIn('lpips',sys.modules)
        self.assertEqual(m.METRICS,('psnr_db','lpips_alex','dinov2_vitl14_cosine'))

    def test_complete_canonical_grid_reuses_arm_aliases(self):
        result=m.validate_visual_rows(self.rows,sources(),nominations())
        self.assertEqual(result,dict(logical_rows=10800,physical_events=7200,sources=200,noises=3))

    def test_missing_and_duplicate_rejected(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[0]]):
            with self.assertRaises(ValueError):m.validate_visual_rows(rows,sources(),nominations())

    def test_foreign_source_condition_and_truth_access_rejected(self):
        for field,value in [('source_id','foreign'),('snr_true',19),('noise_seed',6201),
                            ('source_truth_supplied_to_RX',True),('source_index',-1)]:
            changed=[dict(self.rows[0],**{field:value})]+self.rows[1:]
            with self.subTest(field=field),self.assertRaises(ValueError):m.validate_visual_rows(changed,sources(),nominations())

    def test_keep_drop_cannot_reference_different_actual_event(self):
        rows=copy.deepcopy(self.rows); r=next(r for r in rows if r['receiver_rule']=='DROP' and r['candidate_id']=='single')
        r['actual_rx_key']='z'*64
        with self.assertRaises(ValueError):m.validate_visual_rows(rows,sources(),nominations())

    def test_original_gate_consumes_real_source_three_noise_means(self):
        scored=[]
        for row in self.rows:
            x=dict(row,execution_status='COMPLETE',reconstruction_sha256='h'*64)
            q=.53 if x['candidate_id']=='group' else .5
            q+=(x['noise_seed']-4102)*.01
            if x['snr_true']==x['snr_hat']+3:q=.99
            x.update(psnr_db=20.,lpips_alex=.5,dinov2_vitl14_cosine=q);scored.append(x)
        result=m.policy_selection.select_from_complete_rows(scored,[s['source_id'] for s in sources()],nominations())
        self.assertEqual(result['gate']['status'],'GO_R2')
        self.assertEqual(result['bootstrap_calls'],0)
        for winner in result['winners']:
            self.assertAlmostEqual(winner['cal200_DINO_J'],.53 if winner['candidate_id']=='group' else .5)


class Mechanics(unittest.TestCase):
    def test_original_received_failure_metadata_is_retained(self):
        state=dict(actual_received_kind='gray',actual_received_m=0,actual_received_K=0,actual_failure_state=True)
        self.assertEqual(m.receiver_outcome(state),state)

    def test_engineering_targets_report_tradeoff_without_changing_gate(self):
        selected=dict(gate=dict(status='GO_R2'),winners=[]);means={}
        for hat in (7,10):
            for method,candidate in [('U_two','u'),('S_nom','s')]:
                selected['winners'].append(dict(method=method,snr_hat=hat,candidate_id=candidate,receiver_rule='DROP'))
                for true in (hat-3,hat,hat+3):
                    for metric in m.METRICS:
                        value=(20 if candidate=='s' else 19) if metric=='psnr_db' else .5
                        means[('source',hat,true,candidate,'DROP',metric)]=value
        summary,winners,targets=m.quality_summaries(means,['source'],selected)
        self.assertEqual(targets['classification'],'TRADEOFF_ONLY_ON_CALIBRATION')
        self.assertEqual(selected['gate']['status'],'GO_R2')
        self.assertTrue(all(r['used_for_GO'] is False for r in targets['per_hat']))
        self.assertTrue(all(r['confidence_interval']=='NOT_COMPUTED_R1_SELECTION' for r in summary))

    def test_visual_join_is_by_view_and_rejects_conflicts(self):
        joins=[dict(view_id='a',source_id='s'),dict(view_id='b',source_id='t')]
        images=[dict(view_id='b',status='COMPLETE'),dict(view_id='a',status='COMPLETE')]
        self.assertEqual([r['source_id'] for r in m.join_visual_rows(joins,images)],['s','t'])
        for bad in ([images[0],images[0]],[dict(view_id='a',source_id='wrong'),images[0]],images[:1]):
            with self.assertRaises(ValueError):m.join_visual_rows(joins,bad)

    def test_caps_and_exact_metric_call_accounting(self):
        caps=m.score_caps(10800)
        self.assertEqual(caps['model_constructions'],2);self.assertNotIn('convnext_reference',caps)
        counts=dict(model_constructions=2,reference_preparations=200,dinov2_vitl14_reference=200,
            image_scores=9000,dinov2_vitl14_reconstruction=9000,lpips_pair=9000,lpips_alexnet_backbone_forward=18000)
        receipt=dict(caps=caps,reserved=counts,completed=counts,unresolved=0)
        m.check_score_counts(receipt,10800,9000)
        bad=copy.deepcopy(receipt);bad['completed']['lpips_alexnet_backbone_forward']-=1
        with self.assertRaises(ValueError):m.check_score_counts(bad,10800,9000)
        for n in (0,46801,-1):
            with self.assertRaises(ValueError):m.score_caps(n)

    def test_actual_three_metric_method_calls_original_pair_and_dino(self):
        obj=object.__new__(m.ThreeMetrics);calls=[]
        class Ledger:
            def call(self,kind,fn,**kw):calls.append((kind,kw));return fn()
        obj.ledger=Ledger();obj.identity='metric';obj.tensor=lambda x:x
        obj.torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext,
            autocast=lambda *a,**kw:contextlib.nullcontext())
        obj.perceptual=object();obj.psnr=lambda x,y:21.
        obj.lpips_pair=lambda net,y,x:[types.SimpleNamespace(item=lambda:.4)]
        def dino(x,y,labels,**kwargs):
            self.assertEqual(labels,[5]);self.assertEqual(kwargs,dict(label_conditioned=False,prepared='original'))
            return [dict(dinov2_vitl14_cosine=.7,label_conditioned=False)]
        obj.evaluator=types.SimpleNamespace(score=dino)
        x=np.zeros((3,256,256),np.float32);y=np.ones_like(x)
        prepared=dict(target_sha256=m.array_sha(x),metric_identity='metric',dino='original')
        self.assertEqual(obj.score(x,y,5,prepared,'s','pair'),dict(psnr_db=21.,lpips_alex=.4,dinov2_vitl14_cosine=.7))
        self.assertEqual([x[0] for x in calls],['image_scores','lpips_pair'])
        self.assertEqual(obj.context['feature_kind'],'dinov2_vitl14_reconstruction')

    def test_metric_identity_mismatch_stops_before_pair(self):
        obj=object.__new__(m.ThreeMetrics);obj.identity='metric';obj.tensor=lambda x:x
        obj.ledger=types.SimpleNamespace(call=lambda kind,fn,**kw:fn())
        x=np.zeros((3,256,256),np.float32)
        with self.assertRaises(ValueError):obj.score(x,x,0,dict(target_sha256=m.array_sha(x),metric_identity='wrong'),'s','p')

    def test_no_classifier_or_training_constructed_in_three_metrics(self):
        tree=ast.parse(P.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ThreeMetrics')
        called={n.func.attr for n in ast.walk(cls) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertNotIn('classifier',called);self.assertNotIn('backward',called);self.assertNotIn('step',called)


if __name__=='__main__':unittest.main()
