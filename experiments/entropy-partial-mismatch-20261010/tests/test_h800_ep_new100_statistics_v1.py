"""Fake CPU engineering only: no real scores, random draws or bootstrap calls."""
import copy
import csv
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_new100_statistics_v1 as owner
from test_ep_new100_statistics_core_v1 import fixture,FakePublished


def closure_fixture():
    """Synthetic complete metadata chain; no pixel file exists or is opened."""
    c=owner.core.confirmation;store={};metricroot=owner.METRIC_ROOT/'registered'
    visualroot=owner.io.RT/'qualification/h800_ep_new100_visual_v3_attempt1'
    def put(path,value):
        path=str(path);store[path]=value
        return dict(path=path,sha256=c.digest(value))
    def pin(path):return dict(path=str(path),sha256=c.digest(store[str(path)]))
    expected=fixture()
    for row in expected:row['policy']={'synthetic':True}
    records=[dict(source_index=i,source_id=f'source-{i:03d}',pixels_archive={'path':'NO_PIXEL_READ','sha256':'a'*64}) for i in range(100)]
    sourcepin=put('source_manifest',dict(records=records,source_count=100,policy_bundle={'path':'policy','sha256':c.digest({})}))
    inputs=dict(selection=put('selection',{}),policy_bundle=put('policy',{}),content_gate=put('content',{}),source_manifest=sourcepin)
    meta=dict(population=owner.metric.POPULATION,policy_selection=False,selection_objective=None)
    identity=c.digest(meta);runtime=dict(python=platform.python_version(),numpy=np.__version__)
    identitypin=put('metric_identity',dict(identity=identity,metadata=meta,actual_runtime=runtime))
    pairkey=c.digest(dict(reference_sha256='b'*64,reconstruction_sha256='c'*64,metric_identity=identity))
    values=dict(psnr_db=10.,lpips_alex=.4,dinov2_vitl14_cosine=.5,convnext_top1_source_prediction=1)
    recon=dict(image_archive={'path':'NO_RECON_READ','sha256':'d'*64},image_sha256='c'*64)
    flags=dict(zip(owner.metric.FAILURE_FIELDS,('OK',1,True,True,True,'OK',True,False)))
    pair=put('pair',dict(metrics=values,metric_pair_key=pairkey,reference_archive=records[0]['pixels_archive'],
        reference_pixel_sha256='b'*64,reconstruction_archive=recon['image_archive'],reconstruction_pixel_sha256='c'*64,
        metric_identity=identity,population=owner.metric.POPULATION,historical_score_reuse=False))
    rows=[];reconpins=[]
    for i,event in enumerate(expected):
        rp=put(f'recon/{i}',dict(frame_index=i,logical_event=event,physical_frame=f'actual/{i}',reconstruction=recon,**flags));reconpins.append(rp)
        rows.append(dict(event,**{},metric_pair_key=pairkey,metric_pair_result=pair,reconstruction_result=rp,physical_frame=f'actual/{i}',**flags))
        rows[-1].update(values)
    counts={k:0 for k in owner.metric.CAPS}
    for k in ('reference_preparations','dinov2_vitl14_reference','convnext_reference'):counts[k]=100
    for k in ('image_scores','dinov2_vitl14_reconstruction','convnext_reconstruction','lpips_pair'):counts[k]=1
    counts.update(model_constructions=3,lpips_alexnet_backbone_forward=2)
    result=dict(metric_rows=put('metric_rows',rows),logical_frames=3600,actual_unique_pairs=1,reused_pairs=3599,
        counts=dict(caps=owner.metric.CAPS,reserved=counts,completed=counts,unresolved=0),full_grid_complete=True,
        historical_score_reuse=0,bootstrap_calls=0,policy_selection=False,population=owner.metric.POPULATION)
    vcaps=c.scientific_caps()['visual'];vcounts={k:0 for k in vcaps};vcounts['model_load']=1
    recovery=dict(previous_model_load_attempts=1,previous_model_load_completed=0,previous_unresolved_model_load=1,
        additional_model_load_attempt_cap=1,cumulative_model_load_attempt_cap=2,prior_failed_ledger_preserved=True,
        previous_scientific_calls={k:0 for k in vcaps if k!='model_load'},
        cumulative_scientific_caps={k:v for k,v in vcaps.items() if k!='model_load'},failed_attempt_pins={})
    vr=dict(schema='VISUAL_V2',records=records,inputs=inputs,logical_events=expected,target_index=3,device_identity_binding='GPU3',
        recovery_budget=recovery,CPU_closed={})
    vrp=put(visualroot/'registered/request.json',vr)
    vw=put(visualroot/'registered/run/worker_completion.json',dict(schema='VISUAL_V2',status='VISUAL_PASS',request_sha256=vrp['sha256'],
        quality_scores=0,results=dict(logical_frames=reconpins,counts=dict(caps=vcaps,reserved=vcounts,completed=vcounts,unresolved=0))))
    actual=dict(success=True);put(visualroot/'registered/run/actual_child_wait.json',actual)
    visualclosed=dict(completion=put(visualroot/'registered/run/completion.json',dict(schema='VISUAL_V2',status='VISUAL_PASS',
        request_sha256=vrp['sha256'],worker_completion=vw,actual_wait=actual,actual_children_waited=True,worker_exit_codes=[0])),
        owner_actual_wait=put(visualroot/'run_owner_actual_wait.json',dict(actual_wait=True,returncode=0,timeout=False)))
    r=dict(schema='METRIC_V2',inputs=inputs,visual_closed=visualclosed,records=records,logical_events=expected,reconstruction_results=reconpins,
        metric_config=dict(confirmation_source_manifest=sourcepin,confirmation_policy_bundle=inputs['policy_bundle'],runtime_identity='runtime'),
        spec=dict(python=sys.executable),tool_bindings={owner.METRIC_OWNER_NAME:'f'*64,'ep_new100_metric_core_v1.py':owner.METRIC_CORE_SHA},
        population=owner.metric.POPULATION,bootstrap_calls=0,policy_selection=False,historical_score_reuse_allowed=False,
        target_index=3,device_identity_binding='GPU3')
    rp=put(metricroot/'request.json',r)
    worker=put(metricroot/'run/worker_completion.json',dict(schema='METRIC_V2',status='METRIC_PASS',request_sha256=rp['sha256'],
        results=result,metric_identity=identitypin,population=owner.metric.POPULATION,bootstrap_calls=0,policy_selection=False))
    put(metricroot/'run/actual_child_wait.json',actual)
    pins=dict(completion=put(metricroot/'run/completion.json',dict(schema='METRIC_V2',status='METRIC_PASS',request_sha256=rp['sha256'],
        worker_completion=worker,actual_wait=actual,actual_children_waited=True,worker_exit_codes=[0],bootstrap_calls=0,policy_selection=False)),
        owner_actual_wait=put(owner.METRIC_ROOT/'run_owner_actual_wait.json',dict(actual_wait=True,returncode=0,timeout=False)))
    module=types.SimpleNamespace(SCHEMA='METRIC_V2',PASS='METRIC_PASS',g=types.SimpleNamespace(EXPECTED_RUNTIME=runtime),
        visual_implementation=lambda:types.SimpleNamespace(SCHEMA='VISUAL_V2',PASS='VISUAL_PASS',failed_attempt=lambda:({'CPU_closed':{}},recovery)))
    return store,pins,r,expected,module,pin


class ClosureTests(unittest.TestCase):
    def test_pending_provider_stops_before_any_scientific_or_data_call(self):
        with patch.object(owner,'METRIC_OWNER_SHA','PENDING_FINAL_GPU3_METRIC_OWNER'),patch.object(owner,'read') as read:
            with self.assertRaises(ValueError):owner.metric_implementation()
            read.assert_not_called()

    def closure(self,alter=None):
        store,pins,r,expected,module,pin=closure_fixture()
        if alter:alter(store,pins,r)
        with ExitStack() as stack:
            stack.enter_context(patch.object(owner,'metric_implementation',return_value=module))
            stack.enter_context(patch.object(owner,'read',side_effect=lambda p:store[p['path']]))
            stack.enter_context(patch.object(owner,'pin',side_effect=pin))
            stack.enter_context(patch.object(owner,'METRIC_OWNER_SHA','f'*64))
            stack.enter_context(patch.object(owner.io,'sha',side_effect=lambda p:r['tool_bindings'][Path(p).name]))
            stack.enter_context(patch.object(owner.core.confirmation,'frame_grid',return_value=expected))
            return owner.metric_closure(pins)

    def test_complete3600_closed_chain_and_two_field_upstream_pins(self):
        rows,expected,inputs=self.closure()
        self.assertEqual(len(rows),3600);self.assertEqual(len(expected),3600)
        self.assertIn('visual_closed',inputs);self.assertEqual(inputs['metric_actual_runtime']['numpy'],np.__version__)

    def test_owner_failure_or_changed_pair_or_missing_frame_rejected(self):
        def failed(s,p,r):s[p['owner_actual_wait']['path']]['returncode']=1
        def missing(s,p,r):s['metric_rows'].pop()
        def pair(s,p,r):s['pair']['metrics']=dict(s['pair']['metrics'],lpips_alex=.7)
        def failures(s,p,r):s['metric_rows'][0]['actual_header_ok']=False
        def device(s,p,r):s[str(owner.io.RT/'qualification/h800_ep_new100_visual_v3_attempt1/registered/request.json')]['target_index']=2
        def prior(s,p,r):s[str(owner.io.RT/'qualification/h800_ep_new100_visual_v3_attempt1/registered/request.json')]['recovery_budget']['previous_scientific_calls']['source_rx']=1
        for alter in (failed,missing,pair,failures,device,prior):
            with self.subTest(alter=alter.__name__),self.assertRaises((ValueError,RuntimeError)):self.closure(alter)


class LedgerTests(unittest.TestCase):
    def test_unique_durable_reserve_complete_and_cap(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=owner.CallLedger(Path(d)/'calls',lambda:None)
            ledger.event('reserved','draw_matrix',0,{})
            self.assertEqual(ledger.summary()['unresolved'],1)
            with self.assertRaises(ValueError):ledger.event('reserved','draw_matrix',1,{})
            ledger.event('completed','draw_matrix',0,{})
            with self.assertRaises(ValueError):ledger.event('completed','draw_matrix',0,{})
            with self.assertRaises(FileExistsError):owner.CallLedger(Path(d)/'calls',lambda:None)
            self.assertEqual(ledger.summary()['unresolved'],0)

    def test_deadline_prevents_reservation(self):
        def stop():raise ValueError('STOP')
        with tempfile.TemporaryDirectory() as d:
            ledger=owner.CallLedger(Path(d)/'calls',stop)
            with self.assertRaises(ValueError):ledger.event('reserved','draw_matrix',0,{})
            self.assertEqual(list(ledger.out.iterdir()),[])

    def test_pin_accepts_two_or_three_fields_but_not_changed_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file';p.write_bytes(b'closed');full=owner.pin(p);small={k:full[k] for k in ('path','sha256')}
            self.assertTrue(owner.unchanged(full));self.assertTrue(owner.unchanged(small))
            self.assertFalse(owner.unchanged(dict(full,bytes=999)))
            p.write_bytes(b'changed');self.assertFalse(owner.unchanged(small))


class ExecutionTests(unittest.TestCase):
    def fake_run(self,d,stat=None,wrong_runtime=False):
        out=Path(d)/'attempt';out.mkdir();rows=fixture();rowpin=owner.write(out/'metric_rows.json',rows)
        runtime=dict(numpy='wrong' if wrong_runtime else np.__version__,python=platform.python_version())
        r=dict(CPU_slots=[2,3],max_seconds=900,deadline_unix=time.time()+900,inputs=dict(metric_runtime_python=str(Path(sys.executable)),
            metric_actual_runtime=runtime,metric_rows={k:rowpin[k] for k in ('path','sha256')},visual_closed={}),metric_closed={},
            original_function_source_sha256={},source_bindings={},noise_seeds=list(owner.core.confirmation.SEEDS),
            summaries=owner.core.definitions()[0],pairs=owner.core.definitions()[1])
        request=owner.write(out/'request.json',r);args=types.SimpleNamespace(request=request['path'],request_sha256=request['sha256'])
        stack=ExitStack();stack.enter_context(patch.object(owner,'OUT',out));stack.enter_context(patch.object(owner.io,'inside',side_effect=Path))
        stack.enter_context(patch.object(owner,'registration',return_value=(r,rows,rows,stat or FakePublished())))
        stack.enter_context(patch.object(owner,'source_bindings',return_value={}))
        stack.enter_context(patch.object(owner.sys,'platform','linux'));stack.enter_context(patch.object(owner.os,'sched_getaffinity',return_value={2,3},create=True))
        stack.enter_context(patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}))
        return stack,args,out

    def test_fake_export_complete48_24_and_no_replay(self):
        with tempfile.TemporaryDirectory() as d:
            stack,args,out=self.fake_run(d)
            with stack:
                result=owner.run(args);done=json.loads(Path(result['path']).read_text())
                self.assertEqual(done['status'],owner.PASS);self.assertEqual(done['counts']['unresolved'],0)
                self.assertEqual(done['counts']['completed']['draw_matrix'],1)
                for name,n in [('summary.csv',48),('paired.csv',24),('source_means.csv',4800),('source_paired_differences.csv',2400)]:
                    with (out/'run'/name).open(newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),n)
                with self.assertRaises(ValueError):owner.run(args)

    def test_failed_interval_preserves_unresolved_paid_call_and_blocks_replay(self):
        class Broken(FakePublished):
            def interval(self,*unused):raise RuntimeError('fake interval failed')
        with tempfile.TemporaryDirectory() as d:
            stat=Broken();stack,args,out=self.fake_run(d,stat)
            with stack:
                with self.assertRaises(RuntimeError):owner.run(args)
                failure=json.loads((out/'run/failure.json').read_text());self.assertEqual(failure['counts']['unresolved'],1)
                self.assertEqual(stat.draw_calls,1)
                with self.assertRaises(ValueError):owner.run(args)
                self.assertEqual(stat.draw_calls,1)

    def test_wrong_runtime_rejected_before_draw(self):
        with tempfile.TemporaryDirectory() as d:
            stat=FakePublished();stack,args,out=self.fake_run(d,stat,wrong_runtime=True)
            with stack:
                with self.assertRaises(ValueError):owner.run(args)
                self.assertEqual(stat.draw_calls,0);self.assertFalse((out/'run').exists())

    def test_prepare_does_not_draw_and_registers_only48_24(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'attempt';old=Path(d)/'old.py';old.write_text('# fake provider')
            a=types.SimpleNamespace(out=out,metric_completion_sha256='a'*64,metric_owner_wait_sha256='b'*64,
                statistics_module=old,max_seconds=900,deadline_unix=time.time()+900,cpu_slots=[2,3])
            stat=FakePublished()
            with patch.object(owner,'OUT',out),patch.object(owner.io,'inside',side_effect=Path),patch.object(owner,'metric_closure',return_value=([],[],{})),\
                patch.object(owner.published,'original_statistics',return_value=(stat,{})),patch.object(owner,'source_bindings',return_value={}):
                pin=owner.prepare(a);r=json.loads(Path(pin['path']).read_text())
                self.assertEqual(stat.draw_calls,0);self.assertEqual((len(r['summaries']),len(r['pairs'])),(48,24))
                self.assertFalse((out/'run').exists())
                with patch.object(owner,'read',return_value=r):
                    registered=owner.registration(out/'request.json',pin['sha256'])
                    self.assertIs(registered[-1],stat)
                    r['CPU_slots']=[2,3,4]
                    with self.assertRaises(ValueError):owner.registration(out/'request.json',pin['sha256'])
                with self.assertRaises(ValueError):owner.prepare(a)


if __name__=='__main__':unittest.main()
