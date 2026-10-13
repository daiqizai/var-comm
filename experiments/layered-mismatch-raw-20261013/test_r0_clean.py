"""CPU-only contract/stub tests; no torch import, model, channel or GPU call."""
import ast
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

P=Path(__file__).with_name('r0_clean.py')
spec=importlib.util.spec_from_file_location('r0_clean_under_test',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OLD=((4,0),(5,0),(5,9),(5,18),(6,0),(6,13),(6,16),(6,48),(7,0),(7,3),(7,56),(7,75),(7,82),(8,0),(8,9),(8,126),(8,142))

def fixture():
    records=[dict(source_id=f'fixed/{i}',source_index=999-i,selection_slot=i) for i in range(200)]
    cal=dict(records=records,selection_domain='UEP_MISMATCH_CAL200_V1',uses_quality=False)
    missing=dict(all_candidate_terminal_states=list(OLD)+list(m.MISSING_STATES),missing_pairs=[dict(source_id=r['source_id'],source_index=r['source_index'],m=a,K=b) for r in records for a,b in m.MISSING_STATES])
    historical=[dict(source_id=r['source_id'],m=str(a),K=str(b)) for r in records for a,b in OLD]
    return cal,missing,historical

class Tests(unittest.TestCase):
    def test_exact_grid_source_order(self):
        grid=m.plan(*fixture());self.assertEqual(len(grid),5400);self.assertEqual(grid[0]['source_index'],999)
        self.assertEqual([r['cell_index'] for r in grid],list(range(5400)))
        self.assertEqual(len({(r['source_id'],r['m'],r['K']) for r in grid}),5400)
    def test_missing_duplicate_rejected(self):
        a,b,c=fixture();b['missing_pairs'][-1]=b['missing_pairs'][0];self.assertRaises(RuntimeError,m.plan,a,b,c)
    def test_missing_source_index_rejected(self):
        a,b,c=fixture();b['missing_pairs'][0]['source_index']=0;self.assertRaises(RuntimeError,m.plan,a,b,c)
    def test_selection_order_rejected(self):
        a,b,c=fixture();a['records'][0],a['records'][1]=a['records'][1],a['records'][0];self.assertRaises(RuntimeError,m.plan,a,b,c)
    def test_quality_selected_source_rejected(self):
        a,b,c=fixture();a['uses_quality']=True;self.assertRaises(RuntimeError,m.plan,a,b,c)
    def test_old_overlap_rejected(self):
        a,b,c=fixture();c[0].update(m='1',K='0');self.assertRaises(RuntimeError,m.plan,a,b,c)
    def test_old_missing_rejected(self):
        a,b,c=fixture();c.pop();self.assertRaises(RuntimeError,m.plan,a,b,c)
    def test_unknown_endpoint_rejected(self):
        a,b,c=fixture();b['missing_pairs'][0]['K']=99;self.assertRaises(RuntimeError,m.plan,a,b,c)
    def test_terminal_union_required(self):
        a,b,c=fixture();b['all_candidate_terminal_states'].pop();self.assertRaises(RuntimeError,m.plan,a,b,c)
    def test_raster_prefix_counts(self):
        self.assertEqual(m.OFFSETS[-1],680);self.assertEqual(m.token_count(6,32),123)
        for a,b in m.MISSING_STATES:self.assertEqual(m.token_count(a,b),sum(s*s for s in m.SIZES[:a])+b)
    def test_reject_bool_or_full_partial(self):
        for args in ((True,0),(1,True),(1,4),(0,0),(9,0)):
            self.assertRaises(RuntimeError,m.token_count,*args)
    def test_timing_wait_before_device(self):
        with patch.object(m,'active_timing_processes',return_value=[dict(pid=17)]),patch.object(m,'resource_device',side_effect=AssertionError('device must not initialize')):
            result=m.admission({},Path('unused'))
        self.assertEqual(result['status'],'WAIT');self.assertEqual(result['model_calls'],0)
    def test_busy_wait_before_model(self):
        ResourceBusy=type('ResourceBusy',(Exception,),{})
        device=types.SimpleNamespace(resource_snapshot=lambda *a,**k:(_ for _ in ()).throw(ResourceBusy('occupied')))
        with patch.object(m,'active_timing_processes',return_value=[]),patch.object(m,'resource_device',return_value=(device,types.SimpleNamespace(helper=lambda:None))):
            result=m.admission({},Path('unused'))
        self.assertEqual(result['status'],'WAIT');self.assertEqual(result['model_calls'],0)
    def test_unknown_admission_failure_not_silenced(self):
        device=types.SimpleNamespace(resource_snapshot=lambda *a,**k:(_ for _ in ()).throw(ValueError('identity drift')))
        with patch.object(m,'active_timing_processes',return_value=[]),patch.object(m,'resource_device',return_value=(device,types.SimpleNamespace(helper=lambda:None))):
            self.assertRaises(ValueError,m.admission,{},Path('unused'))
    def test_render_only_accepted_prefix(self):
        calls=[];tokens=np.arange(680,dtype=np.int64);cell=dict(m=6,K=32,source_index=13,cell_index=1)
        def render(native,prefix,a,b):
            self.assertEqual(prefix.tolist(),tokens[:123].tolist());calls.append((a,b));return np.zeros((3,256,256),np.float32)
        ledger=types.SimpleNamespace(call=lambda kind,fn,**meta:fn())
        b=types.SimpleNamespace(t=types.SimpleNamespace(no_grad=contextlib.nullcontext),native=object(),render_function=render)
        image,h=m.render_one(b,tokens,cell,ledger);self.assertEqual(calls,[(6,32)]);self.assertEqual(b.source_index,13)
        self.assertEqual(h,m.array_sha(tokens[:123]));np.testing.assert_array_equal(tokens,np.arange(680))
    def test_render_prefix_mutation_detected(self):
        def render(native,prefix,a,b):prefix[0]=100;return np.zeros((3,256,256),np.float32)
        b=types.SimpleNamespace(t=types.SimpleNamespace(no_grad=contextlib.nullcontext),native=object(),render_function=render)
        ledger=types.SimpleNamespace(call=lambda kind,fn,**meta:fn())
        self.assertRaises(RuntimeError,m.render_one,b,np.arange(680,dtype=np.int64),dict(m=1,K=0,source_index=0,cell_index=0),ledger)
    def test_invalid_render_not_clamped_or_replaced(self):
        for a in (np.zeros((3,256,256),np.float64),np.full((3,256,256),np.nan,np.float32),np.full((3,256,256),2,np.float32)):
            self.assertRaises(RuntimeError,m.validate_image,a)
    def test_source_real_bytes_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);pixels=np.zeros((3,256,256),np.uint8);tokens=np.arange(680,dtype=np.int64)
            np.savez(p/'a.npz',pixels=pixels,tokens=tokens);(p/'c.json').write_text('{}',encoding='utf8')
            r=dict(archive=m.pin(p/'a.npz'),checkpoint=m.pin(p/'c.json'),preprocessing_id=m.hashlib.sha256(pixels.tobytes()).hexdigest(),tokens_sha256=m.hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest())
            got=m.validate_source(r);np.testing.assert_array_equal(got[1],tokens)
            r['tokens_sha256']='0'*64;self.assertRaises(RuntimeError,m.validate_source,r)
    def test_source_archive_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_bytes(b'good');rec=m.pin(p);p.write_bytes(b'evil');self.assertRaises(RuntimeError,m.checked,rec)
    def test_exclusive_evidence_write(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';m.write(p,dict(x='中文'));self.assertEqual(m.read(p)['x'],'中文');self.assertRaises(FileExistsError,m.write,p,{})
    def test_dino_only_requested_model_ast(self):
        tree=ast.parse(P.read_text(encoding='utf8'));nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='MetricEvaluator']
        self.assertEqual(len(nodes),1);kw={k.arg:ast.literal_eval(k.value) for k in nodes[0].keywords if k.arg=='requested'}
        self.assertEqual(kw['requested'],('dinov2_vitl14',))
        self.assertNotIn('FourMetrics(',P.read_text(encoding='utf8'))
    def test_scientific_budgets_no_extra_qualification(self):
        self.assertEqual(m.CAPS['render']['var_render'],5400);self.assertEqual(m.CAPS['render']['decoder_forward'],5400)
        self.assertEqual(m.CAPS['render']['prior_scale'],54000);self.assertEqual(m.CAPS['score']['reconstruction_dino'],5400)
        self.assertEqual(m.CAPS['score']['source_embedding'],200)
    def test_explicit_review_gate_before_registration(self):
        with patch.object(m,'registration',side_effect=AssertionError('must not read')):self.assertRaises(RuntimeError,m.owner,None,None,False)
    def test_no_import_time_models(self):
        tree=ast.parse(P.read_text(encoding='utf8'))
        modules=[a.name.split('.')[0] for n in tree.body if isinstance(n,ast.Import) for a in n.names]
        self.assertNotIn('torch',modules);self.assertNotIn('lpips',modules)
    def test_draft_cannot_launch_with_boolean_flag(self):
        self.assertRaises(RuntimeError,m.review_gate,dict(status='DRAFT_REVIEW_REQUIRED'))
    def test_registered_without_review_cannot_launch(self):
        self.assertRaises(RuntimeError,m.review_gate,dict(status='REGISTERED_REVIEWED'))
    def test_wait_real_exit_not_receipt_only(self):
        child=types.SimpleNamespace(pid=8,wait=lambda timeout:0)
        result=m.wait_child(child,m.time.time()+10);self.assertEqual(result['returncode'],0);self.assertFalse(result['timed_out']);self.assertTrue(result['actual_wait'])
    def test_wait_timeout_terminates_and_waits(self):
        events=[]
        def wait(timeout=None):
            events.append(('wait',timeout))
            if len(events)==1:raise m.subprocess.TimeoutExpired('child',1)
            return -15
        child=types.SimpleNamespace(pid=8,wait=wait,terminate=lambda:events.append(('terminate',None)),kill=lambda:self.fail('kill unnecessary'))
        result=m.wait_child(child,m.time.time()+1);self.assertTrue(result['timed_out']);self.assertTrue(result['terminated']);self.assertEqual(result['returncode'],-15)
    def test_wait_timeout_escalates_and_waits(self):
        events=[]
        def wait(timeout=None):
            events.append('wait')
            if events.count('wait')<=2:raise m.subprocess.TimeoutExpired('child',1)
            return -9
        child=types.SimpleNamespace(pid=8,wait=wait,terminate=lambda:events.append('terminate'),kill=lambda:events.append('kill'))
        result=m.wait_child(child,m.time.time()+1);self.assertTrue(result['killed']);self.assertEqual(result['returncode'],-9);self.assertEqual(events,['wait','terminate','wait','kill','wait'])
    def test_children_inherit_timing_lock(self):
        tree=ast.parse(P.read_text(encoding='utf8'));popen=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
        self.assertEqual(len(popen),1);kw={k.arg:k.value for k in popen[0].keywords};self.assertEqual(ast.unparse(kw['pass_fds']),'lock_fds')
        text=P.read_text(encoding='utf8');self.assertIn('LOCK_SH|fcntl.LOCK_NB',text);self.assertIn('timing/six_method_v6/owner.lock',text)

if __name__=='__main__':unittest.main()
