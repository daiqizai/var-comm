"""Synthetic CPU tests only: no Torch/model construction, real PHY or scoring."""
import collections
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import receiver_visual as v
import numpy as np


def token_state(m=2,k=1,value=123):
    return dict(kind='tokens',m=m,K=k,order='raster',prefix=[[value]*v.SIZES[i]**2 for i in range(m)],partial_values=[value]*k)


def fixture(rule='KEEP',state=None):
    state=state or token_state()
    source=dict(source_id='opaque-source',source_index=43,selection_slot=0,preprocessing_id='a'*64,
                tokens_sha256='b'*64,archive=dict(path='/private/source.npz',sha256='c'*64))
    identity=dict(source=source,snr_hat=7,snr_true=4,noise_seed=4101,physical_key='wire-key',profile_id=123,
        public_counter=0,noise_key=['namespace','cal200','opaque-source',7,4101])
    obs=dict(waveform_sha256='d'*64,standard_noise_sha256='e'*64,received_sha256='f'*64,payload_sha256=['9'*64])
    frame=dict(schema='LAYERED_RAW_CAL200_ACTUAL_PHY_V1',status='COMPLETE',input_identity=identity,
        observation_identity=obs,actual_RX=dict(received_sha256='f'*64,receiver_state=state,header_ok=True,body_crc_accept=True),
        receiver_views={rule:state},view_sha256={rule:v.s.digest(state)},
        transmission=dict(secret_TX_tokens=[99],transmitted_id=123),post_receiver_truth=dict(secret_original_tokens=[99]))
    row=dict(source_id=source['source_id'],source_index=43,selection_slot=0,snr_hat=7,snr_true=4,noise_seed=4101,
        physical_key='wire-key',profile_id=123,public_counter=0,candidate_id='canonical',rule=rule,arms=['S','S_nom'],
        nomination_indices=[0,1],actual_RX=dict(path='/private/physical.json',sha256='1'*64),receiver_state_sha256=v.s.digest(state))
    return row,frame


class Ledger:
    def __init__(self):self.completed=collections.Counter();self.reserved=collections.Counter()
    def call(self,kind,fn,**metadata):
        self.reserved[kind]+=1;result=fn();self.completed[kind]+=1;return result


class StubModels:
    def __init__(self,ledger):self.ledger=ledger;self.calls=[];self.closed=False
    def render(self,row):
        self.calls.append(row)
        def actual_stub():
            for k in range(10):self.ledger.call('prior_scale',lambda:None,view_id=row['view_id'],scale=k)
            self.ledger.call('decoder_forward',lambda:None,view_id=row['view_id'])
            return np.full((3,256,256),.25,np.float32)
        return self.ledger.call('var_render',actual_stub,view_id=row['view_id'])
    def close(self):self.closed=True


class ReceiverStateTests(unittest.TestCase):
    def test_received_errors_preserved(self):
        state=token_state(value=4095);state['transmitted_truth']=[0]
        result=v.normalize_state(state)
        self.assertEqual(result['partial_values'],[4095]);self.assertNotIn('transmitted_truth',result)

    def test_scale10_complete_is_legal(self):
        state=token_state(10,0);self.assertEqual(v.normalize_state(state),state)

    def test_scale9_partial_is_legal(self):
        state=token_state(9,255);self.assertEqual(v.normalize_state(state),state)

    def test_gray_legacy_none_order(self):
        state=dict(kind='gray',m=0,K=0,prefix=[],partial_values=[],order=None)
        self.assertEqual(v.normalize_state(state)['rgb_fill'],.5)

    def test_gray_tokens_prohibited(self):
        with self.assertRaises((ValueError,RuntimeError)):v.normalize_state(dict(kind='gray',m=0,K=0,prefix=[[0]],partial_values=[]))

    def test_gray_other_fill_prohibited(self):
        with self.assertRaises((ValueError,RuntimeError)):v.normalize_state(dict(kind='gray',m=0,K=0,prefix=[],partial_values=[],rgb_fill=0))

    def test_integer_bool_float_and_range_rejected(self):
        for value in (True,0.0,-1,4096):
            with self.subTest(value=value):
                state=token_state();state['partial_values']=[value]
                with self.assertRaises((ValueError,RuntimeError)):v.normalize_state(state)

    def test_endpoint_types_and_bounds(self):
        for m,k in ((True,0),(0,0),(11,0),(10,1),(2,9),(2,-1)):
            with self.subTest(m=m,k=k):
                state=token_state();state['m']=m;state['K']=k
                with self.assertRaises((ValueError,RuntimeError)):v.normalize_state(state)

    def test_raster_and_full_scale_lengths(self):
        for change in ('order','length','partial'):
            state=token_state()
            if change=='order':state['order']='entropy_priority'
            if change=='length':state['prefix'][1].pop()
            if change=='partial':state['partial_values']=[]
            with self.assertRaises((ValueError,RuntimeError)):v.normalize_state(state)


class ProjectionTests(unittest.TestCase):
    def test_pure_allowlist_and_separate_join(self):
        row,frame=fixture();a,b=v.project_one(row,frame,'8'*64)
        self.assertEqual(set(a),v.MODEL_ROW_FIELDS);self.assertEqual(b['arms'],['S','S_nom'])
        text=json.dumps(a)
        for forbidden in ('source.npz','physical.json','tokens_sha256','profile_id','transmission','post_receiver_truth','candidate_id','source_id','secret'):
            self.assertNotIn(forbidden,text)
        self.assertEqual(a['state'],token_state())

    def test_wrong_accepted_values_not_fixed_from_TX(self):
        row,frame=fixture(state=token_state(value=77));frame['transmission']['secret_TX_tokens']=[0]*6
        a,_=v.project_one(row,frame,'8'*64);self.assertEqual(a['state']['prefix'][0],[77])

    def test_keep_drop_identical_cache_one_actual_event(self):
        row,frame=fixture();a,_=v.project_one(row,frame,'8'*64)
        row['rule']='DROP';frame['receiver_views']['DROP']=frame['receiver_views']['KEEP'];frame['view_sha256']['DROP']=row['receiver_state_sha256']
        b,_=v.project_one(row,frame,'8'*64)
        self.assertNotEqual(a['view_id'],b['view_id']);self.assertEqual(v.actual_input_key(a),v.actual_input_key(b))

    def test_failed_keep_and_drop_inputs_differ(self):
        row,frame=fixture();frame['actual_RX']['body_crc_accept']=False;a,_=v.project_one(row,frame,'8'*64)
        state=dict(kind='gray',m=0,K=0,order='raster',prefix=[],partial_values=[],rgb_fill=.5)
        row['rule']='DROP';row['receiver_state_sha256']=v.s.digest(state);frame['receiver_views']['DROP']=state;frame['view_sha256']['DROP']=v.s.digest(state)
        b,_=v.project_one(row,frame,'8'*64)
        self.assertEqual(a['actual_rx_key'],b['actual_rx_key']);self.assertNotEqual(v.actual_input_key(a),v.actual_input_key(b))

    def test_hash_and_identity_tamper_rejected(self):
        for key in ('source_index','public_counter','noise_seed','receiver_state_sha256'):
            row,frame=fixture();row[key]='wrong'
            with self.assertRaises((ValueError,RuntimeError)):v.project_one(row,frame,'8'*64)

    def test_actual_observation_tamper_rejected(self):
        row,frame=fixture();frame['actual_RX']['received_sha256']='0'*64
        with self.assertRaises((ValueError,RuntimeError)):v.project_one(row,frame,'8'*64)

    def test_cache_changes_for_context_observation_rx_model_or_pixels(self):
        row,frame=fixture();base,_=v.project_one(row,frame,'8'*64)
        for field in ('cache_context_key','observation_key','actual_rx_key'):
            other=copy.deepcopy(base);other[field]='6'*64
            self.assertNotEqual(v.actual_input_key(base),v.actual_input_key(other))
        frame['input_identity']['source']['preprocessing_id']='7'*64
        other,_=v.project_one(row,frame,'8'*64)
        self.assertNotEqual(v.actual_input_key(base),v.actual_input_key(other))
        other=copy.deepcopy(base);other['model_identity']='7'*64
        with self.assertRaises((ValueError,RuntimeError)):v.actual_input_key(other)

    def test_extra_model_input_path_rejected(self):
        row,frame=fixture();a,_=v.project_one(row,frame,'8'*64);a['source_path']='/private/image.npz'
        with self.assertRaises((ValueError,RuntimeError)):v.validate_model_row(a)

    def test_model_state_mutation_rejected(self):
        row,frame=fixture();a,_=v.project_one(row,frame,'8'*64);a['state']['prefix'][0][0]=42
        with self.assertRaises((ValueError,RuntimeError)):v.validate_model_row(a)


class RenderTests(unittest.TestCase):
    def test_gray_has_zero_models_and_fixed_pointfive(self):
        gray=dict(kind='gray',m=0,K=0,prefix=[],partial_values=[],order=None)
        row,frame=fixture(state=gray);a,_=v.project_one(row,frame,'8'*64)
        ledger=Ledger()
        with tempfile.TemporaryDirectory() as d:
            rows,counts=v.render_rows([a],d,ledger,lambda:self.fail('Model constructed for gray'),lambda *a:None)
            with np.load(rows[0]['reconstruction']['image_archive']['path']) as z:self.assertTrue((z['image']==.5).all())
        self.assertEqual(counts['var_render'],0);self.assertEqual(dict(ledger.completed),{});self.assertEqual(counts['fixed_gray'],1)

    def test_reuse_does_not_repeat_model_calls_or_drop_denominator(self):
        row,frame=fixture();a,_=v.project_one(row,frame,'8'*64)
        row['rule']='DROP';frame['receiver_views']['DROP']=frame['receiver_views']['KEEP'];frame['view_sha256']['DROP']=row['receiver_state_sha256']
        b,_=v.project_one(row,frame,'8'*64);ledger=Ledger();model=StubModels(ledger)
        with tempfile.TemporaryDirectory() as d:rows,counts=v.render_rows([a,b],d,ledger,lambda:model,lambda *a:None)
        self.assertEqual(len(rows),2);self.assertEqual(counts['exact_reuse'],1);self.assertEqual(ledger.completed['var_render'],1)
        self.assertEqual(ledger.completed['prior_scale'],10);self.assertTrue(model.closed)
        self.assertEqual(rows[0]['reconstruction'],rows[1]['reconstruction'])

    def test_different_actual_event_never_reused_even_same_pixels(self):
        row,frame=fixture();a,_=v.project_one(row,frame,'8'*64);b=copy.deepcopy(a)
        b['view_id']='7'*64;b['actual_rx_key']='6'*64;ledger=Ledger();model=StubModels(ledger)
        with tempfile.TemporaryDirectory() as d:_,counts=v.render_rows([a,b],d,ledger,lambda:model,lambda *a:None)
        self.assertEqual(counts['var_render'],2);self.assertEqual(counts['exact_reuse'],0)

    def test_duplicates_rejected_before_any_model(self):
        row,frame=fixture();a,_=v.project_one(row,frame,'8'*64);ledger=Ledger()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((ValueError,RuntimeError)):v.render_rows([a,a],d,ledger,lambda:self.fail('Unexpected model'),lambda *a:None)
        self.assertEqual(dict(ledger.reserved),{})

    def test_failure_preserves_partial_scene(self):
        row,frame=fixture();a,_=v.project_one(row,frame,'8'*64);ledger=Ledger()
        def failure():raise RuntimeError('synthetic model failure')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'synthetic'):v.render_rows([a],d,ledger,failure,lambda *a:None)
            self.assertEqual(v.s.read(Path(d)/'partial_logical_frames.json'),[])
        self.assertEqual(ledger.reserved['model_load'],1);self.assertEqual(ledger.completed['model_load'],0)


class ClosureAndAccessTests(unittest.TestCase):
    def test_only_absent_stdlib_zip_of_pinned_interpreter_is_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);interpreter=root/'venv/bin/python';interpreter.parent.mkdir(parents=True);interpreter.write_text('bound')
            archive=root/'base/lib'/('python'+str(sys.version_info.major)+str(sys.version_info.minor)+'.zip')
            encoding=archive.parent/('python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor))/'encodings/__init__.py'
            encoding.parent.mkdir(parents=True);encoding.write_text('bound')
            with patch.object(v.sys,'executable',str(interpreter)),patch.object(v.sys,'base_prefix',str(root/'base')):
                f=v.SourceReadFirewall([interpreter,encoding],root/'out',[],[])
                unbound=v.SourceReadFirewall([interpreter],root/'out',[],[])
            f.root=unbound.root=root;f.env=unbound.env=root/'other'
            self.assertTrue(f.permitted(archive));self.assertFalse(f.permitted(archive,True))
            self.assertFalse(unbound.permitted(archive));self.assertFalse(f.permitted(archive.parent/'source.zip'))
            archive.write_text('unbound existing bytes');self.assertFalse(f.permitted(archive))

    def test_pins_compare_identity_ignoring_optional_size(self):
        self.assertTrue(v.same_pin({'path':'a','sha256':'b'},dict(path='a',sha256='b',bytes=3)))
        self.assertFalse(v.same_pin({'path':'a','sha256':'b'},dict(path='a',sha256='c',bytes=3)))

    def test_missing_optional_distribution_metadata_is_not_source_access(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);meta=root/'runtime/x.dist-info/METADATA';meta.parent.mkdir(parents=True);meta.write_text('bound')
            f=v.SourceReadFirewall([meta],root/'out',[],[]);f.root=root;f.env=root/'other'
            self.assertTrue(f.permitted(meta.parent/'entry_points.txt'))
            (meta.parent/'entry_points.txt').write_text('unbound existing bytes')
            self.assertFalse(f.permitted(meta.parent/'entry_points.txt'))
            self.assertFalse(f.permitted(meta.parent/'source.npz'))

    def test_firewall_only_bound_model_source_and_own_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'tmp/visual';out.mkdir(parents=True);code=root/'code/model.py';weight=root/'model/weights.pt'
            f=v.SourceReadFirewall([weight],out,[],[code]);f.root=root;f.env=root/'runtime'
            self.assertTrue(f.permitted(weight));self.assertTrue(f.permitted(code));self.assertTrue(f.permitted(out/'image.npz',True))
            self.assertFalse(f.permitted(weight,True));self.assertFalse(f.permitted(root/'sources/000.npz'))
            self.assertFalse(f.permitted(root/'projection/logical_join.json'));self.assertFalse(f.permitted(root/'phy/frames/actual.json'))
            self.assertTrue(f.permitted(root/'runtime/numpy.py'))

    def test_firewall_rejects_data_open_not_merely_declaration(self):
        with tempfile.TemporaryDirectory() as d:
            f=v.SourceReadFirewall([],Path(d)/'visual',[],[]);f.root=Path(d);f.env=Path(d)/'runtime'
            with self.assertRaises((ValueError,RuntimeError)):f.hook('open',(str(Path(d)/'source.npz'),'rb',0))
            self.assertEqual(len(f.blocked),1)

    def test_wait_requires_exit_zero_actual_wait_and_dead_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'actual_wait.json'
            for data in ({'returncode':0},{'actual_wait':True,'returncode':1,'pid':123}):
                path.write_text(json.dumps(data))
                with self.assertRaises((ValueError,RuntimeError)):v.require_successful_wait(d)
            path.write_text(json.dumps(dict(actual_wait=True,returncode=0,pid=123,start_ticks=456)))
            with patch.object(v.s,'process_identity',return_value=None):self.assertEqual(v.require_successful_wait(d)['path'],str(path))
            with patch.object(v.s,'process_identity',return_value=dict(pid=123,start_ticks=456,state='S')):
                with self.assertRaises((ValueError,RuntimeError)):v.require_successful_wait(d)

    def test_packet_audit_detects_missing_result(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'packets';p.mkdir();fk='a'*64;request=dict(event='header',request={});key=v.s.digest([fk,'header'])
            v.s.write(p/(key+'.reserved.json'),dict(frame=fk,request=request))
            with self.assertRaises((ValueError,RuntimeError)):v.packet_audit(root,{fk})
            v.s.write(p/(key+'.complete.json'),dict(decoded='actual'))
            audit=v.packet_audit(root,{fk});self.assertEqual(audit['packet_decodes'],1)

    def test_grouped_tx_not_counted_decode_and_failure_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'packets';p.mkdir();fk='a'*64
            for direction in ('tx','rx'):
                request=dict(direction=direction,kind='body',counter=0);key=v.s.digest([fk,'grouped',request])
                v.s.write(p/(key+'.reserved.json'),dict(frame=fk,request=request));v.s.write(p/(key+'.complete.json'),dict(request,stage='done'))
            self.assertEqual(v.packet_audit(d,{fk})['packet_decodes'],1)
            v.s.write(p/'bad.failure.json',dict(error='stop'))
            with self.assertRaises((ValueError,RuntimeError)):v.packet_audit(d,{fk})

    def test_packet_orphan_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'packets';p.mkdir();fk='a'*64;key=v.s.digest([fk,'header'])
            v.s.write(p/(key+'.reserved.json'),dict(frame=fk,request=dict(event='header',request={})))
            v.s.write(p/(key+'.complete.json'),dict(decoded='actual'))
            with self.assertRaises((ValueError,RuntimeError)):v.packet_audit(d,{'b'*64})

    def test_ast_extract_no_top_level_source_access(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'math.py';path.write_text("raise RuntimeError('top-level source access')\ndef f(x):\n return x+1\n")
            ns={};proof=v.definitions(v.s.pin(path),{'f'},ns)
            self.assertEqual(ns['f'](2),3);self.assertEqual(len(proof),64)


class ActualVisualClosureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.out=self.root/'visual';self.out.mkdir()
        self.proj=self.root/'projection';self.proj.mkdir()
        r,f=fixture();a,_=v.project_one(r,f,'8'*64)
        r['rule']='DROP';f['receiver_views']['DROP']=f['receiver_views']['KEEP'];f['view_sha256']['DROP']=r['receiver_state_sha256']
        b,_=v.project_one(r,f,'8'*64)
        self.inputs=[a,b];mp=v.s.write(self.proj/'model_inputs.json',self.inputs)
        vp=v.s.write(self.proj/'visual_input_completion.json',dict(synthetic=True))
        v.s.write(self.proj/'completion.json',dict(model_inputs=mp,visual_input_completion=vp))
        common=v.s.load(v.s.ROOT/'experiments/entropy-main-common500-20261013/common.py','_test_real_ledger_only')
        caps=dict(model_load=1,var_render=2,prior_scale=20,decoder_forward=2)
        ledger=common.Ledger(self.out/'calls.sqlite',caps,lambda:None);models=StubModels(ledger)
        self.rows,counts=v.render_rows(self.inputs,self.out,ledger,lambda:models,lambda *a:None)
        summary=ledger.summary();ledger.close()
        bundle=v.s.write(self.root/'bundle.json',dict(synthetic=True))
        access=v.s.write(self.out/'input_access_audit.json',dict(blocked=[],source_data_access_allowed=False,source_or_TX_paths_received=False))
        history=v.s.write(self.out/'resource_history.json',[{'synthetic':True}])
        self.reg=dict(runtime=dict(visual_directory=str(self.out),projection_directory=str(self.proj)),
            expected_logical_var_images=2,visual_registration=dict(path='/fake/registration.json',sha256='a'*64,bytes=10))
        self.done=dict(status='COMPLETE_R1_VISUAL',registration=dict(path='/fake/registration.json',sha256='a'*64),
            model_inputs=mp,projection_completion=vp,image_rows=v.s.write(self.out/'logical_frames.json',self.rows),
            logical_rows=2,counts=counts,ledger=summary,ledger_file=v.s.pin(self.out/'calls.sqlite'),model_identity=v.MODEL_IDENTITY,
            numeric_flags=v.s.FLAGS,source_truth_supplied_to_RX=False,tokenizer_calls=0,extra_qualification_calls=0,
            input_access_audit=access,resource_history=history,model_bundle=bundle)
        self.save()

    def tearDown(self):self.tmp.cleanup()
    def save(self):
        (self.out/'completion.json').write_text(json.dumps(self.done),encoding='utf8')
    def audit(self):
        with patch.object(v,'require_successful_wait',return_value=dict(path='/fake/wait',sha256='b'*64)):
            return v.audit_visual(self.reg)

    def test_complete_nested_ledger_and_exact_reuse_pass(self):
        result=self.audit();self.assertEqual(result['counts']['var_render'],1);self.assertEqual(result['counts']['exact_reuse'],1)

    def test_missing_logical_image_rejected(self):
        p=self.out/'logical_frames.json';p.write_text(json.dumps(self.rows[:1]));self.done['image_rows']=v.s.pin(p);self.save()
        with self.assertRaises((ValueError,RuntimeError)):self.audit()

    def test_false_reuse_rejected(self):
        self.rows[1]['reuse']['from_view_id']='9'*64;p=self.out/'logical_frames.json';p.write_text(json.dumps(self.rows))
        self.done['image_rows']=v.s.pin(p);self.save()
        with self.assertRaises((ValueError,RuntimeError)):self.audit()

    def test_missing_prior_scale_rejected(self):
        import sqlite3
        db=sqlite3.connect(self.out/'calls.sqlite');db.execute("DELETE FROM calls WHERE id=(SELECT MIN(id) FROM calls WHERE kind='prior_scale')");db.commit();db.close()
        self.done['ledger_file']=v.s.pin(self.out/'calls.sqlite');self.save()
        with self.assertRaises((ValueError,RuntimeError)):self.audit()

    def test_unresolved_model_call_rejected(self):
        import sqlite3
        db=sqlite3.connect(self.out/'calls.sqlite');db.execute("UPDATE calls SET state='FAILED' WHERE kind='var_render'");db.commit();db.close()
        self.done['ledger_file']=v.s.pin(self.out/'calls.sqlite');self.save()
        with self.assertRaises((ValueError,RuntimeError)):self.audit()

    def test_changed_actual_image_rejected(self):
        p=Path(self.rows[0]['reconstruction']['image_archive']['path'])
        with p.open('wb') as f:np.savez(f,image=np.zeros((3,256,256),np.float32))
        with self.assertRaises((ValueError,RuntimeError)):self.audit()

    def test_false_model_call_count_rejected(self):
        self.done['counts']['var_render']=2;self.save()
        with self.assertRaises((ValueError,RuntimeError)):self.audit()


if __name__=='__main__':unittest.main()
