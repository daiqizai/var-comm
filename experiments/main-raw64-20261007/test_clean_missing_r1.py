"""CPU-only contract fixtures; actual metadata paths can be explicitly bound."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import clean_missing_r1 as c

HERE=Path(__file__).absolute().parent
RESEARCH=HERE.parents[2]

def metadata():
    if os.environ.get('MAIN64_CLEAN_TEST_ASSETS'):return c.read(os.environ['MAIN64_CLEAN_TEST_ASSETS'])
    return dict(old_clean_source0=str(RESEARCH/'content_real_64qam_20261006/current/H/clean-quality200/sources/0000.json'),
        new_catalogue=str(HERE.parents[1]/'assets/catalogue.json'),
        original_source_driver=str(RESEARCH/'content_real_64qam_20261006/h_source_driver.py'),
        original_raw_cp=str(RESEARCH/'content_real_64qam_20261006/current/S1/export-assets/source_checkpoints/0000.json'))

def put(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d),encoding='utf-8');return str(p)
def rows395():
    prefix=[sum(s*s for s in (1,2,3,4,5,6,8,10,13,16)[:m]) for m in range(11)]
    result=[]
    for T in range(1,396):
        m=max(i for i,v in enumerate(prefix) if v<=T)
        result.append(dict(m=m,K=T-prefix[m],token_count=T))
    return result

def fixture(d):
    d=Path(d);root=d/'repo';old=root/'old';asset=root/'S1/export-assets';clean=old/'clean';runtime=root/'runtime'
    runtime.mkdir(parents=True);ids=[f'cal{i:04d}' for i in range(1000)]
    source={}
    for k in ('source_driver_module','quality_module','closure_module','environment_module'):
        f=runtime/(k+'.py');f.write_text('# fixture never imported\n');source[k]=str(f)
    own={str(HERE/n):c.sha(HERE/n) for n in ('clean_missing_r1.py','test_clean_missing_r1.py')}
    pins={**own,**{p:c.sha(p) for p in source.values()}}
    cal=put(old/'cal.json',dict(stage='m1_calibration',calibration_or_development='m1_calibration',source_ids=ids,preprocessing_ids=['a'*64]*1000,identity={'model':'frozen'}))
    population=put(old/'source200.json',dict(source_ids=ids[:200]))
    model=root/'model.bin';model.write_bytes(b'fake weights not loaded');gpu=root/'nvidia-smi';gpu.write_bytes(b'not executed')
    cfgp=old/'source_config.json';rp=old/'registration.json'
    oc=dict(root=str(root),uep_runtime=str(runtime),native_runtime=str(runtime),S1=str(root/'S1'),source200=population,calibration_registration=cal,registration=str(rp))
    put(cfgp,oc);oldreg=dict(status='H_EXECUTION_REVISION_REGISTERED',source_bindings=pins,input_bindings={str(cfgp):c.sha(cfgp),cal:c.sha(cal),population:c.sha(population)})
    put(rp,oldreg);regsha=c.sha(rp)
    assets={};outputs={}
    for i,sid in enumerate(ids[:200]):
        ap=asset/'sources'/f'{i:04d}.npz';ap.parent.mkdir(parents=True,exist_ok=True);ap.write_bytes(b'bound archive fixture, not loaded')
        cp=dict(source_index=i,source_id=sid,original_calibration_index=i,preprocessing_id='a'*64,archive=str(ap),outputs={str(ap):c.sha(ap)})
        cp['payload_sha256']=c.digest(cp);cp_path=asset/'source_checkpoints'/f'{i:04d}.json';put(cp_path,cp)
        assets.update({str(cp_path):c.sha(cp_path),str(ap):c.sha(ap)})
        rows=clean/'sources'/f'{i:04d}.json';put(rows,rows395())
        op=clean/'source_checkpoints'/f'{i:04d}.json';put(op,dict(source_index=i,source_id=sid,clean_states=395,registration_sha256=regsha,outputs={str(rows):c.sha(rows)}))
        outputs.update({str(rows):c.sha(rows),str(op):c.sha(op)})
    ad=put(asset/'completion.json',dict(status='S1_EXPORT_ASSETS_COMPLETE',outputs=assets))
    done=put(clean/'completion.json',dict(status='H_CLEAN_QUALITY200_COMPLETE',source_count=200,registration_sha256=regsha,development_used=False,holdout_used=False,outputs=outputs))
    log=old/'worker.log';log.write_text('normally closed fixture\n');python='/original/UM/python'
    argv=[python,'-B',source['source_driver_module'],'--config',str(cfgp),'--stage','clean-quality200']
    ident=dict(pid=1,start_ticks=2,uid=1002,argv=argv)
    launch=put(old/'launch.json',dict(identity=ident,argv=argv,registration_sha256=regsha))
    ex=put(old/'exit.json',dict(identity=ident,exit_code=0,closed_log_sha256=c.sha(log),completion=done,completion_sha256=c.sha(done),registration_sha256=regsha))
    num=put(old/'native_q.json',dict(status='REAL_NATIVE_QUALIFICATION_PASS',synthetic=False,frozen_identity={'model':'frozen'},numerical_runtime={'threads':6,'interop_threads':2}))
    new=put(root/'profiles.json',[dict(m=m,K=K,token_count=T) for m,K,T in c.STATES])
    qlog=root/'tests.log';qlog.write_text('Ran 9 tests in 1s\n\nOK\n')
    qp=put(root/'q.json',dict(status='MAIN_RAW64_CLEAN_MISSING_CPU_TESTS_PASS',tests_run=9,test_modules=['test_clean_missing_r1'],python=python,exit_code=0,process_waited=True,new_model_calls=0,new_packet_decodes=0,source_bindings=own,input_bindings={},outputs={str(qlog):c.sha(qlog)},log=str(qlog)))
    request=dict(schema=c.SCHEMA,states=[list(x) for x in c.STATES],source_count=200,images=200,new_packet_decodes=0,selection=False,development_used=False,holdout_used=False,
        **source,source_bindings=pins,input_bindings={p:c.sha(p) for p in [str(cfgp),str(rp),num,done,launch,ex,str(log),ad,new,str(model),str(gpu),qp]},
        old_source_config=str(cfgp),root=str(root),native_qualification=num,native_source_bindings={p:c.sha(p) for p in source.values()},model_bindings={str(model):c.sha(model)},
        old_clean_completion=done,old_clean_launch=launch,old_clean_exit=ex,old_clean_log=str(log),python=python,new_profiles=new,prepared_qualification=qp,
        resources=dict(gpu_device=0,threads=6,interop_threads=2,affinity=[4,5,6,7,8,9],nice=15),max_seconds=3600,
        visual_lock=str(root/'outputs/CONTENT-REAL-64QAM-20261006/shared_visual.lock'),nvidia_smi=str(gpu),execution_dir=str(root/'newexec'),out=str(root/'newout'))
    return request

class CleanTests(unittest.TestCase):
    def test_actual_metadata_gap_and_first_source(self):
        m=metadata();old=c.read(m['old_clean_source0']);new=c.read(m['new_catalogue'])
        self.assertEqual(c.endpoint_gap(old,new['profiles']),[list(x) for x in c.STATES])
        cp=c.read(m['original_raw_cp']);self.assertEqual(cp['original_calibration_index'],171)
        self.assertEqual(cp['source_id'],'n02091032/ILSVRC2012_val_00022684_n02091032')
    def test_register_full_metadata_200_and_real_load_consumer(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(c,'gone',return_value=True):
            req=fixture(t);rp=put(Path(t)/'request.json',req);reg=c.register(rp)
            ctx=c.load_registered(str(Path(req['execution_dir'])/'config.json'))
            self.assertEqual(len(ctx['records']),200);self.assertEqual(reg['images'],200)
            self.assertIn(ctx['records'][199]['archive'],reg['input_bindings'])
            self.assertFalse(Path(req['out']).exists())
            with self.assertRaisesRegex(ValueError,'Fresh'):c.register(rp)
    def test_full_population_or_changed_CP_fail_before_register(self):
        with tempfile.TemporaryDirectory() as t:
            req=fixture(t);cp=Path(t)/'repo/S1/export-assets/source_checkpoints/0000.json';d=c.read(cp);d['source_id']='dev001';put(cp,d)
            with self.assertRaisesRegex(ValueError,'SHA'):c.inspect(req,check_old_process=False)
            self.assertFalse(Path(req['execution_dir']).exists())
    def test_real_qualification_consumer_rejects_wrong_closed_test_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            req=fixture(t);c.qualification(req,req['source_bindings'],dict(req['input_bindings']))
            q=c.read(req['prepared_qualification']);q['tests_run']=8;put(Path(req['prepared_qualification']),q);req['input_bindings'][req['prepared_qualification']]=c.sha(req['prepared_qualification'])
            with self.assertRaisesRegex(ValueError,'qualification'):c.qualification(req,req['source_bindings'],dict(req['input_bindings']))
    def test_missing_state_set_is_exact_no_extra_render(self):
        old=rows395();good=[dict(m=m,K=K,token_count=T) for m,K,T in c.STATES]
        for rows in ([],good+[dict(m=8,K=143,token_count=398)]):
            with self.assertRaisesRegex(ValueError,'Only actual'):c.endpoint_gap(old,rows)
        with self.assertRaisesRegex(ValueError,'395'):c.endpoint_gap(old[:-1],good)
    def test_exact_one_render_call_and_original_PSNR_function(self):
        path=metadata()['original_source_driver'];tree=ast.parse(Path(path).read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='pixel_scores')
        env={'np':np,'require':c.require};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
        tokens=np.arange(680,dtype=np.int64);target=np.ones((3,256,256),np.float32);calls=[];checks=[]
        def render(a,m,K):calls.append((a.copy(),m,K));return np.zeros_like(target)
        rows=c.evaluate(tokens,target,render,env['pixel_scores'],lambda:checks.append(1))
        self.assertEqual([(m,K,len(a)) for a,m,K in calls],list(c.STATES));self.assertEqual(len(checks),1)
        for a,m,K in calls:np.testing.assert_array_equal(a,tokens[:255+K])
        self.assertTrue(all(x['mse']==1 and x['psnr_db']==0 for x in rows))
    def test_pixel_and_token_source_validation(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);pixels=np.zeros((3,256,256),np.uint8);tokens=np.arange(680,dtype=np.int64)
            ap=d/'source.npz';np.savez(ap,pixels=pixels,tokens=tokens);cp=put(d/'cp.json',dict(source_id='cal'))
            rec=dict(source_id='cal',checkpoint=cp,archive=str(ap),preprocessing_id=c.hashlib.sha256(pixels.tobytes()).hexdigest())
            flat,target=c.read_source(rec);np.testing.assert_array_equal(flat,tokens);self.assertEqual(target.dtype,np.float32)
            rec['preprocessing_id']='0'*64
            with self.assertRaisesRegex(ValueError,'uint8'):c.read_source(rec)
    def test_software_error_and_nonfinite_not_zero_score(self):
        def bad(a,m,K):raise RuntimeError('decoder error')
        with self.assertRaises(RuntimeError):c.evaluate(np.arange(680),None,bad,None,lambda:None)
        with self.assertRaisesRegex(ValueError,'float32'):c.evaluate(np.arange(680),None,lambda *a:np.zeros((3,256,256),np.uint8),None,lambda:None)
    def test_worker_wrong_environment_preserves_failure_without_model(self):
        with tempfile.TemporaryDirectory() as t:
            req=dict(out=str(Path(t)/'out'),python='/not/current/python')
            with mock.patch.object(c.sys,'platform','linux'),mock.patch.object(c,'load_registered',return_value={'request':req}),mock.patch.object(c.signal,'signal'),mock.patch.object(c,'module',side_effect=AssertionError('No import allowed')):
                with self.assertRaisesRegex(ValueError,'UM'):c.worker('bound_config')
            self.assertTrue((Path(t)/'out/failure.json').exists());self.assertFalse((Path(t)/'out/completion.json').exists())

if __name__=='__main__':unittest.main()
