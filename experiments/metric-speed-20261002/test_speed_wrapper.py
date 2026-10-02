"""Pure CPU receipt/runtime fixtures; no original modules, weights or GPU."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import wrapper as w
import benchmark as b


def records(own):
    q=dict(status='QUALIFIED',selected_implementation='accelerated',development_read=False,
        training_updates=0,source_bindings=own,base_source_bindings={'base.py':'a'*64},
        strict_equality_passed=True,speedup=1.20,model_identity={'models':'original'},numerical_backend={'flags':'original'})
    p=dict(status='PUSHED',checks='PASS',commit='a'*40,remote_commit='a'*40,source_bindings=own)
    return q,p


class SpeedTests(unittest.TestCase):
    def test_only_original_dedicated_resource_exception_retries75(self):
        class DedicatedBusy(Exception):pass
        class ResourceBusy(Exception):pass
        common=SimpleNamespace(assets=SimpleNamespace(old=SimpleNamespace(b=SimpleNamespace(ResourceBusy=DedicatedBusy))))
        self.assertTrue(w.is_resource_busy(DedicatedBusy('thermal'),common))
        self.assertFalse(w.is_resource_busy(ResourceBusy('thermal'),common))
        self.assertFalse(w.is_resource_busy(RuntimeError('ResourceBusy thermal'),common))
        self.assertFalse(w.is_resource_busy(DedicatedBusy('thermal'),None))
        def throwing(error):
            def main(stage):raise error
            return SimpleNamespace(main=main)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);receipt=dict(status='RUNNING',development_read=False,training_updates=0)
            with self.assertRaises(SystemExit) as caught:
                b.record_exception(receipt,DedicatedBusy('thermal'),common,out)
            self.assertEqual(caught.exception.code,75);self.assertEqual(receipt['status'],'RUNNING')
            self.assertIsNone(receipt['selected_implementation']);self.assertEqual(receipt['resource_wait']['exit_code'],75)
            self.assertEqual(w.read(out/'resource_wait.json')['stage'],'benchmark')
            with self.assertRaises(SystemExit) as caught:
                w.run_stage(throwing(DedicatedBusy('thermal')),common,'calibration',out)
            self.assertEqual(caught.exception.code,75)
            self.assertEqual(w.read(out/'resource_wait.json')['stage'],'calibration')
            for error in (ResourceBusy('same name'),RuntimeError('ResourceBusy same text')):
                failed=dict(status='RUNNING')
                with self.assertRaises(type(error)):b.record_exception(failed,error,common,out)
                self.assertEqual(failed['status'],'FAILED')
                with self.assertRaises(type(error)):w.run_stage(throwing(error),common,'actual',out)

    def test_choice_is_strict_finite_and_publication_matches_all_sources(self):
        own={'speed.py':'1'*64};q,p=records(own)
        self.assertEqual(w.validate_selection(q,p,own),'accelerated')
        for field,value in [('speedup',float('nan')),('speedup',float('inf')),('speedup',None),
                            ('speedup',1.09),('strict_equality_passed',False),('development_read',True),
                            ('training_updates',1),('source_bindings',{}),('base_source_bindings',{})]:
            wrong=dict(q);wrong[field]=value
            with self.assertRaises(RuntimeError):w.validate_selection(wrong,p,own)
        for field,value in [('status','COMMITTED'),('checks','FAIL'),('remote_commit','b'*40),('source_bindings',{})]:
            wrong=dict(p);wrong[field]=value
            with self.assertRaises(RuntimeError):w.validate_selection(q,wrong,own)
        original=dict(q,status='USE_ORIGINAL',selected_implementation='original',strict_equality_passed=False,speedup=None)
        self.assertEqual(w.validate_selection(original,p,own),'original')
        with self.assertRaises(RuntimeError):w.validate_selection(dict(original,selected_implementation='accelerated'),p,own)

    def test_all_case_interleaved_throughput_not_best_case_selection(self):
        row=dict(original_seconds=[2.,2.],accelerated_seconds=[1.,1.])
        decision=b.assess([row]);self.assertEqual(decision['speedup'],2.);self.assertEqual(decision['status'],'QUALIFIED')
        slow=dict(original_seconds=[1.,1.],accelerated_seconds=[2.,2.])
        decision=b.assess([row,slow]);self.assertEqual(decision['speedup'],1.);self.assertEqual(decision['status'],'USE_ORIGINAL')
        for values in ([],[dict(original_seconds=[1.,1.],accelerated_seconds=[0.,0.])],
                       [dict(original_seconds=[np.nan,1.],accelerated_seconds=[1.,1.])]):
            with self.assertRaises(RuntimeError):b.assess(values)

    def test_runtime_receipt_and_profile_registered_for_both_choices(self):
        for selected in ('original','accelerated'):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);source=root/'speed.py';source.write_text('frozen speed source')
                receipt=root/'qualification.json';w.write(receipt,{'engineering':'fixture'})
                bindings={str(p):w.sha(p) for p in (source,receipt)}
                common=SimpleNamespace(OUT=root,source_bindings=lambda:{str(source):w.sha(source)})
                sealed={}
                def seal(path,value):
                    if str(path) in sealed and sealed[str(path)]!=value:raise RuntimeError('Immutable registration differs')
                    sealed[str(path)]=copy.deepcopy(value);w.write(path,value)
                common.seal=seal
                def original_registration(loaded,data,stage,inputs):
                    path=root/(stage+'_registration.json')
                    common.seal(path,dict(identity=loaded['identity'],source_bindings=common.source_bindings(),
                        scientific_field='unchanged',input_artifacts={str(p):w.sha(p) for p in inputs}))
                    return w.sha(path)
                common.registration=original_registration
                loaded={'identity':{'models':'original'},'device':'cpu'}
                common.setup=lambda:loaded
                old_infer=lambda *args,**kw:('original',args,kw)
                receiver=SimpleNamespace(infer=old_infer,torch='CPU-fixture')
                accelerator=SimpleNamespace(infer=lambda module,*args,**kw:('accelerated',module,args,kw))
                profile=dict(selected_implementation=selected,receipt_inputs={str(receipt):w.sha(receipt)})
                q=dict(model_identity=loaded['identity'],numerical_backend={'flags':'original'})
                w.install_runtime(common,receiver,SimpleNamespace(),q,profile,bindings,accelerator)
                if selected=='original':self.assertIs(receiver.infer,old_infer)
                else:self.assertEqual(receiver.infer(1,lam=2)[0],'accelerated')
                before_seal=common.seal
                common.registration(loaded,{},'m2_calibration',[])
                value=w.read(root/'m2_calibration_registration.json')
                self.assertEqual(value['runtime_profile'],profile);self.assertEqual(value['scientific_field'],'unchanged')
                self.assertIn(str(receipt),value['input_artifacts']);self.assertEqual(value['source_bindings'],bindings)
                self.assertIs(common.seal,before_seal)
                with patch.object(w,'numerical_backend',return_value={'flags':'original'}):self.assertIs(common.setup(),loaded)
                with patch.object(w,'numerical_backend',return_value={'flags':'changed'}):
                    with self.assertRaises(RuntimeError):common.setup()
                receipt.write_text('changed selection')
                with self.assertRaises(RuntimeError):common.source_bindings()

    def test_source_inventory_excludes_receipts_and_requires_real_M1_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir()
            for name in ('wrapper.py','controller.py','SPEED_PROTOCOL.md','qualification.json'):(source/name).write_text(name)
            own=w.source_bindings(source);self.assertEqual(len(own),3)
            self.assertTrue(all(Path(p).suffix in ('.py','.md') for p in own))
            parent=root/w.PARENT_REL;parent.mkdir(parents=True)
            pub=dict(status='PUSHED',checks='PASS',commit='b'*40,remote_commit='b'*40,source_bindings=own)
            w.write(parent/'m1_publication.json',pub)
            done=dict(status='M1_COMPLETE',synthetic=False,training_updates=0,publication=pub)
            w.write(parent/'m1_complete.json',done)
            self.assertEqual(len(w.require_m1(root)),2)
            w.write(parent/'m1_complete.json',dict(done,synthetic=True))
            with self.assertRaises(RuntimeError):w.require_m1(root)

    def test_known_original_GPU_entrypoints_checked_without_counting_controller(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);proc=root/'proc';proc.mkdir()
            for pid,path in [(101,root/w.BASE_REL/'m1_runner.py'),(102,root/w.BASE_REL/'supervisor.py'),
                             (103,root/'experiments/metric-speed-20261002/controller.py')]:
                p=proc/str(pid);p.mkdir();(p/'cmdline').write_bytes(b'python\0'+str(path).replace('\\','/').encode()+b'\0')
            # Windows and Linux path separators are normalized by using the
            # actual resolved path in this fixture, not by inspecting processes.
            p=proc/'101';(p/'cmdline').write_bytes(b'python\0'+(str((root/w.BASE_REL).resolve())+'/m1_runner.py').encode()+b'\0')
            self.assertEqual(w.original_gpu_workers(root,proc),[101])

    def test_strict_token_latent_diagnostic_and_KV_checks(self):
        equal=SimpleNamespace(equal=np.array_equal)
        original=dict(tokens=[np.array([[0,1]])],fhat=np.array([1.],dtype=np.float32),diagnostics=[{'gain':1.}],inference='original')
        b.compare(equal,original,copy.deepcopy(original),'fixture')
        for field,value in [('tokens',[np.array([[1,0]])]),('fhat',np.array([1.000001],dtype=np.float32)),
                            ('diagnostics',[{'gain':1.000001}]),('inference','changed')]:
            wrong=copy.deepcopy(original);wrong[field]=value
            with self.assertRaises(b.NumericalMismatch):b.compare(equal,original,wrong,'fixture')
        attn=SimpleNamespace(caching=False,cached_k=None,cached_v=None)
        var=SimpleNamespace(blocks=[SimpleNamespace(attn=attn)])
        b.kv_closed(var,'fixture');attn.caching=True
        with self.assertRaises(b.NumericalMismatch):b.kv_closed(var,'fixture')


if __name__=='__main__':unittest.main()
