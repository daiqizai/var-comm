"""Small actual-file host binding tests and fake RX; no GPU/model calls."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_single_rx_v1 as h


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.base=self.root/'personal';self.base.mkdir()
        self.host=self.root/'system';self.host.mkdir();self.library=self.host/'libfixture.so.1'
        self.library.write_bytes(b'H800 tiny native fixture')
        self.row=dict(path=str(self.library),bytes=self.library.stat().st_size,sha256=h.g.sha(self.library))
        self.doc=dict(schema='H800_NEW_HOST_NATIVE_BINDING_V1',status='CPU_IMPORT_OBSERVED_NOT_NUMERICAL',
            model_calls=0,tensor_calls=0,checkpoint_reads=0,source_image_reads=0,CUDA_VISIBLE_DEVICES='',
            original_runtime_projection_sha256=h.PROJECTION_SHA,
            actual_child_wait=dict(actual_wait=True,returncode=0,interrupted_or_timeout=False),files=[self.row])

    def check(self,doc=None):return h.validate_native_document(self.doc if doc is None else doc,base=self.base)

    def test_actual_external_native_bytes_required_each_time(self):
        result=self.check();self.assertEqual(result,[self.row]);self.assertIsNot(result[0],self.row)
        self.library.write_bytes(b'x'*self.row['bytes'])
        with self.assertRaisesRegex(RuntimeError,'native file bytes differ'):self.check()

    def test_native_scope_and_actual_wait_are_not_optional(self):
        changes=[dict(schema='old'),dict(status='PASS'),dict(CUDA_VISIBLE_DEVICES='0'),
            dict(original_runtime_projection_sha256='0'*64),dict(files=[]),dict(files=[self.row,self.row])]
        changes.extend({k:1} for k in ('model_calls','tensor_calls','checkpoint_reads','source_image_reads'))
        changes.extend({'actual_child_wait':dict(self.doc['actual_child_wait'],**{k:v})} for k,v in
            [('actual_wait',False),('returncode',1),('interrupted_or_timeout',True)])
        for change in changes:
            with self.subTest(change=change),self.assertRaises(RuntimeError):self.check(self.doc|change)

    def test_personal_or_noncanonical_native_substitution_rejected(self):
        personal=self.base/'libfixture.so';personal.write_bytes(self.library.read_bytes())
        rows=[self.row|dict(path=str(personal)),self.row|dict(path=str(self.host/'../system/libfixture.so.1')),
            self.row|dict(sha256='0'*64),self.row|dict(bytes=self.row['bytes']+1)]
        for row in rows:
            with self.subTest(row=row),self.assertRaises(RuntimeError):self.check(self.doc|dict(files=[row]))

    def test_old_host_metadata_is_exact_but_not_reinterpreted_as_current_files(self):
        oldpath='/not-mounted-old-host/libhistorical.so.1'
        raw=dict(path=oldpath,sha256='a'*64,bytes=999)
        normalized=dict(raw,observed_lexical_path=oldpath)
        with mock.patch.object(h.g,'normalize_native_facts',side_effect=AssertionError('Old host path accessed')):
            self.assertEqual(h.historical_native_normalization([raw],{},[normalized],self.base),[normalized])
            for change in [dict(sha256='b'*64),dict(bytes=998),dict(path='/wrong.so')]:
                with self.assertRaises(RuntimeError):h.historical_native_normalization([raw|change],{},[normalized],self.base)
        with self.assertRaises(RuntimeError):h.historical_native_normalization([],{},[normalized],self.base)

    def test_personal_native_still_uses_original_map_and_normalizer(self):
        personal=self.base/'liboriginal.so';personal.write_bytes(b'frozen PFS byte fixture')
        raw=dict(path=str(personal),sha256=h.g.sha(personal),bytes=personal.stat().st_size)
        normalized=dict(raw,observed_lexical_path=str(personal))
        self.assertEqual(h.historical_native_normalization([raw],{str(personal):raw},[normalized],self.base),[normalized])
        with self.assertRaisesRegex(RuntimeError,'absent from active original map'):
            h.historical_native_normalization([raw],{},[normalized],self.base)

    def test_new_host_runtime_retains_real_original_pfs_reverification(self):
        witness=self.base/'original_runtime.bin';witness.write_bytes(b'exact original PFS file')
        row=dict(original_path='/old/original_runtime.bin',path=str(witness),
            sha256=h.g.sha(witness),bytes=witness.stat().st_size)
        generated=[]
        for name in ('venv_a','venv_b'):
            directory=self.base/name;directory.mkdir()
            for filename,content in [('pyvenv.cfg','home = '+str(self.base)+'\ninclude-system-site-packages = false\n'),
                    ('frozen_package_roots.pth',str(self.base)+'\n')]:
                path=directory/filename;path.write_bytes(content.encode())
                generated.append(dict(path=str(path),content=content,bytes=len(content.encode()),
                    sha256=h.g.sha(path),generated_new_bytes=True))
        old_native=[dict(path='/absent-old-system/libold.so',sha256='a'*64,bytes=123)]
        env=dict(files=[row],projection_generated_files=generated,host_native_library_facts=old_native)
        spec=dict(environment=dict(path=str(self.base/'projection.json'),sha256=h.PROJECTION_SHA),fixture_env=env)
        def fixture_validation(spec,base,verify_environment=True):
            # Global RuntimeFiles is intentionally supplied by the isolated gate
            # namespace; it must retain the genuine original reverify method.
            runtime=RuntimeFiles(spec['fixture_env'],base)
            if verify_environment:runtime.reverify()
            return None,runtime,[],{},{}
        def documents(path,digest):return self.doc if str(path)==str(h.NATIVE_PATH) else env
        with mock.patch.object(h.g,'validate_spec',fixture_validation),mock.patch.object(h.g,'checked_json',side_effect=documents),\
                mock.patch.object(h,'NATIVE_PATH',self.base/'native.json'):
            result=h.HOST.validate_spec(spec,self.base)
            self.assertEqual(result[1].host_native,[self.row]);self.assertEqual(result[1].historical_host_native,old_native)
            self.assertIs(result[1].reverify.__func__,h.g.RuntimeFiles.reverify)
            self.assertEqual(env['host_native_library_facts'],old_native)
            witness.write_bytes(b'X'*row['bytes'])
            with self.assertRaisesRegex(RuntimeError,'Frozen runtime bytes changed|Runtime leaf SHA/size mismatch'):
                h.HOST.validate_spec(spec,self.base)


class ReuseTests(unittest.TestCase):
    def test_owner_worker_and_math_functions_reused_without_modifying_old_globals(self):
        old_g=h.original_rx.g;old_device=h.original_rx.DeviceAdapter;old_schema=h.original_rx.SCHEMA
        before={n:h.g.sha(Path(h.__file__).with_name(n)) for n in h.SOURCE_SHA}
        e=h.engine()
        for name in ('run','worker','exercise','actual_bits','wait_prelaunch','adapt_backend','inherited_spec'):
            self.assertIs(getattr(e,name).__code__,getattr(h.original_rx,name).__code__)
        self.assertIs(e.g,h.HOST);self.assertEqual(e.CAPS,dict(model_load=1,encoder=0,source_tx=0,
            source_rx=1,var_render=0,prior_scale=4,decoder_forward=0))
        self.assertIs(h.original_rx.g,old_g);self.assertIs(h.original_rx.DeviceAdapter,old_device)
        self.assertEqual(h.original_rx.SCHEMA,old_schema)
        self.assertEqual(before,{n:h.g.sha(Path(h.__file__).with_name(n)) for n in h.SOURCE_SHA})
        self.assertEqual(e.SCHEMA,h.SCHEMA);self.assertEqual(e.PASS,h.PASS)

    def test_modified_dependency_cannot_create_engine(self):
        with mock.patch.object(h.g,'sha',return_value='0'*64),self.assertRaisesRegex(RuntimeError,'Original implementation changed'):
            h.engine()

    def test_single_rx_still_observes_four_independent_cdfs_before_comparison_tokens(self):
        e=h.engine();events=[];tokens=np.arange(30,dtype=np.int64);bits=np.arange(311,dtype=np.uint8)%2
        hashes={i:hashlib.sha256(str(i).encode()).hexdigest() for i in range(4)}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);ledger=h.g.Ledger(out/'calls',lambda:None,e.CAPS)
            class Backend:
                def __init__(self):self.traces=[]
                def source_rx(self,actual,m):
                    self.assertions=(np.array_equal(actual,bits),m==4,self.tx_cdfs==hashes)
                    events.append('RX')
                    for k in range(4):
                        ledger.call('prior_scale',lambda:None)
                        self.traces.append(dict(source=0,role='RX',m=4,scale=k,cdf_sha256=hashes[k]))
                    return dict(canonical=True,zero_extension_reads=30,received_tokens=tokens)
            backend=ledger.call('model_load',Backend)
            def comparison():self.assertEqual(events,['RX']);events.append('comparison');return tokens.copy()
            self.assertTrue(e.exercise(bits,hashes,backend,ledger,comparison,out/'checks')['exact'])
            self.assertTrue(all(backend.assertions));self.assertEqual(events,['RX','comparison'])
            self.assertEqual(ledger.completed,e.CAPS);self.assertEqual(ledger.summary()['unresolved'],0)
            with self.assertRaises(RuntimeError):ledger.call('source_rx',lambda:None)

    def test_full_tool_closure_includes_old_rx_and_new_owner(self):
        e=h.engine()
        self.assertEqual(e.TOOLS,h.original_rx.TOOLS+('leo_single_rx_diagnostic_v2.py',))
        self.assertEqual(e.check_tools.__globals__['__file__'],h.__file__)
        bindings={name:'a'*64 for name in (*e.TOOLS,Path(h.__file__).name)}
        with mock.patch.object(h.g,'sha',return_value='a'*64):e.check_tools(bindings)
        del bindings['leo_single_rx_diagnostic_v2.py']
        with self.assertRaisesRegex(RuntimeError,'implementation closure'):e.check_tools(bindings)


class DeviceTests(unittest.TestCase):
    def receipt(self):
        return dict(status='READ_ONLY_ENUMERATION_NO_MODEL',model_calls=0,explicit_tensor_calls=0,
            CUDA_driver_devices=[dict(driver_ordinal_unmasked=0,CUDA_UUID='GPU-c61b6bf8-9457-fee7-7b51-761b157e0065',PCI='0000:63:00.0'),
                dict(driver_ordinal_unmasked=2,CUDA_UUID='GPU-ad85ce23-5025-3090-361b-27173fd1ac68',PCI='0000:6B:00.0')],
            NVML_csv='0, GPU-c38d7822-f973-c389-37c9-9ffa97610ac8, 00000000:63:00.0, NVIDIA H800, 29255, 24, 580.95.05\n'
                '2, GPU-70eb08db-3ea5-9394-c385-2125742754fd, 00000000:6B:00.0, NVIDIA H800, 29414, 57, 580.95.05\n')
    def make(self,index=0,document=None):
        with mock.patch.object(h.g,'inside',side_effect=Path),mock.patch.object(h.g,'checked_json',return_value=self.receipt() if document is None else document):
            return h.DeviceAdapter(dict(path='fake.json',sha256=h.DEVICE_SHA),index)

    def test_observed_devices_and_parent_device_remain_separate(self):
        old=(h.g.UUID,h.g.CUDA_UUID,h.g.PCI_BUS_ID)
        for index in (0,2):
            device=self.make(index);binding=device.device_identity_binding()
            self.assertEqual(binding['nvml_gpu_index'],index)
            self.assertEqual(binding['CUDA_VISIBLE_DEVICES'],str(index))
            device.validate_cuda_device_identity(binding['cuda_uuid'],binding['pci_bus_id'])
            self.assertIs(device.validate_spec.__self__,h.HOST)
        self.assertEqual((h.g.UUID,h.g.CUDA_UUID,h.g.PCI_BUS_ID),old)

    def test_wrong_receipt_scope_index_or_pci_stops_before_models(self):
        for index in (1,3,True,'0'):
            with self.subTest(index=index),self.assertRaises(RuntimeError):self.make(index)
        for change in [dict(status='PASS'),dict(model_calls=1),dict(explicit_tensor_calls=1)]:
            with self.assertRaises(RuntimeError):self.make(document=self.receipt()|change)
        document=self.receipt();document['CUDA_driver_devices'][0]['PCI']='0000:6b:00.0'
        with self.assertRaisesRegex(RuntimeError,'PCI mismatch'):self.make(document=document)
        with self.assertRaisesRegex(RuntimeError,'Fixed H800 observation'):
            h.DeviceAdapter(dict(path='fake.json',sha256='0'*64),0)


if __name__=='__main__':unittest.main(verbosity=2)
