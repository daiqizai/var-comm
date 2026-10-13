"""Synthetic source/guard tests; no Torch, GPU, scientific inputs or remote calls."""
import ast
import hashlib
import importlib
import importlib.machinery
import importlib.util
import sys
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import probe as p

SOURCE=Path(__file__).resolve().parents[3]/'experiments/layered-mismatch-raw-20261013/r1_execution_resource_v2/receiver_visual.py'

class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();self.out=self.base/'private_probe';self.out.mkdir()
        self.source=SOURCE.read_text(encoding='utf8');self.ns,self.proof=p.make_definitions(self.source,SOURCE,self.base,self.base/'runtime')
    def test_frozen_original_source_sha_and_ast_exact(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(),p.SOURCE_SHA)
        module,_=p.extracted_definitions(self.source);original=ast.parse(self.source)
        for name in ('SourceReadFirewall','numeric_flags'):
            a=next(n for n in module.body if n.name==name);b=next(n for n in original.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
    def test_changed_original_numeric_flags_refused(self):
        with self.assertRaises(RuntimeError):p.extracted_definitions(self.source.replace('torch.set_num_threads(6)','torch.set_num_threads(5)'))
    def test_private_cache_pins_tempdir_despite_preexisting_cached_value(self):
        before=tempfile.tempdir;self.addCleanup(setattr,tempfile,'tempdir',before)
        with mock.patch.dict(os.environ):
            tempfile.tempdir=str(self.base/'forbidden_old');bindings=p.private_cache(self.out)
            self.assertEqual(tempfile.gettempdir(),str(self.out/'runtime_cache/tmp'))
            self.assertEqual(set(bindings),{'TMPDIR','TEMP','TMP','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'})
            self.assertTrue(all(self.out in Path(v).parents and Path(v).is_dir() for v in bindings.values()))
            with self.assertRaises(FileExistsError):p.private_cache(self.out)
    def test_original_firewall_allows_only_private_cache_among_research_paths(self):
        fw=self.ns['SourceReadFirewall']([],self.out,[],{})
        for name in ('runtime_cache/tmp/probe','runtime_cache/inductor/probe','runtime_cache/triton/probe'):
            self.assertTrue(fw.permitted(self.out/name,True))
        for path in (self.base/'sources/image.npz',self.base/'tokens/tokens.npy',self.base/'phy/TX.npz',self.base/'r1_execution_v1/cache/probe'):
            with self.subTest(path=path):
                self.assertFalse(fw.permitted(path,False));self.assertFalse(fw.permitted(path,True))
                with self.assertRaisesRegex(RuntimeError,'unregistered data access'):fw.hook('open',(str(path),'r',0))
        self.assertEqual(len(fw.blocked),4)  # Deliberate negatives are isolated from the real probe.
    def test_no_weight_whitelist_in_probe(self):
        source=Path(p.__file__).read_text(encoding='utf8')
        self.assertIn("[p['path'] for p in software]",source)
        self.assertNotIn("bundle['weights'].values()",source)
        self.assertNotIn('torch.load(',source)
    def test_exact_numeric_prefix_with_fake_torch(self):
        calls=[];state={};backend=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),cudnn=types.SimpleNamespace(allow_tf32=True,benchmark=True,deterministic=False))
        t=types.SimpleNamespace(backends=backend)
        for name,key in [('set_num_threads','threads'),('set_num_interop_threads','interop_threads'),('use_deterministic_algorithms','deterministic'),('set_float32_matmul_precision','precision')]:
            setattr(t,name,lambda value,n=name,k=key:(calls.append((n,value)),state.update({k:value})))
        t.get_num_threads=lambda:state['threads'];t.get_num_interop_threads=lambda:state['interop_threads']
        t.are_deterministic_algorithms_enabled=lambda:state['deterministic'];t.get_float32_matmul_precision=lambda:state['precision']
        self.ns['apply_numeric_flags'](t);self.assertEqual(self.ns['numeric_flags'](t),p.FLAGS)
        self.assertEqual(calls,[('set_num_threads',6),('set_num_interop_threads',2),('use_deterministic_algorithms',True),('set_float32_matmul_precision','highest')])
    def test_tensor_model_weight_cuda_requests_are_rejected_and_counted(self):
        guard=p.NoScience()
        for module,name,key in [('torch','empty','tensor_requests'),('torch.nn.modules.module','__init__','model_constructor_requests'),('torch.serialization','load','weight_load_requests'),('torch.cuda','_lazy_init','cuda_init_requests')]:
            with self.assertRaisesRegex(RuntimeError,'Zero-science'):guard.inspect_call(module,name)
            self.assertEqual(guard.counts[key],1)
        guard.inspect_call('torch','use_deterministic_algorithms');self.assertEqual(sum(guard.counts.values()),4)
    def test_real_metadata_lookup_preserved_but_loader_execution_rejected(self):
        for module in ('tensorflow','sionna','lpips','models','var_comm'):
            with self.subTest(module=module):
                guard=p.NoScience();loader=mock.Mock();spec=importlib.machinery.ModuleSpec(module,loader,origin='/real/fixture.py')
                finder=mock.Mock();finder.find_spec.return_value=spec
                with mock.patch.object(sys,'meta_path',[guard,finder]),mock.patch.dict(sys.modules,clear=False):
                    sys.modules.pop(module,None)
                    observed=importlib.util.find_spec(module)
                    self.assertEqual(observed.origin,spec.origin);self.assertIsNot(observed,spec)
                    self.assertIs(spec.loader,loader);self.assertIs(observed.loader.original_loader,loader)
                    self.assertEqual(sum(guard.counts.values()),0)
                    self.assertEqual(guard.metadata_lookups[0]['found'],True)
                    with self.assertRaisesRegex(RuntimeError,'channel_import_requests'):importlib.import_module(module)
                    self.assertNotIn(module,sys.modules)
                    loader.create_module.assert_not_called();loader.exec_module.assert_not_called()
                    self.assertEqual(guard.counts['channel_import_requests'],1)
                    with self.assertRaisesRegex(RuntimeError,'channel_import_requests'):observed.loader.exec_module(types.SimpleNamespace())
                    self.assertEqual(guard.counts['channel_import_requests'],2)
    def test_absent_metadata_returns_real_none_without_science(self):
        guard=p.NoScience();first=mock.Mock();first.find_spec.return_value=None
        second=mock.Mock();second.find_spec.return_value=None
        with mock.patch.object(sys,'meta_path',[guard,first,second]):
            self.assertIsNone(guard.find_spec('tensorflow'));self.assertEqual(sum(guard.counts.values()),0)
            self.assertEqual(guard.metadata_lookups,[dict(fullname='tensorflow',found=False,origin=None,loader_type=None)])
        first.find_spec.assert_called_once_with('tensorflow',None,None);second.find_spec.assert_called_once_with('tensorflow',None,None)
    def test_namespace_spec_preserves_search_paths_and_refuses_creation(self):
        guard=p.NoScience();finder=mock.Mock();spec=importlib.machinery.ModuleSpec('models',None,is_package=True)
        spec.submodule_search_locations=['/actual/models'];finder.find_spec.return_value=spec
        with mock.patch.object(sys,'meta_path',[guard,finder]):observed=guard.find_spec('models')
        self.assertEqual(observed.submodule_search_locations,spec.submodule_search_locations)
        self.assertIsNone(spec.loader)
        with self.assertRaisesRegex(RuntimeError,'channel_import_requests'):observed.loader.create_module(observed)
    def test_metadata_failure_not_swallowed_or_converted_to_absence(self):
        guard=p.NoScience();finder=mock.Mock();finder.find_spec.side_effect=ValueError('real finder failure')
        with mock.patch.object(sys,'meta_path',[guard,finder]):
            with self.assertRaisesRegex(ValueError,'real finder failure'):guard.find_spec('tensorflow')
        self.assertEqual(guard.metadata_lookups,[])
        self.assertIsNone(guard.find_spec('torch._inductor'))
    def test_private_write_guard_no_external_tmp_or_network(self):
        audit=p.PrivateWrites(self.out)
        audit.hook('open',(str(self.out/'cache.txt'),'w',os.O_WRONLY))
        self.assertEqual(sum(audit.writes.values()),1)
        for event,args in [('open',(str(self.base/'unbound'),'w',os.O_WRONLY)),('os.mkdir',(str(self.base/'newdir'),)),('socket.connect',()),('subprocess.Popen',())]:
            with self.subTest(event=event),self.assertRaises(RuntimeError):audit.hook(event,args)
        self.assertEqual(len(audit.blocked),4)
    def test_no_symlink_creation_allowed(self):
        audit=p.PrivateWrites(self.out)
        with self.assertRaisesRegex(RuntimeError,'No new PFS'):audit.hook('os.symlink',(str(self.out/'a'),str(self.out/'b')))
    def test_original_runtime_schema_required(self):
        data=dict(schema='R1_MODEL_ONLY_BUNDLE_V1',no_source_or_TX_paths=True,numeric_flags=p.FLAGS,
            expected_runtime=dict(python='3.10.12',torch='2.11.0+cu128',numpy='2.2.6'),runtime_files=[dict(path='/fixture',sha256='0'*64,bytes=1)])
        self.assertEqual(p.metadata_bundle(data),data['runtime_files'])
        data['no_source_or_TX_paths']=False
        with self.assertRaises(RuntimeError):p.metadata_bundle(data)

if __name__=='__main__':unittest.main()
