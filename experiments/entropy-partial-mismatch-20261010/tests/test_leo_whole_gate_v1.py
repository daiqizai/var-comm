"""Fake arrays, file pins and owned CPU subprocesses only; never torch/models."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import leo_whole_gate_v1 as gate
import leo_whole_math_v1 as math_adapter


def fixtures():
    sources=[]
    for i in range(2):
        tokens=(np.arange(680,dtype=np.int64)+i)%4096
        sources.append(dict(tokens=tokens,pixels=np.full((3,2,2),i,np.uint8),
            streams={m:np.asarray([i,m%2,1,0],np.uint8) for m in range(4,10)},
            images={m:np.full((3,2,2),(i+m)/16,np.float32) for m in (7,8,9)}))
    return sources


class Fake:
    def __init__(self,sources,ledger,fail=None):self.sources=sources;self.ledger=ledger;self.i=0;self.fail=fail
    def prior(self,n):
        for k in range(n):self.ledger.call('prior_scale',lambda:None,scale=k)
    def encode_pixels(self,p):
        v=self.sources[self.i]['tokens'].copy()
        if self.fail=='encoder':v[0]+=1
        return v
    def source_tx(self,t):
        self.prior(9);d={m:dict(bits=b.copy()) for m,b in self.sources[self.i]['streams'].items()}
        if self.fail=='tx':d[4]['bits'][0]^=1
        return d
    def source_rx(self,bits,m):
        # The fake RX callback receives only the bit vector and public m.
        self.prior(m);v=self.sources[self.i]['tokens'][:gate.OFFSETS[m]].copy()
        if self.fail=='rx':v[0]+=1
        return dict(received_tokens=v,canonical=True,zero_extension_reads=30)
    def render(self,m):
        self.prior(10);v=self.ledger.call('decoder_forward',lambda:self.sources[self.i]['images'][m].copy())
        if self.fail=='render':v[0,0,0]+=np.float32(.01)
        return v
    def end_source(self):self.i+=1


class Tests(unittest.TestCase):
    def test_complete_exact_scope_and_nested_budget(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);ledger=gate.Ledger(out/'calls',lambda:None)
            ledger.call('model_load',lambda:None);sources=fixtures();backend=Fake(sources,ledger)
            rows=gate.exercise(sources,backend,ledger,out/'checks')
            self.assertEqual(len(rows),32);self.assertEqual(ledger.completed,gate.CAPS)
            self.assertEqual(ledger.summary()['unresolved'],0);self.assertEqual(backend.i,2)
            self.assertTrue(all(x['exact'] for x in rows))

    def test_first_mismatch_each_output_stops_and_retains_difference(self):
        for kind in ('encoder','tx','rx','render'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                out=Path(td);ledger=gate.Ledger(out/'calls',lambda:None);sources=fixtures();backend=Fake(sources,ledger,kind)
                with self.assertRaisesRegex(RuntimeError,'Exact witness mismatch'):
                    gate.exercise(sources,backend,ledger,out/'checks')
                self.assertEqual(backend.i,0);self.assertEqual(ledger.completed['encoder'],1)
                diffs=list((out/'checks').glob('*.npz'));self.assertEqual(len(diffs),1)
                with np.load(diffs[0],allow_pickle=False) as z:self.assertFalse(np.array_equal(z['actual'],z['expected']))

    def test_budget_exhaustion_no_extra_operation(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=gate.Ledger(Path(td)/'calls',lambda:None,{'one':1});called=[]
            ledger.call('one',lambda:called.append(1))
            with self.assertRaises(RuntimeError):ledger.call('one',lambda:called.append(2))
            self.assertEqual(called,[1]);self.assertEqual(ledger.reserved['one'],1)

    def test_failed_call_reservation_is_unresolved_and_no_resume(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'calls';ledger=gate.Ledger(p,lambda:None,{'one':1})
            def fail():raise MemoryError('fake OOM')
            with self.assertRaises(MemoryError):ledger.call('one',fail)
            self.assertEqual(ledger.summary()['unresolved'],1)
            with self.assertRaises(FileExistsError):gate.Ledger(p,lambda:None)
            self.assertEqual(len(list(p.glob('*.reserved.json'))),1)

    def test_resolver_identity_pins_and_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).absolute();p=base/'bytes';p.write_bytes(b'abc')
            row=dict(original_path='/original/a',path=str(p),sha256=hashlib.sha256(b'abc').hexdigest(),bytes=3)
            r=gate.Resolver([row],base);self.assertEqual(r.path('/original/a'),p)
            with self.assertRaises(RuntimeError):r.path('/original/a','0'*64)
            p.write_bytes(b'xyz')
            with self.assertRaises(RuntimeError):r.reverify()

    def test_resolver_duplicate_traversal_escape_and_missing_ref_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).absolute();p=base/'a';p.write_bytes(b'a')
            row=dict(original_path='/old/a',path=str(p),sha256=gate.sha(p),bytes=1)
            with self.assertRaises(RuntimeError):gate.Resolver([row,row],base)
            for key,value in [('original_path','/old/../a'),('path',str(base.parent/'outside'))]:
                with self.subTest(key=key),self.assertRaises(RuntimeError):gate.Resolver([row|{key:value}],base)
            with self.assertRaises(RuntimeError):gate.Resolver([row],base).path('/old/missing')

    def test_json_hash_is_over_actual_parse_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.json';p.write_bytes(b'{"v":1}')
            self.assertEqual(gate.checked_json(p,gate.sha(p)),{'v':1});s=gate.sha(p)
            p.write_bytes(b'{"v":2}')
            with self.assertRaises(RuntimeError):gate.checked_json(p,s)

    def test_dtype_and_signed_zero_not_silently_equated(self):
        for a,b in [(np.array([0],np.int32),np.array([0],np.int64)),
                    (np.array([-0.],np.float32),np.array([0.],np.float32))]:
            with tempfile.TemporaryDirectory() as td,self.assertRaises(RuntimeError):gate.exact(a,b,'exact',td)

    def test_definition_extraction_never_executes_module_loader(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'module.py';p.write_text("raise RuntimeError('module body must not run')\ndef math(x):\n return x+1\n")
            ns={};math_adapter.definitions(p,{'math'},ns);self.assertEqual(ns['math'](3),4)
            with self.assertRaises(RuntimeError):math_adapter.definitions(p,{'absent'},{})

    def test_actual_loaded_library_observation_pins_and_driver_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).absolute();maps=base/'maps'
            library=base/'liboriginal.so';library.write_bytes(b'original bytes')
            # Linux maps paths are absolute POSIX identities; use a mocked reader
            # with real temporary-library byte hashing on this platform.
            fake_path='/lib/liboriginal.so';descriptor=dict(path=fake_path,sha256=gate.sha(library),bytes=library.stat().st_size)
            environment=SimpleNamespace(rows={fake_path:descriptor},host_native=[])
            maps.write_text('abc-def r-xp 0 00:00 1 '+fake_path+'\n')
            original=Path
            def mapped_path(value):
                return library if str(value)==fake_path else original(value)
            with mock.patch.object(math_adapter,'Path',side_effect=mapped_path):
                start=math_adapter.native_linkage(environment,gate,maps)
                self.assertEqual(start['loaded_libraries'][0]['origin'],'original_runtime_file_projected')
                self.assertTrue(math_adapter.native_linkage(environment,gate,maps,start)['previous_observed_files_reverified'])
                library.write_bytes(b'changed bytes')
                with self.assertRaises(RuntimeError):math_adapter.native_linkage(environment,gate,maps,start)

    def test_resources_gpu5_exact_margin_and_not_exclusivity(self):
        shared=mock.Mock();now=0
        shared.collect_snapshot.return_value={'gpu':{'index':5,'free_MiB':20480},'shared':True}
        shared.check_resources.return_value={'cpu_affinity':[2,3]}
        self.assertEqual(gate.resource_snapshot(shared)[1]['cpu_affinity'],[2,3])
        for index,free in [(4,20480),(5,20479)]:
            shared.collect_snapshot.return_value={'gpu':{'index':index,'free_MiB':free}}
            with self.assertRaises(RuntimeError):gate.resource_snapshot(shared)
        shared.collect_snapshot.return_value={'gpu':{'index':5,'free_MiB':30000,'utilization_percent':51}}
        with self.assertRaises(RuntimeError) as error:gate.resource_snapshot(shared,prelaunch=True)
        self.assertEqual(error.exception.resource_snapshot['gpu']['utilization_percent'],51)
        self.assertTrue(gate.resource_snapshot(shared,prelaunch=False))

    def test_original_thread_and_call_contract_stays_fixed(self):
        self.assertEqual(gate.EXPECTED_RUNTIME['threads'],6);self.assertEqual(gate.EXPECTED_RUNTIME['interop_threads'],2)
        self.assertEqual(gate.CAPS['prior_scale'],156);self.assertEqual(gate.CAPS['decoder_forward'],6)
        self.assertNotIn('direct',gate.CAPS);self.assertNotIn('quality',gate.CAPS)

    def test_worker_requires_exact_live_parent_and_launch_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run';out.mkdir();digest='b'*64
            gate.write(out/'intent.json',dict(owner_pid=123,request_sha256=digest))
            argv=['/private/python','-B','-u',str(Path(gate.__file__).absolute()),'_worker','--request',
                  str(out.parent/'request.json'),'--request-sha256',digest,'--owner-pid','123']
            launch=dict(pid=456,owner_pid=123,request_sha256=digest,argv=argv)
            gate.write(out/'child_started.json',launch)
            with mock.patch.object(gate.os,'getppid',return_value=123),mock.patch.object(gate.os,'getpid',return_value=456):
                self.assertEqual(gate.worker_identity(out,digest,123)['pid'],456)
                with self.assertRaises(RuntimeError):gate.worker_identity(out,'c'*64,123)
                with self.assertRaises(RuntimeError):gate.worker_identity(out,digest,999)
            with mock.patch.object(gate.os,'getppid',return_value=123),mock.patch.object(gate.os,'getpid',return_value=457):
                with self.assertRaises(RuntimeError):gate.worker_identity(out,digest,123)

    def test_runtime_alias_copies_and_generated_config_remain_separate(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).absolute();one=base/'old-copy';two=base/'venv-python'
            one.write_bytes(b'original interpreter');two.write_bytes(one.read_bytes());digest=gate.sha(one)
            rows=[dict(original_path='/usr/bin/python3.10',path=str(p),sha256=digest,bytes=p.stat().st_size) for p in (one,two)]
            generated=[]
            for role in ('UM','LDPC'):
                root=base/role;root.mkdir();packages=root/'packages';packages.mkdir()
                for name,content in [('pyvenv.cfg',f'home = {base}\ninclude-system-site-packages = false\nversion = 3.10.12\n'),
                                     ('frozen_package_roots.pth',str(packages)+'\n')]:
                    p=root/name;p.write_bytes(content.encode());generated.append(dict(path=str(p),sha256=gate.sha(p),bytes=p.stat().st_size,content=content,generated_new_bytes=True))
            env=dict(files=rows,projection_generated_files=generated,host_native_library_facts=[])
            r=gate.RuntimeFiles(env,base);r.reverify();self.assertEqual(r.path_for_new(two),two)
            with self.assertRaises(RuntimeError):gate.RuntimeFiles(env|{'files':rows+[rows[0]]},base)
            changed=copy.deepcopy(env);changed['files'][1]['sha256']='0'*64
            with self.assertRaises(RuntimeError):gate.RuntimeFiles(changed,base)
            Path(generated[0]['path']).write_text('changed')
            with self.assertRaises(RuntimeError):r.reverify()

    def test_active_runtime_exclusion_is_exact_unused_ldpc_tree_only(self):
        prefix=gate.OLD+'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment'
        rows=[dict(original_path=p) for p in [prefix+'/lib/unused.so',prefix+'_other/must_keep.so',
            '/usr/lib/python3.11/stdlib.py',gate.OLD+'outputs/UNIFIED-METRICS-20261002/environment/lib/active.so']]
        active,excluded=gate.active_um_files(rows)
        self.assertEqual(excluded,[rows[0]]);self.assertEqual(active,rows[1:])
        self.assertEqual(gate.active_um_files(rows[1:]),(rows[1:],[]))

    def test_explicit_candidate_exact_seven_and_provenance_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).absolute();(root/'models').mkdir();authors={};files=[];maps=[]
            names=['__init__','basic_vae','basic_var','helpers','quant','var','vqvae']
            for rel in ['dist.py','LICENSE']+['models/'+n+'.py' for n in names]:
                path=root/rel;path.write_bytes(('synthetic '+rel).encode());s=gate.sha(path)
                row=dict(path=rel,bytes=path.stat().st_size,sha256=s,
                    url=f'https://raw.githubusercontent.com/FoundationVision/VAR/{gate.UPSTREAM_COMMIT}/{rel}')
                if rel.startswith('models/'):
                    old=gate.AUTHOR+path.name;authors[old]=s
                    row.update(original_sha256=s,exact_original_match=True)
                    maps.append(dict(original_path=old,path=str(path),bytes=path.stat().st_size,sha256=s))
                files.append(row)
            audit=root/'audit.json';commit=root/'commit.json'
            gate.write(audit,dict(status='UPSTREAM_REFERENCE_ONLY_NOT_ORIGINAL_DIST_PROVENANCE',commit=gate.UPSTREAM_COMMIT,
                old_dist_sha_unavailable=True,files=files))
            gate.write(commit,dict(sha=gate.UPSTREAM_COMMIT))
            candidate=dict(root=str(root),audit=dict(path=str(audit),sha256=gate.sha(audit)),
                commit_record=dict(path=str(commit),sha256=gate.sha(commit)),old_dist_sha_unavailable=True)
            with mock.patch.object(gate,'UPSTREAM_AUDIT_SHA',gate.sha(audit)),mock.patch.object(gate,'UPSTREAM_COMMIT_JSON_SHA',gate.sha(commit)),mock.patch.object(gate,'UPSTREAM_DIST_SHA',gate.sha(root/'dist.py')):
                resolver=gate.Resolver(maps,root)
                result=gate.native_candidate(candidate,authors,resolver,root)
                self.assertTrue(result['old_dist_sha_unavailable'])
                with self.assertRaises(RuntimeError):gate.native_candidate(candidate|{'old_dist_sha_unavailable':False},authors,resolver,root)
                (root/'dist.py').write_bytes(b'changed candidate')
                with self.assertRaises(RuntimeError):gate.native_candidate(candidate,authors,resolver,root)

    def test_actual_owned_cpu_child_wait_and_timeout(self):
        shared=gate.helper()
        child=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        result=shared.wait_owned(child,5);self.assertTrue(result['actual_child_waited']);self.assertEqual(result['child_exit_code'],7);self.assertFalse(result['success'])
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        result=shared.wait_owned(child,.05);self.assertIsNotNone(child.poll());self.assertFalse(result['success'])

    def test_symlink_input_rejected_if_platform_allows(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).absolute();p=base/'file';p.write_bytes(b'x');q=base/'link'
            try:q.symlink_to(p)
            except OSError as e:self.skipTest('Platform symlink privilege unavailable: '+str(e))
            with self.assertRaises(RuntimeError):gate.inside(q,base)


if __name__=='__main__':unittest.main(verbosity=2)
