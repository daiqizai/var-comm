"""CPU exact-command/TTL/identity tests; no inference or GPU initialization."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('m2_telemetry_tested',HERE/'telemetry.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


class Fixture:
    def __init__(self):
        self.now=0.;self.calls=[];self.allowed={123:'10',456:'20'}
        self.values={t.COMPUTE_QUERY:'123, python, 3000\n456, python, 7000\n',t.THERMAL_QUERY:'58, Not Active\n'}
        def original(*a,**k):
            self.calls.append((a,k))
            value=self.values.get(tuple(a[0]),'unmatched')
            if isinstance(value,BaseException):raise value
            return value
        self.original=original;self.records=[]
        self.cache=t.QueryCache(original,lambda:dict(self.allowed),clock=lambda:self.now,
            record=lambda s,n:self.records.append((s,n)))
    def call(self,key=t.COMPUTE_QUERY,**kwargs):return self.cache(list(key),**(kwargs or {'text':True}))


class CacheTests(unittest.TestCase):
    def setUp(self):self.f=Fixture()
    def test_hit_keeps_original_bytes_and_one_second_is_exclusive_limit(self):
        f=self.f;first=f.call();f.now=.999;self.assertEqual(first,f.call());self.assertEqual(len(f.calls),1)
        f.now=1.;f.call();self.assertEqual(len(f.calls),2);self.assertEqual(f.cache.stats()['cache_hits'],1)
    def test_ttl_is_from_query_start_including_query_latency(self):
        f=self.f
        def slow(*a,**k):f.now+=.8;return f.original(*a,**k)
        f.cache.original=slow;f.call();f.now=1.01;f.call()
        self.assertEqual(len(f.calls),2)
    def test_clock_rollback_never_reuses_stale_value(self):
        f=self.f;f.call();f.now=-.1;f.call();self.assertEqual(len(f.calls),2)
    def test_reused_peer_pid_changed_ticks_forces_fresh_query(self):
        f=self.f;f.call();f.allowed[456]='999';f.now=.2;f.call();self.assertEqual(len(f.calls),2)
    def test_exited_peer_and_unknown_gpu_invalidate_both_queries(self):
        f=self.f;f.call();f.call(t.THERMAL_QUERY);f.allowed.pop(456);f.now=.1;f.call()
        self.assertEqual(f.cache.entries,{})
        f.values[t.COMPUTE_QUERY]='999, unexpected, 1024\n';f.call();self.assertEqual(f.cache.entries,{})
    def test_hot_or_thermal_slowdown_never_cached_and_invalidates_all(self):
        for value in ('75, Not Active\n','86, Not Active\n','50, Active\n'):
            f=Fixture();f.call();f.values[t.THERMAL_QUERY]=value;f.call(t.THERMAL_QUERY)
            self.assertEqual(f.cache.entries,{})
    def test_malformed_compute_and_temperature_results_not_cached(self):
        for key,values in ((t.COMPUTE_QUERY,['123, python, NaN','123,python','123, python, -1','not available',b'123,python,3']),
                           (t.THERMAL_QUERY,['NaN, Not Active','50, unexpected','50','-1, Not Active'])):
            for value in values:
                f=Fixture();f.values[key]=value;f.call(key);self.assertEqual(f.cache.entries,{})
    def test_empty_compute_list_is_cacheable_during_startup(self):
        f=self.f;f.allowed={};f.values[t.COMPUTE_QUERY]='';f.call();f.now=.1;f.call();self.assertEqual(len(f.calls),1)
    def test_failed_raw_query_propagates_and_never_returns_old_cache(self):
        f=self.f;f.call();f.now=2.;f.values[t.COMPUTE_QUERY]=subprocess.CalledProcessError(1,'nvidia-smi')
        with self.assertRaises(subprocess.CalledProcessError):f.call()
        self.assertEqual(f.cache.entries,{})
    def test_admission_callback_failure_invalidates_all_and_propagates(self):
        f=self.f;f.call();f.call(t.THERMAL_QUERY)
        f.cache.approved_pids=lambda:(_ for _ in ()).throw(RuntimeError('source changed'))
        with self.assertRaises(RuntimeError):f.call()
        self.assertEqual(f.cache.entries,{})
    def test_only_exact_two_argument_vectors_and_exact_bool_keyword_are_wrapped(self):
        f=self.f
        for args,kw in ((list(t.COMPUTE_QUERY),{'text':1}),(list(t.COMPUTE_QUERY),{'text':True,'timeout':1}),
                        (['nvidia-smi','--id=1'],{'text':True}),(list(t.COMPUTE_QUERY),{'encoding':'utf8'})):
            f.cache(args,**kw)
        self.assertEqual(f.cache.stats()['fresh_calls'],0);self.assertEqual(len(f.calls),4)
    def test_formal_timing_disabled_is_always_fresh_even_hot_or_unknown(self):
        f=self.f;f.cache.enabled=False;f.values[t.THERMAL_QUERY]='90, Active';f.values[t.COMPUTE_QUERY]='999, python, 100'
        f.call();f.call();f.call(t.THERMAL_QUERY);f.call(t.THERMAL_QUERY)
        self.assertEqual(f.cache.stats()['fresh_calls'],4);self.assertEqual(f.cache.stats()['cache_hits'],0)
    def test_stats_recorded_first_then_at_most_every_thirty_seconds(self):
        f=self.f;f.call();f.now=15;f.call();f.now=30;f.call()
        self.assertEqual([s for s,n in f.records],['FIRST_QUERY_RECORDED','RUNNING_QUERY_STATS'])
        self.assertEqual(f.records[-1][1]['fresh_calls'],3)
    def test_context_restores_original_and_other_subprocess_arguments_untouched(self):
        f=self.f;original=subprocess.check_output
        with f.cache:
            subprocess.check_output(list(t.COMPUTE_QUERY),text=True)
            subprocess.check_output(['other'],text=True)
        self.assertIs(subprocess.check_output,original);self.assertEqual(f.calls[-1][0][0],['other'])
    def test_source_boundary_stop_is_still_checked_on_every_cache_hit(self):
        f=self.f;stop=[False];checks=[0]
        def boundary():
            checks[0]+=1
            if stop[0]:raise RuntimeError('dedicated original STOP boundary')
            f.call();f.call(t.THERMAL_QUERY)
        boundary();boundary();stop[0]=True
        with self.assertRaises(RuntimeError):boundary()
        self.assertEqual(checks[0],3);self.assertEqual(f.cache.stats()['cache_hits'],2)
    def test_bad_ttl_and_double_install_rejected(self):
        for value in (0,True,1.1,float('nan')):
            with self.assertRaises(ValueError):t.QueryCache(ttl=value)
        f=self.f
        with f.cache:
            with self.assertRaises(RuntimeError):t.QueryCache().__enter__()


class StartupTests(unittest.TestCase):
    def test_other_entry_inert_without_manifest(self):
        with mock.patch.object(sys,'argv',['other.py']),mock.patch.dict(os.environ,{},clear=True),mock.patch.object(os,'_exit') as exit:
            exec(compile((HERE/'sitecustomize.py').read_text(),str(HERE/'sitecustomize.py'),'exec'),{'__file__':str(HERE/'sitecustomize.py')})
            exit.assert_not_called()
    def test_scoped_startup_missing_manifest_explicitly_exits78(self):
        with mock.patch.object(sys,'argv',[str(t.ENTRY)]),mock.patch.dict(os.environ,{},clear=True),mock.patch.object(os,'_exit',side_effect=SystemExit(78)):
            with self.assertRaises(SystemExit) as result:
                exec(compile((HERE/'sitecustomize.py').read_text(),str(HERE/'sitecustomize.py'),'exec'),{'__file__':str(HERE/'sitecustomize.py')})
            self.assertEqual(result.exception.code,78)
    def test_startup_chains_pinned_bool_once_and_does_not_import_torch(self):
        source=(HERE/'telemetry.py').read_text()
        self.assertEqual(source.count("module('_m2_telemetry_pinned_boolean_startup'"),1)
        self.assertNotIn('import torch',source)
    def test_formal_timing_startup_does_not_wrap_raw_subprocess_at_all(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);bound=out/'source.py';bound.write_text('frozen')
            own={str(bound):t.sha(bound)};manifestpath=out/'manifest.json';manifestpath.write_text('{}')
            value=dict(runtime_source_bindings=own,original_source_bindings=own,
                bool_manifest_path=str(bound),bool_manifest_sha256=t.sha(bound),
                bool_sitecustomize_path=str(bound),bool_sitecustomize_sha256=t.sha(bound),
                qualification_path=str(bound),qualification_sha256=t.sha(bound))
            guard=mock.Mock();guard.identity.return_value=dict(pid=123,start_ticks='456')
            original=subprocess.check_output
            with mock.patch.object(t,'OUT',out),mock.patch.object(t,'validate',return_value=value),\
                    mock.patch.object(t,'module',side_effect=[mock.Mock(),guard]) as modules,\
                    mock.patch.object(t,'make_approved_pids',return_value=lambda:{}),\
                    mock.patch.object(t.atexit,'register') as registered:
                cache=t.install(manifestpath,['entry','--root','root','--admission','admission','--stage','timing'],out)
                self.assertIs(subprocess.check_output,original)
                self.assertFalse(hasattr(cache,'wrapped'))
                self.assertEqual(modules.call_args_list[0].args[0],'_m2_telemetry_pinned_boolean_startup')
                receipt=t.read(out/'executions/123_456.json')
                self.assertEqual(receipt['status'],'DISABLED_FOR_FORMAL_TIMING')
                self.assertEqual(receipt['stats']['fresh_calls'],0)
                registered.call_args.args[0]()
                self.assertIs(subprocess.check_output,original)
                self.assertEqual(t.read(out/'executions/123_456.json')['stats']['cache_hits'],0)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.out=Path(self.directory.name);self.frozen=self.out/'benchmark.py';self.frozen.write_text('fixed real fixture')
        self.own={str(self.frozen):t.sha(self.frozen)};self.path=self.out/'telemetry_qualification.json'
        self.value=dict(status='QUALIFIED',selected_implementation='cached',strict_equality_passed=True,speedup=1.2,
            source_bindings=self.own,inputs=self.own,engineering_fixture=True,real_model_weights=True,scientific_result=False,
            development_read=False,training_updates=0,policy_selection_updates=0,original_scientific_outputs_written=False)
    def write(self):
        payload={k:v for k,v in self.value.items() if k!='payload_sha256'}
        self.value['payload_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.path.write_text(json.dumps(self.value))
        return t.sha(self.path)
    def test_complete_real_qualification_is_eligible(self):
        digest=self.write()
        with mock.patch.object(t,'OUT',self.out):self.assertEqual(t.qualification(self.path,digest,self.own),self.value)
    def test_fail_fallback_slow_or_nonexact_never_deploy(self):
        for key,bad in (('status','USE_ORIGINAL'),('selected_implementation','original'),('speedup',1.099),
                        ('strict_equality_passed',False),('development_read',True),('real_model_weights',False),
                        ('training_updates',1),('source_bindings',{'other':'0'*64})):
            with self.subTest(key=key):
                old=self.value[key];self.value[key]=bad;digest=self.write()
                with mock.patch.object(t,'OUT',self.out),self.assertRaises(RuntimeError):t.qualification(self.path,digest,self.own)
                self.value[key]=old
    def test_qualification_hash_payload_and_bound_source_changes_rejected(self):
        digest=self.write()
        with mock.patch.object(t,'OUT',self.out):
            with self.assertRaises(RuntimeError):t.qualification(self.path,'0'*64,self.own)
            self.value['payload_sha256']='0'*64;self.path.write_text(json.dumps(self.value))
            with self.assertRaises(RuntimeError):t.qualification(self.path,t.sha(self.path),self.own)
            digest=self.write();self.frozen.write_text('hotedited')
            with self.assertRaises(RuntimeError):t.qualification(self.path,digest,self.own)


if __name__=='__main__':unittest.main()
