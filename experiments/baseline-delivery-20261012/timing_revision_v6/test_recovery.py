"""No-model CPU tests for explicit failed-slot recovery and closure capture."""
import copy
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path[:0]=[str(HERE),str(BASE/'timing_revision_v5')]
import recovery_adapter as adapter


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


contract=load(BASE/'timing_contract.py','_v6_contract_definitions_only')


def fixtures():
    plan=contract.fixed_cases();records={i:dict(tokens_sha256='frozen-token-'+str(i)) for i in contract.FIXED16}
    rows=[]
    for c in plan[:576]:
        key=c['method']+':'+str(c['source_index'])+':'+str(c['snr_db'])
        identity=dict(fresh_visual_encoder=True,cached_tokens_or_streams=False,batch_size=1,
            fresh_tokens_sha256=records[c['source_index']]['tokens_sha256'],
            wave_sha256='wave-'+key,observation_sha256='obs-'+key,reconstruction_sha256='recon-'+key)
        timing=dict(tx_seconds=.01,rx_seconds=.02,channel_seconds=.001,outer_seconds=.04,
            raw_wall_tx_seconds=.01,raw_wall_rx_seconds=.02,audit_excluded_tx_seconds=0.,audit_excluded_rx_seconds=0.)
        rows.append(dict(case=copy.deepcopy(c),state='COMPLETE',actual_packet_decodes=2,identity=identity,timings=timing))
    rows.append(dict(case=copy.deepcopy(plan[576]),state='RESERVED',actual_packet_decodes=0))
    prior=dict(case_cap=1728,measured_case_cap=864,packet_cap=2304,packet_decodes=1152,cases=rows)
    return plan,records,prior


class Recovery(unittest.TestCase):
    def test_actual_build_compiles_three_exact_changes_with_live_globals(self):
        runtime=load(BASE/'timing_runtime.py','_v6_real_runtime_definitions_only')
        adapted,proof=adapter.adapt_build(runtime)
        self.assertIs(adapted.__globals__,vars(runtime));self.assertTrue(proof['inverse_AST_equal'])
        runtime.visual_backend=object()
        self.assertIs(adapted.__globals__['visual_backend'],runtime.visual_backend)
        self.assertEqual(len(proof['substitutions']),3)
        # Actual constructor remains unexecuted.

    def test_late_binding_failure_and_class_default_capture(self):
        class Provider:
            def __init__(self,native,config):self.native,self.config=native,config
        native=object();codec=SimpleNamespace(entropy=SimpleNamespace(probability_cdf=object()))
        ns={'IndependentProvider':Provider}
        bad=lambda:ns['IndependentProvider'](native,SimpleNamespace(cdf_function=codec.entropy.probability_cdf))
        good=lambda _provider=ns['IndependentProvider']:_provider(native,SimpleNamespace(cdf_function=codec.entropy.probability_cdf))
        ns={'BudgetContinuous':object()}
        with self.assertRaises(KeyError):bad()
        instance=good();self.assertIsInstance(instance,Provider)
        self.assertIs(instance.native,native);self.assertIs(instance.config.cdf_function,codec.entropy.probability_cdf)
        self.assertIs(good.__defaults__[0],Provider)

    def test_actual_nested_alias_shared_visual_audit_order_with_provider_invocation(self):
        prior_test=load(BASE/'timing_revision_v5/test_wrapper_order.py','_v5_order_for_v6')
        case=prior_test.WrapperOrder();original_fixture=case.fixture
        def fixture(base):
            runtime,events,pairs,mods,strict=original_fixture(base)
            source=prior_test.RUNTIME_SOURCE
            old="    events.append('base_build_completed')"
            replacement="""    native=object()
    visual_request={'static':'frozen-static-binding'}
    codec=ep_source_codec.PartialSourceCodec(ROOT)
    ns={'IndependentProvider':lambda n,c: SimpleNamespace(native=n,config=c)}
    codec.provider_factory=lambda:ns['IndependentProvider'](native,SimpleNamespace(cdf_function=codec.entropy.probability_cdf))
    ns={'BudgetContinuous':object()}
    assert codec.provider_factory().native is native
    events.append('provider_default_capture_effective')
    events.append('base_build_completed')"""
            self.assertEqual(source.count(old),1)
            Path(runtime.__file__).write_text(source.replace(old,replacement))
            updated=prior_test.load(Path(runtime.__file__),'_v6_synthetic_provider_runtime')
            updated.events,updated.model,updated.torch_stub=events,runtime.model,runtime.torch_stub
            updated.ROOT=base
            updated.ep_source_codec=SimpleNamespace(PartialSourceCodec=lambda root:SimpleNamespace(entropy=SimpleNamespace(probability_cdf=object())))
            def bind_static(codec,binding):
                self.assertEqual(binding,'frozen-static-binding');codec.static_cdf='frozen-static-table'
                events.append('static_binding_without_new_fit')
            mods['h800_ep48_link_core_v1']=SimpleNamespace(bind_static=bind_static)
            updated.ES=base;mods['timing_runtime']=updated
            return updated,events,pairs,mods,strict
        case.fixture=fixture
        resource,writes=case.run_order(adapter.adapt_build)
        self.assertTrue(resource['shared_GPU']);self.assertIn('user-authorized shared-H800',resource['scheduling_amendment'])
        proof=next(v for n,v in writes if n=='actual_constructor_compatibility.json')
        self.assertTrue(proof['live_globals_preserved'])

    def test_original_static_table_validation_no_encode_or_decode(self):
        import numpy as np
        root=BASE.parents[1]
        link=load(root/'experiments/entropy-partial-mismatch-20261010/scripts/h800_ep48_link_core_v1.py','_v6_static_binding_only')
        core=load(root/'experiments/wcl-evidence-closure-20261009/scripts/t1_entropy_core.py','_v6_static_cdf_validation_only')
        entropy,_,_=core.load_integer_codec(root)
        reg=json.loads((root/'results/baseline_strength_20261011/timing/six_method_v5/registration.json').read_text())
        request=json.loads(Path(reg['visual_request']['path']).read_text())
        codec=SimpleNamespace(entropy=entropy,static_cdf=None,static_identity=None)
        link.bind_static(codec,request['static'])
        self.assertEqual(codec.static_cdf.shape,(10,4097))
        self.assertEqual(codec.static_identity,link.STATIC_SHA)
        self.assertTrue(np.all(codec.static_cdf[:,0]==0))
        # Loads and validates the existing frozen counts/CDF only: no fitting,
        # entropy encoding/decoding, model, packet, or source image operation.

    def test_legal_received_static_family_is_dispatched_in_full_public_domains(self):
        import numpy as np
        from contextlib import nullcontext
        endpoints=load(BASE/'timing_endpoints.py','_v6_received_static_dispatch')
        for count,partial in ((144,False),(360,True)):
            native=SimpleNamespace(torch=SimpleNamespace(no_grad=nullcontext));seen=[]
            profile=dict(family='EC_STATIC_WHOLE',m=7,K=0)
            actual=dict(header=dict(header_ok=True),body=dict(crc_accepted=True,parser_accepted=True,payload=[1,0]),rx_profile=profile)
            runtime=SimpleNamespace(qualified=True,profiles={str(i):{} for i in range(count)},receive=lambda *a,**k:actual)
            codec=SimpleNamespace(native=native,decode=lambda *args:seen.append(args) or dict(received_tokens=np.arange(3)))
            endpoint=endpoints.EntropyEndpoint(native,codec,runtime,None,lambda *a:np.zeros((3,256,256),np.float32),{},partial,None,None,ValueError)
            meter=SimpleNamespace(call=lambda name,fn:fn())
            _,returned=endpoint.receive(np.zeros((1024,2)),4,0,meter,'inert_dispatch_only')
            self.assertEqual(seen[0][0],'EC_STATIC_WHOLE');self.assertEqual(seen[0][2:],(7,0))
            self.assertEqual(returned['status'],'ARITHMETIC_SOURCE_DECODED')

    def test_prior_prefix_exact_grid_and_complete_qualification(self):
        plan,records,prior=fixtures();before=copy.deepcopy(prior)
        rows=adapter.validate_prior(prior,plan,records)
        self.assertEqual(len(rows),576);self.assertEqual(prior,before)
        self.assertEqual(sum(r['case']['phase']=='measurement' for r in rows),288)

    def test_prior_failure_cannot_be_counted_as_a_complete_warmup(self):
        plan,records,prior=fixtures();prior['cases'][-1]['state']='COMPLETE'
        with self.assertRaises(AssertionError):adapter.validate_prior(prior,plan,records)

    def test_changed_previous_identity_is_rejected(self):
        plan,records,prior=fixtures();prior['cases'][1]['identity']['wave_sha256']='changed'
        with self.assertRaises(AssertionError):adapter.validate_prior(prior,plan,records)

    def test_resume_never_enters_first576_endpoints_and_keeps_original_rows(self):
        plan,records,prior=fixtures();original=copy.deepcopy(prior)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prior_path=root/'v5_journal.json';prior_path.write_text(json.dumps(prior));prior_bytes=prior_path.read_bytes()
            class MemoryJournal(contract.CaseReservations):
                def flush(self):pass
            resumed=adapter.resumed_reservations(MemoryJournal,prior,records,dict(path=str(prior_path),sha256='synthetic-fixture-only'))
            source='''def worker():
    reservations=CaseReservations(journal_path)
    first={};results=[]
    for case in fixed_cases():
        endpoint_entries.append(case.copy())
        reservations.begin(case)
        if case['method'].startswith(('RAW_','EC_')):
            reservations.packet_call(lambda:None)
            reservations.packet_call(lambda:None)
        identity=dict(fresh_visual_encoder=True,cached_tokens_or_streams=False,batch_size=1,wave_sha256='w',observation_sha256='y',reconstruction_sha256='x')
        timing=dict(tx_seconds=.01,rx_seconds=.02,channel_seconds=.001,outer_seconds=.04)
        reservations.finish(identity,timing)
        results.append(dict(case=case,identity=identity,timings=timing))
    return results,reservations
'''
            path=root/'stub_owner.py';path.write_text(source);owner=load(path,'_v6_synthetic_resume_owner')
            owner.CaseReservations=resumed;owner.journal_path=root/'v6_journal.json';owner._resume_seed=adapter.seed_completed
            owner.fixed_cases=contract.fixed_cases;owner.endpoint_entries=[]
            worker,proof=adapter.adapt_worker(owner);self.assertIs(worker.__globals__,vars(owner))
            results,journal=worker()
            self.assertTrue(proof['inverse_AST_equal']);self.assertEqual(owner.endpoint_entries,plan[576:])
            self.assertEqual(len(owner.endpoint_entries),1152);self.assertEqual(owner.endpoint_entries[0],plan[576])
            self.assertEqual(len(results),1728);self.assertEqual(sum(r['case']['phase']=='measurement' for r in results),864)
            self.assertEqual(journal.state['packet_decodes'],2304)
            self.assertEqual(journal.state['status'],'COMPLETE_ALL_FIXED_TIMING_CASES')
            self.assertEqual(journal.state['cases'][:576],prior['cases'][:576]);self.assertEqual(prior,original)
            self.assertEqual(prior_path.read_bytes(),prior_bytes)
            self.assertEqual(journal.state['recovery']['failed_fresh_visual_encoder_attempts'],1)
            self.assertTrue(journal.state['recovery']['failed_slot_is_not_a_complete_warmup'])
            first_condition=[r for r in results if r['case']['method']=='EC_VAR_WHOLE' and r['case']['source_index']==0 and r['case']['snr_db']==4]
            self.assertEqual([r['case']['phase'] for r in first_condition],['warmup']*3+['measurement']*3)

    def test_resume_refuses_to_write_original_journal(self):
        plan,records,prior=fixtures()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'v5.json'
            klass=adapter.resumed_reservations(contract.CaseReservations,prior,records,dict(path=str(path),sha256='stub'))
            with self.assertRaises(AssertionError):klass(path)
            self.assertFalse(path.exists())

    def test_actual_cumulative_wall_budget_keeps_four_hour_total(self):
        source=(HERE/'timing_owner_v6.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='elapsed_budget')
        ns=dict(ROOT=BASE.parents[1],math=math,
            read=lambda p:json.loads(Path(p).read_text()),
            pin=lambda p:dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()))
        tree=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(tree)
        exec(compile(tree,'<elapsed_budget_definition_only>','exec'),ns)
        budget=ns['elapsed_budget']()
        self.assertEqual(budget['original_seconds'],14400)
        self.assertEqual(budget['prior_charged_seconds']+budget['remaining_seconds'],14400)
        self.assertEqual([v['version'] for v in budget['attempts']],[2,3,5])
        self.assertEqual(budget['v4_actual_execution_seconds'],0)
        self.assertTrue(budget['watchdog_grace_is_not_additional_scientific_budget'])

    def test_resume_same_device_and_math_cannot_silently_change(self):
        runtime=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/six_method_timing_v5/numerical_runtime.json')
        prior=json.loads(runtime.read_text());current=copy.deepcopy(prior)
        self.assertTrue(adapter.same_numerical_runtime(prior,current)['same_actual_H800_and_math'])
        current['threads']=2
        with self.assertRaises(AssertionError):adapter.same_numerical_runtime(prior,current)


if __name__=='__main__':unittest.main(verbosity=2)
