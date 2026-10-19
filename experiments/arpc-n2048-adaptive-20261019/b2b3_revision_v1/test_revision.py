"""Synthetic amendment/assembly/accounting tests. No real packet/model calls."""
from pathlib import Path
import ast
import hashlib
import importlib.abc
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))


class ModelGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','tensorflow','transformers','timm','sionna','PIL','numpy'}:
            raise AssertionError('Synthetic tests prohibit model/PHY runtime imports: '+fullname)
        return None


sys.meta_path.insert(0, ModelGuard())
import plan
import common
import transport
import prepare
import pair_protocol as protocol

# Local tests read the exact retained original pure-byte protocol. This fallback
# is test-only and never changes the installed production module or file.
if not protocol.OLD_PATH.exists():
    prior = HERE.parents[2]/'.research/arpc_n2048_execution_20261019/pair_protocol.py'
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == '59349ec325b7a285b2cab33f3a8a370ae3b30988c666394a7f2565f158c60fd0'
    protocol.OLD_PATH = prior


def fixture(phase):
    sources, catalogue = [], []
    for i in range(plan.SOURCE_COUNTS[phase]):
        sid = f'synthetic-{i:04d}'
        source = dict(path='/synthetic/source', bytes=3, sha256='a'*64)
        sources.append(dict(source_id=sid, source_index=i, source=source, population=plan.POPULATIONS[phase]))
        for codec in (0, 1):
            for endpoint in (range(25) if codec == 0 else range(1, 8)):
                catalogue.append(dict(population=plan.POPULATIONS[phase], source_id=sid,
                    source_index=i, source=source, codec=codec, endpoint=endpoint,
                    bit_length=8*(20+endpoint*35), object_sha256='b'*64))
    return sources, catalogue


def candidates():
    return [dict(**c, calibration_mean_DINO=0.5, source_mean_count=1000, noise_per_source=3)
            for c in plan.cell_rows()]


def selection():
    rows = candidates()
    return dict(schema='N2048_B2_DINO_FOUR_POLICIES_V1', status='FOUR_POLICIES_FROZEN',
        metric='dinov2_vitl14_cosine', candidate_means=rows, rows=plan.select_four(rows),
        no_additional_candidates=True, calibration_bootstrap_calls=0)


def public():
    return plan.public_job('b2_00000', 4, [2000000, 2000001],
        dict(path='/synthetic/observation.npy', bytes=123, sha256='a'*64))


class Wire:
    class SourceParseError(ValueError):
        pass
    @staticmethod
    def unpack(data, endpoint):
        return (data, endpoint)


def received(data=b'\x12\x34\x56\x78', codec=0, endpoint=7, layout='q2_r1-2'):
    fs = protocol.fragment(data, len(data)*8, codec, endpoint, 1000000, layout)
    return [dict(status='FRAGMENT_PARSED', profile=p, header=dict(header_ok=True),
        body=dict(status='FRAGMENT_PARSED', payload_hex=protocol.parse_body(protocol.pack_body(b,p),p).hex()))
        for b,p in fs]


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources, cls.catalogue = fixture('B2')
        cls.jobs, cls.assets, cls.fallback = plan.make_jobs('B2', plan.cell_rows(), cls.sources, cls.catalogue)

    def test_exact_nine_authorized_cells(self):
        self.assertEqual(len(plan.B2_CELLS),9)
        self.assertEqual(sum(c == 0 and s == 10 for c,s,_ in plan.B2_CELLS),5)
        self.assertEqual(len(set(plan.B2_CELLS)),9)

    def test_exact_b2_caps_and_denominators(self):
        self.assertEqual(len(self.jobs),27000)
        self.assertEqual(len(self.fallback),9000)
        self.assertEqual(2*len(self.jobs),54000)
        self.assertEqual(4*len(self.jobs),108000)
        self.assertEqual({j['noise_seed'] for j in self.jobs},{4101,4102,4103})

    def test_opaque_ids_unique_paid_counters(self):
        counters = [2*j['pair_id']+i for j in self.jobs for i in (0,1)]
        self.assertEqual(len(counters),len(set(counters)))
        self.assertTrue(all(j['source_id'] not in j['key'] for j in self.jobs))

    def test_b2_never_uses_19db_or_test_noise(self):
        self.assertEqual({j['snr_db'] for j in self.jobs},{4,10})
        self.assertFalse({6201,6202,6203} & {j['noise_seed'] for j in self.jobs})

    def test_fallback_changes_actual_endpoint_by_capacity(self):
        js=[j for j in self.jobs if j['source_id']=='synthetic-0000' and j['noise_seed']==4101 and j['codec']==0]
        self.assertGreater(len({j['actual_endpoint'] for j in js}),1)
        for j in js:
            a=self.assets[j['asset_key']]
            self.assertLessEqual((a['bit_length']+7)//8,plan.LAYOUTS[j['layout_id']]['object_bits']//8)

    def test_incomplete_endpoint_catalogue_rejected(self):
        with self.assertRaises(AssertionError):
            plan.make_jobs('B2',plan.cell_rows(),self.sources,self.catalogue[:-1])

    def test_duplicate_endpoint_catalogue_rejected(self):
        with self.assertRaises(AssertionError):
            plan.make_jobs('B2',plan.cell_rows(),self.sources,self.catalogue+[self.catalogue[0]])

    def test_population_mismatch_rejected(self):
        with self.assertRaises(AssertionError):
            plan.make_jobs('B2',plan.cell_rows(),self.sources[:-1],self.catalogue)

    def test_no_fit_keeps_all_six_policy_noise_events(self):
        c=[dict(x,bit_length=90000) if x['source_id']=='synthetic-0000' else x for x in self.catalogue]
        jobs,_,fallback=plan.make_jobs('B2',plan.cell_rows(),self.sources,c)
        r=[j for j in jobs if j['source_id']=='synthetic-0000']
        self.assertEqual(len(r),27)
        self.assertTrue(all(j['source_no_fit'] and j['asset_key'] is None for j in r))
        self.assertEqual(len(jobs),27000)

    def test_new_cells_cannot_silently_expand(self):
        with self.assertRaises(AssertionError):
            plan.make_jobs('B2',plan.cell_rows()+[plan.cell_rows()[0]],self.sources,self.catalogue)

    def test_original_protocol_overheads_unchanged(self):
        for g in plan.LAYOUTS.values():
            self.assertEqual(g['total_complex_symbols'],2048)
            self.assertEqual(g['nominal_symbol_energy'],2)
            self.assertEqual(g['fragment_bytes'],(g['k']-13-16-96)//8)
            self.assertEqual(g['header_symbols_per_fragment']+g['body_symbols_per_fragment'],1024)

    def test_frozen_b3_is_four_specific_winners(self):
        cells=plan.frozen_cells(selection())
        self.assertEqual(len(cells),4)
        self.assertEqual(cells[0]['policy_id'],'nest_t24_q2_r1-2')
        self.assertEqual(cells[1]['policy_id'],'nest_t24_q2_r2-3')

    def test_b3_caps_and_test_noise_once(self):
        sources,catalogue=fixture('B3')
        jobs,_,_=plan.make_jobs('B3',plan.frozen_cells(selection()),sources,catalogue)
        self.assertEqual(len(jobs),1200)
        self.assertEqual({j['noise_seed'] for j in jobs},{6201,6202,6203})
        self.assertFalse({j['pair_id'] for j in self.jobs}&{j['pair_id'] for j in jobs})

    def test_total_caps(self):
        self.assertEqual(sum(x['logical_objects'] for x in plan.CAPS.values()),28200)
        self.assertEqual(sum(x['subframes'] for x in plan.CAPS.values()),56400)
        self.assertEqual(sum(x['decoder_attempts'] for x in plan.CAPS.values()),112800)

    def test_selection_actual_DINO_precedes_resource_tie(self):
        rows=candidates();rows[6]['calibration_mean_DINO']=0.50000001
        self.assertEqual(plan.select_four(rows)[1]['policy_id'],'nest_t24_q4_r2-3')

    def test_selection_cannot_drop_candidate(self):
        with self.assertRaises(AssertionError):plan.select_four(candidates()[:-1])

    def test_selection_cannot_replace_by_quality_estimate(self):
        s=selection();s['metric']='historical_packet_proxy'
        with self.assertRaises(AssertionError):plan.frozen_cells(s)

    def test_selection_cannot_claim_different_winner(self):
        s=selection();s['rows'][1]['policy_id']='nest_t24_q4_r2-3'
        with self.assertRaises(AssertionError):plan.frozen_cells(s)

    def test_selection_nan_and_wrong_noise_count_rejected(self):
        for field,value in [('calibration_mean_DINO',float('nan')),('noise_per_source',3000)]:
            r=candidates();r[0][field]=value
            with self.assertRaises(AssertionError):plan.select_four(r)

    def test_public_no_tx_truth_or_source_id(self):
        for field in ('source_id','source_index','codec','endpoint','asset','source','policy_id','registration'):
            j=public();j[field]='forbidden'
            with self.assertRaises(AssertionError):plan.validate_public_job(j)

    def test_public_keys_cannot_escape_paths(self):
        j=public();j['key']='../../source_truth'
        with self.assertRaises(AssertionError):plan.validate_public_job(j)

    def test_public_counter_shape_and_type(self):
        for counters in ([1,3],[False,1],[-1,0],[1.0,2.0]):
            j=public();j['counters']=counters
            with self.assertRaises(AssertionError):plan.validate_public_job(j)

    def test_pair_assembly_uses_actual_profile_only(self):
        rs=received(codec=0,endpoint=7)
        result,data=transport.pair_result(public(),rs,'a'*64,Wire,protocol)
        self.assertEqual(result['status'],'PAIR_ASSEMBLED')
        self.assertEqual(result['actual_metadata']['endpoint'],7)
        self.assertEqual(data,b'\x12\x34\x56\x78')
        self.assertFalse(result['source_truth_used'])

    def test_wrong_legal_actual_object_not_oracle_corrected(self):
        rs=received(b'\x88\x77\x66\x55',codec=0,endpoint=3)
        result,data=transport.pair_result(public(),rs,'a'*64,Wire,protocol)
        self.assertEqual(result['status'],'PAIR_ASSEMBLED')
        self.assertEqual(result['actual_metadata']['endpoint'],3)
        self.assertEqual(data,b'\x88\x77\x66\x55')

    def test_second_fragment_rejection_keeps_pair_failure(self):
        rs=received();rs[1]=dict(status='BODY_REJECT',body=dict(status='BODY_REJECT'),profile=None,header={})
        r,data=transport.pair_result(public(),rs,'a'*64,Wire,protocol)
        self.assertEqual(r['status'],'PAIR_REJECT');self.assertIsNone(data)

    def test_header_failure_does_not_attempt_source_assembly(self):
        rs=[dict(status='HEADER_REJECT',body=None,profile=None,header={})]*2
        r,data=transport.pair_result(public(),rs,'a'*64,Wire,protocol)
        self.assertEqual(r['status'],'PAIR_REJECT');self.assertIsNone(data)

    def test_null_profiles_pay_two_full_bodies_and_reject(self):
        for codec in (0,1):
            p=protocol.profile(codec,None,'q2_r1-2',None)
            body=protocol.pack_body(b'',p)
            self.assertEqual(len(body),p['k'])
            self.assertEqual(protocol.parse_body(body,p),b'')
            rs=[dict(status='NULL_PROFILE',body={'payload_hex':''},profile=p,header={})]*2
            r,data=transport.pair_result(public(),rs,'a'*64,Wire,protocol)
            self.assertEqual(r['status'],'PAIR_REJECT');self.assertIsNone(data)

    def test_arpc_native_container_failure_is_rejection(self):
        class BadWire(Wire):
            @staticmethod
            def unpack(data,endpoint):raise Wire.SourceParseError('synthetic')
        r,data=transport.pair_result(public(),received(codec=1,endpoint=4),'a'*64,BadWire,protocol)
        self.assertEqual(r['status'],'PAIR_REJECT');self.assertIsNone(data)

    def test_two_fragment_profile_mismatch_rejected(self):
        rs=received();rs[1]['profile']=protocol.profile(0,8,'q2_r1-2',1)
        r,data=transport.pair_result(public(),rs,'a'*64,Wire,protocol)
        self.assertEqual(r['status'],'PAIR_REJECT');self.assertIsNone(data)

    def test_checkpoint_written_before_ledger_complete(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);ledger=common.Ledger(out/'calls.sqlite',dict(header=1,body=1))
            c=transport.RecordedLedger(ledger,out,['b2_00000_0'])
            self.assertEqual(c.call('header','b2_00000_0',lambda:{'header_ok':True}),{'header_ok':True})
            self.assertEqual(common.read(out/'b2_00000_0_header.json'),{'header_ok':True})
            self.assertEqual(ledger.summary()['unresolved'],0)
            ledger.db.close()

    def test_saved_call_never_repeated(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);ledger=common.Ledger(out/'calls.sqlite',dict(header=2,body=1))
            c=transport.RecordedLedger(ledger,out,['b2_00000_0']);calls=[]
            c.call('header','b2_00000_0',lambda:calls.append(1) or {})
            with self.assertRaises(AssertionError):c.call('header','b2_00000_0',lambda:calls.append(2))
            self.assertEqual(calls,[1]);ledger.db.close()

    def test_checkpoint_keys_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);ledger=common.Ledger(out/'calls.sqlite',dict(header=1,body=1))
            c=transport.RecordedLedger(ledger,out,['b2_00000_0'])
            with self.assertRaises(AssertionError):c.call('header','outside',lambda:{})
            self.assertEqual(ledger.summary()['counts'],[]);ledger.db.close()

    def test_scientific_failure_persists_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);ledger=common.Ledger(out/'calls.sqlite',dict(header=1,body=1))
            c=transport.RecordedLedger(ledger,out,['b2_00000_0'])
            def failure():raise RuntimeError('synthetic failure')
            with self.assertRaises(RuntimeError):c.call('header','b2_00000_0',failure)
            self.assertEqual(ledger.summary()['unresolved'],1)
            self.assertEqual(ledger.summary()['counts'][0]['status'],'FAILED');ledger.db.close()

    def test_production_rx_never_reads_registration_or_private_inputs(self):
        tree=ast.parse((HERE/'transport.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rx')
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='registration' for n in ast.walk(fn)))
        strings=[n.value for n in ast.walk(fn) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        self.assertFalse(any('private_jobs' in x or 'private_assets' in x or 'source_catalogue' in x for x in strings))

    def test_rx_allows_exact_old_protocol_directory(self):
        tree=ast.parse((HERE/'transport.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rx')
        self.assertIn('p.OLD_PATH.parent',ast.unparse(fn))

    def test_no_real_runtime_imported(self):
        self.assertFalse({'torch','tensorflow','transformers','sionna','numpy'}&set(sys.modules))


if __name__=='__main__':
    unittest.main(verbosity=2)
