"""Synthetic source-asset contract checks; no model, PHY, or scientific data."""
import argparse
import contextlib
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from t1_entropy_core import OFFSETS,read,write,sha
from t1_source_asset_schema import build_record,exact_received_cache,model_id,token_sha
import t1_merge_sources as merge


class ContractTests(unittest.TestCase):
    def fixture(self,base,short_needed=True):
        base.mkdir();(base/'sources').mkdir();(base/'source_checkpoints').mkdir()
        tokens=np.arange(680,dtype=np.int64)
        arrays={'tokens':tokens}
        for family in ('static','var'):
            for m in range(4 if family=='static' else 6,10):
                n=200 if m<=5 else (950 if short_needed else 900)+100*(m-6)
                arrays[f'{family}_m{m}_bits']=np.arange(n,dtype=np.uint8)%2
                arrays[f'{family}_m{m}_received_tokens']=tokens[:OFFSETS[m]].copy()
        archive=base/'sources'/'0000.npz';np.savez(archive,**arrays)
        record=build_record(source_index=0,source_id='synthetic-source',tokens=tokens,
            preprocessing_id='synthetic-preprocessing',source_assets_checkpoint={'path':'synthetic','sha256':'synthetic'},
            archive=archive,arrays=arrays,static_completion_sha='synthetic-fixed-cdf',
            origins={f:'SYNTHETIC_CONTRACT_ONLY' for f in ('EC_STATIC_WHOLE','EC_VAR_WHOLE')},
            upstream_evidence={f:{} for f in ('EC_STATIC_WHOLE','EC_VAR_WHOLE')})
        cp=base/'source_checkpoints'/'0000.json';write(cp,record)
        mp=base/'manifest.json';write(mp,dict(source_count=1,source_ids=['synthetic-source'],records=[dict(
            source_index=0,source_id='synthetic-source',checkpoint=str(cp),checkpoint_sha256=sha(cp))]))
        done=base/'completion.json';write(done,dict(status='T1_CALIBRATION_SOURCE_CPU_PREPARATION_COMPLETE',outputs={str(mp):sha(mp)}))
        return done,record,arrays

    def short_fixture(self,base,tokens):
        base.mkdir();arrays={}
        for m in (4,5):
            arrays[f'm{m}_bits']=np.arange(200,dtype=np.uint8)%2
            arrays[f'm{m}_received_tokens']=tokens[:OFFSETS[m]].copy()
        p=base/'0000.npz';np.savez(p,**arrays)
        mp=base/'fallback_manifest.json';write(mp,dict(status='T1_CALIBRATION_VAR_FALLBACK_COMPLETE',records=[dict(
            source_index=0,source_id='synthetic-source',tokens_sha256=token_sha(tokens),archive=str(p),sha256=sha(p))]))
        cp=base/'completion.json';write(cp,dict(status='T1_CALIBRATION_VAR_SHORT_PREFIX_PREPARATION_COMPLETE',outputs={str(mp):sha(mp)}))
        return cp

    def test_exact_received_cache_with_no_truth_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            _,record,arrays=self.fixture(Path(temp)/'base')
            entry=record['streams']['EC_STATIC_WHOLE']['7']
            restricted={k:arrays[k] for k in (entry['bits_key'],entry['received_tokens_key'])}
            bits=restricted[entry['bits_key']].copy()
            kwargs=dict(family='EC_STATIC_WHOLE',m=7,received_bits=bits,
                expected_source_model_id=model_id('EC_STATIC_WHOLE','synthetic-fixed-cdf'))
            hit=exact_received_cache(record,restricted,**kwargs)
            self.assertFalse(hit['receiver_truth_used']);self.assertTrue(hit['source_decode_cache_hit'])
            bits[0]^=1
            self.assertIsNone(exact_received_cache(record,restricted,**kwargs))
            kwargs['received_bits']=restricted[entry['bits_key']]
            kwargs['expected_source_model_id']='different-model'
            self.assertIsNone(exact_received_cache(record,restricted,**kwargs))
            kwargs.update(family='EC_VAR_WHOLE',m=4)
            self.assertIsNone(exact_received_cache(record,restricted,**kwargs))

    def test_altered_cached_received_tokens_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            _,record,arrays=self.fixture(Path(temp)/'base')
            arrays['static_m7_received_tokens'][0]^=1
            with self.assertRaisesRegex(ValueError,'Cached independent RX tokens changed'):
                exact_received_cache(record,arrays,family='EC_STATIC_WHOLE',m=7,
                    received_bits=arrays['static_m7_bits'],expected_source_model_id='synthetic-fixed-cdf')

    def test_merge_missing_reachable_short_stream_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);done,_,_=self.fixture(root/'base',short_needed=True)
            with self.assertRaisesRegex(ValueError,'Incomplete registered fallback chain'):
                merge.run(SimpleNamespace(source_completion=str(done),fallback_completion=[],out=str(root/'merged')))

    def test_merge_missing_unreachable_short_stream_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);done,_,_=self.fixture(root/'base',short_needed=False)
            result=merge.run(SimpleNamespace(source_completion=str(done),fallback_completion=[],out=str(root/'merged')))
            self.assertEqual(result['status'],'T1_CALIBRATION_COMPLETE_SOURCE_ASSETS_SEALED')
            self.assertEqual(read(root/'merged'/'manifest.json')['checked_candidate_source_conditions'],72)

    def test_merge_actual_short_streams_closes_all_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);done,_,arrays=self.fixture(root/'base',short_needed=True)
            fallback=self.short_fixture(root/'fallback',arrays['tokens'])
            result=merge.run(SimpleNamespace(source_completion=str(done),fallback_completion=[str(fallback)],out=str(root/'merged')))
            record=read(root/'merged'/'source_checkpoints'/'0000.json')
            self.assertEqual(set(record['streams']['EC_VAR_WHOLE']),set(map(str,range(4,10))))
            self.assertEqual(result['new_entropy_encodes'],0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    write(args.out,dict(status='PASS' if result.wasSuccessful() else 'FAIL',synthetic_only=True,
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        checked_module_sha256={name:sha(Path(__file__).with_name(name)) for name in (
            't1_source_assets_selfcheck.py','t1_source_asset_schema.py','t1_merge_sources.py')},
        new_real_model_calls=0,new_real_channel_decodes=0))
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
