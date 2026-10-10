"""Synthetic read-only reference and source-owner accounting qualification."""
import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from t1_entropy_core import OFFSETS,read,write,sha
from t1_owned_source_gate import array_sha
from t1_reference_cache import RawCalibrationReference
from t1_source_owner import CallLedger,source_budget


class OwnerTests(unittest.TestCase):
    def test_call_count_budget_and_failed_reservation(self):
        with tempfile.TemporaryDirectory() as temp:
            caps=dict(var_source_tx=1,var_source_rx=2,static_source_rx=0,image_render=0)
            ledger=CallLedger(Path(temp)/'calls',caps,lambda:None)
            self.assertEqual(ledger('var_source_tx',lambda:7,source_index=0,m=5),7)
            ledger('var_source_rx',lambda:None,source_index=0,m=4)
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                ledger('var_source_rx',lambda:(_ for _ in ()).throw(RuntimeError('synthetic failure')),source_index=0,m=5)
            self.assertEqual(ledger.summary()['unresolved'],1)
            self.assertEqual(ledger.summary()['new_VAR_prior_scale_evaluations_completed'],9)
            with self.assertRaisesRegex(ValueError,'budget exhausted'):
                ledger('var_source_tx',lambda:None,source_index=1,m=5)

    def test_budget_uses_lengths_not_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);records=[dict(source_index=i,families={'EC_VAR_WHOLE':{'6':927+(i%2)}}) for i in range(1000)]
            mp=base/'manifest.json';write(mp,dict(source_count=1000,records=records))
            done=base/'completion.json';write(done,dict(status='T1_CALIBRATION_SOURCE_CPU_PREPARATION_COMPLETE',outputs={str(mp):sha(mp)}))
            caps,needed=source_budget(done)
            self.assertEqual(needed,list(range(1,1000,2)));self.assertEqual(caps['var_source_tx'],500)
            self.assertEqual(caps['var_source_rx'],1000);self.assertEqual(caps['image_render'],192)

    def fixture_reference(self,base):
        (base/'source_checkpoints').mkdir();(base/'sources').mkdir();(base/'images').mkdir()
        identity={'models':{'var':'synthetic','vae':'synthetic','decoder':'synthetic'}}
        flags={'synthetic':True};tokens=np.arange(OFFSETS[7],dtype=np.int64)
        image=np.full((3,256,256),.5,dtype=np.float32)
        archive=base/'images'/'0000.npz';np.savez(archive,images=np.array([image]))
        state=dict(kind='tokens',m=7,K=0,order='raster',partial_values=[],
            prefix=[tokens[OFFSETS[s]:OFFSETS[s+1]].tolist() for s in range(7)])
        rp=base/'sources'/'0000.json';write(rp,[dict(receiver_state=state,source_id='synthetic-0',source_index=0,
            image_archive=str(archive),image_slot=0,image_sha256=array_sha(image))])
        cp=base/'source_checkpoints'/'0000.json';write(cp,dict(source_index=0,source_id='synthetic-0',images_scored=True,
            outputs={str(rp):sha(rp),str(archive):sha(archive)}))
        done=base/'completion.json';write(done,dict(source_count=1000,images_scored=True,
            frozen_visual_identity=identity,numerical_runtime=flags,
            outputs={str(cp):sha(cp),str(rp):sha(rp),str(archive):sha(archive)}))
        env=base/'environment.json';write(env,dict(original_visual_completion={'path':str(done),'sha256':sha(done)},
            old_visual_identity=identity,old_numerical_runtime=flags,
            records=[dict(source_index=i,source_id=f'synthetic-{i}') for i in range(32)]))
        native=SimpleNamespace(loaded={'identity':identity},flags=flags)
        return RawCalibrationReference(env,native),tokens,archive

    def test_reference_exact_state_and_content(self):
        with tempfile.TemporaryDirectory() as temp:
            lookup,tokens,_=self.fixture_reference(Path(temp))
            hit=lookup(0,7,tokens)
            self.assertEqual(hit['m'],7);self.assertEqual(hit['source_id'],'synthetic-0')
            changed=tokens.copy();changed[0]+=1
            self.assertIsNone(lookup(0,7,changed))
            self.assertIsNone(lookup(0,8,np.arange(OFFSETS[8])))

    def test_changed_reference_archive_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            lookup,tokens,archive=self.fixture_reference(Path(temp));archive.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA256'):lookup(0,7,tokens)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OwnerTests))
    write(a.out,dict(status='PASS' if result.wasSuccessful() else 'FAIL',synthetic_only=True,tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),new_real_model_calls=0,new_real_channel_decodes=0,
        checked_module_sha256={name:sha(Path(__file__).with_name(name)) for name in (
            't1_source_owner.py','t1_reference_cache.py','t1_owned_source_gate.py','t1_owned_shortprefixes.py','t1_owner_selfcheck.py')}))
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
