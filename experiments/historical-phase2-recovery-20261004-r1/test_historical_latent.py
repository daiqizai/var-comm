"""CPU integrity tests; synthetic fixtures never qualify scientific RGB replay."""
import ast
import csv
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import historical_latent as h


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def rows(methods, **extra):
    return [dict(method=m, source_index=str(i), image_id=f'image{i}',
        snr_db=str(s), seed=str(n), N='4084', E='8168', psnr_db='20.0',
        lpips_alex='0.2', dino_cosine='0.6', **extra)
        for i in range(100) for s in h.SNRS for n in h.SEEDS for m in methods]


def fixture(root, study='LINEAR_REPAIRED', records=None):
    folder, table, status = h.SPECS[study]
    folder = root / folder
    folder.mkdir(parents=True)
    records = rows(('source','residual')) if records is None else records
    with (folder/table).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write(folder/'completion.json',dict(status=status,new_holdout_used=False,
        per_frame_sha256=h.sha(folder/table),rows=len(records)))
    write(folder/'projection.json',dict(A_sha256='a'*64,W={'1.0':{'source':.5}}))
    return folder


class LatentIntegrityTests(unittest.TestCase):
    def test_complete_grid_and_exact_csv_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            a=h.create_adapter(root,'LINEAR_REPAIRED')
            self.assertEqual(a.describe()['rows'],3000)
            self.assertEqual(len(a.expected_rows(0)),30)
            self.assertEqual(a.expected_rows(0)[0]['psnr_db'],'20.0')
            row=a.expected_rows(0)[0];row['method']='mutated'
            self.assertEqual(a.expected_rows(0)[0]['method'],'source')

    def test_source_noise_grid_missing_or_duplicate_rejected(self):
        good=rows(('source','residual'))
        for bad in (good[:-1],good+[good[0]], [{**r,'seed':'4101'} for r in good]):
            with self.assertRaisesRegex(ValueError,'incomplete or duplicate'):
                h.validate_grid('LINEAR_REPAIRED',bad)

    def test_original_csv_tamper_fails_before_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=fixture(root)
            with (folder/'per_frame.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'completion SHA differs'):
                h.create_adapter(root,'LINEAR_REPAIRED')

    def test_incomplete_or_holdout_cohort_rejected(self):
        for extra in ({'status':'RUNNING'},{'new_holdout_used':True}):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);folder=fixture(root)
                v=json.loads((folder/'completion.json').read_text());v.update(extra)
                write(folder/'completion.json',v)
                with self.assertRaises(ValueError):h.create_adapter(root,'LINEAR_REPAIRED')

    def test_policy_views_are_distinct_and_alias_join_checks_every_field(self):
        candidate=rows(('raw_N4084_m7_Dc',),family='raw',renderer='Dc',sent_mode='7')[0]
        policy={**candidate,'policy':'quality','selected_mode':'7'}
        other={**policy,'policy':'goodput'}
        self.assertNotEqual(h.method_id('LATENT_FROZEN_POLICIES',policy),h.method_id('LATENT_FROZEN_POLICIES',other))
        self.assertEqual(h.exact_alias_row(policy,[candidate]),candidate)
        with self.assertRaisesRegex(ValueError,'changed candidate'):
            h.exact_alias_row({**policy,'psnr_db':'20.01'},[candidate])
        with self.assertRaisesRegex(ValueError,'one exact'):
            h.exact_alias_row(policy,[candidate,candidate])

    def test_strict_candidate_identity_keeps_mode_and_renderer(self):
        r=dict(family='raw',budget='4084',mode='7',renderer='Dc')
        self.assertEqual(h.method_id('PHASE2_STRICT_DIGITAL',r),'raw_N4084_m7_Dc')
        self.assertNotEqual(h.method_id('PHASE2_STRICT_DIGITAL',r),h.method_id('PHASE2_STRICT_DIGITAL',{**r,'mode':'8'}))

    def test_only_exact_unconsumed_predictor_drift_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/(h.EXP+'followup/src/latent_followup/digital_policy_matrix.py')
            p.parent.mkdir(parents=True);p.write_text('# current strict digital\n')
            adapter=object.__new__(h.LatentAdapter)
            adapter.root=root.resolve();adapter.study='PREDICTOR_REPAIRED';adapter.bindings={}
            adapter.unconsumed_snapshot_drift={};adapter.modules={}
            adapter._verify_consumed_sources({str(p):'0'*64})
            self.assertEqual(adapter.unconsumed_snapshot_drift[str(p)]['registered_sha256'],'0'*64)
            with patch.object(h.importlib,'import_module',return_value=types.SimpleNamespace(__file__=str(p))):
                with self.assertRaisesRegex(ValueError,'Cannot consume'):
                    adapter._module('latent_followup.digital_policy_matrix')
                with self.assertRaisesRegex(ValueError,'Cannot consume'):
                    adapter._module('latent_followup.digital_policy_matrix')
            adapter.study='PHASE2_STRICT_DIGITAL'
            with self.assertRaisesRegex(ValueError,'SHA differs'):
                adapter._verify_consumed_sources({str(p):'0'*64})

    def test_metadata_separates_conditioned_oracle_reference(self):
        a=object.__new__(h.LatentAdapter);a.unconsumed_snapshot_drift={}
        a.study='PHASE2_SELECTED_MODELS'
        m=a.metadata(rows(('pure_continuous',))[0])
        self.assertFalse(m['label_conditioned']);self.assertTrue(m['classification_main_eligible'])
        m=a.metadata(rows(('light_tx',))[0])
        self.assertTrue(m['label_conditioned']);self.assertFalse(m['classification_main_eligible'])
        a.study='PHASE2_DIAGNOSTICS'
        m=a.metadata(rows(('oracle_TX_AWGN',))[0])
        self.assertTrue(m['oracle']);self.assertFalse(m['classification_main_eligible'])
        m=a.metadata(rows(('Dc_F_representation_reference',))[0])
        self.assertTrue(m['reference_only']);self.assertFalse(m['label_conditioned'])

    def test_metric_alias_is_copy_and_native_parity_rejects_drift(self):
        r=dict(psnr_db='20',lpips='.2',dino_cosine='.6')
        mapped=h.original_metrics(r)
        self.assertNotIn('lpips_alex',r)
        proof=h.parity(mapped,dict(psnr_db=20,lpips_alex=.2,dino_cosine=.6))
        self.assertFalse(proof['synthetic'])
        with self.assertRaisesRegex(ValueError,'parity failed'):
            h.parity(mapped,dict(psnr_db=20.1,lpips_alex=.2,dino_cosine=.6))

    def test_actual_channel_events_reject_crc_and_cost_changes(self):
        a=object.__new__(h.LatentAdapter)
        a._verify_info(dict(body_crc_ok='False',N='4084',E='8168'),dict(body_crc_ok=0,N=4084,E=8168))
        for row,actual in ((dict(body_crc_ok='1'),dict(body_crc_ok=0)),
                           (dict(accepted_correct='1'),dict(accepted_correct=0)),
                           (dict(actual_payload_bits='3072'),dict(actual_payload_bits=3000)),
                           (dict(E='8168'),dict(E=8000))):
            with self.assertRaises(ValueError):a._verify_info(row,actual)

    def test_adapter_never_calls_training_calibration_or_old_main(self):
        source=Path(h.__file__).read_text();tree=ast.parse(source)
        called={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
                for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(called & {'main','backward','step','fit_training_norm_range','fit_scalar_w',
                                 'freeze_policies','calibrate_v2','torch_save','save_torch','write_rows'})
        self.assertIn('matched_batch',called);self.assertIn('execute',called)


if __name__=='__main__':unittest.main()
