"""Synthetic closure checks only: no real PHY/model, RNG, or scientific data."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

import r0_finalize as f


class ClosureTests(unittest.TestCase):
    def test_import_is_model_free(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('sionna', sys.modules)

    def test_actual_exit_success(self):
        f.actual_success(dict(actual_wait=True, returncode=0, elapsed_seconds=2))

    def test_launch_is_not_completion(self):
        for bad in (dict(returncode=0, elapsed_seconds=2),
                    dict(actual_wait=True, returncode=1, elapsed_seconds=2),
                    dict(actual_wait=True, returncode=0, elapsed_seconds=2, timed_out=True),
                    dict(actual_wait=True, returncode=0, elapsed_seconds=2, killed=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                f.actual_success(bad)

    def test_readonly_database_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'a.sqlite'
            db = sqlite3.connect(p); db.execute('CREATE TABLE t (x INTEGER)'); db.commit(); db.close()
            before = f.sha(p)
            ro = f.readonly_db(p)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    ro.execute('INSERT INTO t VALUES (1)')
            finally:
                ro.close()
            self.assertEqual(before, f.sha(p))

    def ledger(self, rows):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        p = Path(tmp.name) / 'calls.sqlite'; db = sqlite3.connect(p)
        db.execute('CREATE TABLE calls (id INTEGER PRIMARY KEY,kind TEXT,metadata TEXT,state TEXT)')
        db.executemany('INSERT INTO calls(kind,metadata,state) VALUES (?,?,?)',
                       [(k, json.dumps(m), state) for k, m, state in rows])
        db.commit(); db.close(); return p

    def render_rows(self):
        return [('model_load', {}, 'COMPLETE'), ('var_render', {'cell': 0}, 'COMPLETE'),
                ('var_render', {'cell': 1}, 'COMPLETE')]

    def render_audit(self, rows):
        return f.audit_clean_ledger(self.ledger(rows), 'render', {'model_load': 1, 'var_render': 2},
                                    [{'cell_index': 0}, {'cell_index': 1}], [])

    def test_exact_render_grid(self):
        self.assertEqual(self.render_audit(self.render_rows())['unresolved'], 0)

    def test_duplicate_cell_rejected_even_with_right_counts(self):
        rows = self.render_rows(); rows[-1] = ('var_render', {'cell': 0}, 'COMPLETE')
        with self.assertRaises(ValueError): self.render_audit(rows)

    def test_missing_extra_unknown_calls_rejected(self):
        for rows in (self.render_rows()[:-1], self.render_rows() + [self.render_rows()[-1]],
                     self.render_rows() + [('unknown', {}, 'COMPLETE')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError): self.render_audit(rows)

    def test_failed_and_reserved_calls_rejected(self):
        for status in ('FAILED', 'RESERVED'):
            rows = self.render_rows(); rows[-1] = ('var_render', {'cell': 1}, status)
            with self.subTest(status=status), self.assertRaises(ValueError): self.render_audit(rows)

    def test_prior_scale_and_decoder_source_sequence(self):
        rows = [('model_load', {}, 'COMPLETE'), ('var_render', {'cell': 0}, 'COMPLETE')]
        rows += [('prior_scale', {'source': 42, 'scale': s}, 'COMPLETE') for s in range(10)]
        rows += [('decoder_forward', {'source': 42}, 'COMPLETE')]
        caps = dict(model_load=1, var_render=1, prior_scale=10, decoder_forward=1)
        schedule = [dict(cell_index=0, source_index=42)]
        f.audit_clean_ledger(self.ledger(rows), 'render', caps, schedule, [])
        rows[-2] = ('prior_scale', {'source': 42, 'scale': 8}, 'COMPLETE')
        with self.assertRaises(ValueError):
            f.audit_clean_ledger(self.ledger(rows), 'render', caps, schedule, [])

    def test_decoder_wrong_source_is_not_hidden_by_count(self):
        rows = [('model_load', {}, 'COMPLETE'), ('var_render', {'cell': 0}, 'COMPLETE')]
        rows += [('prior_scale', {'source': 42, 'scale': s}, 'COMPLETE') for s in range(10)]
        rows += [('decoder_forward', {'source': 0}, 'COMPLETE')]
        with self.assertRaises(ValueError):
            f.audit_clean_ledger(self.ledger(rows), 'render', dict(model_load=1, var_render=1, prior_scale=10, decoder_forward=1),
                                  [dict(cell_index=0, source_index=42)], [])

    def score_rows(self):
        return [('dino_model_load', {}, 'COMPLETE'), ('source_embedding', {'source_id': 's'}, 'COMPLETE'),
                ('reconstruction_dino', {'source_id': 's', 'cell': 0}, 'COMPLETE'),
                ('image_score', {'cell': 0}, 'COMPLETE')]

    def score_audit(self, rows):
        return f.audit_clean_ledger(self.ledger(rows), 'score',
            {'dino_model_load': 1, 'source_embedding': 1, 'reconstruction_dino': 1, 'image_score': 1},
            [{'cell_index': 0, 'source_id': 's'}], [{'source_id': 's'}])

    def test_score_source_reference_and_cell_identity(self):
        self.score_audit(self.score_rows())
        rows = self.score_rows(); rows[1] = ('source_embedding', {'source_id': 'foreign'}, 'COMPLETE')
        with self.assertRaises(ValueError): self.score_audit(rows)

    def test_score_reconstruction_foreign_source_rejected(self):
        rows = self.score_rows(); rows[2] = ('reconstruction_dino', {'source_id': 'foreign', 'cell': 0}, 'COMPLETE')
        with self.assertRaises(ValueError): self.score_audit(rows)

    def test_csv_preserves_raw_precision(self):
        f.same_csv_row({'score': '0.123456789012345', 'cell': '5', 'id': 's'},
                       {'score': 0.123456789012345, 'cell': 5, 'id': 's'})
        with self.assertRaises(ValueError):
            f.same_csv_row({'score': '0.1235'}, {'score': 0.123456789012345})

    def trial(self):
        group = {'phy_key': 'a'*64, 'layout': {'layout_id': 'b'*64}, 'symbols': 4}
        identity = dict(namespace=f.bler.NAMESPACE, phy_key=group['phy_key'], layout_id=group['layout']['layout_id'],
                        snr_db=4, trial=0, session=f.bler.BODY_SESSION, group=0, runtime_identity={'runtime': 'frozen'})
        base = [f.bler.NAMESPACE, group['phy_key'], group['layout']['layout_id'], 4, 0, 'public_counter']
        counter = int.from_bytes(f.hashlib.sha256(f.bler.canonical(base).encode()).digest()[:8], 'little') & ((1<<63)-1)
        row = dict(status='COMPLETE', trial_identity=identity, correct=True, rejected=False, undetected=False,
                   payload_equal=True, crc_accept=True, actual_packet_encodes=1, actual_packet_decodes=1,
                   header_calls=0, neural_model_calls=0, public_counter=counter, symbols=4, actual_E=8.,
                   observed_mean_Es=2., noise_variance_per_component=10**(-.4))
        row.update({k:'c'*64 for k in ('payload_sha256','standard_noise_sha256','waveform_sha256','received_sha256',
                                     'hard_payload_sha256','decoded_sha256')})
        return group, identity, row

    def test_actual_trial_identity_and_partition(self):
        g, i, r = self.trial(); f.checked_trial(i, r, g, 4, 0, {'runtime':'frozen'})
        r['undetected'] = True
        with self.assertRaises(ValueError): f.checked_trial(i, r, g, 4, 0, {'runtime':'frozen'})

    def test_trial_counter_noise_and_extra_call_rejected(self):
        for key, value in (('public_counter', 42), ('noise_variance_per_component', .01),
                           ('header_calls', 1), ('actual_packet_decodes', 2)):
            g, i, r = self.trial(); r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                f.checked_trial(i, r, g, 4, 0, {'runtime':'frozen'})

    def test_wrong_trial_population_rejected(self):
        g, i, r = self.trial()
        with self.assertRaises(ValueError): f.checked_trial(i, r, g, 7, 0, {'runtime':'frozen'})

    def test_no_output_when_closure_not_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'nominations'
            with mock.patch.object(f.bler, 'validate', side_effect=ValueError('not complete')):
                with self.assertRaises(ValueError): f.finalize({}, {}, out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
