"""CPU engineering tests. Synthetic fixtures do not certify GPU image quality."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

import numpy as np

import historical_budget as old
import historical_selected_budget as h


def policy_fixture(mcs='QPSK'):
    choices, rows = [], []
    energy = 'per_frame_2N' if mcs == 'QPSK' else 'fixed_constellation_average_2_per_symbol'
    for family in ('raw', 'arithmetic'):
        for n in (2048, 3060, 4084):
            for snr in old.SNRS:
                mode = 7 if snr == 1 else 8
                method = f'token-efficiency-digital-v1/{family}/{mcs}/N{n}/m{mode}'
                choices.append(dict(family=family, N=n, snr_db=snr, method=method,
                    mcs=mcs, context_sha256='context', energy_constraint=energy))
                for i in range(100):
                    for seed in old.SEEDS:
                        row = dict(family=family, N=str(n), snr_db=str(snr), method=method,
                            mcs=mcs, context_sha256='context', energy_constraint=energy,
                            source_index=str(i), source_id=f'image{i}', noise_seed=str(seed),
                            psnr_db='30.0', lpips_alex='0.1', dino_cosine='0.9', mse='0.001',
                            waveform_sha256='wave', observation_sha256='observation',
                            header_ok='True', body_crc_ok='True', E=str(2*n))
                        rows.append(row)
                        rows.append({**row, 'method': method.rsplit('/', 1)[0]+'/m6'})
    return rows, choices


class SelectedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.choices = policy_fixture()

    def test_exact_final_roster_and_no_candidate_rows(self):
        chosen, actions = h.select_policy_rows(self.rows, self.choices, 'QPSK', 'context')
        self.assertEqual(len(chosen), 9000)
        self.assertEqual(len(actions), 30)
        self.assertEqual({int(r['N']) for r in chosen}, {2048, 3060, 4084})
        self.assertTrue(all(not r['method'].endswith('/m6') for r in chosen))
        # Return actual original dictionaries, not manufactured policy rows.
        originals = {id(r) for r in self.rows}
        self.assertTrue(all(id(r) in originals for r in chosen))

    def test_missing_duplicate_and_wrong_policy_rejected(self):
        with self.assertRaisesRegex(old.ReplayMismatch, 'incomplete'):
            h.select_policy_rows(self.rows[2:], self.choices, 'QPSK', 'context')
        with self.assertRaisesRegex(old.ReplayMismatch, 'duplicate final policy'):
            h.select_policy_rows(self.rows+[self.rows[0]], self.choices, 'QPSK', 'context')
        choices = deepcopy(self.choices); choices[0]['method'] += '_not_run'
        with self.assertRaisesRegex(old.ReplayMismatch, 'incomplete'):
            h.select_policy_rows(self.rows, choices, 'QPSK', 'context')
        with self.assertRaisesRegex(old.ReplayMismatch, 'duplicate frozen'):
            h.select_policy_rows(self.rows, self.choices+[self.choices[0]], 'QPSK', 'context')
        with self.assertRaisesRegex(old.ReplayMismatch, 'context'):
            h.select_policy_rows(self.rows, self.choices, '16QAM', 'context')

    def test_energy_protocols_remain_separate(self):
        rows, choices = policy_fixture('16QAM')
        selected, _ = h.select_policy_rows(rows, choices, '16QAM', 'context')
        self.assertEqual(len(selected), 9000)
        bad = deepcopy(choices); bad[0]['energy_constraint'] = 'per_frame_2N'
        with self.assertRaisesRegex(old.ReplayMismatch, 'energy'):
            h.select_policy_rows(rows, bad, '16QAM', 'context')

    def test_forward_receives_selected_actions_only(self):
        selected, actions = h.select_policy_rows(self.rows, self.choices, 'QPSK', 'context')
        rows = deepcopy([r for r in selected if r['source_index'] == '0'])
        a = h.FinalDigital.__new__(h.FinalDigital)
        a.study, a.mcs, a.ready = 'FINAL_DIGITAL_QPSK', 'QPSK', True
        a.actions, a.policy_sha256 = actions, 'frozen-policy'
        pixels = np.full((3, 256, 256), 127, np.uint8)
        target = pixels.astype(np.float32)/255
        image = np.full((3, 256, 256), .5, np.float32)
        for row in rows:
            row['mse'] = str(float(np.mean(np.square(image-target), dtype=np.float64)))
        a.by_source = {0: rows}
        a.records = [dict(pixels=pixels, class_index=1, image_id='image0')]
        a._locations = {old.identity(r): {'test_only': True} for r in rows}
        a.ledger = {}
        for row in rows:
            a.ledger[row['method']] = dict(family=row['family'], N=int(row['N']),
                m_requested=int(row['method'].rsplit('m', 1)[1]))
        calls, batches, txcalls = [], [], []
        def cell(family, n, mode, mcs):
            return SimpleNamespace(family=family, N=n, m=mode,
                name=f'token-efficiency-digital-v1/{family}/{mcs}/N{n}/m{mode}')
        def transmit(prepared, label, cell):
            txcalls.append(cell.name)
            return np.array([0]), {'N': cell.N, 'E': 2*cell.N, 'mcs': 'QPSK', 'energy_constraint': 'per_frame_2N'}
        def receive(observed, snr, cell, *unused):
            calls.append((cell.name, snr, int(observed[1])))
            return image, dict(header_ok=True, body_crc_ok=True)
        def quality(target, slots, *unused):
            batches.append(len(slots))
            return ([dict(psnr_db=30., lpips_alex=.1, dino_cosine=.9) for _ in slots], None, None)
        a.native = SimpleNamespace(Cell=cell, prepare_digital=lambda *args: 'prepared',
            transmit_prepared=transmit, apply_channel=lambda wave, snr, sid, seed, cell: np.array([1, seed]),
            receive=receive, quality_metrics=quality,
            waveform_sha=lambda wave: 'wave' if wave[0] == 0 else 'observation')
        a.torch = SimpleNamespace(no_grad=nullcontext); a.safe = SimpleNamespace(check=lambda: None)
        a.vae = a.var = a.decoder = a.device = a.lp = a.dino = None
        output = list(a.iterate_source(0))
        self.assertEqual(len(calls), 90)
        self.assertEqual(len(output), 90)
        self.assertEqual(len(txcalls), 12)
        self.assertEqual(batches, [15]*12)
        self.assertEqual({x[0] for x in calls}, set(a.ledger))
        self.assertEqual([x[0] for x in output], rows)
        self.assertTrue(all(x[3]['replay_parity_passed'] for x in output))
        self.assertTrue(all(actions[(r['family'], int(r['N']), int(r['snr_db']))] == r['method'] for r in rows))

    def test_selected_continuous_filters_before_parent_setup(self):
        def initialize(a, root, study, *unused):
            a.study = study; a.ready = False; a._loaded = None; a.context = {'bindings': {}}
            a.methods = ['P2048', 'P3060', 'P4084'] if study == 'CONTINUOUS_GRID' else [*h.P4084_METHODS, 'H6-V']
            steps = {'P2048':30000, 'P3060':37500, 'P4084':10000,
                     h.P4084_METHODS[0]:27500, h.P4084_METHODS[1]:30000, 'H6-V':30000}
            a.model_metadata = {m: dict(step=steps[m], selected=dict(step=steps[m]),
                kind='hybrid' if m == 'H6-V' else 'continuous') for m in a.methods}
            a.rows = [dict(method=m, source_index=str(i)) for i in range(100) for m in a.methods]
            a.by_source = {i:[r for r in a.rows if r['source_index'] == str(i)] for i in range(100)}
            a._main = {(i,r['method'],1,2001):r for i in range(100) for r in a.by_source[i]}
        loaded = []
        def setup(a):
            loaded.extend(a.methods)
            return a
        fake_lpips = SimpleNamespace(__file__='/fixture/lpips/__init__.py')
        with patch.object(old.BudgetAdapter, '__init__', initialize), patch.object(old.BudgetAdapter, 'setup', setup), \
             patch.dict('sys.modules', {'lpips': fake_lpips}), patch.object(h, 'registered_quality_weight', return_value='weight-sha'):
            a = h.create_adapter('.', 'FINAL_P2048_P3060'); a.setup()
            b = h.create_adapter('.', 'FINAL_P4084_SELECTED_SEEDS'); b.setup()
        self.assertEqual(loaded, ['P2048', 'P3060', *h.P4084_METHODS])
        self.assertTrue(all(r['method'] in ('P2048', 'P3060') for r in a.rows))
        self.assertTrue(all(r['method'] in h.P4084_METHODS for r in b.rows))
        self.assertNotIn('P4084', loaded)
        self.assertNotIn('H6-V', loaded)

    def test_unknown_and_injected_state_rejected(self):
        with self.assertRaisesRegex(old.ReplayMismatch, 'unregistered'):
            h.create_adapter('.', 'ALL_CANDIDATES')
        for study in ('FINAL_DIGITAL_QPSK', 'FINAL_PHASE2_N4084_DIGITAL'):
            with self.assertRaisesRegex(old.ReplayMismatch, 'injected'):
                h.create_adapter('.', study, loaded={})

    def test_actual_rx_erasure_is_separate_from_tx_ledger(self):
        row = dict(source_overflow_erasure='True', rx_source_overflow_erasure='False',
                   source_complete='False', decoded_label='', source_error='header_failure')
        event = dict(source_overflow_erasure=False, source_complete=False,
                     decoded_label=None, source_error='header_failure')
        verified = h.digital_events(row, event)
        self.assertFalse(verified['rx_source_overflow_erasure'])
        self.assertTrue({**verified, 'source_overflow_erasure': True}['source_overflow_erasure'])
        with self.assertRaisesRegex(old.ReplayMismatch, 'receiver event'):
            h.digital_events(row, {**event, 'decoded_label': 3})

    def test_linear_weight_relocation_requires_identical_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            origin = root/'old/lpips/weights/v0.1/alex.pth'
            actual = root/'new/lpips/weights/v0.1/alex.pth'
            for path in (origin, actual):
                path.parent.mkdir(parents=True); path.write_bytes(b'engineering-fixture')
            a = h.FinalDigital.__new__(h.FinalDigital); a.bindings = {}
            digest = old.sha256(origin)
            a.context = {'bindings': {str(origin): digest}}
            self.assertEqual(h.registered_quality_weight(a, actual), digest)
            self.assertEqual(len(a.bindings), 2)
            actual.write_bytes(b'different')
            with self.assertRaisesRegex(old.ReplayMismatch, 'hash differs'):
                h.registered_quality_weight(a, actual)

    def test_phase2_only_decodes_final_policy_rows(self):
        rows, candidates = [], []
        for family in ('raw', 'arithmetic'):
            for snr in old.SNRS:
                for seed in old.SEEDS:
                    mode = 8 if snr == 1 else 9
                    common = dict(image_id='image0', snr_db=str(float(snr)), seed=str(seed),
                                  psnr_db='30.0', lpips_alex='0.1', dino_cosine='0.9')
                    rows.append(dict(common, method=family+'_adaptive_m789_Dc', source_index='0',
                                     selected_mode=str(mode), N='4084', E='8168'))
                    candidates.append(dict(common, family=family, renderer='Dc', image_index='0',
                                           mode=str(mode), budget='4084'))
        a = h.FinalPhase2.__new__(h.FinalPhase2)
        a.study = 'FINAL_PHASE2_N4084_DIGITAL'; a.policy_sha256 = 'policy'
        a.by_source = {0: rows}; a.aliases = {old.identity(c):r for c,r in zip(candidates, rows)}
        pixels = np.full((3, 256, 256), 127, np.uint8)
        image = np.full((3, 256, 256), .5, np.float32)
        calls, batches = [], []
        def decode(record, source, row, cache):
            calls.append((row['family'], row['mode'], row['snr_db'], row['seed']))
            return image, {'N':4084, 'E':8168.}
        def quality(target, images, *unused):
            batches.append(len(images))
            return ([dict(psnr_db=30., lpips_alex=.1, dino_cosine=.9) for _ in images], None, None)
        a.native = SimpleNamespace(_ready=True, _context=nullcontext,
            runtime=SimpleNamespace(require_available=lambda:None),
            targets=[dict(pixels=pixels, tokens='fixture')],
            progressive=SimpleNamespace(split_prefix=lambda *unused:'split'),
            expected_rows=lambda i:candidates, _digital_image=decode, _verify_info=lambda *unused:None,
            q=SimpleNamespace(quality_metrics=quality), lp=None, dino=None, device=None, models={})
        fake_torch = SimpleNamespace(no_grad=nullcontext,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                cudnn=SimpleNamespace(allow_tf32=False, benchmark=False)))
        with patch.dict('sys.modules', {'torch':fake_torch}):
            output = list(a.iterate_source(0))
        self.assertEqual([x[0] for x in output], rows)
        self.assertEqual(len(calls), 30)
        self.assertEqual({mode for _,mode,_,_ in calls}, {'8','9'})
        self.assertEqual(batches, [15]*4)
        self.assertTrue(all(x[3]['replay_parity_passed'] for x in output))


if __name__ == '__main__':
    unittest.main()
