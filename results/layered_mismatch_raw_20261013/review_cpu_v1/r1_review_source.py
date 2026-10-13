"""Independent CPU review only; actual Runtime construction is forbidden."""
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

W = Path(__file__).resolve().parents[2]
E = W / 'experiments/layered-mismatch-raw-20261013'
O = W / 'results/layered_mismatch_raw_20261013/review_cpu_v1'
sys.path.insert(0, str(E))
import r1_phy as r1
import grouped_wire as wire
import test_r1_phy as base


class IndependentTests(unittest.TestCase):
    def test_real_receiver_static_boundary_has_no_source_payload_or_tx_id(self):
        tree = ast.parse((E / 'r1_phy.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RealRuntime')
        receive = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'receive')
        self.assertEqual([a.arg for a in receive.args.args],
                         ['self', 'group_count', 'received', 'true_snr', 'counter', 'audit_tag'])
        for node in ast.walk(receive):
            if isinstance(node, ast.Name):
                self.assertNotIn(node.id, {'source', 'scales', 'tokens', 'pixels', 'payloads', 'nomination'})
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, {'transmit', 'serialize_raw', 'serialize_groups', 'load', 'loadtxt'})

    def test_legacy_E_receipt_is_accepted_without_extra_receive(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = r1.Journal(Path(tmp) / 'j')
            rt = base.StubRuntime(j)
            transmit = rt.transmit
            def legacy(*args):
                wave, receipt, payload = transmit(*args)
                receipt['E'] = receipt.pop('actual_E')
                return wave, receipt, payload
            rt.transmit = legacy
            r1.execute_frame(base.synthetic_source(), base.synthetic_scales(), base.nomination(),
                             4, 4101, rt, j, {'rt': 'SYNTHETIC'})
            self.assertEqual(len(rt.rx_calls), 1)

    def test_corrupt_wave_receipt_stops_before_channel_and_receiver(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = r1.Journal(Path(tmp) / 'j')
            rt = base.StubRuntime(j)
            transmit = rt.transmit
            def corrupt(*args):
                wave, receipt, payload = transmit(*args)
                receipt['actual_E'] += 1
                return wave, receipt, payload
            rt.transmit = corrupt
            with patch.object(r1.contract, 'standard_noise', side_effect=AssertionError('No channel permitted')):
                with self.assertRaisesRegex(ValueError, 'waveform/energy'):
                    r1.execute_frame(base.synthetic_source(), base.synthetic_scales(), base.nomination(),
                                     4, 4101, rt, j, {'rt': 'SYNTHETIC'})
            self.assertEqual(len(rt.rx_calls), 0)
            self.assertEqual(j.snapshot()['physical_reserved'], 1)
            self.assertEqual(j.snapshot()['physical_complete'], 0)

    def test_declared_scientific_caps_are_unchanged(self):
        self.assertEqual(r1.contract.R1_CAPS, dict(sources=200, physical_frames=36000,
            logical_var_images=46800, packet_decodes=108000))
        rows, profiles = base.PlanTests().rows()
        groups = r1.nomination_groups(rows, profiles)
        self.assertEqual(len(groups) * 200 * 9, 7200)
        self.assertEqual(sum(len(g['logical_receivers']) for g in groups) * 200 * 9, 10800)
        for g in groups:
            self.assertEqual(len({(v['candidate_id'], v['rule']) for v in g['logical_receivers']}),
                             len(g['logical_receivers']))


O.mkdir(parents=True, exist_ok=True)
log = io.StringIO()
suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(base),
                           unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTests)])
with patch.object(r1, 'RealRuntime', side_effect=AssertionError('Real PHY construction forbidden')):
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
text = log.getvalue()
print(text)
(O / 'r1_phy_tests.txt').write_text(text, encoding='utf8')
receipt = dict(status='PASS_CPU_PREPARATION_ONLY' if result.wasSuccessful() else 'FAIL',
    actual_exit_code=0 if result.wasSuccessful() else 1, tests_run=result.testsRun,
    failures=len(result.failures), errors=len(result.errors), reviewed_unix=time.time(),
    source_bindings={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (E/'r1_phy.py', E/'test_r1_phy.py', E/'grouped_wire.py', E/'contract.py')},
    reviewer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    real_PHY_constructed=False, real_packet_encodes=0, real_packet_decodes=0,
    real_source_channel_calls=0, model_calls=0, bootstrap_calls=0,
    synthetic_arrays_only=True, torch_imported='torch' in sys.modules,
    scientific_registration='NOT_CREATED', scientific_run='NOT_RUN',
    source_truth_boundary='Source pixels/tokens are authenticated only in the TX source loader and post-RX audit. '
        'The real receive function accepts received samples, true SNR, public counter, fixed method-domain selector and audit tag; '
        'it does not accept TX profile ID, original tokens or source image.',
    verified=[
        'Candidate IDs are canonical catalogue IDs; one physical wire per hat is sent once, rule aliases retain arms metadata.',
        'Single KEEP and DROP derive from the same actual receive event; DROP never invokes another decode.',
        'Fixed original raw433 single RX domain and mixed raw433+G2 always-DROP domain remain separate.',
        'Source manifest selects the exact hash-ranked original-calibration200; source archive and checkpoint hashes and both arrays are checked.',
        'Source/model-free CPU fixtures cannot constitute scientific qualification.',
        'Immutable frame/packet reservations precede callbacks; unresolved failures block replay and further work; caps unchanged.',
        'Completed exact-identity reuse verifies the saved result pin before returning without transmit/noise/receive.',
        'Both actual_E and frozen legacy E are checked against the actual float64 waveform and its hash.',
        'Frozen raw original array_sha formula equals the new wire array_sha formula; exact real interfaces were read without importing the backend.'
    ],
    limitations=[
        'Only finite PHY preparation is reviewed. No real registration, owner launch, model reconstruction, metrics, selection, or R2 was executed.',
        'Future owner must bind the finalized nominations, source/input/code/dual-domain identities and preserve actual child exit evidence.',
        'Unknown BLER/clean-Q values cannot be inferred or consumed. R0 additional sampling is still awaiting user budget authorization.',
        'R1 output is COMPLETE_R1_PHY_ONLY; it is not a quality result or authorization to launch R2.'
    ])
(O / 'r1_phy_review.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
raise SystemExit(receipt['actual_exit_code'])
