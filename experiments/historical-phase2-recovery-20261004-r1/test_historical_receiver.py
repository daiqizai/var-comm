"""CPU contract tests. Fixtures are synthetic and do not qualify real RGB replay."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('historical_receiver_test_module', HERE / 'historical_receiver.py')
rx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rx)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def score():
    return dict(psnr_db=20., lpips_alex=.2, dino_cosine=.6)


def step2_fixture(root, arm):
    here = root / f'experiments/rx-posterior-step2-{arm}-20260930'
    old = root / 'experiments/rx-posterior-step1-20260929'
    for p in [here / f for f in ('run.py', 'evaluate.py', 'receiver.py', 'train.py', 'protocol.py', 'qualification.py')] + \
             [old / f for f in ('probe.py', 'rx_v3_common.py')] + [root / p for p in (
                 'src/var_comm/quality.py', 'src/var_comm/next_scale_prior.py',
                 'experiments/token_channel_efficiency_20260923/src/token_efficiency/execution.py',
                 'experiments/token_channel_efficiency_20260923/src/token_efficiency/C_evaluate.py',
                 'experiments/var-latent-enhancement-20260917/research/src/latent_research/models.py')]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# synthetic CPU source fixture\n')
    out = root / f'outputs/RX-POSTERIOR-STEP2-{arm}-20260930-R1'
    if arm == 'B':
        out /= 'evaluation'
    result = root / f'results/rx_posterior_step2_{arm}_20260930_R1'
    snrs = [-5, -2, 1, 4, 7, 13] if arm == 'A' else [-5, -2, 1, 4, 13]
    cfg = dict(snrs=snrs, development_seeds=[2001, 2002, 2003], identity={'synthetic_test_only': True})
    write(result / 'config.json', cfg)
    config_sha = rx.sha(result / 'config.json')
    write(result / 'selected_policy.json', dict(development_read=False, levels={}, config_sha256=config_sha))
    policy_sha = rx.sha(result / 'selected_policy.json')
    ident = dict(config_sha256=config_sha, policy_sha256=policy_sha, bindings=[],
                 source_ids=[f'image{i}' for i in range(100)], preprocessing_ids=['prep'] * 100)
    write(out / 'development_identity.json', ident)
    write(out / 'calibration_complete.json', dict(config_sha256=config_sha, policy_sha256=policy_sha))
    base = 'B2' if arm == 'A' else 'P_low'
    methods = [base] + [p + '_' + s for p in ('A1', 'A2', 'V')
                       for s in ('policy', 'raw_fused', 'common', 'tok')] + ['V_lambda1']
    allrows = []
    for i in range(100):
        rows = [dict(source_id=f'image{i}', source_index=i, snr_db=s, noise_seed=seed,
                     method=m, lambda_='', policy_action='BYPASS', **score())
                for s in snrs for m in methods for seed in (2001, 2002, 2003)]
        write(out / 'development_cells' / f'{i:03d}.json', dict(identity=ident, rows=rows, token_rows=[]))
        allrows.extend(rows)
    table = result / 'per_frame.csv'
    with table.open('w', newline='', encoding='utf-8') as f:
        fieldnames = [('lambda' if k == 'lambda_' else k) for k in allrows[0]]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows({('lambda' if k == 'lambda_' else k): v for k, v in r.items()} for r in allrows)
    write(result / 'table_shards/manifest.json', dict(tables={'per_frame.csv':
        dict(sha256=rx.sha(table), rows=len(allrows), parts=[])}))
    write(out / 'completion.json', dict(status='A_COMPLETE_NO_TRAINING_AWAIT_USER_DECISION' if arm == 'A'
          else 'B_SELECTED_P_LOW_EVALUATION_COMPLETE', files=len(allrows), config_sha256=config_sha,
          policy_sha256=policy_sha, frozen_weights_unchanged=True,
          **({'training_updates': 0} if arm == 'A' else {'evaluation_training_updates': 0})))
    return out, result


class ReceiverContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.out_a, cls.result_a = step2_fixture(cls.root, 'A')
        cls.out_b, cls.result_b = step2_fixture(cls.root, 'B')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_original_step2_a_scope(self):
        a = rx.create_adapter(self.root, 'RX_STEP2_A')
        self.assertEqual(a.describe()['rows'], 25200)
        self.assertEqual(a.describe()['sources'], 100)
        self.assertEqual(len(a.describe()['methods']), 14)
        self.assertEqual(len(a.expected_rows(0)), 252)
        self.assertTrue(a.bindings)

    def test_full_step2_b_negative_snr_scope_and_evaluation_not_training(self):
        b = rx.create_adapter(self.root, 'RX_STEP2_B')
        self.assertEqual(len(b.rows), 21000)
        self.assertEqual({r['snr_db'] for r in b.expected_rows(0)}, {-5, -2, 1, 4, 13})
        self.assertEqual(b.metadata(b.expected_rows(0)[0])['snr_role'], 'low_snr_main')
        self.assertEqual(b.metadata(dict(snr_db=13, method='P_low'))['snr_role'], 'side_effect')

    def test_expected_rows_do_not_mutate_sealed_inventory(self):
        a = rx.create_adapter(self.root, 'RX_STEP2_A')
        rows = a.expected_rows(0)
        rows[0]['method'] = 'corrupted'
        self.assertNotEqual(a.expected_rows(0)[0]['method'], 'corrupted')

    def test_lambda_csv_spelling_and_exact_original_values(self):
        row = dict(source_id='x', source_index=0, lambda_=0., flag=True, optional=None)
        want = dict(source_id='x', source_index='0', **{'lambda': '0.0', 'flag': 'True', 'optional': ''})
        rx.validate_csv_rows([row], [want])
        bad = dict(want, source_index='1')
        with self.assertRaisesRegex(ValueError, 'sealed original CSV'):
            rx.validate_csv_rows([row], [bad])

    def test_step2_corrupt_cell_refused_even_when_receipt_count_matches(self):
        p = self.out_a / 'development_cells/000.json'
        raw = p.read_bytes()
        try:
            obj = json.loads(raw)
            obj['rows'][0]['psnr_db'] += .5
            write(p, obj)
            with self.assertRaisesRegex(ValueError, 'sealed original CSV'):
                rx.create_adapter(self.root, 'RX_STEP2_A')
        finally:
            p.write_bytes(raw)

    def test_policy_development_seal_refusal(self):
        p = self.result_b / 'selected_policy.json'
        raw = p.read_bytes()
        try:
            obj = json.loads(raw)
            obj['development_read'] = True
            write(p, obj)
            with self.assertRaises(ValueError):
                rx.create_adapter(self.root, 'RX_STEP2_B')
        finally:
            p.write_bytes(raw)

    def test_parity_accepts_only_complete_real_score_evidence(self):
        got = rx.parity(score(), score())
        self.assertTrue(got['parity_passed'])
        self.assertFalse(got['synthetic'])
        self.assertFalse(got['original_pixel_tensor_available'])
        self.assertEqual(got['training_updates'], 0)
        self.assertEqual(got['policy_selection_updates'], 0)

    def test_parity_rejects_changed_or_nonfinite_or_missing_scores(self):
        for changed in (dict(score(), psnr_db=19), dict(score(), dino_cosine=float('nan')),
                        {'psnr_db': 20., 'lpips_alex': .2}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                rx.parity(score(), changed)

    def test_parity_includes_mismatched_dino_when_original_has_it(self):
        original = dict(score(), dino_mismatched=.1)
        with self.assertRaises(ValueError):
            rx.parity(original, score())
        self.assertIn('dino_mismatched', rx.parity(original, original)['metric_differences'])

    def test_teacher_forcing_never_generates_image_rows(self):
        cl = dict(source_id='x', source_index=0, mode='CL', profile='V_decision',
                  latent_squared_error=3., fused_latent_squared_error=2., scales=[{'acc': .5}],
                  probability_only=[], token_image=score(), fused_image=dict(score(), psnr_db=21.))
        tf = dict(mode='TF', profile='V_decision', scales=[{'acc': .6}])
        rows = rx.image_branches(dict(rows=[tf, cl]), '/original/cell.json', 'a' * 64, 'RX_STEP1_V3_B')
        self.assertEqual([r['method'] for r in rows], ['V_decision_tok', 'V_decision_fuse'])
        self.assertEqual(rows[0]['original_nested_row_index'], 1)
        self.assertEqual(rows[0]['original_nested_row_sha256'], rx.canonical_hash(cl))
        self.assertEqual(rows[0]['latent_sq_err_final'], 3.)
        self.assertEqual(rows[1]['latent_sq_err_final'], 2.)
        self.assertNotEqual(rx.canonical_hash(rows[0]), rx.canonical_hash(rows[1]))
        self.assertEqual(rows[0]['scales'], cl['scales'])

    def test_source_a_reference_is_one_clean_explicit_reference(self):
        cell = dict(rows=[dict(method='B1', **score())], source_id='x', source_index=0,
                    source_identity={'rgb_sha256': 'prep'}, source_A_Dc_true_Fq=score())
        rows = rx.image_branches(cell, '/cell', 'a' * 64, 'RX_STEP1_V3_A')
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['method'], 'Source_A_Dc_true_Fq')
        self.assertEqual(rows[1]['noise_seed'], 0)
        fake = object.__new__(rx.Step1Adapter)
        fake.study, fake.model_identity = 'RX_STEP1_V3_A', {}
        self.assertTrue(fake.metadata(rows[1])['oracle'])
        self.assertFalse(fake.metadata(rows[1])['is_main_conclusion'])

    def test_row_identity_includes_checkpoint_seed_and_nested_output(self):
        base = dict(source_id='x', method='V_policy', noise_seed=2001, base_model_id='selected27500')
        self.assertNotEqual(rx.canonical_hash(base), rx.canonical_hash(dict(base, noise_seed=2002)))
        self.assertNotEqual(rx.canonical_hash(base), rx.canonical_hash(dict(base, base_model_id='selected10000')))

    def test_context_restores_generic_aliases_paths_and_signals(self):
        a = object.__new__(rx.Step2Adapter)
        a.root, a.here, a.oldhere = self.root, self.root, self.root
        a._aliases = {'receiver': types.ModuleType('isolated_fixture_receiver')}
        old = sys.modules.get('receiver')
        before = list(sys.path)
        handler = signal.getsignal(signal.SIGTERM)
        with a._context():
            self.assertIs(sys.modules['receiver'], a._aliases['receiver'])
            self.assertEqual(sys.modules['run_preflight'].__name__, 'historical_inert_preflight')
        self.assertIs(sys.modules.get('receiver'), old)
        self.assertEqual(sys.path, before)
        self.assertEqual(signal.getsignal(signal.SIGTERM), handler)

    def test_no_old_main_or_calibration_training_or_writer_called(self):
        tree = ast.parse((HERE / 'historical_receiver.py').read_text())
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)]
        self.assertFalse(set(calls) & {'main', 'calibrate', 'development', 'observations', 'write_json',
                                      'torch_save', 'save_tensor', 'backward', 'step', 'transmit_truth'})

    def test_step2_receiver_calls_never_receive_truth_or_clean(self):
        tree = ast.parse((HERE / 'historical_receiver.py').read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Step2Adapter')
        infer = [n for n in ast.walk(klass) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'infer']
        self.assertEqual(len(infer), 2)
        for call in infer:
            self.assertFalse({k.arg for k in call.keywords} & {'truth', 'clean'})

    def test_unsupported_failed_revision_not_silently_rerun(self):
        for study in ('RX_STEP1_V1', 'RX_STEP1_V2', 'RX_STEP2_C'):
            with self.assertRaisesRegex(ValueError, 'unsupported completed'):
                rx.create_adapter(self.root, study)


if __name__ == '__main__':
    unittest.main()
