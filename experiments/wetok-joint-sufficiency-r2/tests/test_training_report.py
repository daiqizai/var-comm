import copy
import importlib.util
from pathlib import Path
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('r2_training_report_fixture', EXPERIMENT / 'scripts/report_training.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def fixture():
    config = {'variants': ['single_pass'], 'initial_total_updates': 1, 'inherited_full_steps': [0, 1], 'new_full_steps': [2],
        'unchanged_loss': {'mse': 1., 'lpips': .01, 'bits': .01, 'state': .01}}
    history = [{'step': step, 'loss': .016, 'mse': .01, 'lpips': .2, 'bits': .3, 'state': .1, 'bit_error_rate': .2,
        'step_seconds': 3., 'peak_GPU_allocated_bytes': 1024 ** 3} for step in (1, 2)]
    saved = {'completed_total_updates': 2, 'histories': {'single_pass': history}, 'selected': {'single_pass': {'step': 1}},
        'timings': {'training': {'single_pass': 3.}, 'calibration': {'single_pass': 2.}}, 'elapsed_seconds': 6.,
        'summaries': [{'variant': 'single_pass', 'scope': 'full', 'step': step, 'snr_db': 1., 'source_images': 1000} for step in (0, 1, 2)]}
    metadata = {'prior_training_and_calibration_seconds_by_arm': {'single_pass': {'training': 20., 'calibration': 5.}}}
    return config, saved, metadata


class TrainingReportTests(unittest.TestCase):
    def test_cost_excludes_historical_components_and_keeps_process_overhead(self):
        config, saved, metadata = fixture()
        rows, total = REPORT.continuation_cost(saved, metadata, config)
        self.assertAlmostEqual(rows[0]['R2_recorded_component_hours'], 5 / 3600)
        self.assertAlmostEqual(rows[0]['historical_branch_training_hours'], 20 / 3600)
        self.assertEqual(rows[0]['selected_step'], 1)
        self.assertAlmostEqual(total['R2_unattributed_process_hours'], 1 / 3600)
        self.assertTrue(total['historical_shared_parent7000_training_excluded_not_free'])

    def test_cost_rejects_missing_updates_or_inconsistent_timers(self):
        config, saved, metadata = fixture()
        for mutation in ('elapsed', 'component', 'updates'):
            changed = copy.deepcopy(saved)
            if mutation == 'elapsed':
                changed['elapsed_seconds'] = 1.
            elif mutation == 'component':
                changed['timings']['training']['single_pass'] = 10.
            else:
                changed['histories']['single_pass'].pop()
            with self.assertRaises(RuntimeError):
                REPORT.continuation_cost(changed, metadata, config)

    def test_training_bins_validate_actual_loss_not_scalar_dominance_claims(self):
        config, saved, metadata = fixture()
        rows = REPORT.training_bins(saved, config, width=1)
        self.assertEqual([row['last_step'] for row in rows], [1, 2])
        saved['histories']['single_pass'][1]['loss'] += .01
        with self.assertRaises(RuntimeError):
            REPORT.training_bins(saved, config)

    def test_full_calibration_cannot_be_replaced_by_monitor_subset(self):
        config, saved, metadata = fixture()
        base = {'channel': {'snrs_db': [1.]}}
        self.assertEqual(len(REPORT.complete_calibration(saved, config, base)), 3)
        saved['summaries'][-1]['scope'] = 'monitor'
        saved['summaries'][-1]['source_images'] = 100
        with self.assertRaises(RuntimeError):
            REPORT.complete_calibration(saved, config, base)


if __name__ == '__main__':
    unittest.main()
