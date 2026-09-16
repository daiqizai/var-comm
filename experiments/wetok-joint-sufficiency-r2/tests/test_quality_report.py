import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from test_quality_contracts import EXPERIMENT, fixture
from sufficiency.evaluation import new_names, statistics

SPEC = importlib.util.spec_from_file_location('r2_quality_report_fixture', EXPERIMENT / 'scripts/report_quality.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class QualityReportTests(unittest.TestCase):
    def setUp(self):
        self.config, self.original, self.grid, self.reference, self.base, self.rows = fixture()
        self.summary, self.paired = statistics(self.rows, self.config, self.original, self.grid, self.reference, self.base, 2)
        for row in self.summary:
            row['source_images'] = 100

    def display(self, summary=None, paired=None):
        return REPORT.display_tables(self.summary if summary is None else summary, self.paired if paired is None else paired,
            self.config, self.original, self.grid, self.reference, self.base)

    def test_both_digital_families_and_deep_are_kept_and_scopes_are_separate(self):
        primary, supported, matched, training, systems = self.display()
        self.assertEqual([len(values) for values in (primary, supported, matched, training, systems)], [8, 40, 6, 4, 16])
        self.assertEqual({row['arm'] for row in primary}, set(new_names(self.config)) | set(REPORT.SYSTEM_CONTROLS))
        self.assertEqual({float(row['snrs_db']) for row in supported}, {1., 4., 7., 13., 19.})

    def test_omitted_control_or_relabeled_training_scope_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.display(summary=[row for row in self.summary if row['arm'] != 'wetok_8PSK_FEC'])
        changed = copy.deepcopy(self.paired)
        row = next(row for row in changed if row['comparison_scope'] == 'extra_training_effect_not_new_mechanism' and row['metric'] == 'lpips')
        row['comparison_scope'] = 'matched10000_structure_comparison'
        with self.assertRaises(RuntimeError):
            self.display(paired=changed)

    def test_actual_earlier_selection_is_disclosed_not_replaced_by_available_budget(self):
        selected = REPORT.selected_table(self.rows, self.config)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(row['selected_step'] == 5000 and row['available_total_updates'] == 10000 for row in selected))
        self.assertTrue(all(row['selected_checkpoint_from_previous_trial'] for row in selected))
        changed = copy.deepcopy(self.rows)
        changed[0]['checkpoint_sha256'] = 'unregistered_selection'
        with self.assertRaises(RuntimeError):
            REPORT.selected_table(changed, self.config)

    def test_all_four_predeclared_figures_render_to_png_and_pdf(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            REPORT.draw_figures(output, self.display(), self.config, self.base)
            expected = {stem + suffix for stem in ('supported_snr_quality', 'matched_structure_primary_lpips',
                'extra_training_primary_lpips', 'system_controls_primary_lpips') for suffix in ('.png', '.pdf')}
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            self.assertTrue(all(path.stat().st_size > 0 for path in output.iterdir()))


if __name__ == '__main__':
    unittest.main()
