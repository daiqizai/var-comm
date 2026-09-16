import copy
import unittest

from evaluation_fixtures import driver, fixture
from vector_control.evaluation import learned_names, statistics, support_statistics, validate_rows, validate_selection
from vector_control.references import project_reference
from vector_control.timing import archive_origin, measurement_order, summarize, validate_config, validate_rows as validate_timing

FINISHER = driver('finish')


class EvaluationContractTests(unittest.TestCase):
    def test_all23_methods_and_matched_vector_vs_other_histories(self):
        evaluation, config, r2, original, grid, reference, base, rows = fixture()
        lookup, names = validate_rows(rows, config, r2, original, grid, reference, base, 2)
        self.assertEqual((len(rows), len(names)), (966, 23))
        summary, paired = statistics(rows, config, r2, original, grid, reference, base, 2)
        self.assertEqual(len(paired), 1056)
        self.assertEqual(sum(row['comparison_scope'] == 'matched10000_vector_construction_parameter_and_E_call_comparison' for row in paired), 48)
        self.assertEqual(sum(row['comparison_scope'] == 'same10000_structure_reference_not_equal_parameters_or_E_calls' for row in paired), 144)
        for row in paired:
            self.assertAlmostEqual(row['delta'], -.005 if row['metric'] in ('lpips', 'LPIPS_excess_from_native') else 0.)
        support = [dict(row, arm='perceptual_deepjscc_fixed_support') for row in rows
            if row['arm'] == 'perceptual_deepjscc' and float(row['snr_db']) in (5., 6.)]
        unused, support_paired = support_statistics(rows, support, config, r2, original, grid, reference, base, 2)
        self.assertEqual(len(support_paired), 288)
        self.assertEqual(sum(row['measurement_scope'] == 'new_R3_comparison' for row in support_paired), 18)

    def test_missing_rows_wrong_budget_noise_time_or_opportunity_are_rejected(self):
        evaluation, config, r2, original, grid, reference, base, rows = fixture()
        with self.assertRaises(RuntimeError):
            validate_rows(rows[:-1], config, r2, original, grid, reference, base, 2)
        for key, value in (('total_complex_uses', 3061), ('total_energy', 7000), ('noise_sha256', 'other_noise'),
            ('receiver_seconds', .01), ('new_update_opportunity', 5000), ('communication_parameters', 10)):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            with self.assertRaises(RuntimeError):
                validate_rows(changed, config, r2, original, grid, reference, base, 2)

    def test_earlier_qualified_selection_and_unchanged_reference_projection(self):
        evaluation, config, r2, original, grid, reference, base, rows = fixture()
        for row in rows:
            if row['arm'].startswith('r3__'):
                row['selected_step'], row['selected_global_data_step'] = 1000, 8000
        milestone = {'selected': {'step': 1000, 'checkpoint_sha256': 'selected_checkpoint'}}
        validate_selection(rows, milestone)
        rows[0]['selected_step'] = 5000
        with self.assertRaises(RuntimeError):
            validate_selection(rows, milestone)
        old = {'arm': 'r2__full_grid_innovation', 'quality_origin': 'new_r2_measurement', 'r2_quality_origin': 'new_r2_measurement',
            'receiver_seconds': '', 'online_TX_seconds': '', 'archived_receiver_seconds': '.05'}
        before = copy.deepcopy(old)
        projected = project_reference(old, 'sealed_R2')
        self.assertEqual(old, before)
        self.assertTrue(all(projected[key] == value for key, value in before.items()))
        old['receiver_seconds'] = '.04'
        with self.assertRaises(RuntimeError):
            project_reference(old, 'sealed_R2')

    def test_timing_rotation_full_matrix_and_105_scoped_intervals(self):
        evaluation, config, r2, original, grid, reference, base, unused = fixture()
        names = learned_names(config, r2, original, grid, reference)
        rows = []
        for position, index in enumerate(evaluation['timing_source_indices']):
            for snr_position, snr in enumerate(evaluation['timing_snrs_db']):
                order = measurement_order(names, position * 5 + snr_position)
                for name in names:
                    rows.append({'image_index': index, 'image_id': f'source_{index}', 'snr_db': snr, 'seed': 2001, 'arm': name,
                        'receiver_order_index': order.index(name), 'total_complex_uses': 3060, 'total_energy': 6120.,
                        'online_TX_seconds': .02, 'receiver_seconds': .05 + index * .00001,
                        'timing_scope': 'separate_matched_uncontended_RX_including_visual_Decoder'})
        summary, paired = summarize(rows, evaluation, names, base, config)
        self.assertEqual((len(rows), len(summary), len(paired)), (2560, 112, 105))
        self.assertEqual(sum(row['comparison_scope'] == 'matched10000_vector_construction_parameter_and_E_call_comparison' for row in paired), 7)
        for name in names:
            self.assertEqual({measurement_order(names, frame).index(name) for frame in range(16)}, set(range(16)))
        with self.assertRaises(RuntimeError):
            validate_timing(rows[:-1], evaluation, names)
        for key, value in (('total_energy', 7000), ('receiver_seconds', float('nan')), ('receiver_order_index', -1)):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            with self.assertRaises(RuntimeError):
                validate_timing(changed, evaluation, names)
        self.assertEqual([archive_origin(name) for name in ('r3__new', 'r2__old', 'grid__old', 'joint__old', 'frozen__old')], ['r3', 'r2', 'grid', 'joint', 'joint'])

    def test_only_exact_owned_resource_guard_errors_are_retryable(self):
        for stage, message in (('quality', 'insufficient free GPU memory between quality source images'),
            ('timing', 'other GPU compute workloads are present: [12, 34]; not stopping them')):
            record = {'pid': 99, 'status': 'R3_' + stage.upper() + '_FAILED_OR_INTERRUPTED'}
            log = 'Traceback\nRuntimeError: ' + message + '\n'
            self.assertTrue(FINISHER.retryable_guard(stage, 1, log, record, 99))
            self.assertFalse(FINISHER.retryable_guard(stage, 1, log, record, 98))
            self.assertFalse(FINISHER.retryable_guard(stage, -9, log, record, 99))
            self.assertFalse(FINISHER.retryable_guard(stage, 1, log + 'RuntimeError: changed source\n', record, 99))
            self.assertFalse(FINISHER.retryable_guard('analysis', 1, log, record, 99))


if __name__ == '__main__':
    unittest.main()
