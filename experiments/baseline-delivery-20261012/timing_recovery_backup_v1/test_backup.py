"""Pure CPU tests: explicit scope and no active timing archive members."""
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
entry=HERE/'backup.py'
if not entry.exists():entry=HERE/'timing_recovery_backup.py'
spec=importlib.util.spec_from_file_location('_timing_recovery_backup_under_test',entry)
backup=importlib.util.module_from_spec(spec);spec.loader.exec_module(backup)


class Scope(unittest.TestCase):
    def test_active_v6_runtime_is_always_rejected(self):
        for name in ('case_reservations.json','worker.log','resource_snapshots.jsonl','completion.json'):
            with self.assertRaises(RuntimeError):backup.archive_name(backup.TMP/'six_method_timing_v6'/name)

    def test_v6_only_immutable_file_allowlist(self):
        for name in backup.V6_STATIC_NAMES:
            self.assertTrue(backup.archive_name(backup.RESULTS/'six_method_v6'/name).startswith('repo/'))
        for name in ('launch.json','actual_wait.json','running_status.json','completion.json','timing_summary.csv'):
            with self.assertRaises(RuntimeError):backup.archive_name(backup.RESULTS/'six_method_v6'/name)

    def test_old_failure_and_static_paths_remain_distinct(self):
        self.assertEqual(backup.archive_name(backup.TMP/'six_method_timing_v5/case_reservations.json'),
            'runtime/six_method_timing_v5/case_reservations.json')
        p=backup.RUNTIME/'assets/static.npz'
        self.assertEqual(backup.archive_name(p),'legacy_assets/assets/static.npz')

    def test_external_path_cannot_enter_package(self):
        with self.assertRaises(RuntimeError):backup.archive_name(Path('/etc/passwd'))


if __name__=='__main__':unittest.main(verbosity=2)
