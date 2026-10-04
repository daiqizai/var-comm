"""Real-file tests for the nested manifest contract and publication boundaries."""
import copy
from pathlib import Path
import tempfile
import unittest

import recover_delivery_v1 as r


def manifest_fixture(folder):
    result = Path(folder)
    (result / 'figures').mkdir(parents=True)
    (result / 'REPORT.md').write_text('# Existing report\n', encoding='utf-8')
    (result / 'figures/comparison.png').write_bytes(b'fixture-image-content')
    return dict(status='STATIC_REPORT_COMPLETE', model_inference=False, metric_recomputation=False,
                training_updates=0, outputs={name: dict(sha256=r.sha(result / name), bytes=(result / name).stat().st_size)
                for name in ('REPORT.md', 'figures/comparison.png')})


class ManifestTests(unittest.TestCase):
    def test_valid_builder_nested_manifest_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = manifest_fixture(tmp)
            paths = r.validate_manifest(tmp, manifest)
            self.assertEqual(set(paths), {Path(tmp) / 'REPORT.md', Path(tmp) / 'figures/comparison.png'})
            self.assertTrue(isinstance(manifest['outputs']['REPORT.md'], dict))

    def test_modified_file_is_rejected_even_when_size_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = manifest_fixture(tmp)
            path = Path(tmp) / 'figures/comparison.png'
            before = path.read_bytes()
            path.write_bytes(b'X' + before[1:])
            with self.assertRaisesRegex(RuntimeError, 'SHA256 differs'):
                r.validate_manifest(tmp, manifest)

    def test_exact_byte_count_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = manifest_fixture(tmp)
            manifest['outputs']['REPORT.md']['bytes'] += 1
            with self.assertRaisesRegex(RuntimeError, 'byte count differs'):
                r.validate_manifest(tmp, manifest)

    def test_absolute_parent_and_windows_escape_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = manifest_fixture(tmp)
            for name in ('../outside.txt', 'figures/../../outside.txt', '/etc/passwd', 'C:/outside.txt',
                         'C:outside.txt', '..\\outside.txt', 'figures/../REPORT.md', './REPORT.md', 'figures//comparison.png'):
                manifest = copy.deepcopy(base)
                manifest['outputs'][name] = dict(base['outputs']['REPORT.md'])
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'strict relative'):
                    r.validate_manifest(tmp, manifest)

    def test_symlink_escape_is_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); result = folder / 'result'; manifest = manifest_fixture(result)
            outside = folder / 'outside.bin'; outside.write_bytes(b'outside')
            link = result / 'figures/link.bin'
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest('Symlink creation is unavailable on this host')
            manifest['outputs']['figures/link.bin'] = dict(sha256=r.sha(outside), bytes=outside.stat().st_size)
            with self.assertRaisesRegex(RuntimeError, 'escapes'):
                r.validate_manifest(result, manifest)

    def test_old_bare_digest_and_invalid_metadata_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = manifest_fixture(tmp)
            for replacement in (base['outputs']['REPORT.md']['sha256'],
                                dict(sha256='x' * 64, bytes=1),
                                dict(sha256=base['outputs']['REPORT.md']['sha256'], bytes=True),
                                dict(sha256=base['outputs']['REPORT.md']['sha256'], bytes=r.LIMIT)):
                manifest = copy.deepcopy(base); manifest['outputs']['REPORT.md'] = replacement
                with self.assertRaisesRegex(RuntimeError, 'metadata'):
                    r.validate_manifest(tmp, manifest)

    def test_incomplete_or_computed_report_cannot_be_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = manifest_fixture(tmp)
            for update in (dict(status='RUNNING'), dict(model_inference=True), dict(metric_recomputation=True)):
                manifest = dict(base, **update)
                with self.assertRaises(RuntimeError):
                    r.validate_manifest(tmp, manifest)

    def test_report_mirror_changes_links_only(self):
        text = '# Same\n![x](figures/a.png)\n[table](MAIN_TABLE.csv)\n[web](https://example.com)\n'
        value = r.mirror_report(text)
        self.assertIn('# Same\n', value)
        self.assertIn('](../results/m1_n2048_full_grid_20261004/figures/a.png)', value)
        self.assertIn('](../results/m1_n2048_full_grid_20261004/MAIN_TABLE.csv)', value)
        self.assertIn('[web](https://example.com)', value)


class CompletionTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder); out = root / 'outputs/run'; out.mkdir(parents=True)
        result = root / 'results/report'; manifest = manifest_fixture(result)
        (out / 'metrics').mkdir()
        files = ['registration.json', 'm1_policy.json', 'metrics/registration.json', 'cal.data', 'dev.data', 'score.data']
        for name in files:
            (out / name).write_text('fixture:' + name, encoding='utf-8')
        common = dict(status='COMPLETE', synthetic=False, training_updates=0,
                      registration_sha256=r.sha(out / 'registration.json'), policy_sha256=r.sha(out / 'm1_policy.json'))
        cal = dict(common, stage='calibrate', sources=1000, physical_frames=2070000,
                   outputs={str(out / 'cal.data'): r.sha(out / 'cal.data')})
        dev = dict(common, stage='development', sources=100, method_rows=24000,
                   outputs={str(out / 'dev.data'): r.sha(out / 'dev.data')})
        score = dict(status='M1_N2048_METRICS_COMPLETE', synthetic=False, training_updates=0, sources=100, rows=24000,
                     registration_sha256=r.sha(out / 'metrics/registration.json'),
                     outputs={str(out / 'score.data'): r.sha(out / 'score.data')},
                     input_bindings={str(out / 'm1_policy.json'): r.sha(out / 'm1_policy.json')})
        for name, value in [('calibrate_completion.json', cal), ('development_completion.json', dev), ('score_completion.json', score)]:
            r.write(out / name, value)
        publisher = root / 'publisher.py'; publisher.write_text('# frozen original publisher\n', encoding='utf-8')
        r.write(out / 'delivery_config.json', dict(root=str(root), out=str(out), bindings={str(publisher): r.sha(publisher)},
                                                 publish_source_paths=['publisher.py']))
        manifest.update(sources=100, method_rows=24000, score_completion_sha256=r.sha(out / 'score_completion.json'),
                        policy_sha256=r.sha(out / 'm1_policy.json'),
                        input_bindings={str(out / 'score_completion.json'): r.sha(out / 'score_completion.json')})
        r.write(result / 'MANIFEST.json', manifest)
        return root, out, result

    def test_complete_artifacts_validate_without_loading_scientific_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out, result = self.fixture(tmp)
            config, manifest, paths = r.validate_completed(root, out, result)
            self.assertEqual(config['out'], str(out))
            self.assertEqual(manifest['status'], 'STATIC_REPORT_COMPLETE')
            self.assertIn(result / 'MANIFEST.json', paths)

    def test_changed_completed_stage_output_blocks_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out, result = self.fixture(tmp)
            (out / 'dev.data').write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'SHA256 differs'):
                r.validate_completed(root, out, result)

    def test_changed_report_input_binding_blocks_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out, result = self.fixture(tmp)
            manifest = r.read(result / 'MANIFEST.json')
            manifest['input_bindings'][str(out / 'cal.data')] = '0' * 64
            r.write(result / 'MANIFEST.json', manifest)
            with self.assertRaisesRegex(RuntimeError, 'SHA256 differs'):
                r.validate_completed(root, out, result)

    def test_existing_publication_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out, _ = self.fixture(tmp)
            r.write(out / 'publication_completion.json', dict(status='PUSHED_AND_STOPPED'))
            before = (out / 'publication_completion.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'already has'):
                r.run(root, out)
            self.assertEqual((out / 'publication_completion.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
