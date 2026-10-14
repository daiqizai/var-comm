"""Independent CLI-order qualification, synthetic files and subprocesses only."""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).with_name('build.py')
SYNTHETIC_BUILD = '''def build(args):
    # This stub replaces only the production orchestration. The original
    # save/deduplicate/inner_pins/verify_archive helpers execute unchanged.
    folder = Path(os.environ['R1_SYNTHETIC_CLI_FOLDER']).resolve()
    require(folder.is_dir(), 'Synthetic fixture is missing')
    # The original server's standard library is also on PFS. Preload only
    # gzip's standard-library dependencies before auditing actual file opens;
    # do not add the runtime tree to the allowed PFS roots.
    import gzip
    require(not any(n in sys.modules for n in ('torch', 'numpy', 'tensorflow', 'sionna', 'lpips')), 'Scientific module imported')
    def forbidden_process(*args, **kwargs):
        raise AssertionError('No nested process or network operation is allowed')
    subprocess.run = forbidden_process
    subprocess.Popen = forbidden_process
    subprocess.check_output = forbidden_process
    def fixture_open_only_on_pfs(value):
        # Server TMPDIR is itself on PFS. Allow this one fixture's resolved
        # subtree, never the parent, a same-prefix sibling or a link escape.
        absolute = Path(os.path.abspath(os.fsdecode(value)))
        def on_pfs(path):
            name = path.as_posix()
            return '/mnt/pfs/' in name or name.endswith('/mnt/pfs')
        if on_pfs(absolute):
            require(absolute.is_relative_to(folder), 'Real scientific filesystem read forbidden')
        resolved = absolute.resolve()
        if on_pfs(absolute) or on_pfs(resolved):
            require(resolved.is_relative_to(folder), 'Real scientific filesystem read forbidden')
    def audit(event, args):
        if event.startswith(('socket.', 'subprocess.')):
            raise AssertionError('No network or nested process operation is allowed')
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            fixture_open_only_on_pfs(args[0])
    sys.addaudithook(audit)
    forbidden = os.environ.get('R1_SYNTHETIC_FORBIDDEN_OPEN')
    if forbidden:
        # This is an actual open attempt; only the installed audit hook may
        # reject it. The test never substitutes or swallows the rejection.
        with open(forbidden, 'rb') as stream:
            stream.read(1)
    payload = folder/'data.json'
    save(payload, dict(synthetic=True, scientific_calls=0))
    records, aliases = deduplicate([(payload, 'evidence/data.json')])
    require(len(records) == 1 and aliases == [], 'Synthetic deduplication differs')
    require(list(inner_pins(dict(item=pin(payload)))) == [pin(payload)], 'Synthetic nested pins differ')
    manifest = folder/'manifest.json'
    save(manifest, dict(files=records, synthetic=True))
    archive = folder/'fixture.tar.gz'
    with tarfile.open(archive, 'w:gz', dereference=True) as tar:
        tar.add(payload, arcname='evidence/data.json', recursive=False)
        tar.add(manifest, arcname='manifest.json', recursive=False)
    verify_archive(archive, manifest, records)
    return dict(status='SYNTHETIC_CLI_HELPERS_COMPLETE', helpers=['save', 'deduplicate', 'inner_pins', 'verify_archive'], scientific_calls=0)
'''


def modified_source(force_bad_order=False):
    source = SOURCE.read_text(encoding='utf8')
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build']
    if len(nodes) != 1:
        raise AssertionError('Exactly one production build definition is required')
    original = ast.get_source_segment(source, nodes[0])
    if source.count(original) != 1:
        raise AssertionError('Production build replacement must be unique')
    source = source.replace(original, SYNTHETIC_BUILD.rstrip())
    if force_bad_order:
        tree = ast.parse(source)
        guards = [node for node in tree.body if isinstance(node, ast.If)
                  and ast.dump(node.test) == ast.dump(ast.parse("__name__ == '__main__'", mode='eval').body)]
        if len(guards) != 1:
            raise AssertionError('Exactly one original CLI guard is required')
        main = ast.get_source_segment(source, guards[0])
        source = source.replace(main, '', 1)
        tree = ast.parse(source)
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'inner_pins')
        lines = source.splitlines(keepends=True)
        lines.insert(helper.lineno - 1, main + '\n\n')
        source = ''.join(lines)
    return source


class CLIOrderTests(unittest.TestCase):
    def execute_fixture(self, force_bad_order, simulated_pfs=False, forbidden=None):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            folder = base/'mnt/pfs/synthetic_fixture' if simulated_pfs else base
            folder.mkdir(parents=True, exist_ok=True)
            script = folder/'synthetic_build_cli.py'
            script.write_text(modified_source(force_bad_order), encoding='utf8')
            env = dict(os.environ, R1_SYNTHETIC_CLI_FOLDER=str(folder),
                       CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
            env.pop('R1_SYNTHETIC_FORBIDDEN_OPEN', None)
            if forbidden:
                if forbidden == 'same_prefix_sibling':
                    target = folder.with_name(folder.name+'_other')/'forbidden.txt'
                    target.parent.mkdir();target.write_text('SYNTHETIC OUTSIDE FIXTURE', encoding='utf8')
                elif forbidden == 'traversal':
                    outside = folder.parent/'forbidden.txt'
                    outside.write_text('SYNTHETIC OUTSIDE FIXTURE', encoding='utf8')
                    target = folder/'..'/'forbidden.txt'
                else:
                    target = Path(forbidden)
                env['R1_SYNTHETIC_FORBIDDEN_OPEN'] = str(target)
            result = subprocess.run(
                [sys.executable, '-B', str(script), '--authorization-path', '/synthetic/never-read',
                 '--authorization-sha256', '0'*64, '--authorization-bytes', '0'],
                cwd=folder, env=env, capture_output=True, text=True, encoding='utf8', timeout=20)
            return result, (folder/'fixture.tar.gz').is_file()

    def test_real_cli_order_reaches_original_file_helpers(self):
        result, archive_exists = self.execute_fixture(False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(archive_exists)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt['status'], 'SYNTHETIC_CLI_HELPERS_COMPLETE')
        self.assertEqual(receipt['helpers'], ['save', 'deduplicate', 'inner_pins', 'verify_archive'])
        self.assertEqual(receipt['scientific_calls'], 0)

    def test_exact_synthetic_fixture_on_pfs_is_allowed(self):
        result, archive_exists = self.execute_fixture(False, simulated_pfs=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(archive_exists)

    def test_same_prefix_sibling_and_traversal_are_rejected(self):
        for forbidden in ('same_prefix_sibling', 'traversal'):
            with self.subTest(forbidden=forbidden):
                result, archive_exists = self.execute_fixture(False, simulated_pfs=True, forbidden=forbidden)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Real scientific filesystem read forbidden', result.stderr)
                self.assertNotIn('FileNotFoundError', result.stderr)
                self.assertFalse(archive_exists)

    def test_real_repository_runtime_model_and_source_paths_remain_rejected(self):
        for suffix in (
                'code/liulu/code/VAR_COMM/results/synthetic_denied.json',
                'liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013/r1_cache_recovery_v1/visual/images/synthetic_denied.npz',
                'liberai-checkpoint/liulu/synthetic_denied.pt',
                'code/liulu/var_comm_runtime_20261010/assets/source_restore/synthetic_denied.npz'):
            with self.subTest(suffix=suffix):
                target='/mnt/pfs/pfs-yc2F4O/modelTeam/'+suffix
                result, archive_exists = self.execute_fixture(False, simulated_pfs=True, forbidden=target)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Real scientific filesystem read forbidden', result.stderr)
                self.assertNotIn('FileNotFoundError', result.stderr)
                self.assertFalse(archive_exists)

    def test_old_early_main_order_is_detected_by_actual_subprocess(self):
        result, archive_exists = self.execute_fixture(True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('NameError', result.stderr)
        self.assertIn("name 'save' is not defined", result.stderr)
        self.assertFalse(archive_exists)


if __name__ == '__main__':
    unittest.main()
