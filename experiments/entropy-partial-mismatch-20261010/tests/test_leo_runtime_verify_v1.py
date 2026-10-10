"""Tiny byte/path/race fixtures only; Linux fd tests never import a framework."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import leo_runtime_verify_v1 as verifier


def row(path, data=b'original bytes'):
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


class SchemaTests(unittest.TestCase):
    def test_normal_absolute_rows_and_provenance_are_copied_without_io(self):
        original = row('/personal/nested/file') | {'original_path': '/original/file'}
        with mock.patch.object(verifier.os, 'open', side_effect=AssertionError('No filesystem access')):
            rows, base = verifier.validate_runtime_rows([original], '/personal')
        self.assertEqual(str(base), '/personal')
        self.assertEqual(rows, [row('/personal/nested/file')])
        self.assertEqual(original['original_path'], '/original/file')
        original['path'] = '/changed'
        self.assertEqual(rows[0]['path'], '/personal/nested/file')

    def test_parent_escape_and_noncanonical_spelling_fail_before_io(self):
        paths = ['/personal/../outside', '/personal/a/../b', '/personal//b',
                 '/personal/./b', '/personal/b/', '//personal/b', 'personal/b',
                 '/personal/a\\b', '/personal/a\0b', '/personal_other/b']
        for path in paths:
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                verifier.validate_runtime_rows([row(path)], '/personal')

    def test_scattered_input_is_path_sorted_without_mutating_source_order(self):
        original = [row('/personal/b/x'), row('/personal/a/y'), row('/personal/a/x')]
        original_paths = [r['path'] for r in original]
        result, _ = verifier.validate_runtime_rows(original, '/personal')
        self.assertEqual([r['path'] for r in result], sorted(original_paths))
        self.assertEqual([r['path'] for r in original], original_paths)

    def test_invalid_base_and_base_as_leaf_rejected(self):
        for base in ['/', 'personal', '/personal/../root', '/personal/']:
            with self.subTest(base=base), self.assertRaises(RuntimeError):
                verifier.validate_runtime_rows([row('/personal/file')], base)
        with self.assertRaises(RuntimeError):
            verifier.validate_runtime_rows([row('/personal')], '/personal')

    def test_bad_pin_and_duplicate_file_rejected(self):
        good = row('/personal/file')
        for changes in [{'bytes': -1}, {'bytes': True}, {'sha256': 'X'*64}, {'sha256': ''}]:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                verifier.validate_runtime_rows([good | changes], '/personal')
        with self.assertRaises(RuntimeError):
            verifier.validate_runtime_rows([good, good], '/personal')

    def test_all_rows_validated_before_any_open(self):
        with mock.patch.object(verifier.os, 'open', side_effect=AssertionError('No filesystem access')):
            with self.assertRaisesRegex(RuntimeError, 'outside personal base'):
                verifier.verify_runtime_rows([row('/personal/a'), row('/outside/b')], '/personal')

    def test_non_linux_production_execution_rejected(self):
        with mock.patch.object(verifier.sys, 'platform', 'win32'):
            with self.assertRaisesRegex(RuntimeError, 'requires Linux'):
                verifier.verify_runtime_rows([row('/personal/a')], '/personal')


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux openat/O_NOFOLLOW fixtures')
class LinuxFileTests(unittest.TestCase):
    def make_file(self, root, relative='packages/torch/lib/native.so', data=b'original bytes'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path, row(path, data)

    def mutate_after_first_read(self, operation):
        original = os.read
        called = [False]
        def read(fd, count):
            data = original(fd, count)
            if data and not called[0]:
                called[0] = True
                operation()
            return data
        return mock.patch.object(verifier.os, 'read', side_effect=read)

    def test_nested_regular_and_empty_files_all_bytes_hashed(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _, one = self.make_file(base, data=b'a'*(verifier.READ_BYTES+17))
            _, empty = self.make_file(base, 'empty', b'')
            result = verifier.verify_runtime_rows([one, empty], base)
            self.assertEqual(result['verified_files'], 2)
            self.assertEqual(result['verified_bytes'], one['bytes'])
            self.assertTrue(result['all_file_bytes_hashed'])
            self.assertFalse(result['numerical_qualification'])

    def test_bad_sha_size_and_directory_leaf_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td);path, pin = self.make_file(base)
            for changes in [{'sha256': '0'*64}, {'bytes': pin['bytes']+1},
                            {'path': str(path.parent), 'bytes': path.parent.stat().st_size}]:
                with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                    verifier.verify_runtime_rows([pin | changes], base)

    def test_symlink_base_ancestor_and_leaf_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td);base = root/'personal';base.mkdir()
            target, pin = self.make_file(base)
            leaf = base/'leaf';leaf.symlink_to(target)
            ancestor = base/'ancestor';ancestor.symlink_to(target.parent, target_is_directory=True)
            alias_base = root/'alias';alias_base.symlink_to(base, target_is_directory=True)
            for path, scope in [(leaf, base), (ancestor/target.name, base),
                                (alias_base/target.relative_to(base), alias_base)]:
                with self.subTest(path=path), self.assertRaises((RuntimeError, OSError)):
                    verifier.verify_runtime_rows([pin | {'path': str(path)}], scope)

    def test_fifo_leaf_never_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td);path = base/'fifo';os.mkfifo(path)
            with self.assertRaisesRegex(RuntimeError, 'not regular'):
                verifier.verify_runtime_rows([row(path, b'')], base)

    def test_same_byte_leaf_replacement_during_read_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td);path, pin = self.make_file(base)
            def replace():
                replacement = path.with_name('replacement')
                replacement.write_bytes(b'original bytes')
                os.replace(replacement, path)
            with self.mutate_after_first_read(replace):
                with self.assertRaisesRegex(RuntimeError, 'replaced while reading'):
                    verifier.verify_runtime_rows([pin], base)

    def test_same_byte_directory_replacement_during_read_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td);path, pin = self.make_file(base)
            def replace():
                parent = path.parent
                parent.rename(parent.with_name('retained_old'))
                parent.mkdir()
                path.write_bytes(b'original bytes')
            with self.mutate_after_first_read(replace):
                with self.assertRaisesRegex(RuntimeError, 'Directory namespace changed'):
                    verifier.verify_runtime_rows([pin], base)

    def test_in_place_change_during_read_rejected_even_if_read_bytes_were_original(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td);path, pin = self.make_file(base)
            with self.mutate_after_first_read(lambda: path.write_bytes(b'x'*pin['bytes'])):
                with self.assertRaisesRegex(RuntimeError, 'changed or was replaced'):
                    verifier.verify_runtime_rows([pin], base)

    def test_cache_is_bounded_and_scattered_rows_verified_without_fd_leaks(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            pins = [self.make_file(base, f'd{i}/value', b'x')[1] for i in range(258)]
            pins.append(self.make_file(base, 'd0/second', b'y')[1])
            before = len(os.listdir('/proc/self/fd'))
            result = verifier.verify_runtime_rows(pins, base)
            self.assertEqual(result['verified_files'], 259)
            self.assertEqual(result['verified_bytes'], 259)
            self.assertEqual(result['max_cached_directory_fds'], 256)
            self.assertEqual(len(os.listdir('/proc/self/fd')), before)
            bad = pins[-1] | {'sha256': '0'*64}
            with self.assertRaises(RuntimeError):
                verifier.verify_runtime_rows(pins[:-1]+[bad], base)
            self.assertEqual(len(os.listdir('/proc/self/fd')), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
