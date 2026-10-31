"""CPU-only namespace regression; no real model, PHY or timing measurement."""
import sys
import types
import unittest
from pathlib import Path
from timing_phy_namespace_v2 import isolated_engine_with_explicit_sys


class NamespaceTests(unittest.TestCase):
    def make_function(self, with_sys=False):
        ns = {'R': Path('/registered/root'), 'sentinel': object()}
        if with_sys:
            ns['sys'] = sys
        exec('def isolated_engine():\n return sys, R, sentinel\n', ns)
        return ns['isolated_engine']

    def test_missing_binding_is_reproduced_and_original_unchanged(self):
        original = self.make_function()
        with self.assertRaises(NameError):
            original()
        before = dict(original.__globals__)
        result = isolated_engine_with_explicit_sys(original, '/registered/root')
        self.assertIs(result[0], sys)
        self.assertEqual(result[1], Path('/registered/root'))
        self.assertIs(result[2], original.__globals__['sentinel'])
        self.assertEqual(original.__globals__, before)
        with self.assertRaises(NameError):
            original()

    def test_existing_binding_keeps_result_identity(self):
        original = self.make_function(True)
        self.assertEqual(original(), isolated_engine_with_explicit_sys(original, '/registered/root'))

    def test_root_mismatch_refused(self):
        with self.assertRaises(AssertionError):
            isolated_engine_with_explicit_sys(self.make_function(), '/wrong/root')


if __name__ == '__main__':
    unittest.main()
