"""Engineering checks for the runtime publication gate; no Git writes."""
import copy
from pathlib import Path
import tempfile
import unittest
import publish_source as p

class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);base=root/'base.py';source=root/'speed.py'
        base.write_text('base');source.write_text('speed')
        self.bound={str(source):p.sha(source)};self.base=base
        self.q=dict(status='QUALIFIED',selected_implementation='accelerated',source_bindings=self.bound,
            base_source_bindings={str(base):p.sha(base)},development_read=False,training_updates=0,
            strict_equality_passed=True,speedup=1.3)
    def test_qualified_requires_exact_values_and_measured_gain(self):
        p.gate(self.q,self.bound)
        for change in [dict(strict_equality_passed=False),dict(speedup=1.09),dict(speedup=float('nan')),
                       dict(speedup=float('inf')),dict(development_read=True),dict(training_updates=1)]:
            q=dict(self.q,**change)
            with self.assertRaises(RuntimeError):p.gate(q,self.bound)
    def test_safe_fallback_is_explicit_and_does_not_claim_speedup(self):
        q=dict(self.q,status='USE_ORIGINAL',selected_implementation='original',strict_equality_passed=False,speedup=.9)
        p.gate(q,self.bound)
        q['speedup']=None
        p.gate(q,self.bound)
        q['selected_implementation']='accelerated'
        with self.assertRaises(RuntimeError):p.gate(q,self.bound)
    def test_failed_or_missing_decision_cannot_publish(self):
        for status in ['FAILED','RUNNING','']:
            with self.assertRaises(RuntimeError):p.gate(dict(self.q,status=status),self.bound)
    def test_current_source_must_match_qualified_inventory(self):
        with self.assertRaises(RuntimeError):p.gate(dict(self.q,source_bindings={}),self.bound)
        Path(next(iter(self.bound))).write_text('new source')
        with self.assertRaises(RuntimeError):p.gate(self.q,self.bound)
    def test_base_source_may_not_drift(self):
        self.base.write_text('changed base')
        with self.assertRaises(RuntimeError):p.gate(self.q,self.bound)

    def test_pending_manifest_retry_requires_exact_prior_publication(self):
        rec=dict(source_bindings=self.bound,qualification_sha256='q',base_commit='head',manifest_sha256='m',published_files=self.bound)
        p.verify_pending_manifest(rec,self.bound,'q','head','m','m')
        for changes in [dict(base_commit='new'),dict(manifest_sha256='other'),dict(qualification_sha256='new'),dict(source_bindings={})]:
            with self.assertRaises(RuntimeError):p.verify_pending_manifest(dict(rec,**changes),self.bound,'q','head','m','m')
        with self.assertRaises(RuntimeError):p.verify_pending_manifest(rec,self.bound,'q','head','m','other')
        Path(next(iter(self.bound))).write_text('tampered')
        with self.assertRaises(RuntimeError):p.verify_pending_manifest(rec,self.bound,'q','head','m','m')

if __name__=='__main__':unittest.main()
