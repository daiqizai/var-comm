"""Synthetic source-hash admission tests; no models, images or scientific calls."""
import copy,tempfile,unittest
from pathlib import Path
import t6_project_source_environment as p

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name)
        self.old=root/'old.py';self.extra=root/'other.py';self.new=root/'new.py'
        for f in(self.old,self.extra,self.new):f.write_text('# synthetic metadata-only file\n')
        self.original=dict(root=str(root),schema='ORIGINAL_SCHEMA_UNCHANGED',native_source_bindings={str(self.old):p.sha(self.old)},
            source_bindings={str(self.old):p.sha(self.old),str(self.extra):p.sha(self.extra)},
            old_visual_identity=dict(model='frozen'),old_score_identity=dict(evaluator='frozen'),old_numerical_runtime=dict(tf32=False),
            records=[dict(source_id='old-calibration')],visual_config=dict(frozen='all'))
        self.current={str(self.old):p.sha(self.old),str(self.new):p.sha(self.new)}
    def test_only_new_tracked_addition_and_every_other_field_preserved(self):
        before=copy.deepcopy(self.original);out,additions,digest=p.project(self.original,self.current,[str(self.new)])
        self.assertEqual(self.original,before);self.assertEqual(additions,{str(self.new):p.sha(self.new)})
        self.assertEqual(out['native_source_bindings'],self.current)
        for key in self.original:
            if key not in('source_bindings','native_source_bindings'):self.assertEqual(out[key],self.original[key])
        self.assertEqual(len(digest),64)
    def test_changed_previously_bound_source_is_not_admitted_as_addition(self):
        self.extra.write_text('# changed original source\n')
        with self.assertRaisesRegex(RuntimeError,'Previously registered source changed'):p.project(self.original,self.current,[str(self.new)])
    def test_missing_or_changed_native_binding_is_rejected(self):
        bad=dict(self.current);bad[str(self.old)]='changed'
        with self.assertRaisesRegex(RuntimeError,'Original native source missing or changed'):p.project(self.original,bad,[str(self.new)])
        del bad[str(self.old)]
        with self.assertRaisesRegex(RuntimeError,'Original native source missing or changed'):p.project(self.original,bad,[str(self.new)])
    def test_untracked_runtime_addition_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Only current newly tracked'):p.project(self.original,self.current,[])
    def test_no_additions_is_valid_but_still_audits_all_old_sources(self):
        out,additions,_=p.project(self.original,self.original['native_source_bindings'],[])
        self.assertEqual(additions,{});self.assertEqual(out,self.original)

if __name__=='__main__':unittest.main()
