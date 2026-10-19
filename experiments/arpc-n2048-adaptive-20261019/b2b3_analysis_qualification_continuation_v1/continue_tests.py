"""Resume only the 37 interrupted/unexecuted synthetic tests, with original guard."""
# These standard-library imports intentionally precede the original test audit guard.
import runpy, unittest, zipfile
import argparse, json, sys, types
from pathlib import Path
from combined_review import load_plan, validate_initial

def flatten(suite):
    for test in suite:
        if isinstance(test,unittest.TestSuite):yield from flatten(test)
        else:yield test

class RecordingResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.passed=[]
    def addSuccess(self,test):
        self.passed.append(test.id().split('.',1)[1]);super().addSuccess(test)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    plan,descriptor=load_plan(args.plan)
    assert sys.executable==plan['expected_python'] and sys.dont_write_bytecode
    assert 'runpy' in sys.modules and 'unittest' in sys.modules and 'zipfile' in sys.modules
    assert not any(k.split('.')[0] in {'torch','tensorflow','sionna','transformers','lpips','PIL'} for k in sys.modules)
    validate_initial(plan)
    sys.path.insert(0,str(Path(plan['test_source']['path']).parent))
    # Executes exact original test bytes. Its fixture-only PFS audit hook and model
    # import blocker remain byte-identical and active during every selected test.
    namespace=runpy.run_path(plan['test_source']['path'],run_name='qualification_continuation')
    module=types.ModuleType('qualification_continuation');module.__dict__.update(namespace)
    all_tests=list(flatten(unittest.defaultTestLoader.loadTestsFromModule(module)))
    ordered=[t.id().split('.',1)[1] for t in all_tests]
    assert ordered==plan['ordered_tests'] and len(all_tests)==56
    selected=all_tests[19:];assert len(selected)==37
    selected_ids=[t.id().split('.',1)[1] for t in selected]
    assert selected_ids==plan['continuation_tests']
    assert not set(selected_ids)&set(plan['first_successful_tests'])
    result=unittest.TextTestRunner(verbosity=2,resultclass=RecordingResult).run(unittest.TestSuite(selected))
    imported=sorted(k for k in sys.modules if k.split('.')[0] in {'torch','tensorflow','sionna','transformers','lpips','PIL'})
    success=result.wasSuccessful() and result.testsRun==37 and result.passed==selected_ids and not result.skipped and not imported
    print(json.dumps(dict(schema='B2B3_ANALYSIS_QUALIFICATION_CONTINUATION_RESULT_V1',status='PASS' if success else 'FAIL',
        plan=descriptor,test_source=plan['test_source'],selected_tests=selected_ids,passed_tests=result.passed,
        retained_first_attempt_tests=plan['first_successful_tests'],tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),model_imports=imported,
        original_audit_guard_unchanged=True,production_sources_changed=False,new_scientific_calls=0)))
    raise SystemExit(0 if success else 1)

if __name__=='__main__':main()
