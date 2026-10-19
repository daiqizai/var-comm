"""Read-only combined engineering qualification validator; no test/model execution."""
from pathlib import Path
import ast, hashlib, json, os, re

PLAN_SHA='0ba8880fea742a502fed4d698510cdb8dcdb4ddeb57c99b4b7bd8549b15b054d'
PLAN_BYTES=10971

def pin(path):
    p=Path(path);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def verified(d):
    assert set(d)=={'path','bytes','sha256'}
    p=Path(d['path']);assert p.is_absolute() and not p.is_symlink()
    b=p.read_bytes();assert len(b)==d['bytes'] and hashlib.sha256(b).hexdigest()==d['sha256']
    return b

def load(d):return json.loads(verified(d))

def process_key(p):return p['pid'],p['start_ticks'],p['host']

def require_gone(p):
    assert p['host']==os.uname().nodename
    try:s=Path('/proc')/str(p['pid'])/'stat';tail=s.read_text().split(') ',1)[1].split()
    except FileNotFoundError:return
    assert int(tail[19])!=p['start_ticks'], 'Original process generation is still present'

def strict_wait(w,code):
    assert type(w['returncode']) is int and w['returncode']==code
    assert all(w[k] is False for k in ('timed_out','terminated','killed'))
    assert w['metadata_error'] is None

def test_order(source):
    tree=ast.parse(source)
    return sorted(n.name+'.'+m.name for n in tree.body if isinstance(n,ast.ClassDef)
                  for m in n.body if isinstance(m,ast.FunctionDef) and m.name.startswith('test_'))

def load_plan(path):
    d=dict(path=str(Path(path)),bytes=PLAN_BYTES,sha256=PLAN_SHA)
    plan=load(d);assert plan['schema']=='B2B3_ANALYSIS_QUALIFICATION_CONTINUATION_PLAN_V1'
    assert (plan['total_tests'],plan['retained_passes'],plan['continuation_count'])==(56,19,37)
    return plan,d

def validate_initial(plan):
    for d in plan['source_files']:verified(d)
    assert test_order(verified(plan['test_source']))==plan['ordered_tests']
    first=plan['first_attempt'];raw={k:verified(v) for k,v in first.items()}
    w=json.loads(raw['actual_wait.json']);la=json.loads(raw['launch.json']);supervisor=json.loads(raw['supervisor.json'])
    strict_wait(w,1)
    assert process_key(w['process'])==process_key(la['process'])==process_key(plan['first_child'])
    assert process_key(la['parent_process'])==process_key(plan['first_parent'])==process_key(supervisor['process'])
    assert la['argv']==[plan['expected_python'],'-B',plan['test_source']['path']]
    require_gone(la['process']);require_gone(la['parent_process'])
    assert raw['stdout']==b''
    text=raw['stderr'].decode('utf8')
    ok=re.findall(r'^test_\w+ \(__main__\.([^)]*)\) \.\.\. ok$',text,re.M)
    assert ok==plan['first_successful_tests']==plan['ordered_tests'][:19]
    assert plan['continuation_tests']==plan['ordered_tests'][19:]
    assert len(set(plan['ordered_tests']))==56
    assert plan['first_interrupted_test']==plan['continuation_tests'][0]=='IdentityTests.test_image_hash_and_shape'
    expected='test_image_hash_and_shape (__main__.IdentityTests.test_image_hash_and_shape) ... Traceback (most recent call last):'
    assert expected in text
    assert 'No production PFS assets in synthetic tests:' in text and '__pycache__/zipfile.cpython-311.pyc' in text
    assert text.count(' ... ok\n')==19 and ' ... ERROR' not in text
    return dict(retained_passed_tests=ok,original_failure_preserved=True,first_attempt=first)

def validate_combined(plan_path,control_dir,wrapper_files):
    """Root calls after the continuation supervisor and child are both gone."""
    plan,descriptor=load_plan(plan_path);initial=validate_initial(plan)
    files={Path(d['path']).name:d for d in wrapper_files}
    assert set(files)=={'continue_tests.py','combined_review.py','continuation_plan.json'}
    assert files['continuation_plan.json']==descriptor
    for d in files.values():verified(d)
    out=Path(control_dir);assert out.is_absolute() and not out.is_symlink()
    assert not(out/'failure.json').exists()
    evidence={k:pin(out/k) for k in ('supervisor.json','launch.json','actual_wait.json','stdout','stderr')}
    w=load(evidence['actual_wait.json']);la=load(evidence['launch.json']);sup=load(evidence['supervisor.json'])
    strict_wait(w,0)
    assert process_key(w['process'])==process_key(la['process'])
    assert process_key(la['parent_process'])==process_key(sup['process'])
    require_gone(la['process']);require_gone(la['parent_process'])
    assert la['argv']==[plan['expected_python'],'-B',files['continue_tests.py']['path'],'--plan',descriptor['path']]
    result=load(evidence['stdout'])
    assert result['schema']=='B2B3_ANALYSIS_QUALIFICATION_CONTINUATION_RESULT_V1' and result['status']=='PASS'
    assert result['plan']==descriptor and result['test_source']==plan['test_source']
    assert result['selected_tests']==result['passed_tests']==plan['continuation_tests']
    assert result['retained_first_attempt_tests']==plan['first_successful_tests']
    assert result['tests_run']==37 and result['model_imports']==[]
    assert result['failures']==result['errors']==result['skipped']==0
    text=verified(evidence['stderr']).decode('utf8')
    passed=re.findall(r'^test_\w+ \(qualification_continuation\.([^)]*)\) \.\.\. ok$',text,re.M)
    assert passed==plan['continuation_tests'] and '\nRan 37 tests in ' in text and text.rstrip().endswith('OK')
    assert not set(initial['retained_passed_tests']) & set(passed)
    assert initial['retained_passed_tests']+passed==plan['ordered_tests']
    return dict(schema='B2B3_ANALYSIS_COMBINED_SYNTHETIC_QUALIFICATION_V1',status='PASS',
        total_unique_tests=56,retained_first_attempt_passes=19,continuation_passes=37,
        successful_single_56_test_attempt=False,first_attempt_exitcode=1,continuation_exitcode=0,
        original_failure_preserved=True,plan=descriptor,source_files=plan['source_files'],wrapper_files=list(files.values()),
        first_attempt=plan['first_attempt'],continuation=evidence,retained_tests=initial['retained_passed_tests'],
        continuation_tests=passed,production_sources_changed=False,new_scientific_calls=0,
        process_closure_status='ALL_QUALIFICATION_PROCESSES_GONE_ON_EXECUTION_HOST')
