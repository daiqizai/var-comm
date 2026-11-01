"""Bounded CPU import/signature qualification; no model, packet or timing call."""
from followup_common import *
import types, inspect, traceback, subprocess
from timing_phy_namespace_v2 import isolated_engine_with_explicit_sys


def main(request, expected):
    assert sha(request) == expected
    r = read(request); [checked(p) for p in r['code']]
    out = Path(r['output']); out.mkdir(exist_ok=False)
    os.sched_setaffinity(0, set(r['cpu_affinity']))
    began = time.monotonic()
    try:
        tests = subprocess.run([sys.executable, str(E/'test_timing_phy_namespace_v2.py')],
                               capture_output=True, text=True, timeout=60)
        assert tests.returncode == 0, tests.stderr
        sys.path.insert(0, str(R/'experiments/adatoken-response-20261026'))
        import c_predclass_frame as cp
        original = cp.isolated_engine
        namespace = dict(original.__globals__); namespace.pop('sys', None)
        missing = types.FunctionType(original.__code__, namespace, original.__name__,
                                     original.__defaults__, original.__closure__)
        try:
            missing()
        except NameError as exc:
            assert "sys" in str(exc)
        else:
            raise AssertionError('The historical missing-binding failure was not reproduced')
        corrected = isolated_engine_with_explicit_sys(missing, R)
        reference = isolated_engine_with_explicit_sys(original, R)
        assert corrected.FAMILIES == reference.FAMILIES
        functions = ['pack_body', 'transmit_body', 'transmit_frame', 'receive_body',
                     'receive_frame', 'standard_noise', 'create_runtime']
        proof = {}
        for name in functions:
            a, b = getattr(corrected, name), getattr(reference, name)
            assert a.__code__.co_code == b.__code__.co_code
            assert str(inspect.signature(a)) == str(inspect.signature(b))
            proof[name] = dict(bytecode_sha256=hashlib.sha256(a.__code__.co_code).hexdigest(),
                               signature=str(inspect.signature(a)))
        assert 'configure_threads' in inspect.signature(corrected.create_runtime).parameters
        assert 'sys' not in missing.__globals__
        save(out/'completion.json', dict(status='COMPLETE', CPU_tests=3,
             tests_stdout=tests.stdout, tests_stderr=tests.stderr, real_PHY_module_imported=True,
             constructor_invoked=False, same_PHY_function_bytecode_and_signatures=proof,
             old_missing_sys_reproduced=True, original_namespace_unchanged=True,
             model_calls=0, packet_calls=0, timing_calls=0, real_weight_timing_qualified=False,
             seconds=time.monotonic()-began))
    except BaseException:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False,
             model_calls=0, packet_calls=0, timing_calls=0))
        raise


if __name__ == '__main__':
    main(*sys.argv[1:])
