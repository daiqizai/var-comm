"""Explicit library-path recovery; original executable and science bytes stay frozen."""
import sys
from pathlib import Path

ORIGINAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORIGINAL))
from contract import *
import native

HERE = ORIGINAL / 'library_recovery_v1'
CONTROL = T / 'library_recovery_v1'
REG = O / 'library_recovery_v1/registration.json'
LIBRARY = RT / 'assets/frozen_namespace_v1/external/usr/lib/python3/dist-packages'
OLD_REG = 'd4e7a6bbad058fff98d0c061515c11c562af5d8cc59be5c9e249351363964f31'


def stage_dir(method, role):
    return T/method/('rx_library_recovery_v1' if (method, role)==('NEST','rx') else role)


def stages():
    return [('NEST','rx')] + [
        (m, role) for m in METHODS[1:]
        for role in (('tx','rx') if m in ('P100000','Swin200000') else ('tx','phy','rx'))]


def gone(ident):
    current = process(ident['pid'])
    return current is None or current['start_ticks'] != ident['start_ticks']


def bind(expected):
    assert sha(REG) == expected
    rec = read(REG)
    assert rec['original_registration_sha256'] == OLD_REG
    r = binding(OLD_REG)
    for item in rec['sources'] + rec['evidence']:
        checked(item)
    assert rec['remaining_stages'] == [list(s) for s in stages()]
    return r, rec


def adapted_worker():
    import worker
    # Admit this existing frozen distribution-library directory only. All source,
    # TX witness, output and other PFS restrictions retain the original AST.
    worker.firewall = native.replace_function(worker, 'firewall', [
        ("allowed = [p.resolve() for p in allowed]",
         "allowed += [Path("+repr(str(LIBRARY))+")]\n    allowed = [p.resolve() for p in allowed]")])
    # Only the failed, zero-case NEST receiver receives a new output directory.
    worker.main = native.replace_function(worker, 'main', [
        ("out = T/a.method/a.role",
         "out = T/a.method/('rx_library_recovery_v1' if (a.method,a.role)==('NEST','rx') else a.role)")])
    return worker


def worker_main():
    expected = sys.argv[1]
    r, rec = bind(expected)
    method, role = sys.argv[2:4]
    assert (method, role) in stages()
    sys.argv = [str(HERE/'adapter.py'), method, role, '--registration', OLD_REG]
    adapted_worker().main()


if __name__ == '__main__':
    worker_main()
