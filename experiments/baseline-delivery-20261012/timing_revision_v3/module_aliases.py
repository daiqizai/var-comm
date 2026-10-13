"""Constructor-only aliases for two pinned, byte-identical module copies.

No module is imported by this file. No global sys.modules aliases are replaced.
The raw path/module object survives; only the private T1/EP loader resolves its
canonical path to that same object during construction, after exact checks.
"""
from pathlib import Path
import hashlib
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExactByteAliases:
    def __init__(self, pairs, modules=None):
        self.pairs = {p['module_name']:dict(p) for p in pairs}
        assert len(self.pairs) == len(pairs) == 2
        assert set(self.pairs) == {'main_raw64_plan_only','main_raw64_packet_adapter'}
        self.modules = sys.modules if modules is None else modules
        self.events = []

    def verify(self):
        for row in self.pairs.values():
            a, b = Path(row['loaded_path']).resolve(), Path(row['requested_path']).resolve()
            assert a != b
            assert sha(a) == sha(b) == row['sha256'], str(a)

    def wrap(self, original, loader_name):
        def load_exact(path, name):
            row = self.pairs.get(name)
            requested = Path(path).resolve()
            if row is None or requested != Path(row['requested_path']).resolve():
                return original(path, name)
            module = self.modules.get(name)
            if module is None:
                return original(path, name)
            loaded = Path(module.__file__).resolve()
            if loaded == requested:
                return original(path, name)
            assert loaded == Path(row['loaded_path']).resolve(), 'Unexpected prior module path: '+name
            assert sha(loaded) == sha(requested) == row['sha256'], 'Alias bytes changed: '+name
            if name == 'main_raw64_packet_adapter':
                assert module.p is self.modules['main_raw64_plan_only'], 'Packet planner object split'
            self.events.append(dict(loader=loader_name, module_name=name,
                                    loaded_path=str(loaded), requested_path=str(requested),
                                    sha256=row['sha256'], identical_module_object_reused=True))
            return module
        return load_exact

    def call(self, t1, ep, construct):
        self.verify()
        original_t1, original_ep = t1.load_exact, ep.load_exact
        t1.load_exact = self.wrap(original_t1, 't1_phy.load_exact')
        ep.load_exact = self.wrap(original_ep, 'ep_phy.load_exact')
        try:
            return construct()
        finally:
            t1.load_exact, ep.load_exact = original_t1, original_ep
