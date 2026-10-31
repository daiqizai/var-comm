"""Supply an explicit sys binding without editing the frozen PHY source.

The historical fixed_frame module uses ``from common import *``. In the
combined timing process an earlier module named common can omit sys. Clone
only the original function globals; retain its exact code and all other
bindings. This helper does not construct a Runtime or transmit a packet.
"""
import sys
import types
from pathlib import Path


def isolated_engine_with_explicit_sys(original, expected_root):
    assert original.__name__ == 'isolated_engine'
    assert Path(original.__globals__['R']).resolve() == Path(expected_root).resolve()
    namespace = dict(original.__globals__)
    namespace['sys'] = sys
    cloned = types.FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
    assert cloned.__code__ is original.__code__
    assert all(namespace[k] is v for k, v in original.__globals__.items() if k != 'sys')
    return cloned()
