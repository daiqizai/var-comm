"""Preserve live runtime globals for the one-line constructor compatibility.

The corrected build code/defaults/closure are unchanged. Only the globals
mapping is restored to the real module so later explicit resource hooks remain
visible. This is not a new model or mathematical implementation.
"""
from types import FunctionType
import constructor_compat


def adapt_build_live(runtime):
    compiled,proof=constructor_compat.adapt_build(runtime)
    live=FunctionType(compiled.__code__,vars(runtime),compiled.__name__,
                      compiled.__defaults__,compiled.__closure__)
    live.__kwdefaults__=compiled.__kwdefaults__
    live.__annotations__=dict(compiled.__annotations__)
    live.__doc__=compiled.__doc__
    live.__module__=runtime.__name__
    assert live.__globals__ is vars(runtime)
    assert live.__code__ is compiled.__code__ and live.__defaults__ is compiled.__defaults__
    assert live.__closure__ is compiled.__closure__
    proof.update(runtime_globals_identity_preserved=True,
        compiled_code_object_unchanged=True,defaults_and_closure_unchanged=True,
        later_visual_backend_replacement_visible=True,
        resource_adapter_scope='v2 shared_visual remains dynamically resolved from original runtime module')
    return live,proof
