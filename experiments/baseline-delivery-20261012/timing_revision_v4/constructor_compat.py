"""One-line constructor field compatibility; original mathematics unchanged."""
import ast
import hashlib
import inspect

OLD="s.native.args.multiple_snr=','.join(str(v) for v in range(1,20))"
NEW='s.native.multiple_snr=list(range(1,20))'


def adapt_build(runtime):
    source=inspect.getsource(runtime.build)
    assert source.count(OLD)==1 and NEW not in source
    changed=source.replace(OLD,NEW,1)
    before=ast.dump(ast.parse(source),include_attributes=False)
    assert ast.dump(ast.parse(changed.replace(NEW,OLD,1)),include_attributes=False)==before
    namespace=dict(vars(runtime))
    exec(compile(changed,runtime.__file__,'exec'),namespace)
    proof=dict(original_build_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
               inverse_AST_equal=True,original=OLD,replacement=NEW,
               actual_attribute='native.multiple_snr: list[int]',
               explicit_SNR_TX_RX_calls_unchanged=True,
               original_scientific_files_modified=False)
    return namespace['build'],proof
