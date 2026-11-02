"""Resource-controlled measurements of immutable E5 implementations.

No scientific dependency is edited on disk. All small in-memory adaptations
are explicit, hash-bound, and recorded outside the measured boundaries.
"""
from pathlib import Path
import sys, ast, types, inspect

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0, str(R/'experiments/nestcom-perceptual-followup-20261030'))
from followup_common import *
from timing_dependency_v4 import activate


def replace_once(text, old, new):
    assert text.count(old) == 1, ('Unexpected source revision', old)
    return text.replace(old, new)


def adapted_source(original, bpg=False):
    changes = [
        ('dict(timing_trial=160)', "dict(timing_trial=r['source_count']*5)"),
        ("sources=cr(r['sources']);assert len(sources)==32", "sources=cr(r['sources']);assert len(sources)==r['source_count']"),
        ("assert len(inputrows)==160", "assert len(inputrows)==r['source_count']*5"),
        ("for repetition in range(5):", "for repetition in range(5):\n    ctx['_audit_logits']=[]"),
        ("value,elapsed=ledger.call('timing_trial',event,trial)", "value,elapsed=ledger.call('timing_trial',event,trial)\n    after_trial(ctx,value,event,role)"),
        ("role,si+1,32,flush=True", "role,si+1,r['source_count'],flush=True"),
        ("rows=160,warmups=64,measured=96", "rows=len(rows),warmups=2*len(sources),measured=3*len(sources)"),
    ]
    if bpg:
        changes.append(("shim=B/'five_method_timing_20261020_v1/bpg_threads_qualification_v1/affinity_cpu_count.so'\n assert sha(shim)=='953375c8aa9c3ddd8f3313f99425ee2fe33a08b838492b0fee965dd04211494d'",
                        "shim=checked(BPG_TWO_CORE_SHIM)"))
    modified = original
    for old, new in changes:
        modified = replace_once(modified, old, new)
    reversed_text = modified
    for old, new in reversed(changes):
        reversed_text = replace_once(reversed_text, new, old)
    assert ast.dump(ast.parse(reversed_text)) == ast.dump(ast.parse(original))
    return modified, changes


def install_audit_diagnostic(ctx, out):
    """Only non-protocol audit operations; no token/CDF/decoder math changes."""
    method = ctx['g']['method']
    changes = []
    if method == 'Ada':
        import c_predclass_native
        old = 'and t.isfinite(v).all();return v'
        text = inspect.getsource(c_predclass_native.PredictedClassNative.logits)
        import textwrap
        text = textwrap.dedent(text)
        revised = replace_once(text, old, ";self._fair_audit_logits.append(v);return v")
        ns = dict(c_predclass_native.__dict__)
        exec(compile(revised, '<fairness:deferred-logits-audit>', 'exec'), ns)
        ctx['n'].logits = types.MethodType(ns['logits'], ctx['n'])
        ctx['n']._fair_audit_logits = []
        changes.append(dict(operation='defer_logits_finiteness_to_after_timer', original=text, revised=revised))
        sys.path.insert(0, str(R/'experiments/adatoken-response-20261026/c_predclass_production_rx_v1'))
        import protocol_rx
        text = inspect.getsource(protocol_rx.decode_prefix)
        # CDF values are still consumed by the decoder and canonical encoder.
        revised = replace_once(text, 'traces.append(asha(table))', 'pass # unused CDF audit hash omitted')
        ns = dict(protocol_rx.__dict__)
        exec(compile(revised, '<fairness:no-unused-cdf-hash>', 'exec'), ns)
        protocol_rx.decode_prefix = ns['decode_prefix']
        changes.append(dict(operation='omit_unused_CDF_hashes', original=text, revised=revised))
    elif method == 'Token_NeST':
        import receiver_variants
        text = inspect.getsource(receiver_variants.completed_latent)
        revised = replace_once(text, 'completed_tokens_sha256=asha(np.concatenate(chosen_maps))', 'completed_tokens_sha256=None')
        ns = dict(receiver_variants.__dict__)
        exec(compile(revised, '<fairness:no-unused-completion-hash>', 'exec'), ns)
        receiver_variants.completed_latent = ns['completed_latent']
        changes.append(dict(operation='omit_unused_completed_token_hash', original=text, revised=revised))
    else:
        raise AssertionError('No preregistered audit diagnostic for '+method)
    save(out/'audit_only_adaptations.json', changes)


def main(request, expected):
    assert sha(request) == expected
    r = read(request)
    [checked(p) for p in r['code']]
    baseline = cr(r['baseline_rows'])
    by_event = {v['event']: v for v in baseline}
    assert len(by_event) == len(baseline) == r['source_count']*5
    activate(r['sionna_package'])
    original = checked(r['original_worker']).read_text()
    revised, transformations = adapted_source(original, r['group']['method']=='BPG_LDPC')
    ns = dict(__name__='fair_e5', __file__=str(checked(r['original_worker'])))
    if r['group']['method'] == 'BPG_LDPC':
        ns['BPG_TWO_CORE_SHIM'] = r['BPG_two_core_shim']
    exec(compile(revised, '<fairness:E5-boundaries>', 'exec'), ns)
    # New namespace owns its STOP boundary. No historical queue is invoked.
    ns['L'] = Path(r['study_root'])
    original_setup = ns['setup']

    def setup(cfg, g, out, role, pci):
        ctx = original_setup(cfg, g, out, role, pci)
        t = ctx['t']
        # Model loaders validate their original route before this explicit
        # resource-only setting. No repeated inter-op setter after model load.
        t.set_num_threads(2)
        assert t.get_num_threads() == 2 and t.get_num_interop_threads() == 2
        assert sorted(os.sched_getaffinity(0)) == [38,40]
        topology = []
        for cpu in [38,40]:
            root=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
            topology.append(dict(cpu=cpu, core=(root/'core_id').read_text().strip(),
                                 socket=(root/'physical_package_id').read_text().strip(),
                                 siblings=(root/'thread_siblings_list').read_text().strip()))
        assert len({(v['socket'],v['core']) for v in topology}) == 2
        save(out/'effective_timing_resources.json', dict(cpu_topology=topology, intra_op=2, inter_op=2,
             original_loader_threads=6, original_loader_interop=2, audit_mode=r['audit_mode'],
             comparison_scope='frozen implementations, not algorithmic optimality', transformations=transformations))
        if r['audit_mode']=='lean_diagnostic':
            install_audit_diagnostic(ctx, out)
        return ctx

    def after_trial(ctx, value, event, role):
        if r['audit_mode']=='lean_diagnostic' and ctx['g']['method']=='Ada':
            tensors=ctx['n']._fair_audit_logits
            assert all(bool(ctx['t'].isfinite(v).all()) for v in tensors)
            tensors.clear()
        ref=by_event[event]
        if role=='tx':
            actual=asha(value[0]); key='emitted_symbols_sha256'
        else:
            actual=asha(value[0]); key='image_sha256'
            assert value[1]['status']==ref['status'] and value[1]['success']==ref['success']
        assert actual==ref[key], ('OUTPUT_IDENTITY_MISMATCH',event,key,actual,ref[key])

    ns['setup']=setup
    ns['after_trial']=after_trial
    ns['main'](request, expected)


if __name__=='__main__':
    main(*sys.argv[1:])
