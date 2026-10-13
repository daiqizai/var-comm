"""Reviewed infrastructure-only partial-attempt recovery; no model imports."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import FunctionType

SWIN_OLD="s.native.args.multiple_snr=','.join(str(v) for v in range(1,20))"
SWIN_NEW='s.native.multiple_snr=list(range(1,20))'
PROVIDER_OLD="codec.provider_factory=lambda:ns['IndependentProvider'](native,SimpleNamespace(cdf_function=codec.entropy.probability_cdf))"
PROVIDER_NEW="codec.provider_factory=lambda _provider=ns['IndependentProvider']:_provider(native,SimpleNamespace(cdf_function=codec.entropy.probability_cdf))"
STATIC_OLD='codec=ep_source_codec.PartialSourceCodec(ROOT)'
STATIC_NEW=STATIC_OLD+"\n    import h800_ep48_link_core_v1 as _static_link\n    _static_link.bind_static(codec,visual_request['static'])"
INHERITED_COMPLETE=576


def checked_function(source, substitutions, globals_dict, filename, name):
    changed=source
    for old,new in substitutions:
        assert changed.count(old)==1 and new not in changed,old
        changed=changed.replace(old,new,1)
    inverse=changed
    for old,new in reversed(substitutions):
        assert inverse.count(new)==1
        inverse=inverse.replace(new,old,1)
    before=ast.dump(ast.parse(source),include_attributes=False)
    assert ast.dump(ast.parse(inverse),include_attributes=False)==before
    compiled_namespace=dict(globals_dict)
    exec(compile(changed,filename,'exec'),compiled_namespace)
    compiled=compiled_namespace[name]
    live=FunctionType(compiled.__code__,globals_dict,compiled.__name__,compiled.__defaults__,compiled.__closure__)
    live.__kwdefaults__=compiled.__kwdefaults__
    live.__annotations__=compiled.__annotations__
    assert live.__globals__ is globals_dict
    return live,dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        inverse_AST_equal=True,live_globals_preserved=True,
        substitutions=[dict(original=a,replacement=b) for a,b in substitutions],
        original_files_modified=False)


def adapt_build(runtime):
    """Capture the exact provider class before ns becomes the P namespace."""
    function,proof=checked_function(inspect.getsource(runtime.build),
        [(SWIN_OLD,SWIN_NEW),(PROVIDER_OLD,PROVIDER_NEW),(STATIC_OLD,STATIC_NEW)],vars(runtime),runtime.__file__,'build')
    proof.update(provider_capture='Same IndependentProvider class, captured once as lambda default before P namespace assignment',
        provider_factory_body_math_unchanged=True,shared_visual_late_global_replacement_preserved=True,
        static_receive_domain='Existing qualified bind_static with original relocated frozen table; preserves144/360 public profiles',
        new_source_tokens_or_channel_calls=0)
    return function,proof


def validate_prior(prior,plan,records):
    """Accept exactly the immutable v5 complete prefix and first failed slot."""
    assert (prior['case_cap'],prior['measured_case_cap'],prior['packet_cap'])==(1728,864,2304)
    assert len(plan)==1728 and len(prior['cases'])==577
    rows=prior['cases'];complete=rows[:INHERITED_COMPLETE]
    assert [r['case'] for r in rows]==plan[:577]
    assert all(r['state']=='COMPLETE' for r in complete)
    assert rows[-1]==dict(case=plan[576],state='RESERVED',actual_packet_decodes=0)
    assert plan[576]==dict(method='EC_VAR_WHOLE',source_index=0,snr_db=4,repeat=0,phase='warmup')
    assert prior['packet_decodes']==sum(r['actual_packet_decodes'] for r in rows)==1152
    first={};counts={}
    for row in complete:
        c=row['case'];a=row['identity'];t=row['timings'];key=(c['method'],c['source_index'],c['snr_db'])
        assert c['method'] in ('RAW_WHOLE','RAW_PARTIAL') and row['actual_packet_decodes']==2
        assert a['fresh_visual_encoder'] and not a['cached_tokens_or_streams'] and a['batch_size']==1
        assert a['fresh_tokens_sha256']==records[c['source_index']]['tokens_sha256']
        assert all(a[k] for k in ('wave_sha256','observation_sha256','reconstruction_sha256'))
        assert all(t[k]>=0 for k in ('tx_seconds','rx_seconds','channel_seconds','outer_seconds'))
        for side in ('tx','rx'):
            assert abs(t['raw_wall_'+side+'_seconds']-t['audit_excluded_'+side+'_seconds']-t[side+'_seconds'])<1e-7
        if key in first:assert first[key]==a
        else:first[key]=a
        counts[key]=counts.get(key,0)+1
    assert len(first)==96 and set(counts.values())=={6}
    return complete


def resumed_reservations(base_class,prior,records,prior_binding):
    """New journal copies only complete576; failed reservation stays archived."""
    class ResumedReservations(base_class):
        def __init__(self,path):
            # base_class creates a distinct, previously nonexistent v6 path.
            assert Path(path).resolve()!=Path(prior_binding['path']).resolve()
            super().__init__(path)
            complete=validate_prior(prior,self.plan,records)
            self.state['cases']=copy.deepcopy(complete)
            self.state['packet_decodes']=prior['packet_decodes']
            self.state['status']='RUNNING'
            self.state['recovery']=dict(inherited_journal=copy.deepcopy(prior_binding),
                inherited_complete_frames=576,inherited_packet_decodes=1152,
                archived_failed_slot=copy.deepcopy(prior['cases'][-1]),
                failed_fresh_visual_encoder_attempts=1,
                failed_slot_is_not_a_complete_warmup=True,
                remaining_complete_frames=1152,remaining_measured_frames=576,
                remaining_packet_cap=1152,automatic_retry=False)
            self.flush()
    return ResumedReservations


def seed_completed(reservations):
    first={};results=[]
    assert len(reservations.state['cases'])==576
    for row in reservations.state['cases']:
        c=row['case'];key=(c['method'],c['source_index'],c['snr_db'])
        first[key]=copy.deepcopy(row['identity'])
        results.append(copy.deepcopy({k:row[k] for k in ('case','identity','timings')}))
    assert len(first)==96 and len(results)==576
    return first,results


def adapt_worker(owner):
    """Skip the inherited prefix before entering any endpoint, with AST proof."""
    function,proof=checked_function(inspect.getsource(owner.worker),[
        ('first={};results=[]','first,results=_resume_seed(reservations)'),
        ('for case in fixed_cases():','for case in fixed_cases()[len(results):]:')],
        vars(owner),owner.__file__,'worker')
    proof.update(inherited_frames=576,executed_remaining_frames=1152,
        first_executed_slot_index=576,source_model_noise_receiver_math_unchanged=True,
        completed_prefix_does_not_enter_endpoint=True)
    return function,proof


def same_numerical_runtime(prior,current):
    fields=('python','torch','numpy','cuda','cudnn','threads','interop_threads','affinity',
        'FP32','TF32','deterministic','allocator_cap_bytes','actual_GPU_UUID','device','device_total_bytes')
    for name in fields:assert prior[name]==current[name],name
    assert prior['shared_GPU'] is current['shared_GPU'] is True
    assert prior['timing_measurement'] is current['timing_measurement'] is True
    assert prior['device_identity']['actual_pci_bus_id']==current['device_identity']['actual_pci_bus_id']
    assert prior['device_identity']['same_physical_PCI'] and current['device_identity']['same_physical_PCI']
    return dict(fields=list(fields),same_actual_H800_and_math=True,
        prior_and_current_shared_environment=True,uncontended_claimed=False)
