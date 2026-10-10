"""Original four computations with an honest post-hoc mismatch metric identity.

All provider functions/methods retain their original code objects in a private
namespace. Only population and descriptive selection metadata change. Frozen
pilot modules/globals, four weights, preprocessing and numerical flags stay exact.
"""
from __future__ import annotations
import inspect
import math
from pathlib import Path
import types
import h800_ep_pilot_metrics_v1 as original
import h800_mismatch_input_plan_v1 as inputs

ORIGINAL_SHA='6bde19df620a7dbf391f5d406daa3b4c1de2a66d051c990c26ae9a2474c1ec8d'
SCHEMA='H800_RAW100_ONE_BIN_MISMATCH_FOUR_METRICS_V1'
POPULATION='posthoc_first100_original_common500_holdout_configuration_mismatch'
CAPS=inputs.future_caps()['metrics']
FLAGS=original.FLAGS
require=original.require
array_sha=original.array_sha
pair_cache_key=original.pair_cache_key


def bound_assets(config):
    require(config['diagnostic_population']==POPULATION and config['policy_selection'] is False,
        'Explicit post-hoc diagnostic population required')
    pin=config['raw_frozen_policy']
    require(pin['sha256']==inputs.POLICY_SHA and original.core.sha(pin['path'])==inputs.POLICY_SHA,
        'Original raw calibration-frozen policy anchor required')
    return original.bound_assets(config)


def private_provider():
    require(original.core.sha(original.__file__)==ORIGINAL_SHA,'Frozen four metric computations changed')
    namespace=dict(vars(original));namespace.update(__file__=__file__,SCHEMA=SCHEMA,POPULATION=POPULATION,CAPS=CAPS)
    # Every old numerical function retains the exact same Python code object.
    functions={}
    for name,value in vars(original).items():
        if inspect.isfunction(value) and value.__module__==original.__name__:
            clone=types.FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=value.__kwdefaults__;namespace[name]=clone;functions[name]=clone.__code__ is value.__code__
    attrs={};methods={}
    for name,value in vars(original.FourMetrics).items():
        if inspect.isfunction(value):
            clone=types.FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=value.__kwdefaults__;attrs[name]=clone;methods[name]=clone.__code__ is value.__code__
    require(all(functions.values()) and all(methods.values()),'Numerical function or method changed')
    provider=type('OriginalFourMetricsPrivateMismatchScope',(),attrs)
    proof=dict(original_adapter_sha256=ORIGINAL_SHA,functions_same_code_object=functions,methods_same_code_object=methods,
        new_population=POPULATION,original_globals_modified=False,original_FORWARD_expressions_changed=False,
        only_population_namespace_and_descriptive_metadata_changed=True)
    return provider,proof


class FourMetrics:
    def __new__(cls,config,ledger,guard):
        bound_assets(config);provider,proof=private_provider();value=provider(config,ledger,guard)
        # Correct all pilot selection labels before identity/prepared cache creation.
        value.evaluator.metadata['metric_selection_usage']='Four-metric post-hoc one-bin configuration mismatch diagnostic; no policy selection'
        value.metadata.update(population=POPULATION,selection_objective=None,policy_selection=False,
            diagnostic_raw_policy=config['raw_frozen_policy'],historical_compute_admission_anchor=config['whole_policy'],
            DINO_L_metadata=value.evaluator.metadata,private_population_adapter_proof=proof,
            adapter_source=original.core.descriptor(__file__),original_compute_adapter=dict(path=original.__file__,sha256=ORIGINAL_SHA),
            old_host_metric_values_reused=False)
        value.identity=original.core.digest(value.metadata);guard();return value


def paired_outputs(rows):
    """Source-paired three-noise arithmetic only; no bootstrap or new selection."""
    p=inputs.frozen;metrics=original.core.METRICS
    keys={(r['source_index'],r['actual_snr_db'],r['config_snr_db'],r['family'],r['noise_seed']):r for r in rows}
    expected={(i,s,k,f,n) for i in range(100) for s in p.ACTUAL_SNRS for k in p.LOOKUPS[s] for f in p.FAMILIES for n in p.SEEDS}
    require(len(rows)==len(keys)==5400 and set(keys)==expected,'Complete unique5400 diagnostic metric cells required')
    source_ids={i:{r['source_id'] for r in rows if r['source_index']==i} for i in range(100)}
    require(all(len(v)==1 for v in source_ids.values()) and len(set(next(iter(v)) for v in source_ids.values()))==100,
        'Stable100 source identities required')
    for row in rows:
        require(all(math.isfinite(float(row[m])) for m in metrics) and row['convnext_top1_source_prediction'] in (0,1),
            'Finite metrics and binary source-prediction agreement required')
    paired=[];means=[];failures=[]
    for snr in p.ACTUAL_SNRS:
        for lookup in p.LOOKUPS[snr]:
            for family in p.FAMILIES:
                cell=[keys[i,snr,lookup,family,n] for i in range(100) for n in p.SEEDS]
                failures.append(dict(actual_snr_db=snr,config_snr_db=lookup,family=family,logical_frames=300,
                    header_failures=sum(not r['actual_header_ok'] for r in cell),body_attempted=sum(r['body_attempted'] for r in cell),
                    body_CRC_rejections=sum(r['body_attempted'] and not r['actual_crc_accepted'] for r in cell),
                    fixed_gray=sum(r['gray'] for r in cell),body_CRC_denominator='body_attempted only'))
                for name in metrics:
                    values=[];scale=100 if name=='convnext_top1_source_prediction' else 1
                    for i in range(100):
                        differences=[float(keys[i,snr,lookup,family,n][name])-float(keys[i,snr,snr,family,n][name]) for n in p.SEEDS]
                        value=math.fsum(differences)/3*scale;values.append(value)
                        paired.append(dict(source_index=i,source_id=next(iter(source_ids[i])),actual_snr_db=snr,config_snr_db=lookup,
                            family=family,metric=name,mean=value,noise_count=3,delta_definition='mismatch_minus_same_true_SNR_matched',
                            unit='percentage_points' if scale==100 else 'native_metric_unit'))
                    means.append(dict(actual_snr_db=snr,config_snr_db=lookup,family=family,metric=name,
                        mean=math.fsum(values)/100,source_count=100,noise_count=3,ci_low=None,ci_high=None,
                        unit='percentage_points' if scale==100 else 'native_metric_unit',
                        delta_definition='mismatch_minus_same_true_SNR_matched',LPIPS_sign_not_flipped=True,
                        confidence_intervals_status='NOT_COMPUTED_NO_BOOTSTRAP_ADMITTED'))
    return dict(source_paired_differences=paired,mean_differences=means,failure_counts=failures,
        bootstrap_calls=0,policy_selection=False,matched_zero_differences_retained=True)
