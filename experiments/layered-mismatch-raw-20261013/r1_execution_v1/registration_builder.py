"""Explicit future R1 registration builder, gated by completed authenticated R0.

Defining/importing this entry creates no registration and reads no R0 results.
The build command is only for a later reviewed dispatch after actual R0 closure.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import time

import shared as s
import r0_finalize
import r1_phy
import grouped_wire


def authenticated_r0(audit_pin):
    audit = s.readpin(audit_pin)
    s.require(audit['status'] == 'R0_COMPLETE_AUTHENTICATED_R1_NOT_REGISTERED'
              and audit['schema'] == r0_finalize.SCHEMA and audit['no_new_science'] is True,
              'Completed actual R0 audit required before registration')
    s.checked(audit['finalizer'])
    s.require(audit['finalizer']['sha256'] == s.sha(r0_finalize.__file__), 'R0 authentication code differs')
    nomination = s.readpin(audit['nominations'])
    s.require(nomination['status'] == 'R0_NOMINATION_ARITHMETIC_COMPLETE_NOT_REGISTERED_R1'
              and nomination['R1'] == nomination['R2'] == nomination['GO_decision'] == 'NOT_RUN',
              'R0 shortlist not an authenticated unconsumed nomination')
    # These are read-only audits of actual saved receipts, bytes and ledgers.
    # No ranking, model, channel, tokenization or bootstrap is repeated here.
    body_rows, body_proof = r0_finalize.audit_bler(audit['BLER']['registration'])
    quality_rows, clean_proof = r0_finalize.audit_clean(audit['clean']['registration'])
    s.require(body_proof == audit['BLER'] and clean_proof == audit['clean']
              and len(body_rows) == 1389 and len(quality_rows) == 5400, 'R0 actual closure changed')
    for value in audit['inputs'].values():
        s.checked(value)
    sources = s.readpin(audit['inputs']['source'])
    s.require(len(sources['records']) == 200 and len({r['source_id'] for r in sources['records']}) == 200,
              'Original calibration200 only')
    s.require(nomination['selection_uses_holdout'] is False and nomination.get('bootstrap_calls', 0) == 0,
              'R0 scope differs')
    return audit, nomination, sources


def physical_registration(pins, catalogue, nominations, runtime):
    """Pure structural constructor. It never calls the actual PHY runtime."""
    groups = r1_phy.nomination_groups(nominations, catalogue['profiles'])
    physical = len(groups) * 200 * 3 * 3
    logical = sum(len(g['logical_receivers']) for g in groups) * 200 * 3 * 3
    s.require(physical <= 36000 and logical <= 46800, 'R1 exact grid exceeds existing caps')
    raw = s.readpin(pins['raw_runtime_request'])
    old = s.read(raw['raw_runtime_binding']['roles']['catalogue'])['profiles']
    s.require(catalogue['profiles'][:433] == old and len(old) == 433, 'Original raw433 domain changed')
    return dict(schema=r1_phy.SCHEMA, status='REGISTERED_R1_PHY_V1', plan_id=s.contract.PLAN_ID,
        caps=s.contract.R1_CAPS, namespace=s.contract.NAMESPACE, automatic_successor=False,
        training_updates=0, historical_reuse_admitted=False, pins=pins,
        grouped_catalogue_digest=grouped_wire.Catalogue(catalogue['profiles']).digest,
        single_catalogue_digest=grouped_wire.identity(old), expected_physical_frames=physical,
        expected_logical_var_images=logical,
        runtime=dict(python=s.PY_CPU, threads=2, cpu_affinity=runtime['phy_cpu_affinity'], CUDA_VISIBLE_DEVICES='',
            max_seconds=runtime['max_seconds'], deadline_unix=runtime['deadline_unix'],
            output_directory=runtime['phy_directory'], stop_file=runtime['stop_file']))


def visual_projection(master, model_bundle):
    """No source/selection manifests or original qualified-source recipe in RX child."""
    keys = ('plan_id', 'caps', 'numeric_flags', 'resource_policy', 'automatic_retry', 'automatic_R2',
            'training_updates', 'additional_qualification_frames', 'bootstrap_calls', 'runtime',
            'expected_physical_frames', 'expected_logical_var_images', 'code_bindings', 'models', 'device_receipt')
    result = {key: copy.deepcopy(master[key]) for key in keys}
    result.update(schema=s.VISUAL_SCHEMA, status='REGISTERED_R1_VISUAL_V1', model_bundle=model_bundle,
                  pins={'common_ledger': copy.deepcopy(master['pins']['common_ledger'])})
    return result


def build(audit_pin, review_pin, directory, target_gpu, max_seconds):
    """Explicit future operation. Never called by prepare, owner, import or tests."""
    directory = Path(directory).absolute()
    s.require(directory == s.OUT / 'registered' and not directory.exists(), 'Fresh authoritative registration directory required')
    s.require(type(max_seconds) is int and max_seconds > 0 and type(target_gpu) is int and 0 <= target_gpu < 4,
              'Explicit finite wall limit and selected shared GPU required')
    review = s.readpin(review_pin)
    s.require(review['status'] == 'R1_IMPLEMENTATION_REVIEW_PASS' and review['approved'] is True
              and review['scientific_calls'] == 0 and review['caps'] == s.contract.R1_CAPS,
              'Root-reviewed finite implementation required')
    files = ('shared.py', 'owner.py', 'registration_builder.py', 'receiver_visual.py', 'score_selection.py')
    for name in files:
        path = Path(__file__).with_name(name)
        s.require(review['code_bindings'][str(path)] == s.sha(path), 'Reviewed R1 entry differs: ' + name)
    audit, nomination, sources = authenticated_r0(audit_pin)
    clean_registration = s.readpin(audit['clean']['registration'])
    catalogue = s.readpin(audit['inputs']['catalogue'])
    s.require(len(catalogue['profiles']) == 689, 'Complete paid public catalogue required')
    runtime = dict(root=str(s.TMP), python_cpu=s.PY_CPU, python_visual=s.PY_VISUAL, target_gpu=target_gpu,
        cpu_affinity=[0, 1], phy_cpu_affinity=[2, 3], max_seconds=max_seconds,
        deadline_unix=time.time() + max_seconds, stop_file=str(s.OUT / 'STOP'))
    runtime.update({stage+'_directory': str(s.TMP/stage) for stage in ('phy','projection','visual','score','selection')})
    s.validate_runtime(runtime)
    bindings = dict(clean_registration['code_bindings'])
    bindings.update(review['code_bindings'])
    for path in (Path(r1_phy.__file__), Path(grouped_wire.__file__), Path(s.contract.__file__),
                 s.EXPERIMENT/'policy_selection.py', s.EXPERIMENT/'r0_finalize.py'):
        bindings[str(path)] = s.sha(path)
    for path, expected in bindings.items():
        s.require(s.sha(path) == expected, 'Bound original code changed: ' + path)
    # The private CPU bundle builder performs original source qualification
    # checks here, then projects them away from the RX-only model process.
    import receiver_visual
    directory.mkdir(parents=True)
    frozen = dict(status='FROZEN_R1_NOMINATIONS', rows=nomination['rows'], selection_uses_holdout=False,
        origin_audit=audit_pin, original_nominations=audit['nominations'], rule_id=nomination['rule_id'],
        source_order=[r['source_id'] for r in sources['records']], source_manifest=audit['inputs']['source'],
        candidate_catalogue=audit['inputs']['catalogue'], frozen_before_any_R1=True,
        budget=nomination['budget'], ranking_sha256=s.digest(nomination['ranking']))
    nominations_pin = s.write(directory/'nominations.json', frozen)
    pins = dict(nominations=nominations_pin, cal200_sources=audit['inputs']['source'],
        grouped_catalogue=audit['inputs']['catalogue'],
        raw_runtime_request=s.pin(s.RT/'qualification/h800_mismatch_phy_v1_attempt1/registered/request.json'),
        calibration_request=s.pin(s.RT/'qualification/h800_ep_full_source900_v1_attempt1/registered/request.json'),
        mismatch_core=s.pin(s.ES/'h800_mismatch_phy_core_v1.py'), mismatch_plan=s.pin(s.ES/'mismatch_plan.py'),
        r1_phy=s.pin(r1_phy.__file__), grouped_wire=s.pin(grouped_wire.__file__), contract=s.pin(s.contract.__file__))
    phy = physical_registration(pins, catalogue, frozen['rows'], runtime)
    phy_pin = s.write(directory/'phy_registration.json', phy)
    # Exact frozen worker validation only, never r1_phy.run/RealRuntime.
    r1_phy.validate_registration(phy_pin)
    bundle = receiver_visual.build_model_bundle(clean_registration['visual_request'], directory/'model_bundle.json')
    original = s.readpin(clean_registration['visual_request'])
    master = dict(schema=s.MASTER_SCHEMA, status='REGISTERED_R1_EXECUTION_V1', plan_id=s.contract.PLAN_ID,
        caps=s.contract.R1_CAPS, numeric_flags=s.FLAGS, resource_policy=s.RESOURCE, automatic_retry=False,
        automatic_R2=False, training_updates=0, additional_qualification_frames=0, bootstrap_calls=0,
        namespace=s.contract.NAMESPACE, runtime=runtime, expected_physical_frames=phy['expected_physical_frames'],
        expected_logical_var_images=phy['expected_logical_var_images'], code_bindings=bindings,
        models=clean_registration['models'], visual_request=clean_registration['visual_request'],
        device_receipt=original['device_receipt'], metric_config=clean_registration['metric_config'],
        metric_names=['psnr_db','lpips_alex','dinov2_vitl14_cosine'], phy_registration=phy_pin,
        pins=dict(nominations=nominations_pin, cal200_sources=audit['inputs']['source'],
            grouped_catalogue=audit['inputs']['catalogue'], original_R0_audit=audit_pin,
            root_review=review_pin, common_ledger=s.pin(s.ROOT/'experiments/entropy-main-common500-20261013/common.py')),
        R2='NOT_REGISTERED_NOT_RUN', qualification_policy='No extra qualification frames; finite registered calls only')
    visual_pin = s.write(directory/'visual_registration.json', visual_projection(master, bundle))
    master['visual_registration'] = visual_pin
    master_pin = s.write(directory/'registration.json', master)
    s.validate_registration(master_pin); s.validate_registration(visual_pin)
    return dict(status='REGISTERED_NOT_LAUNCHED', registration=master_pin, phy_registration=phy_pin,
                visual_registration=visual_pin, scientific_calls=0, R1='NOT_RUN', R2='NOT_RUN')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r0-audit', required=True); p.add_argument('--r0-audit-sha256', required=True)
    p.add_argument('--review', required=True); p.add_argument('--review-sha256', required=True)
    p.add_argument('--directory', type=Path, required=True); p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--max-seconds', type=int, required=True)
    a = p.parse_args()
    print(s.canonical(build(dict(path=a.r0_audit,sha256=a.r0_audit_sha256),
        dict(path=a.review,sha256=a.review_sha256), a.directory, a.gpu, a.max_seconds)))
