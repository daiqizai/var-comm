"""One-shot metadata registration. Does not instantiate a PHY or model."""
from common import *
from plan import *
import argparse
import collections
import subprocess


def old_closure():
    b0 = read(OLD_ADAPTIVE_O / 'b0_completion.json')
    assert b0['status'] == 'B0_METADATA_AND_ACTUAL_SOURCE_BYTES_COMPLETE'
    cat = next(d for d in b0['runtime_files'] if Path(d['path']).name == 'source_catalogue.json')
    sources = next(d for d in b0['runtime_files'] if Path(d['path']).name == 'sources.json')
    verify(cat); verify(sources)
    q = read(OLD_ADAPTIVE_O / 'qualification/phy_completion.json')
    assert (q['logical'], q['subframes'], q['header_decodes'], q['body_decodes']) == (24, 48, 48, 48)
    assert q['original_RX_returncode'] == 1 and q['new_packet_calls_in_recovery'] == 0
    verify(q['closure'])
    review = read(OLD_ADAPTIVE_O / 'qualification/review.json')
    assert review['status'] == 'PASS' and review['actual_opened'] is True
    for k in ('preview', 'native_audit', 'transport_audit'):
        verify(review[k])
    for name in ('qualification_close_saved', 'qualification_audit_saved', 'qualification_native'):
        closed_wait(OLD_ADAPTIVE_L / 'controls' / name / 'actual_wait.json')
    oldreg = read(OLD_ADAPTIVE_O / 'b1_registration.json')
    for d in oldreg['code']:
        verify(d)
    eng = read(OLD_ADAPTIVE_L / 'engineering/backend/catalogue.json')
    assert eng['backend_identity'] == oldreg['backend_identity']
    return cat, sources, oldreg, eng


def register(phase, auth_desc, request_path, selection_desc=None):
    assert phase in CAPS
    auth = read(verify(auth_desc))
    assert auth['schema'] == 'NESTCOM_MANUSCRIPT_COMPLETION_AUTHORIZATION_20261019_V1'
    assert auth['new_budget_authorized'] is True and auth['prospective_only'] is True
    assert auth['historical_scope_records_unchanged'] is True
    assert auth['B2_cells'] == cell_rows()
    for name in ('B2', 'B3'):
        assert auth[name+'_caps'] == {k: CAPS[name][k] for k in ('logical_objects','subframes','decoder_attempts')}
    assert auth['calibration_seeds'] == list(SEEDS['B2']) and auth['evaluation_seeds'] == list(SEEDS['B3'])
    # Preserve the original root authorization bytes, whose request path may be
    # Windows-local; bind its exact request bytes at the supplied remote path.
    request = dict(path=str(request_path), bytes=auth['request']['bytes'], sha256=auth['request']['sha256'])
    verify(request)
    cat_desc, sources_desc, oldreg, catalogue = old_closure()
    selection = None
    cells = cell_rows()
    if phase == 'B3':
        assert selection_desc is not None
        selection = read(verify(selection_desc))
        cells = frozen_cells(selection)
        assert Path(selection['B2_registration']['path']) == O / 'B2/registration.json'
        assert Path(selection['B2_transport_completion']['path']) == O / 'B2/transport_completion.json'
        for name in ('B2_registration', 'B2_transport_completion', 'source_evidence',
                     'metric_evidence', 'selection_evidence', 'score_completion', 'score_actual_wait'):
            verify(selection[name])
        closed_wait(selection['score_actual_wait']['path'])
        assert read(selection['score_completion']['path'])['status'] == 'COMPLETE'
        assert read(selection['B2_transport_completion']['path'])['status'] == 'TRANSPORT_COMPLETE_NATIVE_AND_SCORE_SEPARATE'
        closed_wait(L / 'controls/B2_audit/actual_wait.json')
    else:
        assert selection_desc is None
    all_sources = read(sources_desc['path'])
    sources = [s for s in all_sources if s['population'] == POPULATIONS[phase]]
    assert len(sources) == SOURCE_COUNTS[phase]
    if phase == 'B3':
        original = read(OLD_O / 'sources.json')['external_common100_v1']
        assert {(s['source_id'], s['source_index'], s['source']['sha256']) for s in sources} == {
            (s['source_id'], s['source_index'], s['archive']['sha256']) for s in original}
    jobs, assets, fallback = make_jobs(phase, cells, sources, read(cat_desc['path']))
    # Runtime files and registration are never overwritten or silently resumed.
    no_symlink_ancestors(L); no_symlink_ancestors(O)
    L.mkdir(exist_ok=True); O.mkdir(exist_ok=True)
    out, light = L / phase, O / phase
    assert not out.exists() and not light.exists()
    out.mkdir(); light.mkdir()
    put(out / 'private_jobs.json', jobs)
    put(out / 'private_assets.json', assets)
    put(out / 'private_sources.json', sources)
    put(out / 'source_fallback.json', fallback)
    code = {d['path']: d for d in oldreg['code']}
    for name in SOURCE_NAMES:
        d = pin(E / name); code[d['path']] = d
    for d in code.values():
        verify(d)
    capacities = {k: LAYOUTS[k] for k in {c['layout_id'] for c in (POLICIES[r['policy_id']] for r in cells)}}
    reg = dict(schema='N2048_B2B3_REGISTERED_TRANSPORT_V1', phase=phase,
        basis_HEAD=subprocess.check_output(['git', '-C', str(R), 'rev-parse', 'HEAD'], text=True).strip(),
        authorization=auth_desc, user_request=request, code=list(code.values()), caps=CAPS[phase], cells=cells,
        seeds=list(SEEDS[phase]), noise_namespace='NEST_ARPC_N2048_EQUAL_PROTECTION_V1',
        population=POPULATIONS[phase], source_count=len(sources),
        private_jobs=pin(out/'private_jobs.json'), private_assets=pin(out/'private_assets.json'),
        private_sources=pin(out/'private_sources.json'), source_fallback=pin(out/'source_fallback.json'),
        source_catalogue=cat_desc, source_population=sources_desc, selection=selection_desc,
        backend_identity=catalogue['backend_identity'], backend_catalogue=pin(OLD_ADAPTIVE_L/'engineering/backend/catalogue.json'),
        old_B0=pin(OLD_ADAPTIVE_O/'b0_completion.json'), old_B1_review=pin(OLD_ADAPTIVE_O/'qualification/review.json'),
        old_saved_RX_closure=read(OLD_ADAPTIVE_O/'qualification/phy_completion.json')['closure'],
        capacities=capacities, model_routes=oldreg['model_routes'], static=oldreg['static'],
        stage_seconds=dict(tx=7200, rx=43200, audit=1800),
        new_source_probability=0, new_tokenizer=0, training=0,
        native_and_metrics_separate=True, automatic_retry=False, auto_next_phase=False)
    put(light/'registration.json', reg)
    put(light/'preparation_completion.json', dict(status='METADATA_PREPARATION_COMPLETE_NOT_TRANSMITTED',
        registration=pin(light/'registration.json'), logical_objects=len(jobs),
        unique_selected_source_objects=len(assets), no_fit_source_policy=sum(x['source_no_fit'] for x in fallback),
        source_policy_count=len(fallback), scientific_calls=0))
    print(json.dumps(pin(light/'registration.json')), flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('phase', choices=['B2', 'B3'])
    a.add_argument('--authorization', required=True); a.add_argument('--authorization-sha256', required=True)
    a.add_argument('--authorization-bytes', type=int, required=True)
    a.add_argument('--request-path', required=True)
    a.add_argument('--selection'); a.add_argument('--selection-sha256'); a.add_argument('--selection-bytes', type=int)
    x = a.parse_args()
    auth = dict(path=x.authorization, sha256=x.authorization_sha256, bytes=x.authorization_bytes)
    selection = None if x.selection is None else dict(path=x.selection, sha256=x.selection_sha256, bytes=x.selection_bytes)
    try:
        register(x.phase, auth, x.request_path, selection)
    except BaseException:
        fail(L / x.phase)
        raise
