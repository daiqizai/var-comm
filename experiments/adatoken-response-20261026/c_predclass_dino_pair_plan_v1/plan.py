"""Pure logical-to-pixel join after all receiver outputs have closed."""
import copy
import math
from contracts import validate_grid, METRIC


def build_plan(sources, logical, outputs, references, reuse, gray_sha,
               donor_metric_ids, expected_sources=1000):
    assert len(sources)==expected_sources and len(gray_sha)==64
    assert {s['source_index']for s in sources}==set(range(expected_sources))
    ids={s['source_id']:s for s in sources};assert len(ids)==expected_sources
    grid=[dict(f,**{METRIC:0.0})for f in logical]
    validate_grid(grid,sources,expected_sources)
    assert len({f['event_key']for f in logical})==len(logical)
    refs={v['source_id']:v for v in references};assert len(refs)==len(references)==expected_sources
    used={f['RX_key']for f in logical if f['RX_key']is not None}
    oi={v['key']:v for v in outputs};assert len(oi)==len(outputs)and set(oi)==used
    for key,v in oi.items():
        assert v['source_truth_used']is False and v['TX_CDF_used']is False
        assert len(v['pixel_sha256'])==64 and v['image']is not None
        assert v.get('status','ACCEPTED_SOURCE')in ('ACCEPTED_SOURCE','FIXED_GRAY_SOURCE_REJECT')
        if v.get('status')=='FIXED_GRAY_SOURCE_REJECT':assert v['pixel_sha256']==gray_sha
    previous={}
    for v in reuse:
        sid=v['source_id'];assert sid in ids and v['archive']==ids[sid]['archive']
        assert v['metric_identity']in donor_metric_ids and math.isfinite(v['score'])
        key=(sid,v['pixel_sha256']);assert key not in previous
        previous[key]=v['file']
    by_source={sid:{}for sid in ids}
    for f in logical:
        sid=f['source_id'];rx=f['RX_key']
        if rx is None:
            assert f['origin']=='FIXED_GRAY'and f['projection_status']=='FIXED_GRAY'
            pixel=gray_sha;image=None;status='FIXED_GRAY_PHY_OR_CONTAINER'
        else:
            assert f['origin']in ('EXACT_CLOSED_RX_REUSE','NEW_UNIQUE_RX')
            assert f['projection_status']=='SOURCE_INPUT'
            out=oi[rx];pixel=out['pixel_sha256'];image=out['image']
            status=out.get('status','ACCEPTED_SOURCE')
        p=by_source[sid].setdefault(pixel,dict(image=image,pixel_sha256=pixel,
            reuse=previous.get((sid,pixel)),frames=[]))
        # Pixel identity can merge an actual gray source-reject with PHY gray.
        # All individual events, statuses and exact output files stay in frames.
        p['frames'].append(dict(f,source_RX_status=status,actual_image=image,
                                actual_pixel_sha256=pixel))
    result=[]
    for s in sorted(sources,key=lambda v:v['source_index']):
        sid=s['source_id'];ref=refs[sid]
        assert ref['source_index']==s['source_index']and ref['source']==s['archive']
        assert len(s['source_float_sha256'])==len(ref['feature_array_sha256'])==64
        result.append(dict(source_id=sid,source_index=s['source_index'],archive=copy.deepcopy(s['archive']),
            source_float_sha256=s['source_float_sha256'],reference=copy.deepcopy(ref['file']),
            reference_feature_sha256=ref['feature_array_sha256'],pairs=list(by_source[sid].values())))
    pairs=[p for s in result for p in s['pairs']]
    assert sum(len(p['frames'])for p in pairs)==18*expected_sources
    return dict(sources=result,counts=dict(source_count=expected_sources,logical_frames=len(logical),
        source_pair_rows=len(pairs),planned_new_DINO_upper_bound=sum(p['reuse']is None for p in pairs),
        planned_old_pair_reuses=sum(p['reuse']is not None for p in pairs),
        all_failures_retained=True),no_model_or_metric_calls=True)
