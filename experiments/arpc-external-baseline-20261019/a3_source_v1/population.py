"""Fixed evaluation population and calibration-policy gate, no image selection."""
def evaluate_sources(sources,calibration):
    assert len(sources)==100 and len({s['source_id'] for s in sources})==100
    assert not {s['source_id'] for s in sources}&{s['source_id'] for s in calibration}
    assert {s['rank'] for s in sources}==set(range(100))
    return [dict(s,rank=s['source_index']) for s in sources]
def policies(record):
    assert record['status']=='FROZEN_FROM_FULL_CAL1000' and record['sources']==1000 and record['logical_frames']==18000
    assert set(record['policies'])=={'4','10','19'} and record['A3_logical_cap']==900
    return record['policies']
