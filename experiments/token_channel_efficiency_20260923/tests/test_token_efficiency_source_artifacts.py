"""Validate stored actual calibration artifacts; no GPU quality recomputation."""
import csv,gzip,hashlib,io,json
from pathlib import Path
import pytest
from token_efficiency.source import freeze_targets,summarize
from token_efficiency.common import ROOT,EXP

FOLDER=ROOT/'results/token_channel_efficiency_20260923/source_calibration'

def test_actual_calibration_artifacts_targets_and_one_source_object():
    lineage=json.loads((FOLDER/'lineage.json').read_text())
    assert lineage['sources']==1000 and lineage['source_rows']==25000 and not lineage['synthetic'] and not lineage['finite_channel_ranking']
    raw=(FOLDER/'source_codec_per_image.csv').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==lineage['files']['source_codec_per_image.csv']['uncompressed_sha256']
    rows=list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(rows)==25000 and len({(r['source_id'],r['method']) for r in rows})==25000
    assert len({r['source_id'] for r in rows})==1000 and {r['population'] for r in rows}=={'calibration'}
    assert {r['N'] for r in rows}=={'not_applicable_source_reference'}
    for r in rows:
        if r['method'].startswith('codec_m'):
            assert int(r['raw_bits'])=={6:1092,7:1860,8:3060,9:5088,10:8160}[int(r['m'])]
            assert r['roundtrip_tokens_equal']=='True'
    targets=json.loads((EXP/'quality_targets.json').read_text())
    assert targets==freeze_targets(rows,hashlib.sha256(raw).hexdigest())
    actual={r['method']:r for r in csv.DictReader((FOLDER/'source_codec_summary.csv').open())}
    for expected in summarize(rows):
        record=actual[expected['method']]
        for key,value in expected.items():
            if isinstance(value,(float,int)):assert float(record[key])==pytest.approx(value,rel=1e-12,abs=1e-12)
            else:assert record[key]==value
    ledger=(FOLDER/'payload_ledger.csv').read_bytes()
    assert hashlib.sha256(ledger).hexdigest()==lineage['files']['payload_ledger.csv']['uncompressed_sha256']
    costs=list(csv.DictReader(io.StringIO(ledger.decode())))
    assert len(costs)==10000 and len({(r['source_id'],r['m'],r['family']) for r in costs})==10000

def test_actual_development_source_scope_frozen_targets_and_summary():
    folder=ROOT/'results/token_channel_efficiency_20260923/source_development'
    lineage=json.loads((folder/'lineage.json').read_text());targets=json.loads((EXP/'quality_targets.json').read_text())
    assert lineage['sources']==100 and lineage['quality_targets_at_development_start']==targets
    assert not lineage['new_holdout_accessed'] and not lineage['synthetic'] and not lineage['finite_channel_ranking']
    for name,record in lineage['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==record['sha256']
    rows=list(csv.DictReader((folder/'source_codec_per_image.csv').open()))
    assert len(rows)==2500 and len({(r['source_id'],r['method']) for r in rows})==2500 and len({r['source_id'] for r in rows})==100
    assert {r['population'] for r in rows}=={'development'} and {r['N'] for r in rows}=={'not_applicable_source_reference'}
    published={r['method']:r for r in csv.DictReader((folder/'source_codec_summary.csv').open())}
    for expected in summarize(rows):
        for key,value in expected.items():
            actual=published[expected['method']][key]
            if isinstance(value,(float,int)):assert float(actual)==pytest.approx(value,rel=1e-12,abs=1e-12)
            else:assert actual==value
    costs=list(csv.DictReader((folder/'payload_ledger.csv').open()));assert len(costs)==1000
    for r in costs:assert int(r['new_protocol_header_uses'])=={'raw':70,'arithmetic':96}[r['family']]
