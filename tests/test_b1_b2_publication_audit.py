"""Compact exports must preserve complete real frame fields."""
import pytest
from tools.audit_b1_b2_publication import FIELDS, compare_compact

def test_compact_preserves_failure_and_bit_ledger():
    original={'body_crc_ok':False,'payload_bits':5088,'mother_bits':10220}
    row={'source_index':'0','method_index':'0',**{k:str(original.get(k,'')) for k in FIELDS}}
    compare_compact(row,original)
    row['payload_bits']='1092'
    with pytest.raises(RuntimeError):compare_compact(row,original)

def test_compact_missing_field_is_rejected():
    row={'source_index':'0','method_index':'0',**{k:'' for k in FIELDS}}
    del row['observation_sha256']
    with pytest.raises(RuntimeError):compare_compact(row,{})
