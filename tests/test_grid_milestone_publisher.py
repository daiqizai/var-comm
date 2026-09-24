"""CPU engineering checks for lossless publication, not model quality."""
import hashlib
import json
from pathlib import Path
import pytest
from tools.publish_grid_milestones import split_csv, main

def test_csv_parts_preserve_utf8_crlf_and_order(tmp_path):
    source = tmp_path / "source.csv"
    data = "id,text\r\n1,校准\r\n2,\"quoted,field\"\r\n3,end\r\n".encode()
    source.write_bytes(data)
    target = tmp_path / "parts"
    split_csv(source, target, limit=26)
    manifest = json.loads((target / "manifest.json").read_text())
    rebuilt = b"".join((target / p["path"]).read_bytes() for p in manifest["parts"])
    assert rebuilt == data
    assert manifest["original_sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["original_bytes"] == len(data)
    assert all(p["bytes"] <= 26 for p in manifest["parts"])
    for p in manifest["parts"]:
        assert p["sha256"] == hashlib.sha256((target / p["path"]).read_bytes()).hexdigest()

def test_existing_parts_are_never_overwritten(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"a,b\n1,2\n")
    target = tmp_path / "parts"
    split_csv(source, target)
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    with pytest.raises(FileExistsError):
        split_csv(source, target)
    assert before == {p.name: p.read_bytes() for p in target.iterdir()}

def test_publisher_requires_explicit_cpu_mask(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="explicit CUDA mask"):
        main()
