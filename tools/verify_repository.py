#!/usr/bin/env python3
"""Check tracked content, source-export coverage, dependency bytes and no hidden source."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {"publish", "outputs", "checkpoints", "datasets", "data", "weights", "vendor", "third_party", ".venv", "venv", "__pycache__", "_migration_backup"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".model", ".safetensors", ".npy", ".npz", ".pkl", ".pickle", ".so", ".o", ".whl", ".pem", ".key", ".p12", ".zip", ".tgz", ".gz"}
PATTERNS = [re.compile(x) for x in [r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}",r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",r"https?://[^\s/@:]+:[^\s/@]+@",r"\bAKIA[A-Z0-9]{16}\b"]]

def main():
    top=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],cwd=ROOT,text=True).strip()).resolve()
    assert top==ROOT, (top,ROOT)
    entries=subprocess.check_output(["git","ls-files","--stage","-z"],cwd=ROOT).decode().strip("\0").split("\0")
    names=set()
    for entry in entries:
        meta,name=entry.split("\t",1); path=Path(name);names.add(name)
        assert meta.split()[0] not in {"160000","120000"}, f"nested repository or symlink: {name}"
        assert not set(path.parts)&FORBIDDEN_PARTS and path.suffix not in FORBIDDEN_SUFFIXES, f"excluded tracked file: {name}"
        assert not path.name.startswith((".env", "id_rsa", "id_ed25519")), f"secret filename: {name}"
        p=ROOT/path; assert p.is_file(),name
        assert p.stat().st_size<=10_000_000, f"large tracked file: {name}"
        if p.suffix in {".py",".cpp",".sh",".md",".json",".yaml",".yml",".csv",".txt",".toml",".html",".svg"}:
            assert not any(x.search(p.read_text()) for x in PATTERNS), f"possible secret: {name}"
    manifest=ROOT/"results/repository_migration_20260923/source_export_003_coverage.json"
    covered=json.loads(manifest.read_text())
    for item in covered:
        if item["disposition"]=="included_source_or_light_result": assert item["path"] in names,item["path"]
    for item in json.loads((manifest.parent/"self_written_dependencies.json").read_text()):
        assert item["path"] in names,item["path"]
        assert hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest()==item["sha256"],item["path"]
    untracked=subprocess.check_output(["git","ls-files","--others","--exclude-standard","-z"],cwd=ROOT).decode().split("\0")
    missing=[p for p in untracked if p and Path(p).suffix in {".py",".cpp",".h",".sh"}]
    assert not missing, f"untracked self-written source: {missing}"
    print(json.dumps({"status":"PASS","tracked_files":len(names),"source_export_entries_accounted":len(covered),"GPU":"NOT_RUN"},indent=2))

if __name__=="__main__": main()
