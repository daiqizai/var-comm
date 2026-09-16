#!/usr/bin/env python3
"""Check published file integrity and reject credentials or excluded large assets."""

import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".pem", ".key", ".p12", ".jpg", ".jpeg", ".webp"}
IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__", "reproduced", "outputs", ".cache"}
TEXT_SUFFIXES = {".py", ".cpp", ".md", ".csv", ".json", ".yaml", ".yml", ".txt", ".sh"}
PATTERNS = (
    re.compile(r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"(?<!/workspace)/home/(?!\.\.\.)[A-Za-z0-9_.-]+/"),
)


def check_manifest(root=ROOT):
    manifest = json.loads((root / "release_manifest.json").read_text())
    seen = set()
    for entry in manifest["files"]:
        relative = Path(entry["published_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise RuntimeError("unsafe or duplicated manifest path")
        seen.add(relative.as_posix())
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"missing or symlinked publication file: {relative}")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["published_sha256"]:
            raise RuntimeError(f"published artifact changed: {relative}")
    if len(seen) != manifest["exported_files"]:
        raise RuntimeError("manifest file count mismatch")
    return manifest


def check_files(root=ROOT):
    command = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True)
    names = command.stdout.decode().split("\0") if command.returncode == 0 and command.stdout else []
    candidates = [root / name for name in names if name] if names else [path for path in root.rglob("*") if path.is_file() and not set(path.relative_to(root).parts) & IGNORED_PARTS]
    total_bytes = 0
    for path in candidates:
        relative = path.relative_to(root)
        if path.is_symlink() or path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith(".env"):
            raise RuntimeError(f"excluded artifact in publication: {relative}")
        size = path.stat().st_size
        if size > 10_000_000:
            raise RuntimeError(f"oversized artifact: {relative}")
        total_bytes += size
        if path.suffix in TEXT_SUFFIXES:
            text = path.read_text()
            if any(pattern.search(text) for pattern in PATTERNS):
                raise RuntimeError(f"possible credential or original machine-home path: {relative}")
    return {"checked_files": len(candidates), "checked_bytes": total_bytes}


def main():
    manifest = check_manifest()
    checked = check_files()
    print(json.dumps({"status": "PASS", "exported_source_files": manifest["exported_files"], **checked,
                      "GPU_used": False, "network_used": False}, indent=2))


if __name__ == "__main__":
    main()
