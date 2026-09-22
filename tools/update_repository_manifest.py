#!/usr/bin/env python3
"""Hash the explicitly staged root files. Does not copy or publish source."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
old=json.loads((ROOT/'docs/history/release_manifest_before_root_migration.json').read_text())
original={r['published_path']:r['source_sha256'] for r in old['files']}
rows=[]
for name in sorted(filter(None,names)):
    if name=='release_manifest.json': continue
    data=(ROOT/name).read_bytes(); sha=hashlib.sha256(data).hexdigest()
    rows.append(dict(published_path=name,bytes=len(data),source_sha256=original.get(name,sha),published_sha256=sha))
(ROOT/'release_manifest.json').write_text(json.dumps(dict(management_mode='single_project_worktree',source_artifacts_modified=True,model_weights_dataset_pixels_paper_PDFs_and_credentials_included=False,exported_files=len(rows),files=rows),indent=2)+'\n')
