#!/usr/bin/env python3
"""Verify that all Direction 2 PFS directories (code, checkpoints, tmp) contain 0 symlinks and 0 hardlinks."""
from __future__ import annotations

import json
from pathlib import Path
import sys

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from dir2_zpac.hygiene import PFS_CKPT_ROOT, PFS_CODE_ROOT, PFS_TMP_ROOT, verify_no_links


def main() -> None:
    roots = [PFS_CODE_ROOT, PFS_CKPT_ROOT, PFS_TMP_ROOT]
    report = verify_no_links(roots)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
