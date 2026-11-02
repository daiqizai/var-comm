# Complete frame data storage

The complete 18,000-row formal/frame_metrics.csv is delivered as three plain CSV row-range parts, each below the repository's 10 MB per-file limit. The repository also excludes compressed archives, so the temporary gzip attempt is retained only on PFS and is not a Git deliverable. No values, columns, row ordering, or line endings change. data_storage.json pins the raw file, header and three parts; recombination was verified byte-for-byte. The original SHA-bound scientific CSV remains unchanged on PFS.

From this result directory, restore with Python's standard library:

    from pathlib import Path
    import json, hashlib
    m = json.loads(Path('data_storage.json').read_text())
    pieces = [Path(r['path']).read_bytes() for r in m['parts']]
    header = pieces[0].splitlines(keepends=True)[0]
    b = pieces[0] + b''.join(p[len(header):] for p in pieces[1:])
    assert hashlib.sha256(b).hexdigest() == m['raw']['sha256']
    p = Path(m['raw']['path'])
    if p.exists():
        assert p.read_bytes() == b
    else:
        p.write_bytes(b)

This restores data only; do not relaunch completed stages. Preparation stderr logs stay outside Git, with lightweight transport/lineage records retaining their identities.
