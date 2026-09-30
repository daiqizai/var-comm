#!/usr/bin/env python3
"""Archive/restore the two full Step2 A tables under the 10 MB Git file limit."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'results/rx_posterior_step2_A_20260930_R1'
TABLES = {'per_frame.csv': 25200, 'token_accuracy.csv': 108000}
MAX_BYTES = 8_000_000

def digest(data):
    return hashlib.sha256(data).hexdigest()

def row_count(data):
    with io.TextIOWrapper(io.BytesIO(data), encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError('CSV field count changed')
            rows += 1
    return rows

def immutable_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing artifact differs: ' + str(path))
    else:
        path.write_bytes(data)

def export(results):
    original = json.loads((results / 'provenance/original_result_sha256.json').read_text())
    manifest = dict(schema=1, run='RX-POSTERIOR-STEP2-A-20260930-R1',
        max_part_bytes=MAX_BYTES, reconstruction='first header + ordered shard data rows; exact original bytes', tables={})
    for name, expected_rows in TABLES.items():
        data = (results / name).read_bytes()
        if digest(data) != original[name] or row_count(data) != expected_rows:
            raise ValueError('Original table identity/row count differs: ' + name)
        lines = data.splitlines(keepends=True)
        header, data_lines = lines[0], lines[1:]
        if len(data_lines) != expected_rows:
            raise ValueError('Export requires one physical line per CSV record')
        groups, current, size = [], [], len(header)
        for line in data_lines:
            if len(header) + len(line) > MAX_BYTES:
                raise ValueError('Single record exceeds part limit')
            if current and size + len(line) > MAX_BYTES:
                groups.append(current)
                current, size = [], len(header)
            current.append(line)
            size += len(line)
        if current:
            groups.append(current)
        parts, offset = [], 0
        for i, lines_in_part in enumerate(groups):
            part = header + b''.join(lines_in_part)
            relative = f'table_shards/{Path(name).stem}_{i:03d}.csv'
            immutable_write(results / relative, part)
            count = row_count(part)
            parts.append(dict(path=relative, sha256=digest(part), bytes=len(part),
                              rows=count, first_data_row=offset + 1, last_data_row=offset + count))
            offset += count
        manifest['tables'][name] = dict(sha256=digest(data), bytes=len(data), rows=expected_rows,
            header_sha256=digest(header), parts=parts)
    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode()
    immutable_write(results / 'table_shards/manifest.json', encoded)
    verify_or_restore(results, False)

def verify_or_restore(results, restore):
    manifest = json.loads((results / 'table_shards/manifest.json').read_text())
    if set(manifest['tables']) != set(TABLES):
        raise ValueError('Unexpected table set')
    base = results.resolve()
    for name, info in manifest['tables'].items():
        header, blocks, count = None, [], 0
        for part in info['parts']:
            path = (results / part['path']).resolve()
            if not path.is_relative_to(base) or path.parent != base / 'table_shards':
                raise ValueError('Part path outside table_shards')
            data = path.read_bytes()
            if digest(data) != part['sha256'] or len(data) != part['bytes'] or len(data) > MAX_BYTES:
                raise ValueError('Part bytes/hash/size differ: ' + str(path))
            this_header, body = data.split(b'\n', 1)
            this_header += b'\n'
            if header is None:
                header = this_header
            if this_header != header or digest(header) != info['header_sha256']:
                raise ValueError('Part header differs')
            rows = row_count(data)
            if rows != part['rows'] or part['first_data_row'] != count + 1 or part['last_data_row'] != count + rows:
                raise ValueError('Part row count/order differs')
            count += rows
            blocks.append(body)
        rebuilt = header + b''.join(blocks)
        if digest(rebuilt) != info['sha256'] or len(rebuilt) != info['bytes'] or count != TABLES[name] or row_count(rebuilt) != count:
            raise ValueError('Full table identity/rows differ: ' + name)
        if restore:
            immutable_write(results / name, rebuilt)
        print(json.dumps(dict(table=name, status='RESTORED' if restore else 'VERIFIED',
            rows=count, bytes=len(rebuilt), sha256=digest(rebuilt), parts=len(blocks))))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'verify', 'restore'])
    parser.add_argument('--results', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.action == 'export':
        export(args.results)
    else:
        verify_or_restore(args.results, args.action == 'restore')

if __name__ == '__main__':
    main()
