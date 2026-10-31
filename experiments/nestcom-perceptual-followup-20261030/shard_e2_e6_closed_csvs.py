"""Lossless metadata-only publication of closed E2/E6 frame tables."""
from followup_common import *
import io


def main():
    for stage, base in [('e2', 'remaining_evidence_status_v2'), ('e6', 'remaining_evidence_v1')]:
        root = D / base / stage
        closure = read(root / 'execution_closure.json')
        assert closure['actual_wait_returncode'] == 0 and closure['owner_gone'] and closure['child_gone']
        assert not closure['failure_present'] and sha(root / 'summary.csv') == closure['summary_sha256']
        assert (root / 'delivery_manifest.json').is_file()
        src = root / 'frame_metrics.csv'
        original = src.read_bytes()
        with src.open(newline='', encoding='utf8') as f:
            reader = csv.DictReader(f)
            fields, records = reader.fieldnames, list(reader)
        assert len(records) == (15000 if stage == 'e2' else 30000)
        check = io.StringIO(newline='')
        writer = csv.DictWriter(check, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        assert check.getvalue().encode('utf8') == original
        out = root / 'frame_metrics_parts'
        out.mkdir(exist_ok=False)
        parts, combined = [], []
        for start in range(0, len(records), 5000):
            p = out / f'part_{start // 5000:03d}.csv'
            chunk = records[start:start + 5000]
            csvout(p, chunk)
            assert p.stat().st_size < 9_000_000
            with p.open(newline='', encoding='utf8') as f:
                back = list(csv.DictReader(f))
            assert back == chunk
            combined.extend(back)
            parts.append(dict(path=str(p.relative_to(root)), bytes=p.stat().st_size,
                              sha256=sha(p), row_start=start, row_stop=start + len(chunk)))
        assert combined == records and src.read_bytes() == original
        save(root / 'csv_shard_identity.json', dict(original=pin(src), rows=len(records),
             fields=fields, parts=parts, original_csv_serialization_reconstructs_exact_bytes=True,
             scalar_strings_preserved=True, scientific_calls=0))
        text = ('# Closed ' + stage.upper() + ' evidence\n\n'
                'All named execution roles actually exited successfully; owner and child identities '
                'were checked as gone. See execution_closure.json and delivery_manifest.json.\n\n'
                'The original frame_metrics.csv exceeds the unchanged repository 10 MB limit and '
                'remains on PFS, untracked. Published frame_metrics_parts/part_*.csv retain every '
                'full-precision scalar string in original order. csv_shard_identity.json binds the '
                'original and parts; one header plus their rows reconstructs the original CSV bytes. '
                'No scoring, generation, PHY, or bootstrap was repeated for publication.\n\n')
        if stage == 'e2':
            text += ('This is error-free information transfer, not new wireless frames: 15,000 '
                     'logical outputs and 9,000 source/method/budget units. Ada generation repeats '
                     'are averaged within source before pairing. All six budgets and all five metrics '
                     'are reported, including unfavorable high-budget perceptual differences. The '
                     'first metadata-only export expected SOURCE_INPUT instead of the actual '
                     'ACCEPTED_SOURCE label; its partial files and failure diagnosis are preserved. '
                     'This corrected export does not change scientific output or receiver status.\n')
        else:
            text += ('These are 30,000 actual receptions on the registered 100 sources and frozen '
                     '10 dB policies. Overall metrics retain all gray rejections. Successful-only '
                     'quality is descriptive and has an SNR-dependent event denominator; it is '
                     'undefined at 8 dB, where all receptions fail.\n')
        with (root / 'README.md').open('x', encoding='utf8') as f:
            f.write(text)
        print(json.dumps(dict(stage=stage, rows=len(records), parts=len(parts), scientific_calls=0)))


if __name__ == '__main__':
    main()
