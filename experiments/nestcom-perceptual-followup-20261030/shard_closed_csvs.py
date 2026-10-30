"""Lossless CSV publication shards under the repository's 10 MB file limit.

Original files and original completion manifests remain byte-for-byte unchanged.
This performs serialization only, without metrics, renders or statistics.
"""
from followup_common import *
import io

def main():
    root=D/'closed_evidence_v1';dest=root/'e1';index=[]
    for name in ['dists_per_frame.csv','native_output_inventory.csv']:
        src=dest/name;original=src.read_bytes()
        with src.open(newline='',encoding='utf8') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;records=list(reader)
        # Verify exact original serialization, not merely approximate numbers.
        check=io.StringIO(newline='');w=csv.DictWriter(check,fieldnames=fields);w.writeheader();w.writerows(records)
        assert check.getvalue().encode('utf8')==original
        out=dest/(src.stem+'_parts');out.mkdir(exist_ok=False);parts=[];combined=[]
        for start in range(0,len(records),9000):
            p=out/f'part_{start//9000:03d}.csv';chunk=records[start:start+9000];csvout(p,chunk)
            assert p.stat().st_size<=9_000_000
            with p.open(newline='',encoding='utf8') as f:back=list(csv.DictReader(f))
            assert back==chunk;combined.extend(back)
            parts.append(dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size,row_start_inclusive=start,row_stop_exclusive=start+len(chunk),rows=len(chunk)))
        assert combined==records
        index.append(dict(original=pin(src),field_order=fields,rows=len(records),parts=parts,concatenation='Read parts in listed order, retain header once; preserve CSV values as strings.',original_csv_serialization_reconstructs_exact_bytes=True))
    save(root/'csv_shard_identity.json',dict(reason='Unmodified tools/verify_repository.py requires each tracked file <=10,000,000 bytes.',files=index,originals_unchanged=True,new_scientific_calls=0))
    (root/'README.md').write_text('''# Closed E1/E3/E4 scalar delivery

These three stages are actually closed. E2/E5/E6 and the final V7 manuscript are not certified by this package. `delivery_manifest.json` preserves the original inputs, true wait receipts and original export hashes.

The repository limits each tracked file to 10 MB. Therefore the two large E1 tables are published as `e1/dists_per_frame_parts/part_*.csv` and `e1/native_output_inventory_parts/part_*.csv`. `csv_shard_identity.json` lists exact order, fields, rows, hashes and original file identities. Concatenating parsed rows in that order and writing the single original header reconstructs the original CSV bytes exactly. The original unsharded files remain at their pinned runtime paths and are not staged. All scalar strings, including full-precision metrics, are unchanged; no scientific computation was repeated.

Join frame records to native inputs by `dists_pair_key`. There are 66,000 frame rows,28,253 unique native pairs and22,000 source means. The full five-metric summary and direct paired tables have220 rows each. `dists_summary.csv` and `dists_paired_deltas.csv` are DISTS-only subsets. E3/E4 each use their explicitly recorded100-source population, not the500-source population. All methods, failure-gray outcomes and unfavorable directions are retained.

The original BPG7dB stored-gray annotation is preserved separately in `e1/failure_count_annotation.json`; no old metadata or quality score was overwritten. The reports distinguish a confidence interval spanning zero from evidence of a direction. Arrays, weights and logs remain on PFS.
''',encoding='utf8')
    print(json.dumps(dict(status='LOSSLESS_SERIALIZATION_ONLY',tables=len(index),rows=[x['rows'] for x in index],shards=sum(len(x['parts']) for x in index),identity=pin(root/'csv_shard_identity.json'))))
if __name__=='__main__':main()
