"""CPU-only admission of the user's selected saved reconstructions."""
import argparse
import copy
import csv
import json
from pathlib import Path

from build_cached_studies import Builder, COMPLETED_STATUSES, pointer_piece, read_rows
import historical_cached as cache


def narrow(spec, method_keys, rows):
    spec = copy.deepcopy(spec)
    cache.require(set(method_keys) <= set(spec['methods']), 'Unknown selected cached method')
    spec['selected_method_keys'] = list(method_keys)
    spec['methods'] = {k: spec['methods'][k] for k in method_keys}
    selected = [r for r in rows if '|'.join(str(r[k]) for k in spec['method_key_columns']) in method_keys]
    spec['expected_rows'] = len(selected)
    spec['rows_per_source'] = [sum(r[spec['source_id_column']] == x['image_id'] for r in selected) for x in spec['records']]
    cache.require(all(spec['rows_per_source']), 'Selection omitted a source')
    return spec


def build(root):
    root = Path(root).resolve()
    out = root/'outputs/HISTORICAL-METRICS-20261003'
    previous = cache.read(out/'cached_studies_draft_v1.json')
    studies = {}
    authors = copy.deepcopy(previous['studies']['EXTERNAL_AUTHORS'])
    authors['quality']['numerical_runtime'] = dict(matmul_tf32=False, cudnn_tf32=False,
        precision='highest', cudnn_benchmark=False, cudnn_deterministic=False,
        deterministic=False, threads=4, interop_threads=2)
    for key, meta in authors['methods'].items():
        meta.update(condition=key.split('|')[1], phy='author_original', snr_definition='physical_channel_snr_db')
    studies['SELECTED_EXTERNAL_AUTHORS'] = authors
    original = previous['studies']['LATENT_ORIGINAL']
    allrows = [r for table in original['tables'] for r in read_rows(table['path'])]
    studies['LATENT_ORIGINAL'] = narrow(original, ['m8_plus_latent_1024'], allrows)
    # This input is only a loader for OPTIONAL_PHASE2_MAIN; it is not a queued study.
    studies['LATENT_ORIGINAL']['loader_only'] = True
    b = Builder(root, root.parent/'var-next-scale-comm')
    base = root/'outputs/EXTERNAL-BASELINE-POSITIONING-20260916/inputs_001'
    complete = base/'completion.json'
    receipt = b.index_receipt(complete)
    cache.require(receipt['status'] == 'INPUTS_AND_FROZEN_REFERENCES_READY', 'Frozen comparison inputs incomplete')
    COMPLETED_STATUSES.add(receipt['status'])
    table = base/'reference_per_frame.csv'
    b.proof_index[str(table)] = dict(path=str(table), sha256=receipt['reference_csv_sha256'],
        receipt=str(complete), receipt_sha256=b.sha(complete), pointer='/reference_csv_sha256')
    spec = b.base_spec('LEGACY_N3060_FINAL', base, [table], ['method'], 'latent')
    spec['quality']['numerical_runtime'] = dict(matmul_tf32=False, cudnn_tf32=False,
        precision='highest', cudnn_benchmark=False, cudnn_deterministic=False,
        deterministic=False, threads=8, interop_threads=2)
    spec.update(archive_column='image_archive', slot_column='image_slot', image_hash_column='image_sha256')
    spec['metric_columns'].update(lpips_alex='lpips', dino_cosine='dino')
    selected = ['raw_adaptive', 'arithmetic_adaptive', 'perceptual_deepjscc']
    rows = [r for r in read_rows(table) if r['method'] in selected]
    cache.require(len(rows) == 4500, 'Expected three selected N3060 methods')
    for position, row in enumerate(read_rows(table)):
        if row['method'] not in selected:
            continue
        try:
            b.prove(spec, row['image_archive'])
        except RuntimeError as error:
            if 'Missing original file-hash proof:' not in str(error):
                raise
            b.pixel_prove(spec, row['image_archive'], table, position, 'image_sha256',
                          'images', [int(row['image_slot'])], row['image_sha256'])
    for path, digest in receipt['bindings'].items():
        cache.require(b.sha(path) == digest, 'Original final comparison dependency changed')
        spec['evidence_bindings'][path] = digest
    b.methods(spec, rows, study='LEGACY_N3060_FINAL')
    for key, meta in spec['methods'].items():
        meta.update(N=3060, E=6120, phy='QPSK' if key != 'perceptual_deepjscc' else 'continuous',
                    snr_definition='physical_channel_snr_db', condition='paid_class' if key != 'perceptual_deepjscc' else 'unconditional',
                    decoder='D0' if key != 'perceptual_deepjscc' else 'DeepJSCC_decoder',
                    original_protocol='20260916 frozen comparison; not current same-Dc D_U')
    b.finish('LEGACY_N3060_FINAL', spec, rows)
    spec['selected_method_keys'] = selected
    studies['LEGACY_N3060_FINAL'] = spec
    document = dict(status='REGISTERED_ORIGINAL_FLOAT_CACHES', studies=studies,
        training_updates=0, policy_selection_updates=0, gpu_parity_qualification='NOT_RUN',
        selected_scope='2026-10-03 narrowed mandatory plus authorized optional loader only')
    destination = out/'cached_studies.json'
    if destination.exists():
        cache.require(cache.read(destination) == document, 'Existing selected admission differs')
    else:
        destination.write_text(json.dumps(document, indent=2)+'\n')
    proof = []
    for name in studies:
        adapter = cache.CachedAdapter(root, name)
        proof.append(adapter.describe())
    audit = dict(status='SELECTED_CACHE_CPU_ADMISSION_PASS', scientific_result=False,
        gpu_parity='NOT_RUN', studies=proof, manifest_sha256=cache.sha(destination))
    (out/'selected_cache_admission.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    build(p.parse_args().root)
