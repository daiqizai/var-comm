"""All twenty-two R2 quality methods remain unchanged when referenced by the R3 trial."""

import json

from sufficiency.evaluation import load_evaluation as load_r2_evaluation
from sufficiency.references import ImageCache, References as R2ReferenceChain, read_rows
from wetok_comm.common import PROJECT, sha256, verify_sources


def project_reference(row, receipt_sha):
    if row.get('receiver_seconds', '') != '' or row.get('online_TX_seconds', '') != '':
        raise RuntimeError('a frozen R2 quality row contains current latency')
    return {**row, 'r3_quality_origin': 'sealed_R2_reference', 'r3_reference_receipt_sha256': receipt_sha}


class References:
    def __init__(self, evaluation, config, r2, original, grid, reference):
        old_evaluation, old_config, old_original, old_grid, old_reference, base, parent = load_r2_evaluation()
        self.previous = R2ReferenceChain(old_evaluation, old_config, old_original, old_grid, old_reference)
        self.root = PROJECT / 'outputs' / evaluation['reference_quality']
        self.receipt_path = self.root / 'completion.json'
        self.receipt_sha = sha256(self.receipt_path)
        if self.receipt_sha != evaluation['reference_quality_sha256']:
            raise RuntimeError('the frozen R2 quality receipt changed')
        receipt = json.loads(self.receipt_path.read_text())
        if receipt['status'] != 'R2_QUALITY_COMPLETE' or receipt['rows'] != 46200 or not receipt['quality_only_no_current_timing']:
            raise RuntimeError('R2 reference population or measurement scope changed')
        verify_sources(receipt['source_hashes'])
        for relative, expected in receipt['output_hashes'].items():
            if sha256(self.root / relative) != expected:
                raise RuntimeError('a frozen R2 quality artifact changed')
        analysis_path = PROJECT / 'outputs' / r2['outputs']['analysis'] / 'quality_0010000/completion.json'
        if sha256(analysis_path) != evaluation['reference_analysis_sha256']:
            raise RuntimeError('the completed R2 quality analysis changed')
        self.receipt = receipt
        self.raw_rows = read_rows(self.root / 'per_frame.csv')
        self.rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in self.raw_rows}
        self.clean = {int(row['image_index']): row for row in read_rows(self.root / 'native_reference.csv')}
        self.noiseless = read_rows(self.root / 'noiseless_mapping.csv')
        self.support = read_rows(self.root / 'deep_support_supplement.csv')
        self.frozen_models = receipt['frozen_models_before']
        self.source_table_sha = self.previous.source_table_sha

    def validate_source(self, index, identifier, source):
        self.previous.validate_source(index, identifier, source)

    def source_rows(self, index):
        return [project_reference(row, self.receipt_sha) for row in self.raw_rows if int(row['image_index']) == index]

    def validate_reuse(self, rows):
        for row in rows:
            if row['arm'].startswith('r3__'):
                continue
            expected = project_reference(self.rows[int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']], self.receipt_sha)
            if any(str(row.get(key, '')) != str(value) for key, value in expected.items()):
                raise RuntimeError('R3 altered a frozen quality/selection/resource/image reference')
