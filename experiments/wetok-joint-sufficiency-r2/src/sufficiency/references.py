"""Keep all eighteen sealed5000 references unchanged while clearly labeling their older training opportunities."""

import json

from grid_controls.evaluation import load_evaluation as load_grid_evaluation
from grid_controls.references import ImageCache, References as JointReferenceChain, read_rows
from sufficiency.common import load_sources
from wetok_comm.common import PROJECT, sha256, verify_sources


def project_reference(row, receipt_sha):
    result = dict(row)
    if result.get('receiver_seconds', '') != '' or result.get('online_TX_seconds', '') != '':
        raise RuntimeError('the frozen quality reference contains current timing fields')
    result['r2_quality_origin'] = 'sealed_5000_reference'
    result['r2_reference_receipt_sha256'] = receipt_sha
    return result


class References:
    def __init__(self, evaluation, config, original, grid, reference):
        load_sources(config)
        grid_evaluation, grid_config, joint_config, old_reference, base, parent = load_grid_evaluation()
        self.previous = JointReferenceChain(grid_evaluation, grid_config, joint_config, old_reference)
        self.root = PROJECT / 'outputs' / evaluation['reference_quality']
        self.receipt_path = self.root / 'completion.json'
        self.receipt_sha = sha256(self.receipt_path)
        if self.receipt_sha != evaluation['reference_quality_sha256']:
            raise RuntimeError('frozen Grid5000 quality receipt changed')
        receipt = json.loads(self.receipt_path.read_text())
        if receipt['status'] != 'GRID_QUALITY_COMPLETE' or receipt['rows'] != 37800 or not receipt['quality_only_no_current_timing']:
            raise RuntimeError('Grid5000 reference population or measurement scope changed')
        verify_sources(receipt['source_hashes'])
        for relative, expected in receipt['output_hashes'].items():
            if sha256(self.root / relative) != expected:
                raise RuntimeError('a frozen Grid5000 quality artifact changed')
        analysis_path = PROJECT / 'outputs' / grid['outputs']['analysis'] / 'quality_0005000/completion.json'
        if sha256(analysis_path) != evaluation['reference_analysis_sha256']:
            raise RuntimeError('frozen Grid5000 analysis changed')
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
            if row['arm'].startswith('r2__'):
                continue
            expected = project_reference(self.rows[int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']], self.receipt_sha)
            if any(str(row.get(key, '')) != str(value) for key, value in expected.items()):
                raise RuntimeError('an older quality/selection/resource/image reference was altered')
