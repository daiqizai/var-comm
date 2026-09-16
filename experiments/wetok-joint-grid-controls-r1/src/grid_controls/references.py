"""Frozen Joint5000 quality reuse, without importing its old timings into new quality rankings."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from grid_controls.common import completed_joint_reference
from joint_sender.evaluation_io import digest
from wetok_comm.common import PROJECT, assets, sha256, verify_sources


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def project_reference(row, receipt_sha):
    projected = dict(row)
    for key in ('online_TX_seconds', 'receiver_seconds', 'receiver_order_index', 'latency_source'):
        projected['archived_' + key] = row.get(key, '')
        projected[key] = ''
    projected['latency_source'] = 'not_measured_in_this_quality_phase'
    projected['quality_origin'] = 'sealed_joint5000_reference'
    projected['quality_reference_sha256'] = receipt_sha
    return projected


def source_tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(value.dtype).encode())
    hasher.update(json.dumps(list(value.shape)).encode())
    hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


class References:
    def __init__(self, evaluation, config, original, reference):
        binding = completed_joint_reference(config, original, reference)
        self.root = PROJECT / 'outputs' / evaluation['original_joint_evaluation']
        self.receipt_path = self.root / 'completion.json'
        self.receipt_sha = sha256(self.receipt_path)
        if (self.receipt_sha != evaluation['original_joint_evaluation_sha256'] or
            binding['hashes']['joint_analysis'] != evaluation['original_joint_analysis_sha256']):
            raise RuntimeError('registered original Joint evaluation/analysis changed')
        receipt = json.loads(self.receipt_path.read_text())
        if receipt['status'] != 'JOINT_EVALUATION_COMPLETE' or receipt['rows'] != 33600:
            raise RuntimeError('original Joint quality matrix is incomplete')
        verify_sources(receipt['source_hashes'])
        for relative, expected in receipt['output_hashes'].items():
            if sha256(self.root / relative) != expected:
                raise RuntimeError('sealed original Joint artifact changed')
        self.receipt = receipt
        self.raw_rows = read_rows(self.root / 'per_frame.csv')
        self.rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in self.raw_rows}
        self.clean = {int(row['image_index']): row for row in read_rows(self.root / 'native_reference.csv')}
        self.noiseless = read_rows(self.root / 'noiseless_mapping.csv')
        self.support = read_rows(self.root / 'deep_support_supplement.csv')
        self.frozen_models = receipt['frozen_models_before']
        native_root = Path(assets()['native_reference'])
        native_receipt = json.loads((native_root / 'completion.json').read_text())
        source_table = native_root / 'per_image.jsonl'
        self.source_table_sha = sha256(source_table)
        if self.source_table_sha != native_receipt['per_image_jsonl_sha256']:
            raise RuntimeError('the original source tensor identity table changed')
        self.source_identities = {int(row['image_index']): (row['image_id'], row['source_tensor_sha256'])
            for row in (json.loads(line) for line in source_table.read_text().splitlines())}
        if set(self.source_identities) != set(range(100)):
            raise RuntimeError('the original source tensor identity population is incomplete')

    def source_rows(self, index):
        return [project_reference(row, self.receipt_sha) for row in self.raw_rows if int(row['image_index']) == index]

    def validate_reuse(self, rows):
        for row in rows:
            if row['arm'].startswith('grid__'):
                continue
            key = int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']
            expected = project_reference(self.rows[key], self.receipt_sha)
            if any(str(row.get(name, '')) != str(value) for name, value in expected.items()):
                raise RuntimeError('a reused reference quality/selection/resource/image field changed')

    def validate_source(self, index, identifier, source):
        if self.source_identities[index] != (identifier, source_tensor_digest(source)):
            raise RuntimeError('quality input pixels are not the original frozen source tensor')


class ImageCache:
    def __init__(self):
        self.archives = {}

    def image(self, row):
        path = Path(row['image_archive'])
        if path not in self.archives:
            with np.load(path, allow_pickle=False) as archive:
                self.archives[path] = archive['images']
        image = self.archives[path][int(row['image_ref'])]
        if digest(image) != row['image_sha256']:
            raise RuntimeError('referenced reconstruction pixels changed')
        return image
