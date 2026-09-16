"""Read-only qualification and image/waveform access for the completed receiver-only experiment."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import EXPERIMENT, complete_control
from innovation_comm.evaluation import validate_rows, validate_selection, validate_native_reference
from wetok_comm.common import PROJECT, sha256, verify_sources


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def qualify_reference(evaluation, config, reference, base):
    control, control_path, review_path = complete_control(config, reference)
    root = PROJECT / 'outputs' / evaluation['receiver_reference_evaluation']
    receipt_path = root / 'completion.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt['status'] != 'INNOVATION_EVALUATION_COMPLETE' or receipt['milestone_sha256'] != sha256(control_path) or
        receipt['rows'] != 27300 or receipt['noiseless_rows'] != 600 or receipt['separate_fixed_support_rows'] != 600 or
        receipt['frozen_before'] != receipt['frozen_after'] or not receipt['shared_s_y_every_frame']):
        raise RuntimeError('completed receiver reference does not match the required frozen control trial')
    verify_sources(receipt['source_hashes'])
    for relative, expected in receipt['output_hashes'].items():
        if sha256(root / relative) != expected:
            raise RuntimeError('receiver reference artifact changed')
    rows = read_rows(root / 'per_frame.csv')
    validate_rows(rows, reference, base)
    validate_selection(rows, control)
    validate_native_reference(rows + read_rows(root / 'deep_support_supplement.csv'), read_rows(root / 'native_reference.csv'),
                              read_rows(root / 'noiseless_mapping.csv'), reference, base)
    analysis_root = PROJECT / 'outputs' / evaluation['receiver_reference_analysis']
    analysis_path = analysis_root / 'completion.json'
    analysis = json.loads(analysis_path.read_text())
    if (analysis['status'] != 'INNOVATION_DEVELOPMENT_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE' or
        analysis['evaluation_receipt_sha256'] != sha256(receipt_path) or analysis['milestone_sha256'] != sha256(control_path)):
        raise RuntimeError('receiver reference has not completed its registered analysis')
    verify_sources(analysis['source_hashes'])
    for relative, expected in analysis['output_hashes'].items():
        if sha256(analysis_root / relative) != expected:
            raise RuntimeError('receiver reference analysis artifact changed')
    return {'status': 'JOINT_RECEIVER_REFERENCE_QUALIFICATION_PASS', 'reference_rows': len(rows),
        'receiver_milestone_sha256': sha256(control_path), 'receiver_review_sha256': sha256(review_path),
        'reference_evaluation_receipt_sha256': sha256(receipt_path), 'reference_analysis_receipt_sha256': sha256(analysis_path),
        'evaluation_config_sha256': sha256(EXPERIMENT / 'configs/evaluation.yaml'),
        'reference_source_ids': [next(row['image_id'] for row in rows if int(row['image_index']) == index) for index in range(100)],
        'new_model_inference': False, 'GPU_used': False, 'research_goal_complete': False}


class References:
    def __init__(self, evaluation):
        self.evaluation = evaluation
        self.qualification_path = PROJECT / 'outputs' / evaluation['qualification_file']
        self.qualification = json.loads(self.qualification_path.read_text())
        if (self.qualification['status'] != 'JOINT_RECEIVER_REFERENCE_QUALIFICATION_PASS' or
            self.qualification['evaluation_config_sha256'] != sha256(EXPERIMENT / 'configs/evaluation.yaml')):
            raise RuntimeError('Joint reference qualification is missing or belongs to different comparison settings')
        verify_sources(self.qualification['source_hashes'])
        self.root = PROJECT / 'outputs' / evaluation['receiver_reference_evaluation']
        if sha256(self.root / 'completion.json') != self.qualification['reference_evaluation_receipt_sha256']:
            raise RuntimeError('qualified receiver evaluation receipt changed')
        self.receipt = json.loads((self.root / 'completion.json').read_text())
        verify_sources(self.receipt['source_hashes'])
        for filename in ('per_frame.csv', 'native_reference.csv', 'noiseless_mapping.csv', 'deep_support_supplement.csv'):
            if sha256(self.root / filename) != self.receipt['output_hashes'][filename]:
                raise RuntimeError('qualified reference quality, source pairing or physical ledger changed')
        self.rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row
                     for row in read_rows(self.root / 'per_frame.csv')}
        self.clean_rows = {int(row['image_index']): row for row in read_rows(self.root / 'native_reference.csv')}
        self.noiseless_rows = {(int(row['image_index']), row['arm']): row for row in read_rows(self.root / 'noiseless_mapping.csv')}
        self.support_rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed'])): row
                             for row in read_rows(self.root / 'deep_support_supplement.csv')}
        self.index, self.archives, self.signals = None, {}, None

    def _source(self, index, identifier):
        if self.qualification['reference_source_ids'][index] != identifier:
            raise RuntimeError('Joint evaluation source order differs from the qualified receiver reference')
        if self.index == index:
            return
        self.index, self.archives = index, {}
        relative = f'images/{index:03d}/waveforms.npz'
        if sha256(self.root / relative) != self.receipt['output_hashes'][relative]:
            raise RuntimeError('qualified frozen-sender waveform archive changed')
        with np.load(self.root / relative, allow_pickle=False) as archive:
            self.signals = {key: archive[key].copy() for key in archive.files}

    def signal(self, index, identifier, snr):
        self._source(index, identifier)
        return torch.from_numpy(self.signals[f'shared__snr{float(snr)}'].copy())

    def observed(self, index, identifier, snr, seed):
        self._source(index, identifier)
        return torch.from_numpy(self.signals[f'received__snr{float(snr)}__seed{int(seed)}'].copy())

    def _image(self, row):
        path = Path(row['image_archive'])
        if path not in self.archives:
            if path.is_relative_to(self.root):
                relative = str(path.relative_to(self.root))
                if sha256(path) != self.receipt['output_hashes'][relative]:
                    raise RuntimeError('qualified receiver RGB archive changed')
            with np.load(path, allow_pickle=False) as archive:
                self.archives[path] = archive['images'].copy()
        image = self.archives[path][int(row['image_ref'])]
        if (image.dtype != np.float32 or image.shape != (3, 256, 256) or not np.isfinite(image).all() or
            hashlib.sha256(image.tobytes()).hexdigest() != row['image_sha256']):
            raise RuntimeError('qualified source image does not match its actual reference row')
        return torch.from_numpy(image.copy()), dict(row)

    def image(self, index, identifier, snr, seed, name, noise_hash):
        self._source(index, identifier)
        original = 'receiver__' + name.split('__', 1)[1] if name.startswith('frozen__') else name
        row = self.rows[index, float(snr), int(seed), original]
        if row['image_id'] != identifier or row['noise_sha256'] != noise_hash:
            raise RuntimeError('frozen reference source/noise pairing differs')
        header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
        if ((int(row['total_complex_uses']), int(row['header_uses']), int(row['data_uses'])) != (3060, header, 3060 - header) or
            abs(float(row['total_energy']) - 6120) > 1e-5):
            raise RuntimeError('qualified reference physical ledger differs')
        return self._image(row)

    def noiseless(self, index, identifier, variant):
        self._source(index, identifier)
        row = self.noiseless_rows[index, 'receiver__' + variant]
        if row['image_id'] != identifier:
            raise RuntimeError('noiseless reference source differs')
        return self._image(row)

    def support(self, index, identifier, snr, seed, noise_hash):
        self._source(index, identifier)
        row = self.support_rows[index, float(snr), int(seed)]
        if row['image_id'] != identifier or row['noise_sha256'] != noise_hash:
            raise RuntimeError('fixed-support Deep reference source/noise differs')
        return self._image(row)
