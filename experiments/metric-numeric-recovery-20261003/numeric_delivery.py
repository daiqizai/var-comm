"""Publish exact SNR string compatibility evidence after telemetry delivery.

This CPU publication entry point does not reconstruct or rescore any image.
It reuses the already published repository checking and normal push framework.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time


NAME = 'METRIC-NUMERIC-RECOVERY-20261003'
PREVIOUS_NAME = 'M2-TELEMETRY-20261002'
JSON_NAME = 'M2-JSON-RECOVERY-20261002'
MAX_RECEIPT_BYTES = 8_000_000
SCOPE = 'STRICT_INTEGER_SNR_STRING_SUBCLASS_ONLY'
ARTIFACT_SUFFIXES = {'.json', '.md', '.csv', '.txt', '.log'}


class NotReady(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8_000_000), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def verify(mapping, label='bindings'):
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError('Missing ' + label)
    for path, digest in mapping.items():
        if not re.fullmatch(r'[0-9a-f]{64}', str(digest)) or sha(path) != digest:
            raise RuntimeError('Changed ' + label + ': ' + path)


def bindings(paths):
    return {str(Path(path).resolve()): sha(path) for path in sorted(paths)}


def source_files(folder):
    folder = Path(folder).resolve(); answer = []
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.is_symlink() or path.suffix not in {'.py', '.md'}:
            raise RuntimeError('Unexpected runtime source artifact: ' + str(path))
        answer.append(path)
    if not answer:
        raise RuntimeError('Frozen runtime bundle is empty')
    return answer


def published(record):
    if (record.get('status') != 'PUSHED' or record.get('checks') != 'PASS'
            or not re.fullmatch(r'[0-9a-f]{40}', str(record.get('commit', '')))
            or record.get('commit') != record.get('remote_commit')):
        raise RuntimeError('A checked normal-push receipt is required')


def process_snapshot(proc=Path('/proc')):
    if not proc.is_dir():
        raise RuntimeError('Linux process identity evidence is required')
    answer = {}
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / 'stat').read_text(); fields = stat[stat.rfind(')') + 2:].split()
            answer[int(path.name)] = dict(state=fields[0], start_ticks=fields[19],
                command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            answer[int(path.name)] = dict(unreadable=True)
    return answer


def previous_delivery_gate(root, processes):
    root = Path(root).resolve(); out = root / 'outputs' / PREVIOUS_NAME
    paths = [out / 'publication.json', out / 'publication_status.json']
    if any(not path.is_file() for path in paths):
        raise NotReady('Waiting for telemetry publication')
    record, status = map(read, paths)
    if record.get('status') != 'PUSHED' or status.get('status') != 'COMPLETE':
        raise NotReady('Waiting for completed telemetry normal push')
    published(record)
    if (status.get('publication') != record or status.get('stop') is not True
            or type(status.get('pid')) is not int or status['pid'] <= 1):
        raise RuntimeError('Telemetry delivery receipt differs')
    verify(record.get('published_files'), 'published telemetry files')
    verify(record.get('inputs'), 'published telemetry inputs')
    # The earlier publisher binds PID only. Any live occupant, even PID reuse,
    # therefore delays delivery. The receipt cannot prove which occupant exited.
    process = processes.get(status['pid'])
    if process and (process.get('unreadable') or process.get('state') != 'Z'):
        raise NotReady('Telemetry publisher has not exited')
    source = root / 'experiments/m2-json-recovery-20261002/publish_when_complete.py'
    prior = root / 'outputs' / JSON_NAME / 'publication.json'
    old_record = read(prior); published(old_record)
    if old_record.get('source_bindings', {}).get(str(source)) != sha(source):
        raise RuntimeError('Canonical JSON publication helper is not bound')
    verify(old_record.get('published_files'), 'published JSON recovery files')
    return dict(status='TELEMETRY_PUBLISHED_AND_EXITED',
                inputs=bindings([*paths, prior, source]), helper_source=str(source))


def numeric_gate(root, runtime, manifest_path):
    root = Path(root).resolve(); manifest_path = Path(manifest_path).resolve()
    record = read(manifest_path)
    if (record.get('status') != 'NUMERIC_RECOVERY_READY_FOR_PUBLICATION'
            or record.get('scope') != SCOPE
            or type(record.get('training_updates')) is not int or record['training_updates'] != 0
            or type(record.get('policy_selection_updates')) is not int or record['policy_selection_updates'] != 0
            or any(record.get(field) is not False for field in
                   ('inference_functions_changed', 'original_values_changed', 'row_hashes_changed', 'model_weights_changed'))):
        raise RuntimeError('Unexpected numeric compatibility publication scope')
    bound = bindings(source_files(runtime))
    if record.get('source_bindings') != bound:
        raise RuntimeError('Numeric runtime inventory changed')
    verify(record.get('proof_bindings'), 'immutable numeric proof')
    verify(record.get('artifacts'), 'numeric artifacts')
    for name in record['artifacts']:
        path = Path(name)
        if (path.is_symlink() or path.suffix not in ARTIFACT_SUFFIXES
                or path.stat().st_size > MAX_RECEIPT_BYTES
                or not path.resolve().is_relative_to(root)):
            raise RuntimeError('Unexpected numeric proof artifact: ' + name)
    receipts = record.get('qualification_receipts')
    if not isinstance(receipts, dict) or set(receipts) != {'cpu', 'real_first_source'}:
        raise RuntimeError('CPU and real first-source qualification whitelist is required')
    for key, path in receipts.items():
        if (str(Path(path).resolve()) != path or path not in record['proof_bindings']
                or record['artifacts'].get(path) != record['proof_bindings'][path]):
            raise RuntimeError('Qualification is not in the bound artifact whitelist: ' + key)
    cpu = read(receipts['cpu']); real = read(receipts['real_first_source'])
    if cpu.get('status') != 'REAL_NUMERIC_SNR_COMPATIBILITY_PASS':
        raise RuntimeError('Numeric compatibility CPU qualification has not passed')
    if (real.get('status') != 'REAL_FIRST_SOURCE_PARITY_PASS'
            or real.get('parity_passed') is not True or real.get('synthetic') is not False
            or type(real.get('source_index')) is not int or real['source_index'] != 0
            or real.get('original_source_row_ids_preserved') is not True):
        raise RuntimeError('Real first-source parity has not passed')
    verify(real.get('checkpoints_bound'), 'real first-source checkpoints')
    verify(real.get('inputs'), 'real first-source parity inputs')
    if 'inputs' in cpu:
        verify(cpu['inputs'], 'CPU numeric compatibility inputs')
    return dict(status='IMMUTABLE_NUMERIC_COMPATIBILITY_EVIDENCE_READY', source_bindings=bound,
        inputs={**bound, **record['proof_bindings'], **record['artifacts'], **real['inputs'],
                **real['checkpoints_bound'], **cpu.get('inputs', {}), str(manifest_path): sha(manifest_path)},
        artifacts=record['artifacts'], qualification_receipts=receipts, scientific_result=False,
        scope=dict(compatibility=SCOPE, original_values_changed=False, row_hashes_changed=False,
                   model_weights_changed=False, inference_functions_changed=False,
                   training_updates=0, policy_selection_updates=0))


def report_text():
    return '\n'.join(['# 统一指标 SNR 字符串兼容恢复记录', '',
        '本记录在原 M2、统一补充指标、JSON 恢复和遥测记录均已完成推送、相关工作进程退出后发布。原科学源码、策略、模型、缓存和结果保留原字节。', '',
        '独立兼容层仅为严格的整数 SNR 字符串提供字符串子类，以支持原指标流程要求的数值运算。非整数、不合法和其他类型的处理以冻结兼容实现与 CPU 资格证明为准。原字符串内容和数值、源行标识及模型权重保持不变。', '',
        '真实首源的回放与评分一致性证明单独登记；其检查不能代替全部源的最终回放验证。源文件、CPU 兼容资格和真实首源证明见 `proofs/` 及 provenance。', '',
        '本提交记录工程兼容恢复，不重选模型或策略，不新增训练或科学结论。正式指标以统一指标报告和完整结果回执为准。', ''])


def load_base(proof):
    path = Path(proof['helper_source'])
    spec = importlib.util.spec_from_file_location('_published_numeric_json_framework', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def adapted_publisher(base, root):
    """Change declared publication paths and gates; preserve original framework."""
    tree = ast.parse(Path(base.__file__).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    changes = {
        'experiments/m2-json-recovery-20261002': 'experiments/metric-numeric-recovery-20261003',
        'results/m2_json_recovery_20261002': 'results/metric_numeric_recovery_20261003',
        'recovery_publication_manifest.json': 'numeric_publication_manifest.json',
        'RECOVERY_REPORT.md': 'NUMERIC_RECOVERY_REPORT.md',
        'Document exact M2 boolean serialization recovery and cached gate derivation\n':
            'Document exact integer SNR string compatibility and verified metric replay\n'}
    counts = {key: 0 for key in changes}
    class Locations(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in changes:
                counts[node.value] += 1
                return ast.copy_location(ast.Constant(changes[node.value]), node)
            return node
    node = Locations().visit(node)
    expected = {key: (2 if key == 'recovery_publication_manifest.json' else 1) for key in changes}
    if counts != expected:
        raise RuntimeError('Frozen publication framework literals changed: ' + str(counts))
    def completed(root, processes):
        prior = previous_delivery_gate(root, processes)
        try:
            value = base.completion_gate(root, processes)
        except base.NotReady as error:
            raise NotReady(str(error)) from error
        base.no_publishers(processes)
        peers = [pid for pid, item in processes.items() if pid != os.getpid()
                 and item.get('state') != 'Z'
                 and '/numeric_delivery.py' in item.get('command', '').replace('\\', '/')]
        if peers:
            raise NotReady('Another numeric publisher is active: ' + str(peers))
        value['inputs'].update(prior['inputs'])
        return value
    namespace = dict(base.__dict__, NAME=NAME, NotReady=NotReady, recovery_gate=numeric_gate,
                     completion_gate=completed, report_text=report_text)
    compiled = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), __file__, 'exec')
    exec(compiled, namespace)
    return namespace['main']


def main(root, *, runtime=None, manifest=None, cpu_qualification=None,
         real_first_source_qualification=None, poll_seconds=30, once=False):
    root = Path(root).resolve(); runtime = Path(runtime or Path(__file__).parent).resolve()
    out = root / 'outputs' / NAME; out.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest or out / 'numeric_publication_manifest.json').resolve()
    evidence = numeric_gate(root, runtime, manifest)
    for key, requested in (('cpu', cpu_qualification), ('real_first_source', real_first_source_qualification)):
        if requested is not None and str(Path(requested).resolve()) != evidence['qualification_receipts'][key]:
            raise RuntimeError('Requested qualification differs from immutable manifest whitelist: ' + key)
    while True:
        verify(evidence['inputs'], 'frozen numeric inputs')
        try:
            proof = previous_delivery_gate(root, process_snapshot()); break
        except NotReady as error:
            write(out / 'publication_status.json', dict(status='WAITING_FOR_TELEMETRY_PUBLICATION_AND_EXIT',
                  reason=str(error), pid=os.getpid()))
            if once:
                return None
            time.sleep(poll_seconds)
    publisher = adapted_publisher(load_base(proof), root)
    return publisher(root, runtime=runtime, manifest=manifest, poll_seconds=poll_seconds, once=once)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--runtime')
    parser.add_argument('--numeric-manifest'); parser.add_argument('--cpu-qualification')
    parser.add_argument('--real-first-source-qualification')
    parser.add_argument('--poll-seconds', type=int, default=30); parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('poll-seconds must be between 1 and 60')
    result = main(args.root, runtime=args.runtime, manifest=args.numeric_manifest,
                  cpu_qualification=args.cpu_qualification,
                  real_first_source_qualification=args.real_first_source_qualification,
                  poll_seconds=args.poll_seconds, once=args.once)
    if result is None:
        sys.exit(75)
