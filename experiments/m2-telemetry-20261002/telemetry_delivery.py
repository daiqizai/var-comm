"""Publish isolated telemetry optimization evidence after JSON recovery delivery.

This entry point performs CPU repository checks. It neither launches scoring
nor claims that CPU tests qualify reconstructed images or metric models.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


SUFFIXES = {'.py', '.md'}
MAX_RECEIPT_BYTES = 8_000_000
ORIGINS = {'git@github.com:daiqizai/var-comm.git', 'https://github.com/daiqizai/var-comm.git'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8_000_000), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                              allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def verify(mapping, label='bindings'):
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError('Missing ' + label)
    for path, digest in mapping.items():
        if not re.fullmatch(r'[0-9a-f]{64}', str(digest)) or sha(path) != digest:
            raise RuntimeError('Changed ' + label + ': ' + path)


def published(record):
    if (record.get('status') != 'PUSHED' or record.get('checks') != 'PASS'
            or not re.fullmatch(r'[0-9a-f]{40}', str(record.get('commit', '')))
            or record.get('commit') != record.get('remote_commit')):
        raise RuntimeError('A checked normal-push receipt is required')


def source_files(folder):
    folder = Path(folder).resolve(); answer = []
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.is_symlink() or path.suffix not in SUFFIXES:
            raise RuntimeError('Unexpected runtime source artifact: ' + str(path))
        answer.append(path)
    if not answer:
        raise RuntimeError('Frozen runtime bundle is empty')
    return answer


def bindings(paths):
    return {str(Path(p).resolve()): sha(p) for p in sorted(paths)}


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


def require_exited(record, processes, label):
    if (type(record.get('pid')) is not int or record['pid'] <= 1
            or not str(record.get('start_ticks', '')).isdigit()):
        raise RuntimeError('Missing registered ' + label + ' process identity')
    proc = processes.get(record['pid'])
    if proc and proc.get('unreadable'):
        raise RuntimeError('Cannot inspect registered ' + label)
    if proc and proc.get('state') != 'Z' and proc.get('start_ticks') == str(record['start_ticks']):
        raise RuntimeError(label + ' has not exited')





NAME='M2-TELEMETRY-20261002'
JSON_NAME='M2-JSON-RECOVERY-20261002'


class NotReady(RuntimeError):pass


def json_delivery_gate(root, processes):
    out=root/'outputs'/JSON_NAME
    paths=[out/'publication.json',out/'publication_status.json']
    if any(not p.is_file() for p in paths):raise NotReady('Waiting for JSON recovery publication')
    record,status=map(read,paths)
    if record.get('status')!='PUSHED' or status.get('status')!='COMPLETE':
        raise NotReady('Waiting for the completed JSON recovery normal push')
    published(record)
    if status.get('publication')!=record or status.get('stop') is not True or type(status.get('pid')) is not int:
        raise RuntimeError('JSON recovery delivery receipt differs')
    verify(record.get('published_files'),'published JSON recovery files')
    verify(record.get('inputs'),'published JSON recovery inputs')
    proc=processes.get(status['pid'])
    # The frozen JSON publisher recorded PID only. Conservatively wait for any
    # live occupant of that PID, including reuse; never infer exit from a name.
    if proc and (proc.get('unreadable') or proc.get('state')!='Z'):
        raise NotReady('JSON recovery publisher has not exited')
    source=root/'experiments/m2-json-recovery-20261002/publish_when_complete.py'
    if record.get('source_bindings',{}).get(str(source))!=sha(source):
        raise RuntimeError('The canonical JSON publication helper is not bound')
    return dict(status='JSON_RECOVERY_PUBLISHED_AND_EXITED',inputs=bindings([*paths,source]),helper_source=str(source))


def load_base(proof):
    path=Path(proof['helper_source']);spec=importlib.util.spec_from_file_location('_published_json_recovery',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def telemetry_gate(root,runtime,manifest_path):
    record=read(manifest_path)
    ttl=record.get('healthy_query_ttl_seconds')
    if (record.get('status')!='TELEMETRY_READY_FOR_PUBLICATION' or record.get('training_updates')!=0
            or record.get('policy_selection_updates')!=0 or record.get('inference_functions_changed') is not False
            or record.get('scientific_parity_status')!='PASS' or isinstance(ttl,bool)
            or not isinstance(ttl,(int,float)) or not 0<ttl<=1):
        raise RuntimeError('Unexpected telemetry optimization publication scope')
    bound=bindings(source_files(runtime))
    if record.get('source_bindings')!=bound:raise RuntimeError('Telemetry runtime inventory changed')
    verify(record.get('proof_bindings'),'immutable telemetry proof');verify(record.get('artifacts'),'telemetry artifacts')
    for name in record['artifacts']:
        path=Path(name)
        if (path.is_symlink() or path.suffix not in {'.json','.md','.csv','.txt','.log'}
                or path.stat().st_size>MAX_RECEIPT_BYTES or not path.resolve().is_relative_to(root)):
            raise RuntimeError('Unexpected telemetry proof artifact: '+name)
    return dict(status='IMMUTABLE_TELEMETRY_EVIDENCE_READY',source_bindings=bound,
        inputs={**bound,**record['proof_bindings'],**record['artifacts'],str(manifest_path):sha(manifest_path)},
        artifacts=record['artifacts'],scientific_result=False,
        scope=dict(healthy_query_ttl_seconds=ttl,inference_functions_changed=False,scientific_parity_status='PASS',
                   training_updates=0,policy_selection_updates=0))


def report_text():
    return '\n'.join(['# M2 遥测查询优化记录','',
        '本提交在原 M2、统一补充指标和 JSON 恢复记录全部完成推送、相关工作进程退出后发布。原科学源码、策略、缓存和已发布结果保留原字节。','',
        '独立适配器短时复用正常遥测查询响应，固定 manifest 登记有效期不超过 1 秒。异常、热状态、未知 GPU 用户及过期读取失败的处理，以独立工程资格证明为准。监控查询时序改变，不能称为完全没有监控延迟。','',
        '推理函数、模型权重和策略不变；真实等价检查与工程吞吐记录见 proofs。此记录不提供新的科学结果，也不反向选择模型或接收参数。',''])


def adapted_publisher(base, root):
    """Reuse the checked publisher, changing only declared paths and gates.

    The original helper file remains unchanged. AST replacement counts fail
    closed if that frozen framework no longer has the expected literals.
    """
    tree=ast.parse(Path(base.__file__).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    changes={
        'experiments/m2-json-recovery-20261002':'experiments/m2-telemetry-20261002',
        'results/m2_json_recovery_20261002':'results/m2_telemetry_20261002',
        'recovery_publication_manifest.json':'telemetry_publication_manifest.json',
        'RECOVERY_REPORT.md':'TELEMETRY_REPORT.md',
        'Document exact M2 boolean serialization recovery and cached gate derivation\n':'Document verified bounded telemetry query reuse without changing M2 inference\n'}
    counts={k:0 for k in changes}
    class Locations(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str) and n.value in changes:
                counts[n.value]+=1;return ast.copy_location(ast.Constant(changes[n.value]),n)
            return n
    node=Locations().visit(node)
    expected={k:(2 if k=='recovery_publication_manifest.json' else 1) for k in changes}
    if counts!=expected:raise RuntimeError('Frozen publication framework literals changed: '+str(counts))
    def complete(root,processes):
        json_proof=json_delivery_gate(root,processes)
        try:value=base.completion_gate(root,processes)
        except base.NotReady as e:raise NotReady(str(e)) from e
        # The JSON publisher is already exited here. A second telemetry
        # delivery is serialized by its own publication lock; reject writers.
        base.no_publishers(processes)
        value['inputs'].update(json_proof['inputs']);return value
    namespace=dict(base.__dict__,NAME=NAME,NotReady=NotReady,recovery_gate=telemetry_gate,
                   completion_gate=complete,report_text=report_text)
    compiled=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec')
    exec(compiled,namespace)
    return namespace['main']


def main(root,*,runtime=None,manifest=None,poll_seconds=30,once=False):
    root=Path(root).resolve();runtime=Path(runtime or Path(__file__).parent).resolve()
    out=root/'outputs'/NAME;out.mkdir(parents=True,exist_ok=True)
    manifest=Path(manifest or out/'telemetry_publication_manifest.json').resolve()
    evidence=telemetry_gate(root,runtime,manifest)
    while True:
        verify(evidence['inputs'],'frozen telemetry inputs')
        try:json_proof=json_delivery_gate(root,process_snapshot());break
        except NotReady as e:
            write(out/'publication_status.json',dict(status='WAITING_FOR_JSON_RECOVERY_PUBLICATION_AND_EXIT',reason=str(e),pid=os.getpid()))
            if once:return None
            time.sleep(poll_seconds)
    publisher=adapted_publisher(load_base(json_proof),root)
    return publisher(root,runtime=runtime,manifest=manifest,poll_seconds=poll_seconds,once=once)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--runtime');p.add_argument('--telemetry-manifest');p.add_argument('--poll-seconds',type=int,default=30)
    p.add_argument('--once',action='store_true');a=p.parse_args()
    if not 1<=a.poll_seconds<=60:p.error('poll-seconds must be between 1 and 60')
    result=main(a.root,runtime=a.runtime,manifest=a.telemetry_manifest,poll_seconds=a.poll_seconds,once=a.once)
    if result is None:sys.exit(75)
