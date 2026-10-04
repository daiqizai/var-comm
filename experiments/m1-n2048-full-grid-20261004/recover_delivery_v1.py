"""Publish the already completed M1 report after the legacy manifest-reader bug.

No inference, metric computation, image reconstruction or figure generation is
performed. The original publisher and its failure receipt remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import signal
import subprocess
import sys
import time

LIMIT = 10_000_000
VERSION = 'M1_DELIVERY_RECOVERY_V1'
STOP = False


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def seal(path, value):
    if Path(path).exists():
        require(read(path) == value, 'Recovery evidence already exists with different contents: ' + str(path))
    else:
        write(path, value)


def check(out):
    if STOP or (Path(out) / 'STOP').exists():
        raise InterruptedError('Requested publication stop; all prior evidence is retained')


def relative_file(directory, name):
    """Accept canonical POSIX relative files, including on the Windows test host."""
    require(isinstance(name, str) and name and '\\' not in name
            and not PurePosixPath(name).is_absolute() and not PureWindowsPath(name).drive
            and all(part not in ('', '.', '..') for part in name.split('/')),
            'Publication manifest path must be a strict relative path: ' + repr(name))
    directory = Path(directory).resolve()
    path = directory.joinpath(*name.split('/')).resolve()
    require(directory in path.parents and path.is_file(), 'Publication path escapes its directory or is missing: ' + name)
    return path


def verify_bindings(bindings, within=None):
    require(isinstance(bindings, dict) and bool(bindings), 'Nonempty SHA256 bindings required')
    boundary = None if within is None else Path(within).resolve()
    for name, expected in bindings.items():
        require(isinstance(name, str) and digest(expected), 'Invalid file SHA256 binding')
        path = Path(name).resolve()
        require(path.is_file() and (boundary is None or boundary in path.parents), 'Bound file is missing or outside its output directory')
        require(sha(path) == expected, 'Bound file SHA256 differs: ' + name)


def validate_manifest(result, manifest):
    """The builder uses nested {sha256, bytes}, never a bare digest value."""
    result = Path(result).resolve()
    require(manifest.get('status') == 'STATIC_REPORT_COMPLETE', 'Static report is not complete')
    require(manifest.get('model_inference') is False and manifest.get('metric_recomputation') is False
            and manifest.get('training_updates') == 0, 'Unexpected report computation scope')
    outputs = manifest.get('outputs')
    require(isinstance(outputs, dict) and outputs and 'REPORT.md' in outputs, 'Report output inventory is incomplete')
    paths = []
    for name, info in outputs.items():
        path = relative_file(result, name)
        require(isinstance(info, dict) and set(info) == {'sha256', 'bytes'} and digest(info['sha256'])
                and type(info['bytes']) is int and 0 <= info['bytes'] < LIMIT, 'Invalid nested output metadata: ' + name)
        require(path.stat().st_size == info['bytes'] and path.stat().st_size < LIMIT, 'Published output byte count differs: ' + name)
        require(sha(path) == info['sha256'], 'Published output SHA256 differs: ' + name)
        paths.append(path)
    return paths


def validate_completed(root, out, result):
    config_path = out / 'delivery_config.json'
    config = read(config_path)
    require(config.get('root') == str(root) and config.get('out') == str(out), 'Wrong original delivery target')
    verify_bindings(config['bindings'])
    cal = read(out / 'calibrate_completion.json')
    dev = read(out / 'development_completion.json')
    score = read(out / 'score_completion.json')
    require(cal.get('status') == 'COMPLETE' and cal.get('stage') == 'calibrate'
            and cal.get('sources') == 1000 and cal.get('physical_frames') == 2070000,
            'Complete original calibration required')
    require(dev.get('status') == 'COMPLETE' and dev.get('stage') == 'development'
            and dev.get('sources') == 100 and dev.get('method_rows') == 24000,
            'Complete original development required')
    require(score.get('status') == 'M1_N2048_METRICS_COMPLETE' and score.get('sources') == 100
            and score.get('rows') == 24000, 'Complete original metric scoring required')
    for value in (cal, dev, score):
        require(value.get('synthetic') is False and value.get('training_updates') == 0, 'Only real completed inference evidence may be published')
        verify_bindings(value['outputs'], out)
    require(cal['registration_sha256'] == dev['registration_sha256'] == sha(out / 'registration.json'),
            'Calibration/development registration differs')
    require(score['registration_sha256'] == sha(out / 'metrics/registration.json'), 'Metric registration differs')
    require(cal['policy_sha256'] == dev['policy_sha256'] == sha(out / 'm1_policy.json'), 'Frozen selected policy differs')
    verify_bindings(score['input_bindings'])
    manifest_path = result / 'MANIFEST.json'
    manifest = read(manifest_path)
    require(manifest_path.stat().st_size < LIMIT, 'Report manifest exceeds publication limit')
    require(manifest.get('sources') == 100 and manifest.get('method_rows') == 24000
            and manifest.get('score_completion_sha256') == sha(out / 'score_completion.json')
            and manifest.get('policy_sha256') == sha(out / 'm1_policy.json'), 'Report does not refer to the completed original results')
    verify_bindings(manifest['input_bindings'])
    paths = validate_manifest(result, manifest)
    return config, manifest, [*paths, manifest_path]


def mirror_report(text):
    base = '../results/m1_n2048_full_grid_20261004/'
    text = text.replace('](figures/', '](' + base + 'figures/')
    for name in ('SAME_K.md', 'MAIN_TABLE.md', 'MAIN_TABLE.csv', 'POLICIES.csv',
                 'metrics_paired_intervals.csv', 'm1_vs_P_paired.csv', 'resource_summary.csv', 'calibration_summary.csv'):
        text = text.replace('](' + name + ')', '](' + base + name + ')')
    return text


def command(root, args, capture=False):
    result = subprocess.run(args, cwd=root, check=True, text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def staged_paths(root):
    return set(filter(None, command(root, ['git', 'diff', '--cached', '--name-only'], True).splitlines()))


def run(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    require(root in out.parents, 'Original runtime output must be inside the repository')
    check(out)
    require(not (out / 'publication_completion.json').exists(), 'Original publication already has a completion receipt')
    failure_path = out / 'delivery_failure.json'
    require(failure_path.is_file(), 'The original failed delivery receipt must be retained')
    failure_sha = sha(failure_path)
    recovery_out = out / 'delivery_recovery_v1'
    recovery_out.mkdir(exist_ok=True)
    def status(state, **fields):
        write(recovery_out / 'status.json', dict(status=state, recovery_version=VERSION, pid=os.getpid(), time=time.time(), **fields))
    status('VERIFYING_EXISTING_COMPLETE_ARTIFACTS')
    result = root / 'results/m1_n2048_full_grid_20261004'
    config, manifest, published_outputs = validate_completed(root, out, result)
    require(not staged_paths(root), 'Unexpected staged work; preserve it and review before publication')
    check(out)
    here = Path(__file__).resolve().parent
    new_sources = [Path(__file__).resolve(), here / 'test_recover_delivery_v1.py']
    require(all(root in p.parents and p.is_file() for p in new_sources), 'Recovery source and tests must reside in the repository')
    report = root / 'reports/m1_n2048_full_grid_20261004.md'
    report.write_text(mirror_report((result / 'REPORT.md').read_text(encoding='utf-8')), encoding='utf-8')
    recovery = dict(status='VERIFIED_EXISTING_ARTIFACTS_FOR_PUBLICATION', recovery_version=VERSION,
        reason='legacy publisher compared nested output metadata dict to a SHA256 string',
        original_failure_path=str(failure_path), original_failure_sha256=failure_sha,
        original_failure_and_sources_preserved=True, original_delivery_config_sha256=sha(out / 'delivery_config.json'),
        calibration_completion_sha256=sha(out / 'calibrate_completion.json'),
        development_completion_sha256=sha(out / 'development_completion.json'),
        score_completion_sha256=sha(out / 'score_completion.json'), report_manifest_sha256=sha(result / 'MANIFEST.json'),
        source_bindings={str(p): sha(p) for p in new_sources},
        nested_sha256_and_exact_bytes_verified=True, verified_report_files=len(manifest['outputs']),
        model_inference=False, metric_recomputation=False, image_reconstruction=False, figure_regeneration=False,
        training_updates=0, recovery_source=str(Path(__file__).resolve().relative_to(root)))
    recovery_path = result / 'RECOVERY.json'
    seal(recovery_path, recovery)
    source_paths = [relative_file(root, name) for name in config['publish_source_paths']]
    paths = sorted(set([*published_outputs, report, recovery_path, *source_paths, *new_sources]))
    require(all(root in path.parents and path.is_file() and path.stat().st_size < LIMIT for path in paths),
            'Missing, unsafe or oversized publication file')
    inventory = {str(p.relative_to(root)): sha(p) for p in paths}
    seal(recovery_out / 'publication_inventory.json', dict(status='VERIFIED_EXPLICIT_PUBLICATION_INVENTORY', files=inventory))
    status('VERIFYING_AND_PUBLISHING', files=len(paths))
    command(root, ['git', 'fetch', 'origin'])
    command(root, ['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD'])
    check(out)
    command(root, ['git', 'add', '--', *inventory])
    command(root, ['python3', '-B', 'tools/update_repository_manifest.py'])
    command(root, ['git', 'add', '--', 'release_manifest.json'])
    require(staged_paths(root).issubset(set(inventory) | {'release_manifest.json'}), 'Unrelated files entered the staged publication')
    command(root, ['python3', '-B', 'tools/verify_repository.py'])
    command(root, ['python3', '-B', 'tools/run_cpu_checks.py'])
    check(out)
    verify_bindings(config['bindings'])
    verify_bindings({str(root / name): expected for name, expected in inventory.items()})
    require(sha(failure_path) == failure_sha, 'Original failure evidence changed during recovery')
    command(root, ['git', 'commit', '-m', 'Publish completed N2048 M1 evaluation with manifest recovery'])
    command(root, ['git', 'push', 'origin', 'HEAD:main'])
    commit = command(root, ['git', 'rev-parse', 'HEAD'], True)
    remote = command(root, ['git', 'ls-remote', 'origin', 'refs/heads/main'], True).split()[0]
    require(commit == remote and len(commit) == 40, 'Normal push remote SHA mismatch')
    receipt = dict(status='PUSHED_AND_STOPPED', commit=commit, remote_commit=remote, checks='PASS',
        score_completion_sha256=sha(out / 'score_completion.json'), report_manifest_sha256=sha(result / 'MANIFEST.json'),
        training_updates=0, old_queues_resumed=False, figures_structurally_verified=True,
        final_figures_manual_visual_review='NOT_YET_PERFORMED', report=str(report.relative_to(root)),
        recovery_version=VERSION, recovery_source=str(Path(__file__).resolve().relative_to(root)),
        recovery_source_sha256=sha(__file__), recovery_evidence_sha256=sha(recovery_path),
        model_inference=False, metric_recomputation=False, image_reconstruction=False, figure_regeneration=False,
        original_failure_sha256=failure_sha)
    seal(out / 'publication_completion.json', receipt)
    seal(recovery_out / 'completion.json', receipt)
    status('PUSHED_AND_STOPPED', commit=commit)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    import fcntl
    lock = (out / 'delivery.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        run(root, out)
    except BaseException as error:
        write(out / 'delivery_recovery_v1/failure.json', dict(status='RECOVERY_FAILED_REQUIRES_REVIEW',
              error=repr(error), time=time.time(), automatic_retry=False, original_failure_preserved=True))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
