"""Commit authorized N1024 source and, after real completion, measured results."""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/EXTREME-BW-20261001-R1-N1024'
RESULT = ROOT / 'results/extreme_bandwidth_20261001_R1_N1024'
TRAIN = OUT / 'training/p1024_2026093001'
REPORT = ROOT / 'reports/extreme_bandwidth_probe_20261001_R1_N1024.md'
RUN = 'EXTREME-BW-20261001-R1-N1024'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise RuntimeError('Existing provenance differs: ' + str(target))
    shutil.copyfile(source, target)


def command(args, log, cpu=False):
    environment = dict(os.environ)
    if cpu:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
    log.flush()


def source_bindings():
    bindings = {str(p): sha(p) for p in HERE.iterdir() if p.suffix in ('.py', '.json', '.md', '.csv')}
    archiver = ROOT / 'experiments/rx-posterior-step2-B-20260930/tables.py'
    bindings[str(archiver)] = sha(archiver)
    return bindings


def prepare_results():
    training = read(TRAIN / 'completion.json')
    plugin = read(OUT / 'plugin/completion.json')
    digital = read(OUT / 'digital/evaluation_complete.json')
    analysis = read(RESULT / 'analysis_completion.json')
    if training['development_read'] or not training['state']['finished']:
        raise RuntimeError('Calibration-only training incomplete')
    if not plugin['frozen_weights_unchanged'] or not digital['frozen_weights_unchanged']:
        raise RuntimeError('Frozen evaluation assets changed')
    if (analysis['status'] != 'N1024_SOURCE_PAIRED_ANALYSIS_COMPLETE'
            or analysis['automatic_further_experiments'] or analysis.get('synthetic') is not False):
        raise RuntimeError('Authorized analysis incomplete')
    if analysis['metric_rows'] != 45000 or plugin['metric_rows'] != 21000 or digital['rows'] != 24000:
        raise RuntimeError('The complete registered development comparison is required')
    if len(set(analysis['methods'])) != 30 or 'P1024' not in analysis['methods']:
        raise RuntimeError('All thirty registered outputs are required')
    for name, expected in analysis['files'].items():
        if sha(RESULT / name) != expected:
            raise RuntimeError('Analyzed result changed: ' + name)
    configs = [read(RESULT / name) for name in ('plugin_config.json', 'digital_config.json')]
    bindings = source_bindings()
    for config in configs:
        for path, expected in config['source_bindings'].items():
            if path in bindings and bindings[path] != expected:
                raise RuntimeError('Executed source changed: ' + path)
            bindings[path] = expected
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise RuntimeError('Bound source changed: ' + path)
    provenance = RESULT / 'provenance'
    for folder, label in ((TRAIN, 'training'), (OUT / 'plugin', 'plugin'), (OUT / 'digital', 'digital')):
        for path in folder.glob('*.json'):
            if path.name in ('status.json', 'latest.json') or path.name.startswith('failure_'):
                continue
            copy_exact(path, provenance / label / path.name)
    for path in OUT.glob('qualification_launch*.json'):
        copy_exact(path, provenance / 'qualification_attempts' / path.name)
    for path in OUT.glob('qualification_attempt_*.log'):
        copy_exact(path, provenance / 'qualification_attempts' / path.name)
    for role in ('calibration', 'development'):
        for name in ('identity.json', 'completion.json'):
            source = OUT / 'digital/references' / role / name
            if not source.exists():
                raise RuntimeError('Source reference evidence missing: ' + str(source))
            copy_exact(source, provenance / 'source_references' / role / name)
    for path in (OUT / 'source_publication_history').glob('*.json'):
        copy_exact(path, provenance / 'source_publication_history' / path.name)
    copy_exact(OUT / 'source_publication.json', provenance / 'source_publication.json')
    copy_exact(OUT / 'supervisor_registration.json', provenance / 'supervisor_registration.json')
    for name in ('reference_import.json', 'prior_budget_import.json'):
        copy_exact(OUT / name, provenance / name)
    for path in (TRAIN / 'calibration').glob('full_*.csv'):
        copy_exact(path, RESULT / 'training_calibration' / path.name)
        copy_exact(path.with_suffix('.json'), RESULT / 'training_calibration' / path.with_suffix('.json').name)
    complete = read(OUT / 'digital/calibration_complete.json')
    csv_path = OUT / 'digital/calibration_per_frame.csv'
    if not csv_path.exists():
        raise RuntimeError('The full digital calibration table is missing')
    if complete['per_frame_sha256'] != sha(csv_path) or complete['frames'] != 210000 or complete['development_read']:
        raise RuntimeError('Digital calibration identity or selection boundary changed')
    with csv_path.open(newline='') as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    if rows != 210000:
        raise RuntimeError('Full1000 digital calibration table is incomplete')
    copy_exact(csv_path, RESULT / 'digital_calibration_per_frame.csv')
    write(provenance / 'digital_calibration_archive.json', dict(rows=rows, sha256=sha(csv_path),
          completion_sha256=sha(OUT / 'digital/calibration_complete.json'), completion=complete))
    original = {str(p.relative_to(RESULT)): sha(p) for p in RESULT.rglob('*')
                if p.is_file() and 'provenance' not in p.relative_to(RESULT).parts
                and 'table_shards' not in p.relative_to(RESULT).parts
                and p.name not in ('table_registry.json', 'README.md')}
    write(provenance / 'original_result_sha256.json', original)
    tables = {}
    for path in RESULT.glob('*.csv'):
        if path.stat().st_size > 8_000_000:
            with path.open(newline='') as handle:
                tables[path.name] = sum(1 for _ in csv.DictReader(handle))
    write(RESULT / 'table_registry.json', dict(run=RUN, tables=tables,
        scientific_completeness='Validated by training, digital, plugin and analysis receipts before byte archiving',
        archiver_source=str(ROOT / 'experiments/rx-posterior-step2-B-20260930/tables.py'),
        archiver_sha256=sha(ROOT / 'experiments/rx-posterior-step2-B-20260930/tables.py')))
    import importlib.util
    spec = importlib.util.spec_from_file_location('_extreme_bw_tables', HERE / 'tables.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.export(RESULT)
    ignore = ROOT / '.gitignore'
    text = ignore.read_text()
    for name in tables:
        line = '/' + str((RESULT / name).relative_to(ROOT))
        if line not in text.splitlines():
            text += '\n' + line + '\n'
    ignore.write_text(text)
    report = (RESULT / 'report.md').read_text()
    report = re.sub(r'(!\[[^\]]*\]\()((?!https?://)[^)]*)(\))',
        lambda match: match[1] + '../results/extreme_bandwidth_20261001_R1_N1024/' + match[2] + match[3], report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)
    selected = training['selected']['P1024']
    completed_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
    note = (f"## N1024 extreme bandwidth study completed {completed_date}\n\n"
        f"One fresh P1024 completed {training['state']['step']} updates and selected step {selected['step']} "
        "using complete1000-source calibration at1/4/7/13/19dB. Digital raw N1024 policies use paid68-symbol headers "
        "and were independently frozen on1000 calibration sources; P1024 receiver policies use200 calibration sources. "
        "All systems then used the same100 development sources and three noise repeats. "
        f"P1024 budget_truncated={training['budget_truncated']}; no convergence claim is made. "
        "QPSK/continuous have actual per-frameE2048;16QAM retains original fixed constellation scaling and actual energy. "
        "H_D/H_R decisions, paid-class conditions, distortion, source specificity, failures and computation are reported. "
        "Development is reused and the intervals do not cover training-seed variance. "
        "This authorized N1024 stage is complete. N512 remains frozen at its 40000-update budget cap; further experiments await a separate decision.\n\n"
        "[Scientific report](reports/extreme_bandwidth_probe_20261001_R1_N1024.md).\n\n")
    for filename in ('RESEARCH_STATUS.md', 'PROGRESS.md', 'EXPERIMENTS.md'):
        path = ROOT / filename
        text = path.read_text()
        if note.splitlines()[0] not in text:
            path.write_text(note + text)
    (RESULT / 'README.md').write_text(
        '# N1024 extreme bandwidth study\n\n'
        '- [Scientific report](report.md)\n- [Frozen protocol and identities](config.json)\n'
        '- [H_D/H_R decisions](decisions.json)\n- [Training evidence](provenance/training/completion.json)\n'
        '- [Full digital calibration evidence](provenance/digital_calibration_archive.json)\n\n'
        'Large measured tables are stored as exact byte-preserving parts below the repository file limit. '
        'Restore them with:\n\n```bash\npython3 experiments/extreme-bandwidth-20261001-N1024/tables.py restore\n```\n\n'
        'Models, checkpoints and tensor caches remain on the execution host. '
        'Source-level intervals use the reused development set and one training seed.\n')
    return bindings


def main(register_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT / ('source_publication.json' if register_only else 'publication.json')
    if receipt.exists():
        previous = read(receipt)
        if previous.get('status') == 'PUSHED':
            if register_only and previous.get('source_bindings') == source_bindings():
                return
            if not register_only:
                for path, expected in previous.get('source_bindings', {}).items():
                    if sha(path) != expected:
                        raise RuntimeError('Published execution source changed: ' + path)
                return
        if register_only:
            history_name = previous['commit'] + '_' + previous['status'] + '_' + str(previous['time']).replace('.', '_') + '.json'
            copy_exact(receipt, OUT / 'source_publication_history' / history_name)
    bindings = source_bindings() if register_only else prepare_results()
    own = [str(HERE.relative_to(ROOT))]
    if not register_only:
        own += [str(RESULT.relative_to(ROOT)), str(REPORT.relative_to(ROOT)), '.gitignore',
                'RESEARCH_STATUS.md', 'PROGRESS.md', 'EXPERIMENTS.md']
    with (OUT / ('source_publication_checks.log' if register_only else 'publication_checks.log')).open('a') as log:
        command(['git', 'add', '--', *own], log)
        staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()
        if any(p != 'release_manifest.json' and not any(p == s or p.startswith(s + '/') for s in own) for p in staged):
            raise RuntimeError('Unrelated staged changes need separate review')
        command([sys.executable, 'tools/update_repository_manifest.py'], log)
        command(['git', 'add', 'release_manifest.json'], log)
        command([sys.executable, 'tools/verify_repository.py'], log)
        command([sys.executable, 'tools/run_cpu_checks.py'], log)
        for path in sorted(HERE.glob('test_*.py')):
            command([sys.executable, str(path)], log, cpu=True)
        if not register_only:
            command([sys.executable, str(HERE / 'tables.py'), 'verify'], log)
        command(['git', 'fetch', 'origin'], log)
        command(['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD'], log)
        for path, expected in bindings.items():
            relative = str(Path(path).relative_to(ROOT))
            data = subprocess.check_output(['git', 'show', ':' + relative], cwd=ROOT)
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError('Staged source differs from execution: ' + relative)
        message = OUT / ('source_commit_message.txt' if register_only else 'result_commit_message.txt')
        message.write_text('Register authorized N1024 extreme bandwidth protocol and execution\n' if register_only
                           else 'Publish N1024 extreme bandwidth calibration and paired receiver results\n')
        if subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT).strip():
            command(['git', 'commit', '-F', str(message)], log)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        write(receipt, dict(status='COMMITTED', commit=commit, checks='PASS', source_bindings=bindings, time=time.time()))
        command(['git', 'push', 'origin', 'main'], log)
        remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT, text=True).split()[0]
        if remote != commit:
            raise RuntimeError('Remote main differs from committed results')
    write(receipt, dict(status='PUSHED', commit=commit, remote_commit=remote, checks='PASS', source_bindings=bindings, time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args.register_only)
    except Exception as error:
        import traceback
        write(OUT / f'publication_failure_{time.time_ns()}.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise
