"""Publish only closed ARPC comparison artifacts; no scientific work is run here."""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

R = pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E = R / 'experiments/arpc-external-baseline-20261019'
O = R / 'results/arpc_external_baseline_20261019_v1'
L = pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/arpc_external_20261019_v1')
sys.path.insert(0, str(E))
import a1_owner as evidence

A = L / 'final_publication_v1'
PY = '/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
CODE_DIRS = [
    'phy_v1', 'phy_rx_environment_v1', 'phy_rx_environment_v2',
    'a2b_source_v1', 'a2b_link_v1', 'a2b_rx_v1', 'a2b_metrics_v1',
    'a3_source_v1', 'a3_link_v1', 'a3_rx_v1', 'a3_metrics_v1',
    'final_figures_v1', 'final_report_v1', 'final_publication_v1', 'final_cpu_execution_v1',
]
RESULT_DIRS = [
    'a2_phy_engineering_v1', 'a2_phy_rx_environment_v2',
    'a2b_source_v1', 'a2b_link_v1', 'a2b_rx_v1', 'a2b_metrics_v1',
    'a3_source_v1', 'a3_link_v1', 'a3_rx_v1', 'a3_metrics_v1',
    'final_figures_v1', 'final_report_v1',
]
GATES = dict(zip(RESULT_DIRS[2:10], [
    'SOURCE1000_CLOSED', 'PHY18000_CLOSED', 'ACTUAL_RX_CLOSED', 'CAL1000_FROZEN_CLOSED',
    'EVAL100_SOURCE_CLOSED', 'EVAL900_PHY_CLOSED', 'EVAL_ACTUAL_RX_CLOSED', 'EVAL100_FINAL_ANALYSIS_CLOSED']))
SCOPES = [
    'a2_phy_dispatch_control_v1', 'a2_phy_engineering_v1', 'a2_phy_environment_diagnostic_v1',
    'a2_phy_initial_control_v1', 'a2_phy_launcher_import_observation_v1', 'a2_phy_prepare_control_v1',
    'a2_phy_qualification_control_v1', 'a2_phy_rx_environment_dispatch_control_v2',
    'a2_phy_rx_environment_initial_control_v2', 'a2_phy_rx_environment_prepare_control_v2',
    'a2_phy_rx_environment_qualify_control_v1', 'a2_phy_rx_environment_qualify_control_v2',
    'a2_phy_rx_environment_v1', 'a2_phy_rx_environment_v2', 'a2_phy_synthetic_qualification_v1',
]
for stage in ['a2b_source', 'a2b_link', 'a2b_rx', 'a2b_metrics', 'a3_source', 'a3_link', 'a3_rx', 'a3_metrics']:
    SCOPES += [stage + suffix for suffix in ['_v1', '_initial_control_v1', '_prepare_control_v1',
        '_dispatch_control_v1', '_qualification_v1', '_qualify_control_v1']]
DOCS = ['README.md', 'RESEARCH_STATUS.md', 'PROGRESS.md', 'EXPERIMENTS.md']


def git(*argv):
    return subprocess.check_output(['git', '-c', 'core.quotepath=false', *argv], cwd=R, text=True).strip()


def load(path):
    return json.loads(path.read_text())


def check(name, argv, seconds=600):
    evidence.RUN = A
    before = pathlib.Path.cwd()
    os.chdir(R)
    try:
        evidence.run_stage(name, argv, dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='6', MKL_NUM_THREADS='6'), seconds)
    finally:
        os.chdir(before)
    return evidence.desc(A / (name + '.actual_wait.json'))


def copy_closed_receipts(extra_scopes):
    target = O / 'final_execution_receipts_v1'
    assert not target.exists()
    target.mkdir()
    copies = []
    scopes = SCOPES + extra_scopes
    assert len(scopes) == len(set(scopes))
    for name in scopes:
        assert re.fullmatch('[a-z0-9_]+', name)
        root = L / name
        assert root.is_dir() and not root.is_symlink()
        files = [p for p in root.iterdir() if p.is_file()
                 and p.suffix in ['.json', '.log', '.txt', '.stdout', '.stderr']
                 and not re.fullmatch('[a-f0-9]{64}.json', p.name)]
        # Named closed phase summaries and compact packet ledgers only.
        for sub in ['tx', 'rx', 'audit', 'rx_environment_v1', 'probe', 'constructor_probe']:
            for filename in ['completion.json', 'failure.json', 'calls.jsonl']:
                p = root / sub / filename
                if p.is_file():
                    files.append(p)
        for p in sorted(files):
            source = evidence.desc(p)
            assert source['bytes'] < 15_000_000 and not p.is_symlink()
            rel = p.relative_to(root)
            # Plain text extension retains original stdout/stderr/log bytes.
            name_parts = list(rel.parts)
            if p.suffix in ['.log', '.stdout', '.stderr']:
                name_parts[-1] += '.txt'
            dest = target / name / pathlib.Path(*name_parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as stream:
                stream.write(p.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            evidence.verify(source)
            result = evidence.desc(dest)
            assert (source['bytes'], source['sha256']) == (result['bytes'], result['sha256'])
            copies.append({'source': source, 'copy': result})
    evidence.put(target / 'copy_manifest.json', {'scope': 'Closed lightweight evidence only; no restoration backup.', 'files': copies})
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-parent', required=True)
    parser.add_argument('--expected-report-manifest', required=True)
    parser.add_argument('--expected-final-closure', required=True)
    parser.add_argument('--extra-closed-control', action='append', required=True)
    args = parser.parse_args()
    assert git('branch', '--show-current') == 'main' and git('rev-parse', 'HEAD') == args.expected_parent
    assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only') and not A.exists()
    assert evidence.desc(O / 'a3_metrics_v1/root_closure_review.json')['sha256'] == args.expected_final_closure
    for stage, status in GATES.items():
        review = load(O / stage / 'root_closure_review.json')
        assert review['status'] == status and review['all_processes_gone'] and review['unresolved'] == 0
        for pin in review['controls'] + review['outputs']:
            evidence.verify(pin)
    report_path = O / 'final_report_v1/manifest.json'
    assert evidence.desc(report_path)['sha256'] == args.expected_report_manifest
    report = load(report_path)
    assert report['status'] == 'CLOSED_RESULTS_REPORTED_PENDING_PUBLICATION'
    for pin in report['inputs'] + report['outputs']:
        evidence.verify(pin)
    visual = load(O / 'final_figures_v1/actual_visual_review.json')
    assert visual['status'] == 'PASS' and visual['actual_opened_both_pngs'] is True
    # Controls supplied here must be closed CPU figure/report supervisors on this host.
    for name in args.extra_closed_control:
        root = L / name
        assert re.fullmatch('[a-z0-9_]+', name) and not (root / 'failure.json').exists()
        sup = load(root / 'supervisor.json')
        wait = load(root / 'child.actual_wait.json')
        assert evidence.gone(sup['process']) and evidence.gone(wait['process'])
        assert type(wait['returncode']) is int and wait['returncode'] == 0
        assert not any(wait.get(k) for k in ['timed_out', 'terminated', 'killed', 'metadata_error'])
    lock = (L / 'arpc_science.lock').open('r')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    A.mkdir()
    evidence.put(A / 'authorization.json', {'parent': args.expected_parent, 'final_closure': args.expected_final_closure,
        'report_manifest': evidence.desc(report_path), 'source': evidence.desc(pathlib.Path(__file__)),
        'extra_closed_controls': args.extra_closed_control, 'normal_publication': True, 'created_unix': time.time()})
    light = copy_closed_receipts(args.extra_closed_control)
    body = '''## 2026-10-19 ARPC fixed100 matched comparison completed

Official adapted ARPC weights passed independent source decoding and the full
finite cal1000 strategy calibration. Frozen policies then completed900 logical
frames at4/10/19dB on external_common100_v1, followed by all seven metrics and
one10000-replicate source-paired analysis. The same100 source IDs select existing
final baseline scores; NeST-Com19dB uses T1-C m10. All failures remain in their
denominators. Results in every direction are reported without test-set selection.
See results/arpc_external_baseline_20261019_v1/final_report_v1/REPORT_zh.md and
final_figures_v1/. This is not the fullcommon500 result or a new blind test.
Original failed initialization and receiver-environment evidence is preserved.
No training, old baseline scoring, old timing or reconstruction was repeated.
Restoration backups remain deferred. Historical experiments/figures are unchanged.

'''
    for name in DOCS:
        p = R / name
        (A / ('before_' + name)).write_bytes(p.read_bytes())
        p.write_bytes(body.encode() + p.read_bytes())
    tracked = set(git('ls-files').splitlines())
    paths = []
    for directory in CODE_DIRS:
        root = E / directory
        assert root.is_dir()
        for p in sorted(root.iterdir()):
            if p.is_file():
                assert p.suffix in ['.py', '.md', '.json'] and not p.is_symlink()
                rel = p.relative_to(R).as_posix()
                if rel not in tracked:
                    paths.append(rel)
    for root in [O / name for name in RESULT_DIRS] + [light]:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                assert not p.is_symlink() and p.suffix in ['.json', '.jsonl', '.csv', '.md', '.txt', '.tex', '.pdf', '.svg', '.png']
                rel = p.relative_to(R).as_posix()
                if rel not in tracked:
                    paths.append(rel)
    paths += DOCS
    assert len(paths) == len(set(paths))
    evidence.put(A / 'reviewed_paths.json', paths)
    evidence.put(A / 'reviewed_input_files.json', [evidence.desc(R / p) for p in paths])
    subprocess.run(['git', 'add', '--sparse', '--', *paths], cwd=R, check=True)
    check('manifest_update', [PY, '-B', 'tools/update_repository_manifest.py'])
    subprocess.run(['git', 'add', '--sparse', '--', 'release_manifest.json'], cwd=R, check=True)
    paths.append('release_manifest.json')
    assert set(git('diff', '--cached', '--name-only').splitlines()) == set(paths)
    checks = {
        'diff_check': check('diff_check', ['git', 'diff', '--cached', '--check']),
        'verify_repository': check('verify_repository', [PY, '-B', 'tools/verify_repository.py']),
        'run_cpu_checks': check('run_cpu_checks', [PY, '-B', 'tools/run_cpu_checks.py']),
    }
    evidence.put(A / 'checks.json', checks)
    files = []
    for path in sorted(paths):
        data = subprocess.check_output(['git', 'show', ':' + path], cwd=R)
        files.append({'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    evidence.put(A / 'staged_files.json', files)
    proof = {'parent': args.expected_parent, 'tree': git('write-tree'),
        'checks_sha256': evidence.desc(A / 'checks.json')['sha256'],
        'staged_files_sha256': evidence.desc(A / 'staged_files.json')['sha256']}
    evidence.put(A / 'checks_complete.json', dict(proof, status='PASS'))
    message = A / 'commit_message.txt'
    message.write_text('Complete bounded ARPC calibration and fixed100 matched evaluation\n\n'
        'Preserve actual independent reception, all seven source-balanced metrics and directly paired intervals. '
        'Record frozen policies, paid source costs, historical failures and verified figures without extending the scientific scope.\n')
    check('commit', ['git', '-c', 'user.name=Codex', '-c', 'user.email=codex@openai.com', 'commit', '-F', str(message)])
    head = git('rev-parse', 'HEAD')
    assert git('rev-parse', 'HEAD^') == args.expected_parent
    evidence.put(A / 'commit_verified.json', dict(proof, head=head, normal_commit=True))
    print(json.dumps({'head': head, 'publication': str(A), 'push_pending': True}))


if __name__ == '__main__':
    main()
