"""Exact continuation after CRLF-only audit-copy diff failure; no data rewrite."""
from pathlib import Path
import hashlib,importlib.util,json,re,subprocess
E=Path(__file__).parent
s=importlib.util.spec_from_file_location('plan_publication',E/'publish.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
R,A,O=m.R,m.A,m.O
assert m.git('rev-parse','HEAD')==m.PARENT
old=json.loads((A/'diff_check.actual_wait.json').read_text());assert old['returncode']==2
assert not (A/'verify_repository.launch.json').exists() and not (A/'commit.launch.json').exists()
name='experiments/arpc-n2048-feasibility-20261019/inputs/asset_audit.json'
warnings=re.findall(r'^([^:\r\n]+):\d+: trailing whitespace\.$',(A/'diff_check.stdout.txt').read_text(),re.M)
assert len(warnings)==10960 and set(warnings)=={name}
data=(R/name).read_bytes();assert data.count(b'\r\n')==10960 and b'\n' not in data.replace(b'\r\n',b'')
assert all(not line.endswith((b' ',b'\t')) for line in data.split(b'\r\n'))
manifest=json.loads((O/'package_manifest.json').read_text())
for row in manifest['files']:
    p=R/row['path'];assert m.pin(p)['sha256']==row['sha256']
note=O/'publication_format_review.json'
m.put(note,dict(status='EXACT_CRLF_INPUT_COPY_PRESERVED',file=m.pin(R/name),original_default_failure=m.pin(A/'diff_check.actual_wait.json'),original_stdout=m.pin(A/'diff_check.stdout.txt'),warnings=10960,
    resolution='One command-scoped core.whitespace=cr-at-eol for this exact input audit JSON only; all other staged paths keep default checking. Original audit bytes and persistent Git configuration unchanged.',continuation_source=m.pin(Path(__file__)),initial_metadata_guard_note=m.pin(A/'line_guard_preflight_note.json')))
subprocess.run(['git','add','--sparse','--',str(Path(__file__).relative_to(R)),str(note.relative_to(R))],cwd=R,check=True)
m.check('manifest_update_continuation',[m.PY,'-B','tools/update_repository_manifest.py'])
subprocess.run(['git','add','--sparse','--','release_manifest.json'],cwd=R,check=True)
paths=m.git('diff','--cached','--name-only').splitlines()
expected={x['path'] for x in manifest['files']}|{'results/arpc_n2048_feasibility_20261019_v1/package_manifest.json','README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md','release_manifest.json',str(Path(__file__).relative_to(R)),str(note.relative_to(R))}
assert set(paths)==expected
checks={'diff_check':m.check('diff_exact_CRLF',['git','-c','core.whitespace=cr-at-eol','diff','--cached','--check','--',name]),
 'default_other_paths':m.check('diff_default_other_paths',['git','diff','--cached','--check','--',*sorted(expected-{name})]),
 'verify_repository':m.check('verify_repository',[m.PY,'-B','tools/verify_repository.py']),
 'run_cpu_checks':m.check('run_cpu_checks',[m.PY,'-B','tools/run_cpu_checks.py'])}
m.put(A/'checks.json',checks)
staged=[]
for name in sorted(paths):
    data=subprocess.check_output(['git','show',':'+name],cwd=R);staged.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
m.put(A/'staged_files.json',staged)
proof=dict(parent=m.PARENT,tree=m.git('write-tree'),checks_sha256=m.pin(A/'checks.json')['sha256'],staged_files_sha256=m.pin(A/'staged_files.json')['sha256'])
m.put(A/'checks_complete.json',dict(proof,status='PASS'))
(A/'commit_message.txt').write_text('Document ARPC payload-quality and bounded N2048 feasibility\n\nPreserve N1024 results; account for two paid subframes and reassembly. Report calibration fit, checked source reuse and proposed compute caps without running new science.\n')
m.check('commit',['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-F',str(A/'commit_message.txt')])
head=m.git('rev-parse','HEAD');assert m.git('rev-parse','HEAD^')==m.PARENT
m.put(A/'commit_verified.json',dict(proof,head=head,normal_commit=True));print(json.dumps(dict(head=head,push_pending=True)))
