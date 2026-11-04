from pathlib import Path
import hashlib,json,subprocess
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');L=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/rdvq_prefix_20261104_v1');S=R/'experiments/rdvq-prefix-20261104';CPY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
role='p1_control_analysis_v1';assert not (L/role).exists();gate=L/'p1_source_gate.json';assert json.loads(gate.read_text())['passed'];wait=L/'p1_source_chain_owner_control_v1/actual_wait.json';w=json.loads(wait.read_text());assert w['returncode']==0 and w['child_gone']
roots=[L/f'p1_reliability_{i:02d}' for i in range(8)]
for d in roots:
 w=json.loads((d/'actual_wait.json').read_text());assert w['returncode']==0 and w['child_gone']
P=L/'p1_control_analysis_registration_v1';P.mkdir(exist_ok=False)
r=dict(output=str(L/role),source_gate=str(gate),source_outer_wait=str(wait),manifest=str(L/'p1_registration_v1/evaluation_manifest.json'),scores=str(L/'p1_source_metrics_v1/scores.csv'),statistics_identity=str(L/'p1_source_analysis_v1/statistics_identity.json'),source_bank=str(L/'p1_source_analysis_v1/source_bootstrap_bank.npy'),reliability_roots=list(map(str,roots)))
rp=P/'request.json';rp.write_text(json.dumps(r,indent=2))
files=[S/'analyze_reliability_control.py',S/'owned_process.py',rp,gate,wait]+[Path(r[k]) for k in ['manifest','scores','statistics_identity','source_bank']]+[d/'events.jsonl' for d in roots]
bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(P/'bindings.json').write_text(json.dumps(bindings,indent=2))
code='import os,sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(S))+')\nfrom owned_process import run_role\nrun_role('+repr(str(L))+','+repr(role)+',[sys.executable,"-u",'+repr(str(S/'analyze_reliability_control.py'))+',"--request",'+repr(str(rp))+'],dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2"),[16,17],[],3600,'+repr(bindings)+')\n'
owner=P/'owner.py';owner.write_text(code);compile(code,str(owner),'exec')
p=subprocess.Popen([CPY,'-u',str(owner)],stdin=subprocess.DEVNULL,stdout=(P/'stdout.log').open('xb'),stderr=(P/'stderr.log').open('xb'),start_new_session=True)
receipt=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],role=role,request_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),new_GPU_calls=0)
(P/'launch.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
