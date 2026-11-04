from pathlib import Path
import json,hashlib,subprocess,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');L=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/rdvq_prefix_20261104_v1');S=R/'experiments/rdvq-prefix-20261104';PY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
rp=L/'p2_registration_v1/queue_request.json';r=json.loads(rp.read_text());assert not (L/'p2_study_v1').exists() and not (L/'p2_launch.json').exists() and not (L/'STOP_P2').exists()
assert time.time()<r['absolute_deadline'];assert json.loads(Path(r['p1_gate']).read_text())['passed'];assert json.loads(Path(r['dataset_gate']).read_text())['passed']
for p,h in r['bindings'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
w=json.loads(Path(r['p1_outer_wait']).read_text());assert w['returncode']==0 and w['child_gone']
control=L/'p2_owner_control_v1';control.mkdir(exist_ok=False)
code='import os,sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(S))+')\nfrom owned_process import run_role\nrun_role('+repr(str(control))+',"study",[sys.executable,"-u",'+repr(str(S/'p2_study_controller.py'))+','+repr(str(rp))+'],dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2"),list(range(180)),[],280800,'+repr({str(rp):hashlib.sha256(rp.read_bytes()).hexdigest(),str(S/'p2_study_controller.py'):hashlib.sha256((S/'p2_study_controller.py').read_bytes()).hexdigest()})+')\n'
owner=control/'owner.py';owner.write_text(code);compile(code,str(owner),'exec')
p=subprocess.Popen([PY,'-u',str(owner)],stdin=subprocess.DEVNULL,stdout=(control/'stdout.log').open('xb'),stderr=(control/'stderr.log').open('xb'),start_new_session=True)
receipt=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],request=str(rp),request_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),reserved_gpu_hours=190,absolute_deadline=r['absolute_deadline'],first_only=True)
(L/'p2_launch.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
