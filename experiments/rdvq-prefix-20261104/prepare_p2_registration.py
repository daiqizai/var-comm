"""Remote payload: first registration only; deliberately does not launch science."""
from pathlib import Path
import hashlib,json,subprocess,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');L=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/rdvq_prefix_20261104_v1');S=R/'experiments/rdvq-prefix-20261104'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);assert not p.exists();p.write_text(json.dumps(v,indent=2));return p
assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()=='0fc52c4ef47e16f9c1db3a1c7690615b33552567'
gate=L/'p1_source_gate.json';assert json.loads(gate.read_text())['passed']
wait=L/'p1_source_chain_owner_control_v1/actual_wait.json';w=json.loads(wait.read_text());assert w['returncode']==0 and w['child_gone']
D=L/'data_union_verified_v1';dg=D/'complete.json';assert json.loads(dg.read_text())['passed']
P=L/'p2_registration_v1';P.mkdir(exist_ok=False);CP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/rdvq_prefix_20261104_v1/p2_matched');CP.mkdir(exist_ok=False)
clean=lambda v:{k:v[k] for k in ['source_id','split','name','path','sha256']}
dev=json.loads((D/'development_manifest.json').read_text())['sources'][:128];test=json.loads((D/'test_manifest.json').read_text())['sources'];train=json.loads((D/'train_manifest.json').read_text())['sources']
assert [len(dev),len(test),len(train)]==[128,500,20000]
p1=json.loads((L/'p1_registration_v1/evaluation_manifest.json').read_text())['sources'];p1dev=[v for v in p1 if v['split']=='development'];p1test=[v for v in p1 if v['split']=='test']
assert [(v['source_id'],v['sha256']) for v in dev]==[(v['source_id'],v['sha256']) for v in p1dev]
assert [(v['source_id'],v['sha256']) for v in test]==[(v['source_id'],v['sha256']) for v in p1test]
dm=write(P/'development128.json',dict(sources=list(map(clean,dev))));tm=write(P/'test500.json',dict(sources=list(map(clean,test))));tr=write(P/'train20000.json',dict(sources=list(map(clean,train))))
rp=L/'p1_r0_rate_proxy_v1/complete.json';assert json.loads(rp.read_text())['passed'];checkpoint=json.loads((L/'checkpoint.json').read_text())
init={k:checkpoint[k] for k in ['path','sha256']};assert sha(init['path'])==init['sha256']
deps=[p for p in S.iterdir() if p.suffix in ['.py','.md']]+[gate,wait,dg,D/'duplicate_audit.json',dm,tm,tr,rp,L/'p1_p2_protocol.json',L/'metric_assets.json',Path(init['path'])]
bindings={str(p):sha(p) for p in deps};cpus=[[2,3],[4,5],[90,91],[92,93]];gpus=[]
for index,c in zip([1,2,4,5],cpus):
 line=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,pci.bus_id','--format=csv,noheader'],text=True).strip().split(',');gpus.append(dict(index=index,uuid=line[0].strip(),pci=line[1].strip(),cpus=c))
specs=[('R1',2026110401),('R2',2026110401),('R3',2026110401),('R3_estimated',2026110401),('R2',2026110402),('R3',2026110402),('R3_estimated',2026110402),('R2',2026110403),('R3',2026110403),('R3_estimated',2026110403)]
experiment=dict(protocol_sha256=sha(L/'p1_p2_protocol.json'),data_gate_sha256=sha(dg),training_manifest_sha256=sha(tr),initialization=init,arms=specs,steps=[2000,4000],global_batch=16,microbatch=4,cap_gpu_hours=256,reserved_gpu_hours=190,head='0fc52c4ef47e16f9c1db3a1c7690615b33552567',numerical_route='FP32, TF32off, mathSDPA; native bicubic backward nondeterminism disclosed')
ep=write(P/'experiment.json',experiment);bindings[str(ep)]=sha(ep);arms=[]
source_roots=[str(L/x) for x in ['data','data_priority','data_archive_repair','data_last2','data_train_complement','data_exact_gap_repair','data_union_verified_v1','p1_registration_v1','p2_registration_v1']]
for i,(arm,seed) in enumerate(specs):
 name=f'p2_{arm.lower()}_seed{seed}';gpu=gpus[i%4]
 training=dict(arm=arm,seed=seed,p1_gate=str(gate),training_manifest=str(tr),audit_manifest=str(dm),initial_rate_audit=str(rp),initialization=init,metric_assets=str(L/'metric_assets.json'),experiment_sha256=sha(ep))
 req=dict(root=str(L),output_root=str(L/name),reserved_gpu_hours=19,p1_gate=str(gate),gpu=gpu,bindings=bindings,training=training,checkpoint_dir=str(CP/name),development_manifest=str(dm),test_manifest=str(tm),source_roots=source_roots,metric_assets=str(L/'metric_assets.json'))
 path=write(P/(name+'.json'),req);arms.append(dict(name=name,request=str(path),output_root=req['output_root'],gpu_index=gpu['index'],arm=arm,seed=seed,sha256=sha(path)))
queue=dict(root=str(L),output=str(L/'p2_study_v1'),reserved_gpu_hours=190,cap_gpu_hours=256,p1_gate=str(gate),dataset_gate=str(dg),p1_outer_wait=str(wait),bindings=dict(bindings,**{v['request']:v['sha256'] for v in arms}),arms=arms,gpus=gpus,absolute_deadline=time.time()+72*3600)
qp=write(P/'queue_request.json',queue)
analysis=write(P/'analysis_request.json',dict(output=str(L/'p2_matched_analysis_v1'),statistics_identity=str(L/'p1_source_analysis_v1/statistics_identity.json'),source_bank=str(L/'p1_source_analysis_v1/source_bootstrap_bank.npy'),r0_scores=str(L/'p1_source_metrics_v1/scores.csv'),arms=arms))
print(json.dumps(dict(prepared=True,not_started=True,registration=str(qp),sha256=sha(qp),arms=len(arms),reserved_gpu_hours=190,cap=256,absolute_deadline=queue['absolute_deadline'])))
