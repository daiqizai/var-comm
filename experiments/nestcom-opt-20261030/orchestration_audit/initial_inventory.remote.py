from pathlib import Path
import subprocess,json,hashlib,os,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
def cmd(args):
 p=subprocess.run(args,cwd=R,capture_output=True,text=True,timeout=90)
 return dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
v=dict(observed_unix=time.time(),repo_exists=R.exists(),git=[cmd(['git',*a])for a in [['rev-parse','HEAD'],['branch','--show-current'],['status','--short'],['diff','--stat','9d55aacfcd740034ce85014ae65f20a33893b59c','HEAD'],['diff','--cached','--stat']]],gpu=cmd(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits']),gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits']),filesystem=cmd(['df','-B1',str(R),str(B)]),affinity=sorted(os.sched_getaffinity(0)),load=os.getloadavg())
v['requested_new_directories']={str(p):p.exists()for p in [R/'experiments/nestcom-opt-20261030',R/'results/nestcom_opt_20261030_v1',B/'nestcom_opt_20261030_v1']}
v['root_documents']={n:dict(bytes=(R/n).stat().st_size,sha256=hashlib.sha256((R/n).read_bytes()).hexdigest(),text=(R/n).read_text())for n in ['AGENTS.md','README.md','RESEARCH_STATUS.md','PROGRESS.md','EXPERIMENTS.md']}
for x in ['cpu.max','memory.max']:
 p=Path('/sys/fs/cgroup')/x
 if p.exists():v[x]=p.read_text()
print(json.dumps(v,ensure_ascii=False))
