"""Read-only ledger/status snapshot. Never launches or resumes any role."""
from pathlib import Path
import json,os,time,subprocess
from runtime import L,OLD,R,sha,write
def read(p):return json.loads(p.read_text()) if p.exists() else None
def live(identity):
    if not identity:return False
    p=Path('/proc')/str(identity['pid'])/'stat'
    return p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==str(identity['start_ticks'])
def collect():
    now=time.time();ledger={}
    for role in (L/'controllers').iterdir():
        reg=read(role/'registration.json');owner=read(role/'owner.json');child=read(role/'child.json');actual=read(role/'actual_wait.json')
        ledger[role.name]=dict(owner=owner,owner_live=live(owner),child=child,child_live=live(child),actual_wait=actual,
            registered_gpu_devices=reg['gpu_devices'] if reg else [],max_seconds=reg['max_seconds'] if reg else None,
            charged_gpu_device_hours=actual['charged_gpu_device_hours'] if actual else max(0,now-owner['started'])*len(reg['gpu_devices'])/3600,
            cpu_process_seconds=actual['cpu_process_seconds'] if actual else None,
            active_cpu_accounting='final descendant CPU process time recorded at actual wait' if not actual else None)
    data={}
    for name in ['corpus_coco116k_v2','corpus_openimages100k_v1']:
        out=L/name;data[name]={n:read(out/n) for n in ['progress.json','complete.json','index_identity.json']}
    active='training_v2' if (L/'training_v2/training_started.json').exists() else 'training_v1'
    o=dict(time=now,active_training_role=active,branch=subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
        old_P2_stop=(OLD/'STOP_P2').exists(),old_pause=read(OLD/'user_pause_verification_v1/complete.json')['P2_status'],
        ledger=ledger,training={n:read(L/active/n) for n in ['admission.json','training_started.json','progress.json','latest_checkpoint.json','segment_complete.json']},
        data=data,source_bindings=read(L/active/'request.json')['bindings'])
    o['charged_gpu_device_hours_total_as_of_snapshot']=sum(v['charged_gpu_device_hours'] for v in ledger.values())
    return o
if __name__=='__main__':print(json.dumps(collect(),ensure_ascii=False))
