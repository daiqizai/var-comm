"""Exact legacy status-label admission; preserve the zero-science first failure."""
from common import *
import asset_audit as old,ast
failure=read(T/'asset_audit_control_v1/actual_wait.json')
assert failure['returncode']==1 and not failure['timed_out'] and gone(failure['process']) and gone(failure['parent_process'])
prepared=read(T/'asset_audit_control_v1/prepared.json');checked(prepared['source'])
source=Path(old.__file__).read_text()
changes={
 "assert not (O/'assets').exists();(O/'assets').mkdir(parents=True);(T/'assets').mkdir(parents=True)":"assert (O/'assets').is_dir() and (T/'assets').is_dir()",
 "assert vdone['status']=='COMPLETE'":"assert vdone['status']=='COMPLETE_R1_VISUAL'",
 "assert sdone['status']=='COMPLETE'":"assert sdone['status']=='COMPLETE_R1_SCORE'"}
changed=source
for a,b in changes.items():assert changed.count(a)==1;changed=changed.replace(a,b)
restored=changed
for a,b in changes.items():restored=restored.replace(b,a)
assert ast.dump(ast.parse(source))==ast.dump(ast.parse(restored))
reused=[]
def keep_exact(p,v):
 p=Path(p)
 if p.exists():assert read(p)==v;reused.append(pin(p))
 else:save(p,v)
ns=dict(old.__dict__);exec(compile(changed,str(E/'asset_audit.py'),'exec'),ns)
ns['save']=keep_exact;ns['run']()
save(O/'assets/status_label_recovery.json',dict(status='PASS_EXACT_LEGACY_LABEL_ADAPTER',old_source=prepared['source'],new_source=pin(__file__),changes=list(changes.items()),inverse_AST_exact=True,original_failed_wait=pin(T/'asset_audit_control_v1/actual_wait.json'),existing_metadata_reused_exact=reused,model_calls=0,scientific_calls=0))
