"""Repair only host path serialization after a diagnosed zero-score failure."""
from pathlib import Path,PurePosixPath
import argparse,ast,copy,hashlib,importlib.util,json,os,sqlite3,sys,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/quality-metrics-supplement-20261014'
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1')
O=R/'results/quality_metrics_supplement_20261014_v1/hifi_fixed16'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(p.read_text())
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def normalize(old):
    v=old.replace('\\','/')
    while '//' in v:v=v.replace('//','/')
    v=PurePosixPath(v).as_posix()
    assert v.startswith(str(T/'metric_assets')+'/') and '..' not in PurePosixPath(v).parts and ':' not in v
    return v
def prepare():
    assert os.name=='posix'
    assert sha(O/'registration.json')=='8e9b6103b73b4255852187a4eab3b702bcae3801e7dc510b6194123b56838c75'
    assert sha(E/'hifi_score.py')=='208466c81cd590abcbd0039a71e25415a140e86524b8769ca1402b4c53a5dec9'
    assert sha(T/'metric_assets_manifest.json')=='1f58102ecffd1f167ba7dcad34c3612340737ea93c3726e2e0dafdecb64cc24a'
    d=T/'path_recovery_preparation_v1';assert not d.exists() and not (T/'hifi_score_path_recovery_v1').exists()
    wait=read(T/'hifi_score_dispatch_v1/actual_wait.json');assert wait['actual_wait'] is True and type(wait['returncode']) is int and wait['returncode']==1 and wait['metadata_error'] is None
    for ident in [wait['process'],wait['parent_process'],read(T/'hifi_score_initial_control_v1/launch.json')['parent_process']]:
        p=Path('/proc')/str(ident['pid'])/'stat';assert not p.exists() or int(p.read_text().rsplit(')',1)[1].split()[19])!=ident['start_ticks']
    assert not (T/'hifi_score_v1/completion.json').exists() and not list((T/'hifi_score_v1').glob('row_*.json'))
    db=sqlite3.connect('file:'+str(T/'hifi_score_v1/calls.sqlite')+'?mode=ro',uri=True)
    calls=db.execute('SELECT kind,state,count(*) FROM calls GROUP BY kind,state').fetchall();db.close();assert calls==[('load_dists','FAILED',1)]
    assert 'FileNotFoundError' in read(T/'hifi_score_v1/failure.json')['traceback']
    old=read(T/'metric_assets_manifest.json');new=copy.deepcopy(old);changes=[]
    def walk(v):
        if isinstance(v,dict):
            if 'path' in v:
                original=v['path'];v['path']=normalize(original);changes.append(dict(before=original,after=v['path']))
            for a in v.values():walk(a)
        elif isinstance(v,list):
            for a in v:walk(a)
    walk(new['models']);assert changes and all(Path(x['after']).exists() for x in changes)
    # All weight/source bytes are rechecked; no model loader or tensor is invoked.
    for row in new['files']:
        p=T/'metric_assets'/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    spec=importlib.util.spec_from_file_location('unchanged_metric_loader',R/'experiments/unified-metrics-20261002/metric_models.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for name,model in new['models'].items():
        m.verify_tree(model['implementation'],expected_revision=m.SOURCE_REVISIONS[name])
        for field in ('weights','vgg16_weights','dino_weights'):
            if field in model:m.verify_file(model[field])
        if name=='dreamsim':m.verify_tree(model['cache'],source=False);m.verify_tree(model['dino_implementation'],expected_revision=m.SOURCE_REVISIONS['dino'])
    assert not any(k in sys.modules for k in ('torch','tensorflow','dreamsim'))
    # Exact same scientific functions, with only distinct registration/runtime routes.
    def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=funcs(E/'hifi_score.py'),funcs(E/'hifi_score_path_recovery_v1.py')
    unchanged=[n for n in a if n!='register'];assert all(a[n]==b[n] for n in unchanged)
    d.mkdir();save(T/'metric_assets_posix_manifest.json',new)
    review=dict(status='PASS',reviewer='Codex root diagnosed path serialization and inspected scoped recovery',observed_unix=time.time(),
        diagnosis='Windows Path serialization wrote backslashes in relocated Linux model paths; failure occurred at first DISTS verify_tree before weight loading',
        changed_metadata_only=True,path_changes=changes,all61_original_asset_SHA_PASS=True,torch_and_metric_model_imports=[],
        old_registration=pin(O/'registration.json'),old_assets=pin(T/'metric_assets_manifest.json'),corrected_assets=pin(T/'metric_assets_posix_manifest.json'),
        old_failure=pin(T/'hifi_score_v1/failure.json'),old_ledger=pin(T/'hifi_score_v1/calls.sqlite'),old_actual_wait=pin(T/'hifi_score_dispatch_v1/actual_wait.json'),
        old_failed_loader_reservations=1,old_model_weights_loaded=0,old_metric_forwards=0,old_scores=0,original_processes_gone=True,
        unchanged_scientific_function_AST=unchanged,source={n:pin(E/n) for n in ['hifi_score.py','hifi_score_path_recovery_v1.py','execute_path_recovery_v1.py','record_cpu_path_recovery_v1.py','path_recovery_adapter.py']},
        old_stage_never_restarted=True,one_distinct_recovery_attempt=True,new_channel=0,new_receiver=0,new_training=0,common500='WAITING_FOR_ORIGINAL_FILES')
    save(d/'review.json',review);print(json.dumps(dict(review=pin(d/'review.json'),corrected_manifest=pin(T/'metric_assets_posix_manifest.json'),scientific_calls=0)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=('prepare',));p.parse_args();prepare()
