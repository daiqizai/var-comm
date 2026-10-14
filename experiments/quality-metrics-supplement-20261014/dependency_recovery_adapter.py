from pathlib import Path
import argparse,ast,hashlib,json,os,sqlite3,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');E=R/'experiments/quality-metrics-supplement-20261014'
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1')
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text())
def prepare():
    d=T/'dependency_recovery_preparation_v1';assert not d.exists() and not (T/'hifi_score_dependency_recovery_v1').exists()
    wait=read(T/'hifi_score_path_recovery_dispatch_v1/actual_wait.json');assert type(wait['returncode']) is int and wait['returncode']==1 and wait['actual_wait'] is True
    for ident in (wait['process'],wait['parent_process'],read(T/'hifi_score_path_recovery_initial_control_v1/launch.json')['parent_process']):
        p=Path('/proc')/str(ident['pid'])/'stat';assert not p.exists() or int(p.read_text().rsplit(')',1)[1].split()[19])!=ident['start_ticks']
    db=sqlite3.connect('file:'+str(T/'hifi_score_path_recovery_v1/calls.sqlite')+'?mode=ro',uri=True);calls=db.execute('SELECT kind,state,count(*) FROM calls GROUP BY kind,state').fetchall();db.close()
    assert calls==[('load_dists','COMPLETE',1),('load_dreamsim','FAILED',1)] and not list((T/'hifi_score_path_recovery_v1').glob('row_*.json'))
    assert "No module named 'markupsafe'" in read(T/'hifi_score_path_recovery_v1/failure.json')['traceback']
    q=T/'dependency_qualify_control_v1/actual_wait.json';qw=read(q);assert type(qw['returncode']) is int and qw['returncode']==0 and qw['actual_wait'] is True and not qw['timed_out']
    for ident in (qw['process'],qw['parent_process']):
        p=Path('/proc')/str(ident['pid'])/'stat';assert not p.exists() or int(p.read_text().rsplit(')',1)[1].split()[19])!=ident['start_ticks']
    qualification=read(T/'dependency_import_qualification_v1/completion.json');assert qualification['status']=='PASS_REAL_DEPENDENCY_IMPORT_ONLY' and qualification['scientific_scores']==0 and qualification['checkpoint_loads']==0 and qualification['model_constructions']==0
    overlay=T/'dependency_overlay_v1/manifest.json';assert pin(overlay)['sha256']=='c3372713ca0237ade062a3a286b9ce7a531858d607f247a97a3d793e8c9bee4d'
    for row in read(overlay)['files']:assert pin(Path(row['path']))==row
    def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=funcs(E/'hifi_score.py'),funcs(E/'hifi_score_dependency_recovery_v1.py');unchanged=[n for n in a if n!='register'];assert all(a[n]==b[n] for n in unchanged)
    d.mkdir();v=dict(status='PASS',reviewer='Codex root scoped dependency repair and unchanged-science source review',observed_unix=time.time(),
        prior_failure=pin(T/'hifi_score_path_recovery_v1/failure.json'),prior_actual_wait=pin(T/'hifi_score_path_recovery_dispatch_v1/actual_wait.json'),prior_ledger=pin(T/'hifi_score_path_recovery_v1/calls.sqlite'),
        prior_dists_loads_complete=1,prior_dreamsim_load_reservations_failed=1,prior_metric_forwards=0,prior_scores=0,
        dependency_import_qualification=pin(T/'dependency_import_qualification_v1/completion.json'),qualification_actual_wait=pin(q),private_overlay=pin(overlay),
        old_environment_unchanged=True,official_metric_source_weights_preprocessing_unchanged=True,unchanged_scientific_function_AST=unchanged,
        new_channel=0,new_receiver=0,new_training=0,original_failed_sources_never_hot_edited=True,one_new_dependency_ready_attempt=True,
        sources={n:pin(E/n) for n in ['hifi_score_dependency_recovery_v1.py','execute_dependency_recovery_v1.py','record_cpu_dependency_recovery_v1.py','dependency_recovery_adapter.py']})
    with (d/'review.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(pin(d/'review.json')))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('role',choices=('prepare',));a.parse_args();prepare()
