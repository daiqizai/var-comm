"""Read-only native-source inventory projection for first confirmation100 access.

All existing source hashes and every non-binding environment field stay intact.
Only newly tracked files already discovered by the original native inventory
function may be added. No model modules, images, source codec or PHY are run.
"""
import argparse,copy,hashlib,importlib.util,json,os,subprocess
from pathlib import Path

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def write(path,value):
    with Path(path).open('x',encoding='utf-8')as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def project(original,current,tracked):
    old=original['native_source_bindings'];sources=original['source_bindings'];tracked=set(tracked)
    require(isinstance(old,dict)and old and isinstance(sources,dict)and sources,'Bound original source inventories required')
    for path,expected in sources.items():require(sha(path)==expected,'Previously registered source changed: '+path)
    for path,expected in old.items():
        require(sources.get(path)==expected and current.get(path)==expected==sha(path),'Original native source missing or changed: '+path)
    additions={p:h for p,h in current.items()if p not in old}
    require(set(additions)<=tracked,'Only current newly tracked source paths may extend the original native inventory')
    for path,expected in additions.items():
        require(sha(path)==expected,'Current tracked addition changed during projection: '+path)
        require(path not in sources or sources[path]==expected,'Addition conflicts with an original source binding: '+path)
    projected=copy.deepcopy(original);projected['native_source_bindings']=dict(old,**additions)
    projected['source_bindings']=dict(sources,**additions)
    require(projected['native_source_bindings']==current,'Exact original-plus-added inventory required')
    excluded={'source_bindings','native_source_bindings'}
    old_other={k:v for k,v in original.items()if k not in excluded};new_other={k:v for k,v in projected.items()if k not in excluded}
    require(old_other==new_other,'Projection may not change visual/scorer/numerical/model/data identity or other environment fields')
    return projected,additions,digest(old_other)

def run(a):
    original=read(a.environment_request);root=Path(original['root']).resolve();out=Path(a.out).resolve()
    require(not out.exists()and out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009'),'Fresh separate WCL projection directory required')
    v=original['visual_config'];path=Path(v['static_closure_module']).resolve()
    require(original['source_bindings'].get(str(path))==sha(path),'Original read-only native inventory helper must be pinned')
    # This bound helper imports only the standard library and queries git ls-files.
    spec=importlib.util.spec_from_file_location('_t6_bound_readonly_native_inventory',path);closure=importlib.util.module_from_spec(spec);spec.loader.exec_module(closure)
    require(tuple(closure.GIT_PATTERNS)==('*.py','*.cpp','configs/*.yaml','configs/*.json'),'Original tracked-path rule changed')
    current=closure.collect_bindings(root,v['native_runtime'],v['var_source'],v['dino_source'],v['uep_runtime'])
    tracked=[str(root/p)for p in closure.tracked_paths(root)]
    projected,additions,preserved=project(original,current,tracked)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    descriptor=pin(a.environment_request);out.mkdir(parents=True)
    addition_record=dict(schema='T6_NEWLY_TRACKED_NATIVE_SOURCE_ADDITIONS_V1',original_environment_request=descriptor,
        old_native_count=len(original['native_source_bindings']),current_native_count=len(current),added_count=len(additions),additions=additions,
        original_native_hashes_all_exact=True,original_source_hashes_all_exact=True,unchanged_nonbinding_fields_sha256=preserved,
        tracked_patterns=list(closure.GIT_PATTERNS),inventory_helper=pin(path),execution_commit=commit)
    write(out/'added_tracked_sources.json',addition_record)
    projected['source_inventory_projection']=dict(original_environment_request=descriptor,added_tracked_sources=pin(out/'added_tracked_sources.json'),
        original_native_hashes_all_exact=True,original_source_hashes_all_exact=True,unchanged_nonbinding_fields_sha256=preserved,
        model_identity_unchanged=True,numerical_flags_unchanged=True,metric_identity_unchanged=True,projection_only=True)
    write(out/'environment_request.json',projected)
    value=dict(status='T6_CONFIRMATION_SOURCE_ENVIRONMENT_PROJECTION_COMPLETE',original_environment_request=descriptor,
        environment_request=pin(out/'environment_request.json'),added_tracked_sources=pin(out/'added_tracked_sources.json'),
        previous_native_files=len(original['native_source_bindings']),previous_source_files=len(original['source_bindings']),
        added_tracked_files=len(additions),unchanged_nonbinding_fields_sha256=preserved,script=pin(__file__),inventory_helper=pin(path),
        original_visual_identity=original['old_visual_identity'],original_score_identity=original['old_score_identity'],
        original_numerical_runtime=original['old_numerical_runtime'],original_environment_modified=False,
        source_model_calls=0,image_reads=0,new_packet_decodes=0,new_metric_calls=0,new_bootstrap_calls=0,
        outputs={str(out/n):sha(out/n)for n in('environment_request.json','added_tracked_sources.json')})
    write(out/'completion.json',value);return value

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--environment-request',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(run(a),sort_keys=True),flush=True)
