"""Resume only interrupted lightweight evidence copying, then original publication."""
import ast
import importlib.util
import json
import os
import pathlib
import re
import sys
import time

E=pathlib.Path(__file__).parent.parent
sys.path.insert(0,str(E))
import a1_owner as evidence
L,O,R=evidence.L,evidence.O,evidence.R
path=E/'final_layout_v2/publish.py'
assert evidence.desc(path)['sha256']=='53441b01506b6c08dce106d61df594062750c29d797bb3cc34fe337ee0a48769'
spec=importlib.util.spec_from_file_location('reviewed_final_layout_route',path)
route=importlib.util.module_from_spec(spec);spec.loader.exec_module(route)
m=route.module
EXCLUDED={'path':str(L/'a2b_link_v1/tx/completion.json'),'bytes':19048505,
          'sha256':'5cd4be7a8cf66c0c1a5acf7a1c03333ea326bcc8212d1c191e5064dafcc9ca7a'}

def main():
    d=L/'final_publication_execution_v1'
    wait=json.loads((d/'child.actual_wait.json').read_text())
    assert type(wait['returncode']) is int and wait['returncode']==1 and evidence.gone(wait['process'])
    assert evidence.gone(json.loads((d/'supervisor.json').read_text())['process'])
    a=json.loads((m.A/'authorization.json').read_text())
    assert m.git('rev-parse','HEAD')==a['parent'] and m.git('branch','--show-current')=='main'
    assert not m.git('diff','--name-only') and not m.git('diff','--cached','--name-only')
    assert not (m.A/'reviewed_paths.json').exists() and not (m.A/'before_README.md').exists()
    assert evidence.desc(O/'a3_metrics_v1/root_closure_review.json')['sha256']==a['final_closure']
    evidence.verify(a['report_manifest'])
    assert json.loads((O/'final_figures_v2/actual_visual_review.json').read_text())['status']=='PASS'
    evidence.verify(EXCLUDED)
    target=O/'final_execution_receipts_v1';assert target.is_dir() and not(target/'copy_manifest.json').exists()
    extra=a['extra_closed_controls']
    m.CODE_DIRS += ['final_receipt_resume_v1']
    # Preserve the closed failed publication supervisor, too. Its live continuation
    # is deliberately excluded; actual publication exits are accompanying receipts.
    m.SCOPES += ['final_publication_execution_v1','final_publication_initial_control_v1']
    copies=[];expected=set();reused=0
    scopes=m.SCOPES+extra;assert len(scopes)==len(set(scopes))
    for name in scopes:
        assert re.fullmatch('[a-z0-9_]+',name)
        root=L/name;assert root.is_dir() and not root.is_symlink()
        files=[p for p in root.iterdir() if p.is_file() and p.suffix in ['.json','.log','.txt','.stdout','.stderr'] and not re.fullmatch('[a-f0-9]{64}.json',p.name)]
        for sub in ['tx','rx','audit','rx_environment_v1','probe','constructor_probe']:
            for filename in ['completion.json','failure.json','calls.jsonl']:
                p=root/sub/filename
                if p.is_file():files.append(p)
        for p in sorted(files):
            source=evidence.desc(p)
            if str(p)==EXCLUDED['path']:
                assert source==EXCLUDED
                continue
            assert source['bytes']<15_000_000 and not p.is_symlink()
            parts=list(p.relative_to(root).parts)
            if p.suffix in ['.log','.stdout','.stderr']:parts[-1]+='.txt'
            dest=target/name/pathlib.Path(*parts);expected.add(dest)
            if dest.exists():
                result=evidence.desc(dest);assert result['sha256']==source['sha256'] and result['bytes']==source['bytes'];reused+=1
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
                result=evidence.desc(dest)
            evidence.verify(source);assert result['sha256']==source['sha256'] and result['bytes']==source['bytes']
            copies.append({'source':source,'copy':result})
    assert {p for p in target.rglob('*') if p.is_file()}==expected
    evidence.put(target/'copy_manifest.json',{'scope':'Closed lightweight evidence only; no restoration backup.','files':copies,
      'large_closed_metadata_retained_on_PFS':[EXCLUDED],'exclusion_reason':'Exact full 18,000-frame TX completion is 19 MB; preserved on PFS and bound by the scientific closure. No array or scientific bytes modified.',
      'initial_copy_failure_preserved':evidence.desc(d/'failure.json'),'verified_existing_copies':reused,'source':evidence.desc(__file__)})
    evidence.put(m.A/'copy_resume_review.json',{'source':evidence.desc(__file__),'manifest':evidence.desc(target/'copy_manifest.json'),'original_failed_execution':evidence.desc(d/'child.actual_wait.json'),'only_exact_oversize_excluded':EXCLUDED,'observed_unix':time.time()})
    # Continue at the original document/staging/check/commit body. Earlier gates and
    # copy phase are explicitly verified above; original checks are not weakened.
    original=E/'final_publication_v1/publish.py'
    assert evidence.desc(original)['sha256']=='aec83acc71aee6016078ba84a11b90b1dc2519511c03901c5b0b1ae9c5cfdae8'
    tree=ast.parse(original.read_text().replace('final_figures_v1','final_figures_v2'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
    fn.name='continue_verified_publication';fn.args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='args'),ast.arg(arg='light')],kwonlyargs=[],kw_defaults=[],defaults=[]);fn.body=fn.body[start:]
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    exec(compile(module,'<exact original final publication continuation>','exec'),m.__dict__)
    from types import SimpleNamespace
    args=SimpleNamespace(expected_parent=a['parent'])
    m.continue_verified_publication(args,target)

if __name__=='__main__':main()
