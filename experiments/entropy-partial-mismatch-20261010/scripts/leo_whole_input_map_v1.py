"""Build the finite gate input map from actually verified migration receipts.

Metadata and SHA reads only. No model import, CUDA, PHY, subprocess worker or
automatic successor. The separately invoked gate prepare/run own those stages.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import leo_whole_gate_v1 as g

RT=g.BASE/'var_comm_runtime_20261010'
HEAD='ca26ceb383e93fc4ee56298dcf896720188f86f6'


def descriptor(path):
    p=g.inside(path);return dict(path=str(p),sha256=g.sha(p))


def source_rows(receipt):
    g.require(receipt['status']=='FIXED_EXISTING32_BYTES_RESTORED_VERIFIED_NOT_SCIENTIFIC_QUALIFICATION'
              and receipt['new100_read'] is False and receipt['model_calls']==0,'Existing32 restore not complete')
    rows=[dict(original_path=r['original'],path=r['restored'],sha256=r['sha256'],bytes=r['bytes']) for r in receipt['mapping']]
    g.require(receipt['count']==len(rows)==547 and sum(r['bytes'] for r in rows)==receipt['total_bytes']==98236269,'Existing32 restore scope changed')
    g.require(len({r['original_path'] for r in rows})==len(rows) and len({r['path'] for r in rows})==len(rows),'Ambiguous existing32 restore mapping')
    g.require(all(r['original_path'].startswith(g.OLD+'outputs/') for r in rows),'Unexpected existing32 original namespace')
    return rows


def model_rows(receipt,root):
    g.require(receipt['status']=='MODEL_BYTES_AND_PUBLISHED_SPARSE_SOURCE_VERIFIED_NOT_RUNTIME_QUALIFICATION'
              and receipt['gpu_calls']==0 and len(receipt['files'])==5,'Original model seed verification missing')
    rows=[];seen=set();base=RT/'assets/model_seed_v1'
    for r in receipt['files']:
        p=g.inside(r['path']);g.require(p.is_relative_to(base) and r['verified'] is True,'Unverified model seed input')
        rel=p.relative_to(base).as_posix()
        if rel.startswith('VAR_COMM/'):
            old=g.OLD+rel[len('VAR_COMM/'):]
        elif rel.startswith('external/home/'):
            old='/'+rel[len('external/'):]
        else:raise RuntimeError('Unknown original model namespace')
        g.require(old not in seen,'Duplicate original model seed identity');seen.add(old)
        # Published identical source code is used at the actual repository path.
        target=root/old[len(g.OLD):] if old==g.OLD+'src/var_comm/next_scale_prior.py' else p
        rows.append(dict(original_path=old,path=str(target),sha256=r['sha256'],bytes=r['bytes']))
    expected=set(g.WEIGHTS.values())|{
        g.OLD+'src/var_comm/next_scale_prior.py',
        g.OLD+'outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v2_repaired_20260921/decoder_gate.json',
        g.OLD+'outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_A_v1/checkpoints/update_0038000_1789635001035591298.pt'}
    g.require(seen==expected,'Model seed identity set changed');return rows


def candidate_rows(receipt):
    g.require(receipt['status']=='UPSTREAM_CANDIDATE_VERIFIED_PENDING_NUMERICAL_QUALIFICATION'
              and receipt['original_dist_sha_available'] is False and receipt['original_models_exact_matches']==7
              and receipt['upstream_commit']==g.UPSTREAM_COMMIT and receipt['dist_sha256']==g.UPSTREAM_DIST_SHA
              and receipt['audit_sha256']==g.UPSTREAM_AUDIT_SHA and receipt['model_calls']==0,'Unreviewed native candidate')
    root=g.inside(receipt['root']);rows=[]
    for r in receipt['files']:
        rel=PurePosixPath(r['path'])
        g.require(not rel.is_absolute() and '..' not in rel.parts and str(rel)==r['path'],'Unsafe candidate relative path')
        if str(rel).startswith('models/'):
            g.require(r['exact_original_match'] is True and r['sha256']==r['original_sha256'],'Original model bytes differ')
            rows.append(dict(original_path=g.AUTHOR+rel.name,path=str(root/str(rel)),sha256=r['sha256'],bytes=r['bytes']))
    g.require(len(rows)==7 and len({r['original_path'] for r in rows})==7,'Candidate model map incomplete')
    return rows,root


def merge_rows(*groups):
    rows={}
    for group in groups:
        for r in group:
            old=r['original_path']
            g.require(old not in rows or rows[old]==r,'Conflicting source/provider mapping: '+old)
            rows[old]=r
    return list(rows.values())


def optional_host_native_pin(a):
    path=getattr(a,'host_native_addendum',None);digest=getattr(a,'host_native_addendum_sha256',None)
    g.require(bool(path)==bool(digest),'Host-native addendum path and SHA must be supplied together')
    if not path:return None
    g.require(isinstance(digest,str) and g.SHA_RE.fullmatch(digest),'Exact host-native addendum SHA required')
    return dict(path=str(g.inside(path)),sha256=digest)


def verify_published_head(root,head):
    g.require(isinstance(head,str) and g.re.fullmatch('[0-9a-f]{40}',head),'Invalid actual repository HEAD')
    if head==HEAD:return
    result=subprocess.run(['git','-c','core.hooksPath=/dev/null','-C',str(root),
        'merge-base','--is-ancestor',HEAD,head],capture_output=True,text=True,timeout=15)
    g.require(result.returncode==0,'Actual repository HEAD is not a verified descendant of the published baseline')


def build(a):
    addendum=optional_host_native_pin(a)
    root=g.inside(a.project_root);out=g.inside(a.out)
    g.require(not out.exists(),'Fresh metadata output required')
    g.require(0<a.max_seconds<=1800 and time.time()<a.deadline_unix,'Finite future gate deadline required')
    head=subprocess.check_output(['git','-c','core.hooksPath=/dev/null','-C',str(root),'rev-parse','HEAD'],text=True,timeout=15).strip()
    verify_published_head(root,head)
    pins={name:dict(path=str(g.inside(getattr(a,name))),sha256=getattr(a,name+'_sha256'))
          for name in ('source_receipt','model_receipt','candidate_receipt','runtime_receipt')}
    receipts={name:g.checked_json(p['path'],p['sha256']) for name,p in pins.items()}
    source=g.Resolver(source_rows(receipts['source_receipt']));records,h=g.source_metadata(source)
    # Only the first two fixed source closure is registered. No new-source arrays are read.
    model=model_rows(receipts['model_receipt'],root);authors,croot=candidate_rows(receipts['candidate_receipt'])
    candidate=dict(root=str(croot),audit=dict(path=str(g.inside(a.candidate_audit)),sha256=g.UPSTREAM_AUDIT_SHA),
        commit_record=dict(path=str(g.inside(a.candidate_commit)),sha256=g.UPSTREAM_COMMIT_JSON_SHA),old_dist_sha_unavailable=True)
    code=[]
    for rel,digest in g.CODE.items():
        p=g.inside(root/rel);g.require(p.is_file() and g.sha(p)==digest,'Published mathematical source changed: '+rel)
        code.append(dict(original_path=g.OLD+rel,path=str(p),sha256=digest,bytes=p.stat().st_size))
    rows=merge_rows(list(source.used.values()),model,authors,code);mapped=g.Resolver(rows)
    g.native_candidate(candidate,{p:s for p,s in h['source_bindings'].items() if p.startswith(g.AUTHOR)},mapped)
    environment=g.runtime_from_receipt(pins['runtime_receipt'])
    # Actual file verification belongs to gate.prepare; this stage binds the
    # complete original relocation map and validates its semantic provenance.
    runtime=g.RuntimeFiles(environment)
    out.mkdir(parents=True)
    env_path=out/'runtime_projection.json';g.write(env_path,environment)
    if addendum:g.attach_host_native_addendum(runtime,addendum,g.sha(env_path))
    spec=dict(schema='LEO_WHOLE_GATE_INPUT_MAP_V1',source_indices=[0,1],caps=g.CAPS,gpu_uuid=g.UUID,
        deadline_unix=a.deadline_unix,max_seconds=a.max_seconds,project_root=str(root),files=rows,
        python=environment['python'],environment=descriptor(env_path),native_dependency_candidate=candidate,
        receipts=pins,published_source_head=head,builder=descriptor(__file__))
    if addendum:spec['host_native_addendum']=addendum
    spec_path=out/'spec.json';g.write(spec_path,spec)
    result=dict(status='TWO_SOURCE_GATE_INPUT_MAP_BUILT_NOT_PREPARED_OR_EXECUTED',spec=descriptor(spec_path),
        runtime_projection=descriptor(env_path),mapped_original_files=len(rows),source_indices=[0,1],source_ids=g.IDS,
        model_calls=0,packet_calls=0,new_source_arrays_read=0,automatic_successor=False,
        next_prepare_argv=[sys.executable,'-B',str(Path(g.__file__).absolute()),'prepare','--spec',str(spec_path),
                           '--spec-sha256',g.sha(spec_path),'--out',str(out/'registered_gate')])
    g.write(out/'map_receipt.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',default=str(g.BASE/'code/VAR_COMM'))
    defaults=dict(source_receipt=RT/'receipts/fixed_existing32_seed_verified_v1.json',
        model_receipt=RT/'receipts/model_seed_remote_verified_v1.json',
        candidate_receipt=RT/'receipts/var_upstream_candidate_verified_v1.json',
        runtime_receipt=RT/'receipts/frozen_runtime_relocation_v1/completion.json')
    for name,path in defaults.items():
        p.add_argument('--'+name.replace('_','-'),default=str(path))
        p.add_argument('--'+name.replace('_','-')+'-sha256',required=True)
    p.add_argument('--candidate-audit',default=str(RT/'assets/var_upstream_candidate_v1/audit.json'))
    p.add_argument('--candidate-commit',default=str(RT/'assets/var_upstream_candidate_v1/commit.json'))
    p.add_argument('--host-native-addendum')
    p.add_argument('--host-native-addendum-sha256')
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--max-seconds',type=float,default=1800)
    p.add_argument('--out',required=True)
    print(json.dumps(build(p.parse_args()),sort_keys=True))


if __name__=='__main__':main()
