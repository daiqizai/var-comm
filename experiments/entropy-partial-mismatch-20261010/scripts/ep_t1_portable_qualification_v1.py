"""Read-only exact-byte relocation proof for the original completed T1-241 gate.

No qualification case is executed here. Original241 qualification provenance is
retained separately from the new-host CPU environment qualification. Formal
runtime admission still checks all original request/source/input/output bytes,
the closed241 ledger, actual current constructors and full144 catalogue.
"""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import ep_new100_content_io_v1 as io

SCHEMA='T1_ORIGINAL241_EXPLICIT_RELOCATION_SPEC_V1'
T1_SHA='47f75430235c98432a513dbbcb492556d4e1977f3631f1806e648f558b1b583e'
OLD_ROOT='/home/liulu/projects/VAR_COMM'
require=io.require


def inspect_binding(desc):
    spec=io.read(desc)
    require(spec['schema']==SCHEMA and spec['status']=='ORIGINAL_T1_241_BYTES_RELOCATED_SHA_CLOSED_NO_NEW_QUALIFICATION' and
        spec['original_actual_decodes']==241 and spec['new_qualification_calls']==spec['new_actual_decodes']==0 and
        spec['original_ledger_read_only'] is True and spec['original_receipt_bytes_modified'] is False and
        spec['qualified_runtime_admitted'] is False and spec['original_root']==OLD_ROOT,
        'Explicit original241 relocation only; no newly claimed qualification')
    mapping={x['original_path']:x for x in spec['mapping']}
    require(len(mapping)==len(spec['mapping']),'Ambiguous old-to-current qualification mapping')
    for row in mapping.values():
        p=io.inside(row['path']);require(p.is_file() and p.stat().st_size==row['bytes'] and io.sha(p)==row['sha256'],
            'Relocated original qualification file changed: '+row['original_path'])
    completed=io.read(spec['completion']);request=io.read(spec['request'])
    require(completed['status']=='PASS' and completed['schema']=='WCL_T1_REAL_PHY_QUALIFICATION_20261009_V1' and
        completed['actual_constructor_validation'] is True and completed['profile_count']==144 and completed['actual_layout_count']==12 and
        completed['actual_body_decodes']==96 and completed['actual_header_decodes']==145 and completed['packet_decode_count']==241 and
        completed['ledger']==dict(total=241,unresolved=0,cap=241),'Original real241 qualification did not close')
    require(request['root']==OLD_ROOT and completed['request_sha256']==spec['request']['sha256'] and
        mapping[completed['request_path']]['sha256']==spec['request']['sha256'] and
        mapping[completed['request_path']]['path']==spec['request']['path'] and
        request['source_bindings']==completed['source_bindings'] and request['input_bindings']==completed['input_bindings'],
        'Original request/root/source/input binding changed')
    required={**completed['source_bindings'],**completed['input_bindings'],**completed['outputs']}
    require(all(k in mapping and mapping[k]['sha256']==digest for k,digest in required.items()),
        'Every actual original source/input/output must be available with its original SHA')
    require(request['catalogue']['catalogue_sha256']==completed['catalogue_sha256']==spec['original_catalogue_sha256'] and
        request['backend_identity']==completed['backend_identity']==spec['original_backend_identity'] and
        len(request['catalogue']['profiles'])==144 and len(request['catalogue']['layouts'])==12 and
        request['header_ids']==list(range(144))+[4095] and request['packet_cap']==241 and
        request['body_decoder_calls']==96 and request['header_decoder_calls']==145,
        'Original full144 receive domain and241-case request required')
    ledger=spec['ledger'];require({k:mapping[completed['ledger_path']][k] for k in ('path','sha256','bytes')}==ledger,
        'Original closed ledger mapping required')
    ledger_path=io.inside(ledger['path'])
    db=sqlite3.connect(ledger_path.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        require(db.execute('SELECT hash,cap FROM config WHERE id=1').fetchone()==(completed['request_sha256'],241),
            'Original qualification ledger registration differs')
        rows=db.execute('SELECT event_id,event,request,result,status FROM events').fetchall()
        require(len(rows)==241 and all(row[4]=='COMPLETE' and row[3] is not None for row in rows),'Original241 actual callbacks incomplete')
        events={row[0]:tuple(json.loads(x) for x in row[1:4]) for row in rows}
        kinds=[event[0]['kind'] for event in events.values()]
        require(kinds.count('header')==145 and kinds.count('body')==96,'Original actual header/body counts differ')
    finally:db.close()
    used=set()
    for old in completed['outputs']:
        if '/body_cases/' in old:
            value=io.read(mapping[old]);eid=value['row']['event_id'];event,req,result=events[eid]
            require(event['kind']=='body' and value['rx']==result and value['row']['correct'] is True and
                req['received_sha256']==value['row']['received_sha256'] and req['catalogue_sha256']==completed['catalogue_sha256'],
                'Original actual body qualification output does not match paid ledger');used.add(eid)
        elif '/header_cases/' in old:
            value=io.read(mapping[old]);eid=value['event_id'];event,req,result=events[eid]
            require(event['kind']=='header' and all(value[k]==v for k,v in result.items()) and value['correct'] is True and
                req['received_sha256']==value['received_sha256'] and req['catalogue_sha256']==completed['catalogue_sha256'],
                'Original actual header qualification output does not match paid ledger');used.add(eid)
    require(used==set(events),'Every original241 paid decode must have matching actual output')
    require(io.sha(ledger_path)==ledger['sha256'],'Read-only qualification ledger changed')
    return spec,completed,request,mapping


def create_runtime(desc,root,phy,current_cpu_environment,configure_threads=True):
    require(io.sha(phy.__file__)==T1_SHA,'Unmodified original T1 implementation required')
    spec,done,request,mapping=inspect_binding(desc);root=io.inside(root)
    io.bytes_checked(current_cpu_environment)
    require(str(root)==spec['actual_project_root'],'Explicit relocated project root differs')
    current={mapping[old]['path']:digest for old,digest in done['source_bindings'].items()}
    require(phy.collect_source_bindings(root)==current,'Current full T1 source closure differs from exact relocated originals')
    reference=[mapping[p]['path'] for p,h in done['input_bindings'].items() if h==phy.LEGACY_QUALIFICATION_SHA]
    require(len(reference)==1,'Exactly one unchanged original backend qualification input required')
    runtime=phy.Runtime(root,reference[0],configure_threads=configure_threads)
    runtime.use_catalogue(request['catalogue'])
    require(runtime.backend.identity==done['backend_identity']==request['backend_identity'] and
        runtime.catalogue['catalogue_sha256']==done['catalogue_sha256'] and len(runtime.profiles)==144,
        'Actual current backend/constructors/full144 catalogue differ from original241 qualification')
    # The original factory performs the same final assignment only after these
    # checks. Here the absolute-path test is replaced solely by the explicit
    # independently checked old->current exact-byte mapping above.
    runtime.qualified=True
    runtime.qualification=dict(original_completion=spec['completion'],original_request=spec['request'],
        relocation=desc,original_packet_calls=241,new_qualification_calls=0,
        current_CPU_environment=current_cpu_environment,historical_qualification_preserved=True)
    return runtime
