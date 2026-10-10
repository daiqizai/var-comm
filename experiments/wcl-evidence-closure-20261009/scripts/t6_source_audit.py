"""Metadata-only exclusion audit; never chooses images or reads image pixels."""
import argparse,csv,json,re
from pathlib import Path
import t2_pilot as h

REQUIRED={'train20k':20000,'calibration1000':1000,'development100':100,'holdout500':500}

def source_key(value):
    value=str(value).replace('\\','/');match=re.search(r'ILSVRC2012_val_(\d{8})(?:_|\.|/|$)',value)
    if match:return 'imagenet-val:'+match.group(1)
    return value

def records(value):
    if isinstance(value,list):rows=value
    elif isinstance(value,dict):
        field=next((key for key in('records','entries','source_ids')if isinstance(value.get(key),list)),None)
        h.require(field is not None,'Explicit source list required; no recursive guessed identifiers');rows=value[field]
    else:raise ValueError('Unsupported source manifest')
    result=[]
    for row in rows:
        if isinstance(row,str):identifier=row;path=None
        else:
            identifier=row.get('source_id',row.get('image_id'));path=row.get('path')
            h.require(isinstance(identifier,str) and identifier,'Explicit source/image identity required')
        result.append(dict(source_id=identifier,canonical_source_id=source_key(identifier),path=path))
    h.require(len({r['canonical_source_id']for r in result})==len(result),'Source manifest has duplicate/aliased IDs')
    return result

def audit(exclusions,pool,out):
    roles={};pins={};used={}
    for role,path in exclusions:
        h.require(role in REQUIRED or role.startswith('history'),'Unknown exclusion role')
        h.require(role not in roles,'Duplicate exclusion role')
        rows=records(h.read(path));pins[str(Path(path).resolve())]=h.sha(path)
        if role in REQUIRED:h.require(len(rows)==REQUIRED[role],'Required source count differs: '+role)
        roles[role]=dict(count=len(rows),manifest=h.desc(path))
        for row in rows:used.setdefault(row['canonical_source_id'],set()).add(role)
    h.require(set(REQUIRED)<=set(roles) and any(k.startswith('history')for k in roles),'All original data roles and known historical manifests required')
    output=Path(out);output.mkdir(parents=True,exist_ok=True);rows=[]
    if pool is not None:
        poolrows=records(h.read(pool));pins[str(Path(pool).resolve())]=h.sha(pool)
        for row in poolrows:
            excluded=sorted(used.get(row['canonical_source_id'],[]))
            rows.append(dict(**row,excluded_by=';'.join(excluded),outside_known_exclusions=not bool(excluded)))
    remaining=sum(row['outside_known_exclusions']for row in rows)
    status='BLOCKED_NO_DOCUMENTED_INDEPENDENT_POOL'if pool is None else('BLOCKED_NO_UNSEEN_POOL_SOURCES'if remaining<100 else'POOL_CANDIDATES_REQUIRE_HISTORY_AND_SOURCE_AVAILABILITY_CLOSURE')
    result=dict(schema='WCL_T6_CONFIRMATION_SOURCE_AUDIT_V1',status=status,known_exclusion_roles=roles,known_excluded_unique_ids=len(used),
        candidate_pool=None if pool is None else h.desc(pool),pool_source_count=len(rows),outside_known_exclusion_count=remaining,
        source_alias_rule='ImageNet val image number canonicalizes class/path/extension aliases; other exact identifiers preserved',
        global_history_inventory_exhaustive=False,raw_source_availability_verified=False,content_duplicates_across_manifests_checked=False,
        source_selection_made=False,confirmation_source_ids=[],entropy_family_selected=False,new_image_reads=0,new_scientific_calls=0,input_bindings=pins,
        limitation='Known manifests establish exclusion only. No independent source pool or complete all-history source-use closure is inferred. Do not label existing seen sources as new blind confirmation.')
    h.save(output/'source_audit.json',result)
    if rows:
        path=output/'pool_exclusions.csv'
        with path.open('x',newline='',encoding='utf-8')as stream:
            writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {k:v for k,v in result.items()if k not in('input_bindings','known_exclusion_roles')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--exclude',action='append',required=True,help='role=manifest.json; original four roles plus history labels')
    p.add_argument('--pool');p.add_argument('--out',required=True);a=p.parse_args()
    pairs=[]
    for item in a.exclude:
        role,sep,path=item.partition('=');h.require(sep and path,'Use role=path');pairs.append((role,path))
    print(h.canonical(audit(pairs,a.pool,a.out)))
if __name__=='__main__':main()
