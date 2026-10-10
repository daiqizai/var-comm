"""Read known old metadata/JPEG bytes for a finite content-hash reference index.

Never opens newly registered100 files and never decodes any image/model. Coverage
is the five audited role manifests, not unrecorded use elsewhere.
"""
import argparse
from pathlib import Path
from t1_entropy_core import read,require,sha,verify
from t1_holdout_metadata import pin,pinned
from t6_register_confirmation_metadata import canonical_id
import t2_pilot as shared

def run(a):
    registration=read(a.registration);excluded=pinned(registration['excluded_manifest']);image_root=Path(registration['image_root']).resolve()
    require(registration['schema']=='WCL_N2048_CONFIRMATION100_METADATA_V1' and excluded['schema']=='WCL_KNOWN_PRIOR_SOURCE_EXCLUSIONS_V1','Original registered exposure audit required')
    supplied=dict(x.split('=',1) for x in a.manifest);require(set(supplied)==set(excluded['roles']),'Every audited role must be supplied')
    new_ids={r['canonical_source_id'] for r in registration['records']};used=set(excluded['canonical_source_ids'])
    require(not(new_ids&used),'Registered IDs overlap exclusions')
    content={};inputs={};role_counts={};jpeg_reads=0
    for role,path in supplied.items():
        verify(path,excluded['roles'][role]['manifest']['sha256']);inputs[str(Path(path).resolve())]=sha(path);value=read(path)
        rows=value if isinstance(value,list) else value.get('records',value.get('entries',[]));require(rows,'Audited manifest has no records')
        role_counts[role]=len(rows)
        for row in rows:
            sid=row.get('source_id',row.get('image_id'));require(isinstance(sid,str),'Reference source ID absent')
            cid=canonical_id(sid);require(cid in used and cid not in new_ids,'Only predeclared old sources can be opened')
            r=content.setdefault(cid,dict(source_id=sid,canonical_source_id=cid,raw_JPEG_sha256=[],preprocessing_sha256=[],roles=[]))
            if role not in r['roles']:r['roles'].append(role)
            metadata=[row]
            for depth in range(3):
                current=metadata[depth] if depth<len(metadata) else None
                if current is None:break
                cp=current.get('checkpoint');digest=current.get('checkpoint_sha256',current.get('sha256'))
                ref=current.get('source_assets_checkpoint')
                if isinstance(ref,dict):cp,digest=ref['path'],ref['sha256']
                if isinstance(cp,str) and isinstance(digest,str) and Path(cp).is_file():
                    verify(cp,digest);inputs[cp]=digest;metadata.append(read(cp))
            for d in metadata:
                if isinstance(d.get('preprocessing_id'),str):r['preprocessing_sha256'].append(d['preprocessing_id'])
                if isinstance(d.get('image_file_sha256'),str):r['raw_JPEG_sha256'].append(d['image_file_sha256'])
            if role=='train20k':
                require('pixel_sha256' in row and 'flip_pixel_sha256' in row,'Original training CHW pixel/flip hashes required')
                r['preprocessing_sha256']+= [row['pixel_sha256'],row['flip_pixel_sha256']]
            if cid.startswith('imagenet-val:') and not r['raw_JPEG_sha256']:
                path=Path(row.get('image_path',row.get('path',str(image_root/(sid+'.JPEG')))))
                if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(image_root):
                    digest=sha(path);r['raw_JPEG_sha256'].append(digest);inputs[str(path.resolve())]=digest;jpeg_reads+=1
            for name in ('raw_JPEG_sha256','preprocessing_sha256'):
                r[name]=sorted(set(r[name]));require(all(len(x)==64 for x in r[name]),'Malformed source content hash')
    require(set(content)==used,'Content inventory must represent every excluded source, including missing hashes')
    rows=list(content.values());available=sum(bool(r['raw_JPEG_sha256'] or r['preprocessing_sha256']) for r in rows)
    result=dict(status='T6_AUDITED_OLD_CONTENT_HASH_INVENTORY_COMPLETE',scope='all_available_audited_source_content_hashes',
        confirmation_registration=pin(a.registration),excluded_manifest=registration['excluded_manifest'],source_count=len(rows),records=rows,
        available_reference_hash_count=available,unavailable_reference_hash_count=len(rows)-available,
        source_without_raw_JPEG_hash=sum(not r['raw_JPEG_sha256'] for r in rows),
        source_without_preprocessing_hash=sum(not r['preprocessing_sha256'] for r in rows),
        comparable_hash_domains=['original_JPEG_bytes','uint8_CHW_256_pixels_and_registered_horizontal_flips'],
        role_counts=role_counts,old_JPEG_files_hashed=jpeg_reads,new_confirmation_files_opened=0,
        image_decodes=0,new_model_calls=0,new_packet_decodes=0,input_bindings=inputs,
        historical_exclusions_complete_beyond_supplied_manifests=False)
    shared.save(a.out,result);return {k:result[k] for k in ('status','source_count','available_reference_hash_count','unavailable_reference_hash_count','source_without_raw_JPEG_hash','source_without_preprocessing_hash','old_JPEG_files_hashed')}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',required=True);p.add_argument('--manifest',action='append',required=True,help='role=remote_manifest_path, all five registered roles');p.add_argument('--out',required=True)
    print(run(p.parse_args()))
