"""Lossless expansion of deduplicated registration metadata; no science calls."""
import hashlib,json,pathlib

def expand(compact):
    result=dict(compact['registration_skeleton']);rows=[]
    for sid,i,j,r,l in compact['selected_sources_compact']:
        rows.append(dict(split='train',source_id=sid,source_index=i,within_shard=j,rgb_shard=compact['shard_descriptors'][r],latent_shard=compact['shard_descriptors'][l]))
    result['selected_sources']=rows
    data=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
    assert len(data)==compact['original']['bytes'] and hashlib.sha256(data).hexdigest()==compact['original']['sha256']
    return data

def main():
    root=pathlib.Path(__file__).resolve().parents[2]
    directory=root/'results/progressive_rd_20261022_v1';c=json.loads((directory/'expansion_registration_compact.json').read_text());data=expand(c);target=directory/'expansion_registration.json'
    if target.exists():assert target.read_bytes()==data
    else:
        with target.open('xb')as f:f.write(data)
    print(json.dumps(dict(status='EXACT_METADATA_RESTORED_OR_ALREADY_PRESENT',sha256=c['original']['sha256'],scientific_calls=0)))

if __name__=='__main__':main()
