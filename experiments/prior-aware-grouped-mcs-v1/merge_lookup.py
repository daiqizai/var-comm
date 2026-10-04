"""Verify all eight disjoint physical lookup shards before CPU optimization."""
from __future__ import annotations
import argparse
from pathlib import Path
from uep_common import require,read,sha,identity,seal
import phy_lookup as p


def merge(profiles_path,shard_dirs,phase,selection_path=None):
    require(phase in ('coarse','refine'),'Unknown lookup phase')
    folders=[Path(x).resolve() for x in shard_dirs]
    require(len(folders)==len(set(folders))==8,'Exactly eight distinct canonical shard directories required')
    profiles_path=Path(profiles_path).resolve();profiles=read(profiles_path);catalog=p.physical_catalog(profiles)
    points={(key,snr) for key in catalog for snr in p.SNRS};bindings={str(profiles_path):sha(profiles_path)}
    requested=points
    if phase=='refine':
        require(selection_path is not None,'Frozen refinement selection required')
        selection_path=Path(selection_path).resolve();selected=read(selection_path)
        require(selected.get('synthetic') is False and selected.get('development_read') is False,'Actual calibration refinement selection required')
        requested={(r['phy_key'],int(r['snr_db'])) for r in selected['refinement_requirements']}
        require(requested and requested<=points,'Refinement includes unknown physical configurations')
        bindings[str(selection_path)]=sha(selection_path)
    status='COARSE_COMPLETE' if phase=='coarse' else 'REFINEMENT_COMPLETE'
    found={};rows=[];common=None;registrations=[]
    for folder in folders:
        require(not (folder/'failure.json').exists(),'A shard has an unresolved failure')
        regpath=folder/'registration.json';reg=read(regpath);bindings[str(regpath)]=sha(regpath)
        index=reg['shard_index'];require(type(index)is int and 0<=index<8 and index not in found,'Duplicate or invalid shard index')
        found[index]=str(folder);require(reg['shard_count']==8 and reg['N']==1024 and reg['synthetic'] is False
            and reg['development_read'] is False and reg['quality_read'] is False,'Shard scope or scientific identity differs')
        require(reg['profiles_sha256']==sha(profiles_path) and reg['catalog_sha256']==identity(catalog)
            and reg['snrs']==list(p.SNRS) and reg['full_catalog_point_count']==len(points),'Shard catalog differs')
        require(reg['coarse_blocks']==256 and reg['refine_max_blocks']==20000 and reg['refine_block_errors']==100
            and reg['batch_size'] in (32,64,128),'Registered Monte Carlo sample rules differ')
        shared={k:v for k,v in reg.items() if k not in ('shard_index','owned_points')}
        if common is None:common=shared
        require(shared==common,'Shard backend, batch qualification, counters or source bindings differ')
        for path,digest in reg['source_bindings'].items():require(sha(path)==digest,'Bound lookup source changed');bindings[path]=digest
        owned=set(p.shard_points(catalog,points,index,8));phase_points=owned & requested
        require(reg['owned_points']==[list(x) for x in sorted(owned)],'Shard ownership differs from stable assignment')
        name='coarse_completion.json' if phase=='coarse' else 'refine_'+sha(selection_path)+'_completion.json'
        donepath=folder/name;done=read(donepath);bindings[str(donepath)]=sha(donepath)
        require(done['status']==status and done['phase']==phase and done['synthetic'] is False
            and done['registration_sha256']==identity(reg) and done['shard_index']==index and done['shard_count']==8,
            'Missing or mismatched shard phase completion')
        require(done['points']==len(phase_points) and done['phase_points']==[list(x) for x in sorted(phase_points)]
            and done['owned_points']==reg['owned_points'] and done['full_catalog_point_count']==len(points),'Shard phase coverage differs')
        if phase=='refine':
            requestpath=folder/'refinement_requests'/(sha(selection_path)+'.json');request=read(requestpath)
            require(request['selection_sha256']==sha(selection_path) and Path(request['selection_path']).resolve()==selection_path
                and request['points']==done['phase_points'],'Refinement request changed')
            bindings[str(requestpath)]=sha(requestpath)
        bundlepath=folder/'bler_lookup.json';bundle=read(bundlepath);bindings[str(bundlepath)]=sha(bundlepath)
        require(sha(bundlepath)==done['lookup_sha256'] and bundle['status']==status and bundle['synthetic'] is False
            and bundle['registration_sha256']==identity(reg) and bundle['conditional_group_independence'] is True,
            'Shard lookup changed after phase completion')
        observed={(r['phy_key'],int(r['snr_db'])) for r in bundle['rows']}
        require(len(observed)==len(bundle['rows']) and observed==owned,'Duplicate, missing or foreign shard rows')
        expected_cp={str(folder/'points'/identity(dict(phy_key=k,snr_db=s))/'checkpoint.json') for k,s in owned}
        require(set(bundle['checkpoint_bindings'])==expected_cp,'Shard checkpoint directory coverage differs')
        for row in bundle['rows']:
            key,snr=row['phy_key'],int(row['snr_db']);point=dict(phy_key=key,snr_db=snr)
            cp_path=folder/'points'/identity(point)/'checkpoint.json';cp=p.verify_checkpoint(read(cp_path),identity(reg),point)
            require(cp.get('synthetic') is False and sha(cp_path)==bundle['checkpoint_bindings'][str(cp_path)],'Nonphysical or changed point checkpoint')
            require(all(0<b['n_blocks']<=reg['batch_size'] for b in cp['batches']),'Unregistered Monte Carlo batch size')
            require(row==p.lookup_row(cp,catalog[key]),'Exported probability/count/interval differs from actual block receipts')
            snapname='refined.json' if phase=='refine' and (key,snr) in requested else 'coarse.json'
            snapshot=cp_path.with_name(snapname);saved=p.verify_checkpoint(read(snapshot),identity(reg),point)
            require(saved.get('synthetic') is False and saved['batches']==cp['batches'][:len(saved['batches'])], 'Immutable phase snapshot differs from sampled block prefix')
            n=saved['counts']['n_blocks'];errors=n-saved['counts']['n_correct']
            if snapname=='coarse.json':require(n==256,'Coarse snapshot must contain exactly256 independent blocks')
            else:require(n>=256 and (errors>=100 or n==20000) and saved==cp,'Refinement stopping rule incomplete')
            bindings[str(cp_path)]=sha(cp_path);bindings[str(snapshot)]=sha(snapshot);rows.append(row)
        registrations.append(dict(shard_index=index,directory=str(folder),registration_sha256=identity(reg),completion_sha256=sha(donepath)))
    require(set(found)==set(range(8)) and {(r['phy_key'],r['snr_db']) for r in rows}==points,'Incomplete eight-shard catalog coverage')
    registration=dict(version='UEP-EIGHT-SHARD-MERGE-V1',phase=phase,shards=sorted(registrations,key=lambda x:x['shard_index']),
        input_bindings=bindings,source_bindings={str(Path(__file__).resolve()):sha(__file__)},
        full_catalog_point_count=len(points),requested_refinement_points=[list(x) for x in sorted(requested)] if phase=='refine' else [],
        actual_bit_simulation=True,synthetic=False,development_read=False)
    return dict(version=p.VERSION,status=status,synthetic=False,conditional_group_independence=True,
        source_probability_mode='random raw payload and public coding scrambler; image payload applicability not yet verified',
        real_image_payload_probability_validation_complete=False,registration_sha256=identity(registration),
        merge_registration=registration,rows=sorted(rows,key=lambda r:(r['phy_key'],r['snr_db'])),checkpoint_bindings=bindings)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profiles',required=True)
    parser.add_argument('--shards',nargs=8,required=True);parser.add_argument('--phase',choices=('coarse','refine'),required=True)
    parser.add_argument('--selection');parser.add_argument('--output',required=True);args=parser.parse_args()
    result=merge(args.profiles,args.shards,args.phase,args.selection);seal(args.output,result)
    print(result['status'],len(result['rows']),'independently simulated physical/SNR points')


if __name__=='__main__':main()
