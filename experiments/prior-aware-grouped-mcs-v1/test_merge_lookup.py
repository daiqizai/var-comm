"""On-disk synthetic receipts exercise the merge validator, never science data."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import merge_lookup as m
import phy_lookup as p
from uep_common import read,write,identity,sha


def fixture(folder,phase='coarse'):
    folder=Path(folder);profiles=[]
    for i in range(8):
        g=dict(phy_key='body'+str(i),source_bits=24,crc_bits=16,tail_bits=0,information_bits=40,
            transmitted_bits=128,modulation='QPSK',layout={})
        profiles.append(dict(N=1024,encoder_qualified=True,header_phy_key='header',groups=[g]))
    pp=folder/'profiles.json';write(pp,profiles);catalog=p.physical_catalog(profiles)
    points={(k,s) for k in catalog for s in p.SNRS};selection=folder/'selection.json'
    requests={('body0',7),('header',4)};write(selection,dict(synthetic=False,development_read=False,
        refinement_requirements=[dict(phy_key=k,snr_db=s) for k,s in sorted(requests)]))
    dirs=[]
    for i in range(8):
        out=folder/('shard'+str(i));dirs.append(out);owned=p.shard_points(catalog,points,i,8)
        reg=dict(version=p.VERSION,N=1024,snrs=list(p.SNRS),profiles_sha256=sha(pp),catalog_sha256=identity(catalog),
            shard_index=i,shard_count=8,synthetic=False,development_read=False,quality_read=False,
            owned_points=[list(x) for x in owned],full_catalog_point_count=len(points),batch_size=128,
            coarse_blocks=256,refine_max_blocks=20000,refine_block_errors=100,source_bindings={})
        write(out/'registration.json',reg);rows=[];bindings={}
        for k,s in owned:
            point=dict(phy_key=k,snr_db=s);batches=[]
            for start in (0,128):
                correct=np.arange(128)%2==0
                meta=dict(correct=correct,rejected=~correct,undetected=np.zeros(128,dtype=bool),actual_E=np.full(128,136. if k=='header' else 128.))
                batches.append(p.batch_receipt(meta,start,128))
            cp=dict(binding=identity(reg),point=point,batches=batches,counts=dict(n_blocks=256,n_correct=128,n_reject=128,n_undetected=0),synthetic=False)
            cp['payload_sha256']=identity(cp);path=out/'points'/identity(point)/'checkpoint.json';write(path,cp);write(path.with_name('coarse.json'),cp)
            if phase=='refine' and (k,s) in requests:write(path.with_name('refined.json'),cp)
            bindings[str(path)]=sha(path);rows.append(p.lookup_row(cp,catalog[k]))
        status='COARSE_COMPLETE' if phase=='coarse' else 'REFINEMENT_COMPLETE'
        lookup=dict(status=status,synthetic=False,registration_sha256=identity(reg),conditional_group_independence=True,rows=rows,checkpoint_bindings=bindings)
        write(out/'bler_lookup.json',lookup);phasepoints=[x for x in owned if phase=='coarse' or x in requests]
        done=dict(status=status,phase=phase,synthetic=False,registration_sha256=identity(reg),shard_index=i,shard_count=8,
            points=len(phasepoints),phase_points=[list(x) for x in phasepoints],owned_points=reg['owned_points'],full_catalog_point_count=len(points),lookup_sha256=sha(out/'bler_lookup.json'))
        name='coarse_completion.json' if phase=='coarse' else 'refine_'+sha(selection)+'_completion.json';write(out/name,done)
        if phase=='refine':write(out/'refinement_requests'/(sha(selection)+'.json'),dict(selection_path=str(selection),selection_sha256=sha(selection),points=done['phase_points']))
    return pp,dirs,selection


class MergeTests(unittest.TestCase):
    def test_exact_eight_shards_and_header_once(self):
        with tempfile.TemporaryDirectory() as folder:
            pp,dirs,_=fixture(folder);result=m.merge(pp,dirs[::-1],'coarse')
            self.assertEqual(len(result['rows']),54);self.assertEqual(sum(r['kind']=='header' for r in result['rows']),6)
            self.assertEqual(result['status'],'COARSE_COMPLETE')

    def test_refinement_accepts_empty_owner_subsets_but_full_catalog(self):
        with tempfile.TemporaryDirectory() as folder:
            pp,dirs,selection=fixture(folder,'refine');result=m.merge(pp,dirs,'refine',selection)
            self.assertEqual(result['status'],'REFINEMENT_COMPLETE');self.assertEqual(len(result['rows']),54)

    def test_duplicate_directories_missing_shard_and_lookup_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            pp,dirs,_=fixture(folder)
            with self.assertRaises(RuntimeError):m.merge(pp,dirs[:-1]+[dirs[0]],'coarse')
            with self.assertRaises(RuntimeError):m.merge(pp,dirs[:-1],'coarse')
            path=dirs[0]/'bler_lookup.json';value=read(path);value['rows'][0]['p_correct']=1.;write(path,value)
            with self.assertRaises(RuntimeError):m.merge(pp,dirs,'coarse')

    def test_exported_probability_cannot_disagree_even_with_resealed_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            pp,dirs,_=fixture(folder);path=dirs[0]/'bler_lookup.json';value=read(path);value['rows'][0]['p_correct']=1.;write(path,value)
            done=dirs[0]/'coarse_completion.json';receipt=read(done);receipt['lookup_sha256']=sha(path);write(done,receipt)
            with self.assertRaises(RuntimeError):m.merge(pp,dirs,'coarse')

    def test_phase_requires_its_exact_frozen_refinement_request(self):
        with tempfile.TemporaryDirectory() as folder:
            pp,dirs,selection=fixture(folder,'refine');value=read(selection);value['refinement_requirements'].append(dict(phy_key='body1',snr_db=7));write(selection,value)
            with self.assertRaises((RuntimeError,FileNotFoundError)):m.merge(pp,dirs,'refine',selection)


if __name__=='__main__':unittest.main()
