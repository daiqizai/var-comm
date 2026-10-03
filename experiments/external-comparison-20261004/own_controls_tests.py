"""CPU contracts; fixtures never represent scientific measurements."""
import ast
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import types
import unittest
import sys
import numpy as np
import own_controls_common as c
import own_controls as run


HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('OWN_CONTROLS_TEST_ROOT',str(HERE.parents[1]/'historical_eval_20261003/source'))).resolve()
OLD = SOURCE/'experiments/scale-causal-partial-residual-20261002'
sys.path.insert(0,str(SOURCE/'src'))
import own_controls_phy as phy


def native_rank():
    module = ast.parse((OLD/'m1_runner.py').read_text())
    fn = next(x for x in module.body if isinstance(x, ast.FunctionDef) and x.name == 'rank')
    ns = {}; exec(compile(ast.Module(body=[fn], type_ignores=[]), str(OLD/'m1_runner.py'), 'exec'), ns)
    return ns['rank']


def population(count):
    rows = [dict(image_id=f'source{i}', preprocessing_id=f'pre{i}', class_index=i % 1000) for i in range(count)]
    data = dict(records=rows, bindings={'cache':'sha'})
    original = dict(source_ids=[r['image_id'] for r in rows], preprocessing_ids=[r['preprocessing_id'] for r in rows],
                    data_bindings=data['bindings'])
    return data, original


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_exact_and_no_new_search_or_holdout(self):
        protocol = c.read(HERE/'own_controls_protocol.json')
        self.assertEqual(c.validate_protocol(protocol), protocol)
        for key, value in [('snrs',[1,4,7,13]), ('calibration_seeds',[4101]), ('whole_grid',[4,5,6,7,8,9]),
                           ('holdout_access',True), ('development_selection',True), ('decoder','D0')]:
            altered = copy.deepcopy(protocol); altered[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): c.validate_protocol(altered)

    def test_derived_phy_changes_only_registered_four_literals(self):
        raw = (OLD/'partial_phy.py').read_bytes()
        self.assertEqual(c.derived_phy_bytes(raw), (HERE/'own_controls_phy.py').read_bytes())
        with self.assertRaises(RuntimeError): c.derived_phy_bytes(raw+b'\n')

    def test_original_population_order_cache_and_count_are_bound(self):
        for role, count in [('calibration',1000), ('development',100)]:
            data, original = population(count)
            self.assertEqual(len(c.data_proof(data, original, role)['source_ids']),count)
            for corrupt in ('order','binding','count'):
                bad = copy.deepcopy(data)
                if corrupt == 'order': bad['records'][0],bad['records'][1] = bad['records'][1],bad['records'][0]
                if corrupt == 'binding': bad['bindings']['cache'] = 'other'
                if corrupt == 'count': bad['records'].pop()
                with self.subTest(role=role,corrupt=corrupt), self.assertRaises(RuntimeError): c.data_proof(bad, original, role)

    def test_grid_counts_and_calibration_order(self):
        grid = phy.action_grid(2048,'QPSK',orders=('entropy',))
        self.assertEqual((len(grid),sum(a.q==0 for a in grid)),(21,5))
        choices = {s:grid for s in c.SNRS}; source = population(1000)[0]['records'][0]
        rows = [dict(source_index=0,source_id=source['image_id'],preprocessing_id=source['preprocessing_id'],
                     N=2048,E=4096,phy_family='QPSK',snr_db=s,noise_seed=seed,action_id=c.action_id(a),
                     psnr_db=20.,lpips_alex=.2,dino_cosine=.8)
                for s,acts in choices.items() for a in acts for seed in (4101,)]
        self.assertEqual(len(rows)*200,12600); c.validate_rows(rows,0,source,choices,(4101,))
        for field, value in [('noise_seed',2001),('E',2048),('source_id','different'),('psnr_db',float('nan'))]:
            bad = copy.deepcopy(rows);bad[0][field]=value
            with self.subTest(field=field), self.assertRaises(RuntimeError): c.validate_rows(bad,0,source,choices,(4101,))
        with self.assertRaises(RuntimeError): c.validate_rows(rows[::-1],0,source,choices,(4101,))

    def test_choice_is_exact_native_rank_and_single_feasible_is_legal(self):
        rank = native_rank(); runner = run.Runner.__new__(run.Runner)
        runner.native = types.SimpleNamespace(m1=types.SimpleNamespace(rank=rank))
        grid = phy.action_grid(2048,'QPSK',orders=('entropy',)); mapping = {c.action_id(a):a for a in grid}
        summary = []
        for snr in c.SNRS:
            for j, a in enumerate(grid):
                summary.append(dict(snr_db=snr,action_id=c.action_id(a),m=a.m,q=a.q,order=a.order,
                    failure_fraction=0. if a.m==8 else .1,psnr_db=25. if a.m==8 else 20.,lpips_alex=.8-j/100))
        selected = runner.choose(summary,True,mapping)
        self.assertEqual(len(selected),6)
        self.assertTrue(all(len(x['action_ids'])==1 for x in selected))
        policy = runner.choose(summary,False,mapping,{'cells':selected})
        self.assertTrue(all(x['action']['m']==8 and x['action']['q']==0 for x in policy))
        # Feasible low LPIPS wins even if DINO happens to favor another candidate.
        group = [dict(action_id='a',m=4,q=0,failure_fraction=0.,psnr_db=20.,lpips_alex=.4,dino_cosine=.99),
                 dict(action_id='b',m=4,q=6,failure_fraction=.1,psnr_db=19.8,lpips_alex=.3,dino_cosine=.5)]
        self.assertEqual(rank(group)[0][0]['action_id'],'b')

    def test_source_mean_accumulation_matches_old_calibration_order(self):
        a=phy.action_grid(2048,'QPSK',orders=('entropy',))[0]
        rows=[dict(N=2048,phy_family='QPSK',snr_db=1,action_id=c.action_id(a),m=a.m,q=a.q,order=a.order,
                   E=4096,psnr_db=20.+j,lpips_alex=.2+j/10,dino_cosine=.7,header_ok=True,body_crc_ok=j!=1)
              for j in range(3)]
        r=c.summarize(rows)[0]
        self.assertEqual(r['rows'],3);self.assertEqual(r['psnr_db'],21.)
        self.assertEqual(r['failure_fraction'],1/3)


class CacheTests(unittest.TestCase):
    def test_checkpoint_reseal_cannot_change_bound_registration(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'cp.json'; v=c.sealed_checkpoint(p,dict(binding='registered',source_index=4,rows=[]))
            self.assertEqual(c.checkpoint(p,'registered',4),v)
            bad=dict(v,binding='other');bad['payload_sha256']=c.identity({k:v for k,v in bad.items() if k!='payload_sha256'})
            c.write(p,bad)
            with self.assertRaises(RuntimeError): c.checkpoint(p,'registered',4)

    def test_lossless_distinct_reference_targets_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            source=np.full((3,256,256),.5,np.float32); target=source.copy()
            target[0,0,0]=np.nextafter(target[0,0,0],np.float32(1))
            image=np.full_like(source,.25);path=Path(folder)/'cache.npz'
            receipt=c.save_float_cache(path,[image],source,target)
            rows=[dict(image_slot=0,image_sha256=c.rgb_sha(image),reference_sha256=c.rgb_sha(target))]
            c.verify_float_cache(receipt,rows,source,target)
            self.assertNotEqual(receipt['native_reference_sha256'],receipt['comparison_reference_sha256'])
            self.assertEqual(c.save_float_cache(path,[image],source,target),receipt)
            with self.assertRaises(RuntimeError): c.save_float_cache(path,[source],source,target)
            with self.assertRaises(RuntimeError): c.verify_float_cache(receipt,rows,source,source)
            for slot in (-1,True,1):
                with self.subTest(slot=slot),self.assertRaises(RuntimeError):
                    c.verify_float_cache(receipt,[dict(rows[0],image_slot=slot)])
            self.assertEqual(c.rgb_sha(source),hashlib.sha256(b'float32:3,256,256:RGB\0'+source.tobytes()).hexdigest())

    def test_resealed_pixels_still_rejected_against_scientific_row(self):
        with tempfile.TemporaryDirectory() as folder:
            image=np.full((3,256,256),.2,np.float32);source=np.full_like(image,.5);path=Path(folder)/'cache.npz'
            receipt=c.save_float_cache(path,[image],source,source)
            rows=[dict(image_slot=0,image_sha256=c.rgb_sha(image),reference_sha256=c.rgb_sha(source))]
            altered=image.copy();altered[0,0,0]=.1
            c.atomic_npz(path,images=altered[None],native_reference=source,comparison_reference=source)
            receipt.update(sha256=c.sha(path),image_sha256=[c.rgb_sha(altered)])
            with self.assertRaises(RuntimeError): c.verify_float_cache(receipt,rows)


class ProvenanceTests(unittest.TestCase):
    def build(self, root):
        out=root/'outputs/UNIFIED-METRICS-20261002'; result=root/'results/unified_metrics_20261002'
        reg=result/'metrics_registration.json'; c.write(reg,{'ENGINEERING_FIXTURE':True})
        data,_=population(100); cp={}
        for i, source in enumerate(data['records']):
            p=out/'source_checkpoints'/f'{i:03d}.json'
            c.sealed_checkpoint(p,dict(binding=c.identity(c.read(reg)),source_index=i,rows=[{'replay_row_id':'id'}],
                baseline=dict(source_index=i,source_id=source['image_id'],preprocessing_id=source['preprocessing_id'],true_class_index=source['class_index'])))
            cp[str(p)]=c.sha(p)
        inventory=result/'scoring_inventory.json'
        c.write(inventory,dict(sources=100,parity_passed=True,source_checkpoint_sha256=cp))
        score=out/'scoring_completion.json';copy_path=result/'provenance/scoring_completion.json'
        value=dict(status='COMPLETE',synthetic=False,parity_passed=True,training_updates=0,policy_selection_updates=0,
                   outputs={str(p):c.sha(p) for p in (reg,inventory)})
        c.write(score,value);c.write(copy_path,value)
        c.write(out/'completion.json',dict(status='UNIFIED_METRICS_COMPLETE',publication=dict(
            status='PUSHED',checks='PASS',commit='a'*40,remote_commit='a'*40,published_files={str(copy_path):c.sha(copy_path)})))
        return out,result,data

    def test_published_inventory_binds_every_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out,result,data=self.build(root); inv=c.old_source_inventory(root)
            self.assertEqual(len(inv['checkpoints']),100)
            path=out/'source_checkpoints/000.json'
            c.original_checkpoint(path,inv,0,data['records'][0])
            altered=c.read(path);altered['rows'][0]['replay_row_id']='edited'
            altered['payload_sha256']=c.identity({k:v for k,v in altered.items() if k!='payload_sha256'});c.write(path,altered)
            with self.assertRaises(RuntimeError): c.original_checkpoint(path,inv,0,data['records'][0])

    def test_publication_and_registration_cannot_be_replaced(self):
        for target in ('scoring_inventory.json','metrics_registration.json','publication'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);out,result,_=self.build(root)
                if target=='publication':
                    value=c.read(out/'completion.json');value['publication']['remote_commit']='b'*40;c.write(out/'completion.json',value)
                else:c.write(result/target,{'replaced':True})
                with self.assertRaises(RuntimeError):c.old_source_inventory(root)

    def test_original_checkpoint_source_class_and_preprocessing_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out,_,data=self.build(root);inv=c.old_source_inventory(root)
            for key in ('image_id','preprocessing_id','class_index'):
                altered=dict(data['records'][0]);altered[key]='wrong' if key!='class_index' else 333
                with self.subTest(key=key),self.assertRaises(RuntimeError):
                    c.original_checkpoint(out/'source_checkpoints/000.json',inv,0,altered)

    def test_development_frozen_action_and_no_label_side_channel(self):
        source=population(100)[0]['records'][0]
        grid=phy.action_grid(2048,'QPSK',orders=('entropy',));cells={}
        rows=[]
        for snr in c.SNRS:
            for method,a in zip(c.METHODS,(grid[0],grid[-1])):cells[snr,method]={'action':c.action_dict(a)}
            for seed in c.DEV_SEEDS:
                for method in c.METHODS:
                    action=cells[snr,method]['action']
                    rows.append(dict(**{k:action[k] for k in ('m','q','order')},N=2048,E=4096,phy_family='QPSK',
                        source_index=0,source_id=source['image_id'],preprocessing_id=source['preprocessing_id'],source_class_index=0,
                        snr_db=snr,noise_seed=seed,method=method,policy_sha256='policy',policy_selected_on_development=False,
                        semantic_side_information=False,receiver_class_embedding=1000))
        c.validate_development_rows(rows,0,source,cells,'policy')
        self.assertEqual(len(rows)*100,1800)
        for key,value in [('q',999),('semantic_side_information',True),('receiver_class_embedding',0),('policy_sha256','other')]:
            altered=copy.deepcopy(rows);altered[0][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):c.validate_development_rows(altered,0,source,cells,'policy')

    def test_n1024_saved_export_preserves_exact_old_row_identity(self):
        source=population(100)[0]['records'][0];originals={};measured={};rows=[]
        for method in ('P1024','D_U_QPSK','entropy_policy'):
            for snr in c.SNRS:
                for seed in c.DEV_SEEDS:
                    old=dict(method=method,snr_db=str(snr),noise_seed=str(seed),psnr_db='24.0')
                    rid=c.identity(old);originals[rid]=old
                    measured[rid]=dict(image_sha256='image'+rid,reference_sha256='originaltarget')
                    rows.append(dict(original_replay_row_id=rid,original_scientific_row=old,
                        original_scientific_row_sha256=c.identity(old),
                        original_completed_metrics=measured[rid],source_index=0,source_id=source['image_id'],
                        source_class_index=0,N=1024,method=method if method!='entropy_policy' else 'M1_entropy_N1024_frozen',
                        snr_db=snr,noise_seed=seed,image_sha256='image'+rid,native_reference_sha256='originaltarget',
                        original_scalar_parity=dict(status='PASS',missing_fields=[]),policy_selection_updates=0))
        c.validate_export_rows(rows,0,source,originals,measured)
        self.assertEqual(len(rows)*100,2700)
        bad=copy.deepcopy(rows);bad[0]['original_completed_metrics']['image_sha256']='replacement'
        with self.assertRaises(RuntimeError):c.validate_export_rows(bad,0,source,originals,measured)
        bad=copy.deepcopy(rows);bad[0]['original_scalar_parity']['missing_fields']=['lpips_alex']
        with self.assertRaises(RuntimeError):c.validate_export_rows(bad,0,source,originals,measured)
        with self.assertRaises(RuntimeError):c.validate_export_rows(rows[1:]+[rows[1]],0,source,originals,measured)

    def test_published_original_population_copy_is_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=c.locations(root);name='m1_calibration_registration.json'
            original=p['original_out']/name;published_copy=p['original_result']/'provenance'/name
            fixture={'ENGINEERING_FIXTURE':True};c.write(original,fixture);c.write(published_copy,fixture)
            pub=p['original_result']/'provenance/m1_publication.json'
            c.write(pub,dict(status='PUSHED',checks='PASS',commit='a'*40,remote_commit='a'*40,
                            published_files={str(published_copy):c.sha(published_copy)}))
            self.assertEqual(c.original_registration(root,'calibration')[0],fixture)
            c.write(original,{'ENGINEERING_FIXTURE':'changed'})
            with self.assertRaises(RuntimeError):c.original_registration(root,'calibration')


if __name__=='__main__':unittest.main()
