"""Synthetic CPU fixtures only; no scientific outputs or model imports."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import materialize_gate_reuse as m


def locate(name):
    relatives={'m2_runner.py':['.research/m2_runner_remote.py','experiments/scale-causal-partial-residual-20261002/m2_runner.py'],
        'common.py':['.research/methods_20261002/source/common.py','experiments/scale-causal-partial-residual-20261002/common.py'],
        'residual_receiver.py':['.research/methods_20261002/source/residual_receiver.py','experiments/scale-causal-partial-residual-20261002/residual_receiver.py']}
    for root in (Path.cwd(),*Path(__file__).resolve().parents):
        for relative in relatives[name]:
            path=root/relative
            if path.is_file():return path
    raise FileNotFoundError(name)


def functions():return m.frozen_functions(*(locate(name) for name in ('m2_runner.py','common.py','residual_receiver.py')))


def rows_for(index,f):
    rows=[]
    for g,ch in f['projections']:
        for seed in f['seeds']:
            choices=[('UNGUIDED',0.),('LIKELIHOOD',0.),('DIRECT',0.)]
            choices += [(control,lam) for control in ('VAR_GUIDED','STATIC') for lam in f['lambdas']]
            for control,lam in choices:
                guided=control in ('VAR_GUIDED','STATIC')
                rows.append(dict(source_index=index,source_id=f'source-{index}',preprocessing_id='frozen-pixels',
                    projection=f'g{g}_c{ch}',noise_seed=seed,control=control,**{'lambda':lam},snr_db=7,
                    stage='calibration_screen_reduced_noisy',psnr_db=21. if guided else 20.,
                    lpips_alex=.20+abs(lam-.5)/100 if guided else .3,dino_cosine=.795 if guided else .8,
                    measurement_y_sha256=f'observation-{index}-{g}-{ch}-{seed}',decoder_applied=True))
    return rows


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)


def fixture(root):
    root=Path(root);base=root/'experiments/scale-causal-partial-residual-20261002';base.mkdir(parents=True)
    for name in ('m2_runner.py','common.py','residual_receiver.py'):(base/name).write_bytes(locate(name).read_bytes())
    f=m.frozen_functions(*(base/name for name in ('m2_runner.py','common.py','residual_receiver.py')))
    registration=dict(stage='m2_calibration',training_updates=0,source_ids=[f'source-{i}' for i in range(1000)],
        preprocessing_ids=['frozen-pixels']*1000,source_bindings={str(p):m.sha(p) for p in base.iterdir()})
    parent=root/'outputs'/m.RUN;reg=parent/'m2_calibration_registration.json';m.atomic_preserve(reg,registration)
    calibration=parent/'m2/calibration';screen=[]
    for index in range(200):
        rows=rows_for(index,f);screen.extend(rows)
        m.atomic_preserve(calibration/'screen'/f'source_{index:04d}.json',dict(registration_sha256=m.sha(reg),rows=rows))
    csv_write(calibration/'screen_per_frame.csv',screen)
    ledger=[f['resource'](N,g,ch,phy) for g,ch in f['projections'] for N in (512,1024) for phy in ('QPSK','16QAM')]
    csv_write(root/'results/scale_causal_partial_residual_20261002/m2_resource_ledger.csv',ledger)
    return calibration,registration


class GateReuseTests(unittest.TestCase):
    def test_actual_ast_pair_order_rows_and_bool_only_policy_proof(self):
        f=functions();f['count']=3;screen=sum([rows_for(i,f) for i in range(3)],[])
        untouched=copy.deepcopy(screen);names={'g4_c8','g4_c16'}
        original=f['choose'](screen,'VAR_GUIDED')
        with self.assertRaises(TypeError):json.dumps(original)
        outputs,selection=m.derive(screen,f,names)
        self.assertEqual(screen,untouched);self.assertEqual(selection['var'],original)
        json.dumps(selection,allow_nan=False)
        for index,rows in outputs.items():
            self.assertEqual(len(rows),12)
            self.assertEqual([x['control'] for x in rows],['UNGUIDED','VAR_GUIDED']*6)
            self.assertTrue(all(r['stage']==m.STAGE for r in rows))
            for row in rows:
                self.assertIn(dict(row,stage='calibration_screen_reduced_noisy'),screen)
            self.assertEqual(selection['var']['g4_c8']['selected_lambda'],.5)

    def test_missing_or_duplicate_pair_rejected(self):
        f=functions();f['count']=1;screen=rows_for(0,f)
        missing=[row for row in screen if not(row['projection']=='g4_c8' and row['noise_seed']==4101 and row['control']=='UNGUIDED')]
        with self.assertRaises(RuntimeError):m.derive(missing,f,{'g4_c8'})
        duplicate=screen+[next(row for row in screen if row['projection']=='g4_c8' and row['control']=='UNGUIDED')]
        with self.assertRaises(RuntimeError):m.derive(duplicate,f,{'g4_c8'})

    def test_atomic_preserve_keeps_original_bytes_and_rejects_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.json';path.write_text('{"rows": [], "registration_sha256": "same"}')
            before=path.read_bytes()
            self.assertFalse(m.atomic_preserve(path,{'registration_sha256':'same','rows':[]}))
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(RuntimeError):m.atomic_preserve(path,{'registration_sha256':'different','rows':[]})
            self.assertEqual(path.read_bytes(),before)

    def test_realistic200_source_qualification_materialization_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calibration,registration=fixture(root)
            proof=m.execute(root,'qualify')
            self.assertEqual(proof['source_indices'],list(range(200)))
            self.assertFalse(proof['fresh_gpu_inference']);self.assertFalse(proof['torch_imported'])
            self.assertFalse((calibration/'gate').exists())
            receipt=m.execute(root,'apply');m.verify(receipt['outputs']);m.verify(receipt['inputs'])
            before=dict(receipt['outputs']);self.assertEqual(len(before),200)
            repeated=m.execute(root,'apply');self.assertEqual(receipt,repeated);m.verify(before)
            one=m.read(calibration/'gate/source_0000.json')
            self.assertEqual(set(one),{'registration_sha256','rows'})
            self.assertTrue(all(row['stage']==m.STAGE for row in one['rows']))
            corrupted=calibration/'screen/source_0003.json';body=m.read(corrupted);body['registration_sha256']='stale'
            corrupted.write_bytes(m.encode(body))
            with self.assertRaisesRegex(RuntimeError,'another calibration'):m.prepare(root)
            m.verify(before)

    def test_csv_original_values_and_ledger_feasibility_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ledger.csv';f=functions();rows=[f['resource'](512,4,8,'QPSK')]
            csv_write(path,rows);m.validate_csv(path,rows)
            modified=copy.deepcopy(rows);modified[0]['feasible']=not modified[0]['feasible'];csv_write(path,modified)
            with self.assertRaises(RuntimeError):m.validate_csv(path,rows)


if __name__=='__main__':unittest.main()
