"""Synthetic metadata/CSV tests only; no active scientific data or rendering."""
import ast
import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import figures as f


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8');return f.pin(path)


def table_fixture(folder,master):
    rows=[];winners=[]
    for mi,method in enumerate(f.METHODS):
        for hat in f.HATS:
            winners.append(dict(method=method,snr_hat=hat,candidate_id=method+str(hat),receiver_rule='DROP'))
            for true in (hat-3,hat,hat+3):
                for ki,metric in enumerate(f.METRICS):
                    value=(mi+true)/10 if ki==0 else (0. if mi==0 else -.2+mi/20) if ki==2 else .45-mi/30
                    rows.append(dict(method=method,snr_hat=hat,snr_true=true,metric=metric,mean=value,
                        source_count=200,noise_count=3,candidate_id=method+str(hat),receiver_rule='DROP'))
    path=folder/'frozen_winner_summary.csv'
    with path.open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    selected=dict(status='COMPLETE_R1_CALIBRATION_SELECTION',authenticated_actual_closure=True,
        source_count=200,noise_count=3,no_bootstrap=True,R2='NOT_REGISTERED_NOT_RUN',registration=master,
        winners=winners,gate=dict(status='ONE_POINT_PROMISING',comparisons={
            'S_robust':dict(per_hat={'7':-.001,'10':.002},mean=.0005),
            'S_back3':dict(per_hat={'7':0.,'10':.003},mean=.0015)}))
    selection=save(folder/'selection.json',selected)
    return dict(frozen_winner_summary_csv=f.pin(path),selection=selection),rows,selected


def provenance_fixture(root,master):
    return dict(recovery_registration=master,original_registration=dict(
        path=str(root/'results/layered_mismatch_raw_20261013/r1_execution_v1/registered/registration.json'),
        sha256=f.ORIGINAL_MASTER_SHA),original_owner_actual_exit=1,new_owner_actual_exit=0,
        all_original_and_new_processes_gone=True,scientific_calls=0,bootstrap_calls=0,reselection_calls=0,
        inherited_successful_stages=[dict(stage=x,actual_exit=0) for x in ('phy','projection')],
        new_successful_stages=[dict(stage=x,actual_exit=0) for x in ('visual','score','selection')],
        original_failed_visual=dict(stage='visual',actual_exit=1),
        original_failure=dict(original_failure_preserved=True,failed_model_load_reservations=1,
            completed_original_model_loads=0,completed_original_forward_calls=0,partial_images=0,
            audit=dict(sha256=f.FAILURE_AUDIT_SHA)))


def closure_fixture(base):
    root=base/'repo';temporary=base/'tmp';control=temporary/'r1_cache_recovery_delivery_dispatch_v1'
    output=root/'results/layered_mismatch_raw_20261013/r1_cache_recovery_delivery_v1';heavy=temporary/'r1_cache_recovery_delivery_v1'
    output.mkdir(parents=True);heavy.mkdir(parents=True);control.mkdir()
    entry=root/'experiments/layered-mismatch-raw-20261013/r1_cache_recovery_delivery_v1/deliver.py';entry.parent.mkdir(parents=True);entry.write_text('frozen synthetic CPU entry')
    dispatch=root/'experiments/layered-mismatch-raw-20261013/r1_cache_recovery_delivery_dispatch_v1/dispatch_once.py';dispatch.parent.mkdir();dispatch.write_text('frozen synthetic parent')
    master=save(root/'results/layered_mismatch_raw_20261013/r1_cache_recovery_v1/registered/registration.json',{'fixture':True})
    inputs,rows,selected=table_fixture(output,master)
    audit=save(output/'closure_audit.json',dict(status='PASS_R1_COMPLETED_CPU_DELIVERY',registration=master,
        scientific_calls=0,bootstrap_calls=0,reselection_calls=0,old_scientific_results_modified=False,
        process_closure=provenance_fixture(root,master)))
    (heavy/'heavy.csv').write_text('all_saved_rows_fixture\n')
    manifest=save(output/'delivery_manifest.json',dict(registration=master,files=[*inputs.values(),audit,f.pin(heavy/'heavy.csv')],
        copy_keys=dict(frozen_winner_summary_csv='frozen_winner_summary.csv',selection='selection.json'),heavy_not_for_git=str(heavy),scientific_arrays_copied=False))
    save(output/'completion.pending.json',dict(status='COMPLETE_R1_CPU_DELIVERY',requires_actual_delivery_exit_zero=True,
        registration=master,R2='NOT_REGISTERED_NOT_RUN',scientific_calls=0,bootstrap_calls=0,reselection_calls=0,manifest_sha256=manifest['sha256']))
    os.link(output/'completion.pending.json',output/'completion.json')
    draft=temporary/'r1_cache_recovery_delivery_v1.preparing_fixture';draft.mkdir()
    result=dict(status='DELIVERED_R1_RECOVERED_CALIBRATION_ONLY',output=str(output),heavy=str(heavy),scientific_calls=0,
        bootstrap_calls=0,reselection_calls=0,original_staging_preserved=str(draft),manifest=manifest)
    stdout=save(control/'stdout.txt',result);(control/'stderr.txt').write_text('')
    common=dict(schema='R1_RECOVERED_CPU_DELIVERY_DISPATCH_V1',registration=master,delivery_entry=f.pin(entry),dispatcher=f.pin(dispatch),
        argv=[str(f.RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(entry),'--expected-registration',master['sha256']],
        owner=dict(pid=101,start_ticks=11),CUDA_VISIBLE_DEVICES='',scientific_calls=0,max_seconds=7200)
    save(control/'prepared.json',common);save(control/'launch.json',dict(common,child=dict(pid=102,start_ticks=12)))
    save(control/'actual_wait.json',dict(common,child=dict(pid=102,start_ticks=12),actual_wait=True,returncode=0,
        terminated=False,killed=False,timed_out=False,metadata_error=None,stdout=stdout,stderr=f.pin(control/'stderr.txt')))
    (control/'dispatch.lock').write_text('')
    return dict(root=root,temporary=temporary,control=control,output=output,heavy=heavy,inputs=inputs,rows=rows,selected=selected,
        master=master,entry=f.pin(entry),dispatcher=f.pin(dispatch),draft=draft)


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.x=closure_fixture(Path(self.temp.name));self.addCleanup(mock.patch.stopall)
        mock.patch.object(f,'MASTER_SHA',self.x['master']['sha256']).start()
        mock.patch.object(f,'DELIVERY_SHA',self.x['entry']['sha256']).start()
        mock.patch.object(f,'DISPATCH_SHA',self.x['dispatcher']['sha256']).start()
        mock.patch.object(f,'unheld_lock',lambda path:None).start()

    def guard(self,probe=lambda pid:None):
        return f.guard_delivery(self.x['root'],self.x['temporary'],self.x['control'],probe)

    def test_complete_metadata_verifies_every_manifest_artifact_without_output(self):
        before={str(p) for p in Path(self.temp.name).rglob('*')};inputs,proof=self.guard()
        self.assertEqual(proof['all_manifest_files_sha_verified'],4);self.assertEqual(inputs,self.x['inputs'])
        self.assertEqual(before,{str(p) for p in Path(self.temp.name).rglob('*')})

    def test_invalid_exit_stops_before_quality_or_matplotlib(self):
        path=self.x['control']/'actual_wait.json';record=f.read(path);record['returncode']=False;save(path,record)
        original=f.sha
        def hashes(path):
            self.assertNotIn(Path(path).name,('frozen_winner_summary.csv','selection.json'));return original(path)
        with mock.patch.object(f,'sha',hashes),self.assertRaises(RuntimeError):self.guard()
        self.assertNotIn('matplotlib.pyplot',__import__('sys').modules)

    def test_timeout_metadata_failure_and_bool_identity_reject(self):
        path=self.x['control']/'actual_wait.json';original=f.read(path)
        for changed in (dict(original,timed_out=True),dict(original,metadata_error='metadata failure'),
                        dict(original,child=dict(pid=True,start_ticks=12))):
            save(path,changed)
            with self.assertRaises(RuntimeError):self.guard()

    def test_live_original_identity_including_zombie_rejects(self):
        for state in ('R','S','Z'):
            with self.assertRaises(RuntimeError):self.guard(lambda pid:dict(pid=pid,start_ticks=11 if pid==101 else 12,state=state))
        self.guard(lambda pid:dict(pid=pid,start_ticks=999,state='R'))

    def test_publication_failure_blocks_even_with_valid_marker(self):
        save(self.x['draft']/'publication_failure.json',dict(error='fsync failed'))
        with self.assertRaises(RuntimeError):self.guard()

    def test_any_manifest_tamper_rejects_before_quality_parse(self):
        (self.x['heavy']/'heavy.csv').write_text('changed')
        with self.assertRaises(RuntimeError):self.guard()

    def test_fixed_code_or_wrong_control_directory_reject(self):
        with self.assertRaises(RuntimeError):f.guard_delivery(self.x['root'],self.x['temporary'],self.x['control'].parent)
        Path(self.x['dispatcher']['path']).write_text('changed')
        with self.assertRaises(RuntimeError):self.guard()

    def test_exact_90_grid_real_x_negative_zero_and_stored_gate_preserved(self):
        inputs,_=self.guard();data=f.load_tables(inputs)
        self.assertEqual(len(data['values']),90)
        self.assertEqual({k[2] for k in data['values'] if k[1]==7},{4,7,10})
        self.assertEqual({k[2] for k in data['values'] if k[1]==10},{7,10,13})
        self.assertLess(min(data['values'].values()),0);self.assertIn(0.,data['values'].values())
        self.assertEqual(data['selection']['gate'],self.x['selected']['gate'])
        self.assertEqual(data['go_rows'][0]['stored_DINO_J_difference'],-.001)
        self.assertIn('non-blind calibration',f.caption(data));self.assertIn('no confidence intervals',f.caption(data))

    def test_duplicate_missing_method_or_wrong_noise_count_reject(self):
        path=Path(self.x['inputs']['frozen_winner_summary_csv']['path'])
        for mutation in ('duplicate','noise'):
            rows=copy.deepcopy(self.x['rows'])
            if mutation=='duplicate':rows[0]=rows[1].copy()
            else:rows[0]['noise_count']=2
            with path.open('w',newline='',encoding='utf8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            inputs=dict(self.x['inputs'],frozen_winner_summary_csv=f.pin(path))
            with self.assertRaises(RuntimeError):f.load_tables(inputs)

    def test_axis_limits_retain_negative_zero_and_degenerate_cells(self):
        for values in ([-.2,0.,.1],[0.,0.],[-1.,-1.]):
            low,high=f.axis_limits(values);self.assertLess(low,min(values));self.assertGreater(high,max(values))

    def test_precision_saved_gate_and_lower_is_better_never_recomputed(self):
        inputs,rows,selected=table_fixture(self.x['output'],self.x['master'])
        rows[0]['mean']='0.12345678901234566'
        path=Path(inputs['frozen_winner_summary_csv']['path'])
        with path.open('w',newline='',encoding='utf8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        selected['gate']['comparisons']['S_robust']['mean']=0.009999999999999998
        # Deliberately stored status does not follow displayed rounding. It is preserved.
        selected['gate']['status']='NO_GO_STOP'
        selection=save(Path(inputs['selection']['path']),selected)
        data=f.load_tables(dict(frozen_winner_summary_csv=f.pin(path),selection=selection))
        key=(rows[0]['method'],int(rows[0]['snr_hat']),int(rows[0]['snr_true']),rows[0]['metric'])
        self.assertEqual(data['values'][key],float('0.12345678901234566'))
        self.assertEqual(data['raw'][0]['mean'],'0.12345678901234566')
        self.assertEqual(data['go_rows'][2]['stored_DINO_J_difference'],0.009999999999999998)
        self.assertEqual(data['selection']['gate']['status'],'NO_GO_STOP')
        self.assertGreater(data['values'][('U_two',7,4,'lpips_alex')],0)

    def test_original_failure_cannot_be_relabelled_as_success(self):
        closure=provenance_fixture(self.x['root'],self.x['master'])
        f.recovery_provenance(dict(process_closure=closure),self.x['root'],self.x['master'])
        for field,value in [('original_owner_actual_exit',0),('new_owner_actual_exit',1),('new_owner_actual_exit',False)]:
            bad=copy.deepcopy(closure);bad[field]=value
            with self.assertRaises(RuntimeError):f.recovery_provenance(dict(process_closure=bad),self.x['root'],self.x['master'])
        for field,value in [('failed_model_load_reservations',0),('completed_original_forward_calls',1),('original_failure_preserved',False)]:
            bad=copy.deepcopy(closure);bad['original_failure'][field]=value
            with self.assertRaises(RuntimeError):f.recovery_provenance(dict(process_closure=bad),self.x['root'],self.x['master'])

    def test_new_stages_must_all_be_closed(self):
        closure=provenance_fixture(self.x['root'],self.x['master'])
        for mutate in ('missing','failed','bool'):
            bad=copy.deepcopy(closure)
            if mutate=='missing':bad['new_successful_stages'].pop()
            else:bad['new_successful_stages'][0]['actual_exit']=1 if mutate=='failed' else False
            with self.assertRaises(RuntimeError):f.recovery_provenance(dict(process_closure=bad),self.x['root'],self.x['master'])

    def test_foreign_manifest_path_is_rejected_before_hash(self):
        manifest_path=self.x['output']/'delivery_manifest.json';manifest=f.read(manifest_path)
        manifest['files'][0]['path']=str(self.x['temporary']/'r1_cache_recovery_v1/visual/active.csv')
        manifest=save(manifest_path,manifest)
        complete=f.read(self.x['output']/'completion.json');complete['manifest_sha256']=manifest['sha256']
        save(self.x['output']/'completion.pending.json',complete)
        result=f.read(self.x['control']/'stdout.txt');result['manifest']=manifest;stdout=save(self.x['control']/'stdout.txt',result)
        wait=f.read(self.x['control']/'actual_wait.json');wait['stdout']=stdout;save(self.x['control']/'actual_wait.json',wait)
        with self.assertRaisesRegex(RuntimeError,'Foreign or duplicate'):self.guard()

    def test_dispatch_failure_blocks_even_with_completion(self):
        save(self.x['control']/'failure.json',dict(error='late metadata error'))
        with self.assertRaisesRegex(RuntimeError,'retained a failure'):self.guard()

    def test_style_methods_distinct_and_no_inference_bootstrap_or_interval_calls(self):
        self.assertEqual(len({v[1] for v in f.STYLE.values()}),5)
        source=Path(f.__file__).read_text(encoding='utf8');tree=ast.parse(source)
        calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
               if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
        self.assertTrue({'errorbar','fill_between','bootstrap','calibration_gate','select_from_complete_rows','score_worker','render_rows'}.isdisjoint(calls))
        self.assertNotIn('import torch',source);self.assertNotIn('import score_selection',source)
        self.assertIn("'svg.fonttype':'none'",source);self.assertIn("'pdf.fonttype':42",source);self.assertIn('dpi=600',source)


if __name__=='__main__':unittest.main()
