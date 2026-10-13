"""Independent synthetic fixtures only; never access the real R1 run."""
import copy
import csv
import json
import os
from pathlib import Path
import sys
import subprocess
import stat
import tempfile
import types
import unittest
from unittest import mock

import deliver as d


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8');return d.pin(path)


def fixture(base):
    root=base/'repo';temporary=base/'personal';run=temporary/'r1_execution_v1'
    out=root/'results/layered_mismatch_raw_20261013/r1_execution_v1';resource=root/'experiments/layered-mismatch-raw-20261013/r1_execution_resource_v2'
    dispatch=temporary/'r1_resource_dispatch_v2';control=temporary/'r1_initial_dispatch_control_v1'
    runtime=dict(root=str(run),python_cpu='cpu',python_visual='visual')
    runtime.update({s+'_directory':str(run/s) for s in d.STAGES})
    reg=dict(runtime=runtime,expected_physical_frames=32400,expected_logical_var_images=43200,
             phy_registration=dict(path='phy',sha256='p'),visual_registration=dict(path='visual',sha256='v'))
    registration=save(out/'registered/registration.json',reg)
    owner=dict(pid=200,start_ticks=20,state='S');supervisor=dict(pid=100,start_ticks=10,state='S')
    save(run/'owner_launch.json',dict(process=owner,registration=registration))
    save(dispatch/'launch.json',dict(process=supervisor,registration=registration,
        argv=['cpu','-B','-u',str(resource/'owner.py'),'--registration',registration['path'],'--registration-sha256',registration['sha256']]))
    save(control/'launch.json',dict(process=supervisor,registration=registration,
        argv=['cpu','-B','-u',str(resource/'dispatch_once.py'),'--expected-registration',registration['sha256']]))
    save(dispatch/'owner_identity.json',dict(process=owner,registration=registration))
    save(dispatch/'actual_wait.json',dict(actual_wait=True,pid=200,returncode=0,identity_at_launch=owner,
        registration=registration,scientific_owner_launch_exists=True))
    stages=[]
    for index,stage in enumerate(d.STAGES):
        identity=dict(pid=300+index,start_ticks=30+index,state='S')
        wait=dict(actual_wait=True,pid=identity['pid'],returncode=0,stage=stage)
        wp=save(run/stage/'actual_wait.json',wait);save(run/(stage+'_actual_wait.json'),wait)
        child_registration=reg['phy_registration'] if stage=='phy' else (reg['visual_registration'] if stage=='visual' else registration)
        cp=save(run/stage/'completion.json',dict(status=d.STAGE_STATUS[stage],channel_crc_accept=False,
            registration=child_registration,ledger=dict(physical_complete=32400,physical_reserved=32400,
                packet_decodes_complete=90000,packet_decodes_reserved=90000,unresolved_packets=0)))
        if stage=='phy':
            script=resource.parent/'r1_phy.py';sub=reg['phy_registration'];py='cpu'
            args=['run','--registration',sub['path'],'--registration-sha256',sub['sha256']]
        elif stage in ('projection','visual'):
            script=resource/'receiver_visual.py';sub=reg['visual_registration'] if stage=='visual' else registration
            py='visual' if stage=='visual' else 'cpu'
            args=['visual' if stage=='visual' else 'project','--registration',sub['path'],'--expected-registration',sub['sha256'],'--parent-pid','200','--parent-ticks','20']
        else:
            script=resource/'score_selection.py';py='visual' if stage=='score' else 'cpu'
            args=['score' if stage=='score' else 'select','--registration',registration['path'],'--registration-sha256',registration['sha256'],'--parent','200']
        save(run/(stage+'_launch.json'),dict(stage=stage,registration=registration,process=identity,argv=[py,'-B','-u',str(script),*args]))
        stages.append(dict(stage=stage,actual_wait=wp,completion=cp))
    save(out/'completion.json',dict(status='COMPLETE_R1_CHAIN',actual_exit_zero_all=True,automatic_R2=False,
        R2='NOT_REGISTERED_NOT_RUN',registration=registration,stages=stages,frozen_policy_completion=stages[-1]['completion']))
    return root,temporary,registration


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root,self.temporary,self.registration=fixture(Path(self.temp.name))
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(d,'MASTER_SHA',self.registration['sha256']).start()
        mock.patch.object(d,'routed',lambda reg,original,args:[str(original),*args]).start()
        mock.patch.object(d,'unheld_locks',lambda paths:None).start()

    def call(self,probe=lambda pid:None):return d.preflight(self.root,self.temporary,self.registration,probe)

    def test_complete_real_wait_shape_passes_without_output(self):
        before={str(x) for x in Path(self.temp.name).rglob('*')}
        reg,proof=self.call();self.assertTrue(proof['all_original_processes_gone'])
        self.assertIn('no child-wait receipt',proof['initial_supervisor_observation'])
        self.assertEqual(before,{str(x) for x in Path(self.temp.name).rglob('*')})

    def test_live_and_zombie_original_identity_rejected(self):
        for state in ('R','S','Z'):
            with self.subTest(state=state),self.assertRaises(RuntimeError):
                self.call(lambda pid:dict(pid=pid,start_ticks=20 if pid==200 else 10,state=state))

    def test_pid_reuse_is_not_original_process(self):
        self.call(lambda pid:dict(pid=pid,start_ticks=9999,state='R'))

    def test_same_identity_may_have_different_observed_process_states(self):
        for name in ('launch.json','owner_identity.json'):
            path=self.temporary/'r1_resource_dispatch_v2'/name;value=d.read(path);value['process']['state']='R';save(path,value)
        path=self.temporary/'r1_resource_dispatch_v2/actual_wait.json';value=d.read(path);value['identity_at_launch']['state']='D';save(path,value)
        self.call()
        value['identity_at_launch']['start_ticks']+=1;save(path,value)
        with self.assertRaises(RuntimeError):self.call()

    def test_unclosed_or_failed_does_not_read_quality_or_make_output(self):
        original=d.read;reads=[]
        def read_spy(path):
            reads.append(str(path));self.assertNotIn('scored_rows',str(path));self.assertNotIn('source_means',str(path));self.assertNotIn('selection.json',str(path))
            return original(path)
        save(self.temporary/'r1_resource_dispatch_v2/actual_wait.json',dict(actual_wait=True,pid=200,returncode=1))
        before={str(x) for x in Path(self.temp.name).rglob('*')}
        with mock.patch.object(d,'read',read_spy),self.assertRaises(RuntimeError):self.call()
        self.assertEqual(before,{str(x) for x in Path(self.temp.name).rglob('*')})

    def test_failure_marker_rejected_but_channel_crc_false_is_not_failure(self):
        self.call();save(self.temporary/'r1_execution_v1/phy/failure.json',{})
        with self.assertRaises(RuntimeError):self.call()

    def test_missing_stage_completion_and_wrong_argv_rejected(self):
        path=self.temporary/'r1_execution_v1/phy_launch.json';value=d.read(path);value['argv'][-1]='WRONG';save(path,value)
        with self.assertRaises(RuntimeError):self.call()

    def test_missing_final_completion_stops_early(self):
        (self.root/'results/layered_mismatch_raw_20261013/r1_execution_v1/completion.json').unlink()
        with self.assertRaises(FileNotFoundError):self.call()

    def test_rehashed_wrong_stage_status_or_child_registration_stops_before_quality(self):
        final_path=self.root/'results/layered_mismatch_raw_20261013/r1_execution_v1/completion.json'
        complete=self.temporary/'r1_execution_v1/visual/completion.json';original=d.read(complete);final_original=d.read(final_path)
        real_read=d.read
        def no_quality(path):
            self.assertNotIn('scored_rows',str(path));self.assertNotIn('source_means',str(path));self.assertNotIn('selection.json',str(path))
            return real_read(path)
        for kind in ('status','registration'):
            changed=copy.deepcopy(original)
            if kind=='status':changed['status']='STILL_RUNNING'
            else:changed['registration']=dict(path='wrong-child',sha256='wrong')
            newpin=save(complete,changed);final=copy.deepcopy(final_original);final['stages'][2]['completion']=newpin;save(final_path,final)
            before={str(x) for x in Path(self.temp.name).rglob('*')}
            with mock.patch.object(d,'read',no_quality),self.assertRaisesRegex(RuntimeError,'Stage status or child registration'):self.call()
            self.assertEqual(before,{str(x) for x in Path(self.temp.name).rglob('*')})

    def test_all_interruption_flags_rejected(self):
        identity=dict(pid=1,start_ticks=3)
        for flag in ('timeout','timed_out','terminated','killed','stopped'):
            with self.subTest(flag=flag),self.assertRaises(RuntimeError):
                d.actual_zero(dict(actual_wait=True,pid=1,returncode=0,**{flag:True}),identity)
        with self.assertRaises(RuntimeError):d.actual_zero(dict(actual_wait=True,pid=1,returncode=False),identity)
        with self.assertRaises(RuntimeError):d.actual_zero(dict(actual_wait=True,pid=True,returncode=0),identity)
        with self.assertRaises(RuntimeError):d.actual_zero(dict(actual_wait=True,pid=1,returncode=0,start_ticks=True),dict(pid=1,start_ticks=1))


def synthetic_gate(j):
    comparisons={r:dict(per_hat={h:j['U_two'][h]-j[r][h] for h in (7,10)},
        mean=sum(j['U_two'][h]-j[r][h] for h in (7,10))/2,
        both_hats_positive=all(j['U_two'][h]>j[r][h] for h in (7,10))) for r in ('S_robust','S_back3')}
    go=all(x['mean']>=.01 and x['both_hats_positive'] for x in comparisons.values())
    return dict(status='GO_R2' if go else 'NO_GO_STOP',comparisons=comparisons,r2_authorized=go)


def selection_fixture():
    names={'U':'U_two','S':'S_robust','S_nom':'S_nom','S_back3':'S_back3','EEP':'E_two'}
    policy=types.SimpleNamespace(ARM_NAMES=names)
    nominations=[];expected={};scores={};winners=[];ranking=[]
    values={'U':.65,'S':.60,'S_nom':.59,'S_back3':.58,'EEP':.61}
    for hat in (7,10):
        for arm in names:
            nominations.append(dict(snr_hat=hat,arm=arm,candidate_id=arm,receiver_variants=['DROP']))
            expected[(hat,arm)]=dict(arms={arm},physical_key='physical_'+arm);scores[(hat,arm,'DROP')]=[values[arm]]*200
        for arm,method in names.items():
            entries=['S','S_nom','S_back3'] if arm=='S' else [arm]
            for index,candidate in enumerate(entries):ranking.append(dict(method=method,snr_hat=hat,rank=index+1,candidate_id=candidate,receiver_rule='DROP',DINO_J=values[candidate]))
            winners.append(dict(method=method,snr_hat=hat,candidate_id=arm,receiver_rule='DROP',cal200_DINO_J=values[arm],
                physical_key='physical_'+arm,selection_metric='dinov2_vitl14_cosine',selection_p=.25,
                pressure_e_minus3_used_for_selection=False,convnext_used_for_selection=False))
    j={m:{h:values[arm] for h in (7,10)} for arm,m in names.items() if m in ('U_two','S_robust','S_back3')}
    selected=dict(authenticated_actual_closure=True,no_bootstrap=True,bootstrap_calls=0,source_count=200,noise_count=3,
        logical_rows=43200,R2='NOT_REGISTERED_NOT_RUN',stress_true_snr_plus3_used_for_selection=False,
        winners=winners,ranking=ranking,gate=synthetic_gate(j))
    return json.loads(json.dumps(selected)),scores,expected,nominations,policy


class ArithmeticTests(unittest.TestCase):
    def frame_fixture(self):
        row=dict(source_id='s',source_index=1,selection_slot=0,snr_hat=7,snr_true=4,noise_seed=4101,candidate_id='c',physical_key='physical',receiver_rule='DROP',
                 execution_status='COMPLETE',psnr_db=10.,lpips_alex=.8,dinov2_vitl14_cosine=-.1)
        for key in ('view_id','observation_key','actual_rx_key','actual_input_key','receiver_state_sha256','model_identity','metric_identity',
                    'metric_pair_key','metric_reuse','first_scored_view_id','reconstruction_sha256'):row[key]=key
        join=dict(view_id=row['view_id'],actual_received_kind='tokens',actual_received_m=1,actual_received_K=0,actual_failure_state=False)
        event=dict(header_ok=True,header_crc_ok=True,header_fields_legal=True,profile_id=500,receive_status='BASE_ONLY',received_group_count=2,
                   groups=[dict(crc_accept=True),dict(crc_accept=False)])
        frame=dict(actual_RX=event,input_identity=dict(G=2,profile_id=500),receiver_views={'DROP':dict(prefix=[[2]],partial_values=[])},
            actual_frame_energy=2048.,actual_frame_mean_Es=2.,channel_noise_variance_per_component=10**(-.4),
            post_receiver_truth=dict(header_false_accept=False,body_comparison_available=True,groups=[]))
        return row,join,frame

    def test_flatten_keeps_crc_failure_quality_and_real_energy(self):
        row,join,frame=self.frame_fixture();flat=d.flatten(row,join,frame)
        self.assertEqual(flat['state_class'],'BASE_ONLY');self.assertFalse(flat['g2_crc_accept'])
        self.assertEqual(flat['software_execution_status'],'COMPLETE');self.assertEqual(flat['dinov2_vitl14_cosine'],-.1)
        self.assertEqual(flat['retained_token_count'],1);self.assertEqual(flat['actual_frame_energy'],2048.)

    def test_wrong_legal_id_and_header_rejection_not_full_success(self):
        row,join,frame=self.frame_fixture();frame['actual_RX']['profile_id']=0
        frame['actual_RX']['receive_status']='SINGLE_FORMAT_ACCEPT';frame['actual_RX']['groups']=[dict(crc_accept=True)]
        frame['post_receiver_truth'].update(header_false_accept=True,body_comparison_available=False)
        flat=d.flatten(row,join,frame);self.assertEqual(flat['state_class'],'WRONG_LEGAL_ID');self.assertIsNone(flat['g2_crc_accept'])
        frame['actual_RX'].update(header_ok=False,header_crc_ok=False,profile_id=None,receive_status='HEAD_REJECT',groups=[])
        frame['receiver_views']['DROP']=dict(prefix=[],partial_values=[]);join.update(actual_received_kind='gray',actual_received_m=0,actual_failure_state=True)
        flat=d.flatten(row,join,frame);self.assertIsNone(flat['g1_crc_accept']);self.assertEqual(flat['retained_token_count'],0)

    def test_raw_flags_are_required_exact_original_top_level_evidence(self):
        row,join,frame=self.frame_fixture();frame['input_identity'].update(G=1,profile_id=12)
        frame['actual_RX']=dict(header_ok=True,header_crc_ok=True,header_fields_legal=True,rx_profile_id=12,body_crc_accept=False)
        flat=d.flatten(row,join,frame);self.assertTrue(flat['header_crc_ok']);self.assertFalse(flat['g1_crc_accept'])
        self.assertEqual(flat['state_class'],'SINGLE_CRC_REJECT_DROP')
        del frame['actual_RX']['header_crc_ok']
        with self.assertRaises(KeyError):d.flatten(row,join,frame)

    def test_non_go_report_is_tex_escaped_without_rewriting_values(self):
        selected,*_=selection_fixture();selected['gate']['status']='NO_GO_STOP'
        selected['separate_engineering_targets']=dict(per_hat=[dict(snr_hat=7,matched_U_minus_S_nom_psnr_db=-.2,same_U_overestimate_psnr_loss_db=1.4)])
        report=d.report_text(dict(selected=selected));tex=d.tex_fragment(report)
        self.assertIn('NO_GO_STOP',report);self.assertIn(r'NO\_GO\_STOP',tex);self.assertIn('-0.200000',tex)

    def test_physical_aliases_deduplicate_only_exact_shared_event(self):
        row,join,frame=self.frame_fixture();flat=d.flatten(row,join,frame)
        self.assertEqual(len(d.physical_rows([flat,dict(flat,receiver_rule='KEEP',view_id='other')])),1)
        with self.assertRaises(RuntimeError):d.physical_rows([flat,dict(flat,actual_frame_energy=0.)])

    def test_oversize_copy_keeps_complete_bytes_in_personal_tmp(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);light=base/'light';heavy=base/'heavy';light.mkdir();heavy.mkdir()
            source=base/'source.json';source.write_bytes(b'1234567890')
            with mock.patch.object(d,'LIMIT',10):target=d.copy_selected(source,light,heavy)
            self.assertEqual(target.parent,heavy);self.assertEqual(target.read_bytes(),source.read_bytes())
            self.assertEqual(list(light.iterdir()),[])

    def test_absence_gate_refuses_existing_empty_directory_or_file(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'target';target.mkdir();file=Path(td)/'file';file.write_bytes(b'keep')
            for path in (target,file):
                with self.assertRaises(FileExistsError):d.require_absent(path)
            self.assertTrue(target.is_dir());self.assertEqual(file.read_bytes(),b'keep')

    def test_absence_gate_refuses_dangling_symlink(self):
        # Do not create real PFS symbolic links even in synthetic tests. lstat
        # returns link metadata without following the dangling destination.
        target=Path('synthetic-dangling-link');link=types.SimpleNamespace(st_mode=stat.S_IFLNK)
        with mock.patch.object(Path,'lstat',return_value=link) as metadata,\
             mock.patch.object(Path,'open',side_effect=AssertionError('Followed link via open')),\
             mock.patch.object(Path,'is_file',side_effect=AssertionError('Followed link via is_file')):
            with self.assertRaises(FileExistsError):d.require_absent(target)
        metadata.assert_called_once_with()

    def test_stored_json_gate_keys_and_ranking_are_verified_without_reselection(self):
        selected,scores,expected,nominations,policy=selection_fixture()
        with mock.patch.dict(sys.modules,{'contract':types.SimpleNamespace(calibration_gate=synthetic_gate)}):
            out=d.verify_existing_selection(selected,{},scores,expected,nominations,[],policy)
            self.assertEqual(len(out),10)
            selected['winners'][0]['cal200_DINO_J']+=.001
            with self.assertRaises(RuntimeError):d.verify_existing_selection(selected,{},scores,expected,nominations,[],policy)

    def test_missing_candidate_and_changed_gate_rejected(self):
        selected,scores,expected,nominations,policy=selection_fixture()
        with mock.patch.dict(sys.modules,{'contract':types.SimpleNamespace(calibration_gate=synthetic_gate)}):
            selected['gate']['comparisons']['S_robust']['mean']=1
            with self.assertRaises(RuntimeError):d.verify_existing_selection(selected,{},scores,expected,nominations,[],policy)
            selected,scores,expected,nominations,policy=selection_fixture();selected['ranking'].pop()
            with self.assertRaises(RuntimeError):d.verify_existing_selection(selected,{},scores,expected,nominations,[],policy)

    def test_csv_value_validation_is_exact(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'means.csv';expected=[dict(source_id='x',mean=-.0123,noise_count=3)]
            d.write_csv(path,expected);d.compare_csv(path,expected)
            with self.assertRaises(RuntimeError):d.compare_csv(path,[dict(expected[0],mean=.0123)])

    def test_zero_states_retained_no_fake_conditional_quality(self):
        rows=[]
        for source in range(200):
            for seed in range(3):rows.append(dict(source_id=str(source),snr_hat=7,snr_true=4,candidate_id='c',receiver_rule='DROP',
                tx_group_count=2,state_class='HEAD_REJECT',post_receiver_header_false_accept=False,g1_crc_accept=None,g2_crc_accept=None,
                psnr_db=10.,lpips_alex=.7,dinov2_vitl14_cosine=-.1))
        output=d.state_summary(rows,('snr_hat','snr_true','candidate_id','receiver_rule'))
        lookup={r['state_class']:r for r in output};self.assertEqual(lookup['BASE_ONLY']['frames'],0)
        self.assertIsNone(lookup['BASE_ONLY']['psnr_db_conditional_mean'])
        self.assertEqual(lookup['HEAD_REJECT']['fraction'],1.);self.assertEqual(lookup['HEAD_REJECT']['g1_attempted'],0)
        self.assertEqual(lookup['HEAD_REJECT']['dinov2_vitl14_cosine_conditional_mean'],-.1)
        with self.assertRaises(RuntimeError):d.state_summary(rows[:-1],('snr_hat','snr_true','candidate_id','receiver_rule'))

    def test_import_and_readonly_guard_fail_closed(self):
        guard=d.CPUReadOnly()
        for name in ('torch','torch.nn','tensorflow','sionna','lpips'):
            with self.assertRaises(RuntimeError):guard.find_spec(name)
        for event,args in [('subprocess.Popen',()),('os.mkdir',()),('open',('x','w',os.O_WRONLY)),('open',('x','r+',os.O_RDWR))]:
            with self.assertRaises(RuntimeError):guard.audit(event,args)
        guard.audit('open',('x','r',os.O_RDONLY));self.assertIsNone(guard.find_spec('numpy'))

    def test_real_installed_audit_guard_in_isolated_cpu_interpreter(self):
        code="""import sys
from pathlib import Path
import deliver as d
guard=d.CPUReadOnly();sys.meta_path.insert(0,guard);sys.addaudithook(guard.audit)
for name in ('torch','sionna','tensorflow','lpips'):
    try:__import__(name)
    except RuntimeError:pass
    else:raise AssertionError('scientific import allowed')
try:Path('forbidden_write.txt').write_text('x')
except RuntimeError:pass
else:raise AssertionError('write allowed')
assert Path(d.__file__).read_text()
print('INSTALLED_GUARD_PASS')
"""
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(d.__file__).parent,text=True,capture_output=True,check=False)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('INSTALLED_GUARD_PASS',result.stdout)

    def test_source_has_no_scientific_or_selection_calls(self):
        source=Path(d.__file__).read_text()
        for forbidden in ('select_from_complete_rows(','.select(registration','.score_worker(','.RealRuntime(','.ThreeMetrics(','.render_rows('):
            self.assertNotIn(forbidden,source)
        self.assertNotIn('torch',sys.modules)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        self.root=self.base/'repo';self.temporary=self.base/'tmp';self.temporary.mkdir()
        (self.root/'results/layered_mismatch_raw_20261013').mkdir(parents=True)
        self.output=self.root/'results/layered_mismatch_raw_20261013/r1_delivery_v1'
        self.rows=[dict(view_id=str(i),snr_hat=7,candidate_id=str(i//1800),receiver_rule='DROP') for i in range(43200)]
        self.data=dict(rows=self.rows,joins=[dict(view_id=r['view_id'],actual_RX=dict(sha256='frame')) for r in self.rows],
            frames={'frame':{}},winners={('U_two',7):dict(candidate_id='0',receiver_rule='DROP')},selection={},audit={},selected=dict(gate=dict(status='NO_GO_STOP')))
        for key in ('source_means','source_means_csv','candidate_summary_csv','frozen_winner_summary_csv','selection'):
            path=self.base/(key+'.json');path.write_text('{}');self.data['selection'][key]=d.pin(path)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(d,'flatten',lambda row,join,frame:row).start()
        mock.patch.object(d,'state_summary',lambda *a:[dict(state='test')]).start()
        mock.patch.object(d,'physical_rows',lambda flat:[{}]*32400).start()
        mock.patch.object(d,'method_parameters',lambda *a:[dict(method='test')]).start()
        mock.patch.object(d,'report_text',lambda *a:'NO_GO_STOP\n').start()
        mock.patch.object(d,'write_csv',lambda path,rows:path.write_text('synthetic\n')).start()
        if not sys.platform.startswith('linux'):mock.patch.object(d,'fsync_directory',lambda path:None).start()

    def call(self):return d.publish(self.root,self.temporary,dict(path='fixture',sha256='fixture'),self.data,{})

    def test_existing_claim_prevents_publication_and_preserves_prior_bytes(self):
        claim=self.temporary/'r1_delivery_v1.claim.json';claim.write_bytes(b'prior-claim')
        with self.assertRaises(FileExistsError):self.call()
        self.assertEqual(claim.read_bytes(),b'prior-claim');self.assertFalse(self.output.exists())
        self.assertEqual(list(self.temporary.iterdir()),[claim])

    def test_second_publish_failure_retains_payload_claim_and_failure_receipt(self):
        actual=d.publish_directory;calls=[]
        def second_fails(source,target,claim,exclude=()):
            calls.append(str(target))
            if len(calls)==2:raise OSError('synthetic second publish failure')
            return actual(source,target,claim,exclude)
        with mock.patch.object(d,'publish_directory',second_fails),self.assertRaisesRegex(OSError,'synthetic second publish failure'):self.call()
        self.assertEqual(len(calls),2)
        self.assertFalse(self.output.exists());self.assertTrue((self.temporary/'r1_delivery_v1').is_dir())
        self.assertTrue((self.temporary/'r1_delivery_v1.claim.json').is_file())
        failure=list(self.temporary.glob('r1_delivery_v1.preparing_*/publication_failure.json'))
        self.assertEqual(len(failure),1);receipt=d.read(failure[0]);self.assertTrue(receipt['heavy_files_complete'])
        self.assertFalse(receipt['light_directory_exists']);self.assertFalse(receipt['completion_marker_exists'])
        self.assertFalse(receipt['automatic_retry']);self.assertEqual(receipt['phase'],'light_directory')
        before=(self.temporary/'r1_delivery_v1/physical_events.csv').read_bytes()
        with self.assertRaises(FileExistsError):self.call()
        self.assertEqual((self.temporary/'r1_delivery_v1/physical_events.csv').read_bytes(),before)

    def test_hardlink_publishes_completion_only_after_all_files_pass(self):
        original=d.publish_completion;calls=[]
        def observed(source,target,identity,claim):
            self.assertFalse(target.exists());self.assertTrue((self.output/'delivery_manifest.json').is_file())
            d.verify_published_files(self.output/'delivery_manifest.json',source);calls.append(str(target))
            return original(source,target,identity,claim)
        actual_link=d.os.link
        def same_directory_only(source,target,**kwargs):
            self.assertEqual(Path(source).parent,Path(target).parent,'Cross-area link is forbidden')
            return actual_link(source,target,**kwargs)
        with mock.patch.object(d,'publish_completion',observed),mock.patch.object(d.os,'link',same_directory_only):result=self.call()
        self.assertEqual(len(calls),1);self.assertEqual(result['status'],'DELIVERED_R1_CALIBRATION_ONLY')
        marker=self.output/'completion.json';draft=Path(result['original_staging_preserved'])
        self.assertTrue(draft.is_dir());self.assertEqual(marker.stat().st_ino,(self.output/'completion.pending.json').stat().st_ino)
        self.assertEqual(marker.read_bytes(),(draft/'light/completion.json').read_bytes())
        before=marker.read_bytes();identity=d.filesystem_identity(self.output,'directory')
        claimpath=self.temporary/'r1_delivery_v1.claim.json';claim=dict(file=d.pin(claimpath),identity=d.filesystem_identity(claimpath,'file'))
        with self.assertRaises(FileExistsError):d.publish_completion(draft/'light/completion.json',marker,identity,claim)
        self.assertEqual(marker.read_bytes(),before)

    def test_link_unsupported_keeps_both_directories_incomplete_without_fallback(self):
        with mock.patch.object(d.os,'link',side_effect=OSError(22,'synthetic unsupported hardlink')),self.assertRaisesRegex(OSError,'unsupported hardlink'):self.call()
        self.assertTrue(self.output.is_dir());self.assertFalse((self.output/'completion.json').exists())
        self.assertTrue((self.temporary/'r1_delivery_v1').is_dir())
        receipt=d.read(next(self.temporary.glob('r1_delivery_v1.preparing_*/publication_failure.json')))
        self.assertEqual(receipt['errno'],22);self.assertEqual(receipt['phase'],'completion_marker')
        self.assertTrue(receipt['heavy_files_complete'] and receipt['light_files_complete'])
        self.assertFalse(receipt['completion_marker_exists']);self.assertFalse(receipt['automatic_retry'])

    def test_exclusive_copy_and_changed_claim_never_overwrite(self):
        targetdir=self.base/'exclusive';targetdir.mkdir();identity=d.filesystem_identity(targetdir,'directory')
        source=self.base/'copy-source';source.write_bytes(b'new');target=targetdir/'target';target.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):d.exclusive_copy(source,target,identity)
        self.assertEqual(target.read_bytes(),b'keep')
        claimpath=self.temporary/'claim';claimpath.write_bytes(b'claim');claim=dict(file=d.pin(claimpath),identity=d.filesystem_identity(claimpath,'file'))
        claimpath.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):d.verify_claim(claim)
        claimpath.write_bytes(b'claim');claim['identity']['inode']+=1
        with self.assertRaises(RuntimeError):d.verify_claim(claim)

    def test_published_payload_mutation_is_rejected_before_completion_link(self):
        actual=d.publish_directory
        def mutate_after_heavy(source,target,claim,exclude=()):
            identity=actual(source,target,claim,exclude)
            if target==self.temporary/'r1_delivery_v1':(target/'physical_events.csv').write_text('changed')
            return identity
        with mock.patch.object(d,'publish_directory',mutate_after_heavy),mock.patch.object(d.os,'link') as link,self.assertRaisesRegex(RuntimeError,'Bound file changed'):
            self.call()
        link.assert_not_called();self.assertFalse((self.output/'completion.json').exists())
        receipt=d.read(next(self.temporary.glob('r1_delivery_v1.preparing_*/publication_failure.json')))
        self.assertEqual(receipt['phase'],'final_artifact_verification')

    def test_post_link_fsync_failure_retains_marker_but_returns_failure(self):
        actual=d.fsync_directory
        def fail_after_marker(path):
            if Path(path)==self.output and (self.output/'completion.json').exists():raise OSError(5,'synthetic post-link fsync failure')
            return actual(path)
        with mock.patch.object(d,'fsync_directory',fail_after_marker),self.assertRaisesRegex(OSError,'post-link fsync failure'):self.call()
        self.assertTrue((self.output/'completion.json').is_file())
        receipt=d.read(next(self.temporary.glob('r1_delivery_v1.preparing_*/publication_failure.json')))
        self.assertTrue(receipt['completion_marker_exists']);self.assertEqual(receipt['errno'],5);self.assertFalse(receipt['automatic_retry'])


if __name__=='__main__':unittest.main()
