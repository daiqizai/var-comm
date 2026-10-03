"""Synthetic CPU fixtures for the separately versioned historical handoff."""
from pathlib import Path
import tempfile
import unittest
import external_controller as c


class Phase2GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.old=self.root/'outputs'/c.OLD_NAME
        self.out=self.root/'outputs/HISTORICAL-PHASE2-RECOVERY-20261004-R1'
        source=self.root/'experiments/historical-phase2-recovery-20261004-r1/fixture.py'
        source.parent.mkdir(parents=True);source.write_text('# engineering fixture\n')
        self.bound={str(source):c.sha(source)}
        names=['FINAL_P2048_P3060','FINAL_P4084_SELECTED_SEEDS','FINAL_DIGITAL_QPSK',
            'FINAL_DIGITAL_16QAM','FINAL_PHASE2_N4084_DIGITAL','SELECTED_EXTERNAL_AUTHORS','LEGACY_N3060_FINAL',
            'SELECTED_NOISELESS_REFERENCES','OPTIONAL_RX_STEP2_A','OPTIONAL_RX_STEP2_B','OPTIONAL_H6_13DB','OPTIONAL_PHASE2_MAIN']
        frames=[3000,3000,9000,9000,3000,13500,4500,1800,2400,2400,600,4500]
        inherited={n:dict(out=str(self.root/'outputs/HISTORICAL-METRICS-R2-20261003'/n)) for n in names[:7]}
        self.original=dict(jobs=[dict(study=n,adapter='historical_fixture') for n in names],coverage_manifest={},
            contrasts=[],inherited_studies=inherited)
        c.write(self.old/'queue_registration.json',self.original);oldsha=c.sha(self.old/'queue_registration.json')
        self.oldlaunch=dict(pid=22,start_ticks='1')
        c.write(self.old/'bootstrap_launch.json',dict(pid=23,start_ticks='1'))
        self.queue=dict(self.original,status='REGISTERED',training_updates=0,policy_selection_updates=0,
            source_bindings=self.bound,input_proof_bindings=self.bound,shared_metric_bindings=self.bound,
            completed_r3_studies={n:dict(out=str(self.old/n)) for n in names[7:11]},
            replay_adapter_overrides={'OPTIONAL_PHASE2_MAIN':'historical_phase2_alias'},
            phase2_recovery=dict(original_queue_sha256=oldsha,inherited_sources=81,remaining_sources=19,
                scope_changed=False,new_metric_offset_applied=False,strict_grid_parity_tolerances_changed=False))
        c.write(self.out/'queue_registration.json',self.queue);qsha=c.sha(self.out/'queue_registration.json')
        self.config=dict(root=str(self.root),old=dict(queue_registration_sha256=oldsha,
            phase2_recovery=dict(output=str(self.out),queue_registration_sha256=qsha)))
        self.launch=dict(pid=33,start_ticks='2',queue_registration_sha256=qsha,source_bindings=self.bound)
        c.write(self.out/'controller_launch.json',self.launch)
        inputs={}
        for index,(name,count) in enumerate(zip(names,frames)):
            folder=Path(inherited[name]['out']) if index<7 else self.old/name if index<11 else self.out/name
            path=folder/'completion.json'
            c.write(path,dict(status='HISTORICAL_STUDY_METRICS_COMPLETE',study=name,sources=100,frames=count,
                parity_passed=True,synthetic=False,training_updates=0,policy_selection_updates=0))
            inputs[str(path)]=c.sha(path)
        self.audit=dict(status='PHASE2_RECOVERY_100_SOURCES_VERIFIED',sources=100,rows=4500,
            inherited_sources=81,new_sources=19,inherited_rows=3645,new_rows=855,new_pure_strict_grid_proofs=285,
            historical_alias_rows_disclosed=1500,new_metric_offset_applied=False,original_files_written=False,
            synthetic=False,bindings=self.bound)
        c.write(self.out/'phase2_completion_audit.json',self.audit)
        inputs[str(self.out/'phase2_completion_audit.json')]=c.sha(self.out/'phase2_completion_audit.json')
        self.publication=dict(status='PUSHED',checks='PASS',commit='1'*40,remote_commit='1'*40,phase='results',
            queue_registration_sha256=qsha,runtime_source_bindings=self.bound,source_bindings=self.bound,
            published_files=self.bound,inputs=inputs)
        self.save_publication()

    def tearDown(self):self.temp.cleanup()

    def save_publication(self):
        c.write(self.out/'results_publication.json',self.publication)
        c.write(self.out/'completion.json',dict(self.launch,status='HISTORICAL_METRICS_COMPLETE',stop=True,
            training_updates=0,policy_selection_updates=0,publication=self.publication))

    def gate(self,reader=lambda pid:None):
        return c.phase2_recovery_gate(self.config,reader,{},self.original,self.oldlaunch)

    def test_all_twelve_and_81_plus_19_accept(self):
        value=self.gate();self.assertFalse(value['original_r3_complete_claimed'])

    def test_original_owner_must_exit(self):
        with self.assertRaises(c.Waiting):self.gate(lambda pid:dict(state='S',start_ticks='1') if pid==22 else None)

    def test_old_bootstrap_must_exit(self):
        with self.assertRaises(c.Waiting):self.gate(lambda pid:dict(state='S',start_ticks='1') if pid==23 else None)

    def test_new_owner_must_exit(self):
        with self.assertRaises(c.Waiting):self.gate(lambda pid:dict(state='S',start_ticks='2') if pid==33 else None)

    def test_partial_new_pure_proofs_fail(self):
        self.audit['new_pure_strict_grid_proofs']=284;c.write(self.out/'phase2_completion_audit.json',self.audit)
        self.publication['inputs'][str(self.out/'phase2_completion_audit.json')]=c.sha(self.out/'phase2_completion_audit.json')
        self.save_publication()
        with self.assertRaisesRegex(RuntimeError,r'81\+19'):self.gate()

    def test_shared_release_manifest_is_not_immutable_publication_artifact(self):
        path=self.root/'release_manifest.json';c.write(path,{})
        self.publication['published_files']={**self.bound,str(path):c.sha(path)};self.save_publication()
        with self.assertRaisesRegex(RuntimeError,'shared status/release'):self.gate()

    def test_unpushed_result_refused(self):
        self.publication['status']='COMMITTED';self.save_publication()
        with self.assertRaises(RuntimeError):self.gate()

    def test_missing_source_count_refused(self):
        path=self.out/'OPTIONAL_PHASE2_MAIN/completion.json';value=c.read(path);value['sources']=99;c.write(path,value)
        self.publication['inputs'][str(path)]=c.sha(path);self.save_publication()
        with self.assertRaises(RuntimeError):self.gate()


if __name__=='__main__':unittest.main()
