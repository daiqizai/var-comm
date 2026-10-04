"""Synthetic CPU contracts only; no development files, learned models or GPU."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import evaluate as e
from actual_phy import accepted_state,indices_to_bits
from uep_common import SIZES,identity,sha,write,csv_write


def profile(pid=0,groups=1,k=0):
    m=4;j=4 if groups==2 else None;first=12*sum(s*s for s in SIZES[:4])
    bits=[first] if groups==1 else [first,12*k]
    if groups==1:bits[0]+=12*k
    return dict(profile_id=pid,stable_id='candidate'+str(pid),wire_key='wire'+str(pid),N=1024,G=groups,m=m,K=k,j=j,
        encoder_qualified=True,groups=[dict(source_bits=b,phy_key='phy'+str(i)) for i,b in enumerate(bits)])


def fixture(k=0):
    p=profile(k=k);codebook=dict(entries=[p]);labels={str(snr):[dict(family=f,status='FROZEN',profile_id=0,
        stable_id='candidate0',wire_key='wire0') for f in ('B0','B3','B4')] for snr in e.SNRS}
    return dict(codebook=codebook,labels_by_snr=labels,required_ms=[4] if k else [])


def event_from(p,payloads,accepts,counter):
    groups=[dict(index=i,crc_accept=accepts[i],accepted_payload=payloads[i].tolist() if accepts[i] else None,
        source_bits=p['groups'][i]['source_bits'],phy_key=p['groups'][i]['phy_key'],
        truth_correct_after_receiver=accepts[i],false_accept_after_receiver=False) for i in range(p['G'])]
    return dict(header_ok=True,profile_id=p['profile_id'],header_crc_ok=True,header_fields_legal=True,
        N=1024,frame_counter=counter,received_sha256='received',groups=groups,state=accepted_state(p,payloads,accepts),
        codebook_sha256=identity([p]))


class ReceiverTests(unittest.TestCase):
    def test_offline_truth_stripped_before_renderer(self):
        p=profile();payloads=[np.zeros(p['groups'][0]['source_bits'],dtype=np.uint8)]
        raw=event_from(p,payloads,[True],1);clean,offline=e.clean_receiver_event(raw,dict(header_correct=True))
        self.assertIn('truth_correct_after_receiver',raw['groups'][0]);self.assertNotIn('truth_correct_after_receiver',clean['groups'][0])
        self.assertTrue(offline['groups'][0]['truth_correct_after_receiver'])
        e.check_receiver_state(clean,dict(entries=[p]))

    def test_later_crc_accept_does_not_restore_failed_earlier_group(self):
        p=profile(groups=2,k=1);payloads=[np.zeros(g['source_bits'],dtype=np.uint8) for g in p['groups']]
        raw=event_from(p,payloads,[False,True],1);clean,_=e.clean_receiver_event(raw,{})
        e.check_receiver_state(clean,dict(entries=[p]));self.assertEqual(clean['state']['kind'],'gray')
        wrong=copy.deepcopy(clean);wrong['state']=accepted_state(p,payloads,[True,True])
        with self.assertRaises(RuntimeError):e.check_receiver_state(wrong,dict(entries=[p]))

    def test_true_label_or_tx_position_in_event_is_rejected(self):
        for field in ('class_index','true_label','tx_positions','source_id'):
            with self.assertRaises(RuntimeError):e.clean_receiver_event(dict(groups=[],**{field:0}),{})

    def test_rejected_hard_bits_are_prohibited(self):
        p=profile();payloads=[np.zeros(p['groups'][0]['source_bits'],dtype=np.uint8)]
        raw=event_from(p,payloads,[False],1);raw['groups'][0]['accepted_payload']=payloads[0].tolist()
        clean,_=e.clean_receiver_event(raw,{})
        with self.assertRaises(RuntimeError):e.check_receiver_state(clean,dict(entries=[p]))


class TXTests(unittest.TestCase):
    def test_npz_contains_true_tokens_and_only_required_entropy_orders(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tx.npz';tokens=np.arange(sum(s*s for s in SIZES),dtype=np.uint16)
            np.savez(path,tokens=tokens,entropy_order_m4=np.arange(25,dtype=np.int64)[::-1])
            record=dict(archive=str(path),archive_sha256=sha(path));scales,orders=e.load_tx_source(record,[4])
            self.assertEqual([len(s) for s in scales],[s*s for s in SIZES]);self.assertEqual(orders[4][0],24)
            with self.assertRaises(RuntimeError):e.load_tx_source(record,[])

    def test_tx_extra_label_and_invalid_order_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tx.npz';tokens=np.zeros(sum(s*s for s in SIZES),dtype=np.uint16)
            np.savez(path,tokens=tokens,label=np.array(5))
            with self.assertRaises(RuntimeError):e.load_tx_source(dict(archive=str(path),archive_sha256=sha(path)),[])
            np.savez(path,tokens=tokens,entropy_order_m4=np.zeros(25,dtype=np.int64))
            with self.assertRaises(RuntimeError):e.load_tx_source(dict(archive=str(path),archive_sha256=sha(path)),[4])

    def test_same_wire_methods_share_exactly_one_physical_frame(self):
        schedule=e.schedule_for_source(dict(source_index=95,source_id='original95'),fixture()['labels_by_snr'])
        self.assertEqual(len(schedule),12);self.assertEqual(sum(len(x['methods']) for x in schedule),36)
        self.assertEqual({x['spec']['snr_db'] for x in schedule},{4,7,10,13})
        self.assertEqual({x['spec']['noise_seed'] for x in schedule},{2001,2002,2003})
        self.assertEqual(len({x['spec']['frame_counter'] for x in schedule}),12)


class RunnerTests(unittest.TestCase):
    def test_safe_resume_uses_actual_events_and_never_regenerates_completed_wire(self):
        calls=[];frozen=fixture(k=1);p=frozen['codebook']['entries'][0]
        class FakeLink:
            def execute(self,pid,payloads,N,snr,counter,seed,source):
                calls.append((snr,seed));event=event_from(p,payloads,[True],counter)
                return event,dict(N=1024,E=2048,waveform_sha256='wave'),dict(header_correct=True,header_false_accept=False,undetected_body_errors=0,noise_sha256='noise')
        with tempfile.TemporaryDirectory() as folder:
            tx=Path(folder)/'tx.npz';np.savez(tx,tokens=np.arange(sum(s*s for s in SIZES),dtype=np.uint16),entropy_order_m4=np.arange(25)[::-1])
            record=dict(source_index=95,source_id='original95',archive=str(tx),archive_sha256=sha(tx))
            with self.assertRaises(e.StopRequested):e.run_source(record,frozen,FakeLink(),folder,'binding',stop=lambda:len(calls)>=1,synthetic=True)
            cp=e.run_source(record,frozen,FakeLink(),folder,'binding',synthetic=True)
            self.assertEqual(len(calls),12);self.assertTrue(cp['complete']);self.assertTrue(cp['synthetic'])
            self.assertEqual(cp['frames'][0]['receiver_event']['state']['partial_values'],[int(sum(s*s for s in SIZES[:4])+24)])
            self.assertTrue(all(not f['actual_bit_chain_executed'] for f in cp['frames']))
            e.run_source(record,frozen,FakeLink(),folder,'binding',synthetic=True);self.assertEqual(len(calls),12)
            with self.assertRaises(RuntimeError):e.run_source(record,frozen,FakeLink(),folder,'binding')

    def test_nonfrozen_or_incomplete_calibration_policy_rejected(self):
        for change in (dict(status='SCREEN_ONLY_NOT_DEPLOYABLE'),dict(source_count=300),dict(development_read=True)):
            value=dict(status='CALIBRATION_POLICIES_SELECTED',synthetic=False,stage='final',source_count=1000,
                development_read=False,source_ids=['cal'+str(i) for i in range(1000)],cells=[])
            value.update(change)
            with self.assertRaises(RuntimeError):e.selected_schedule(value)


class AdmissionTests(unittest.TestCase):
    def bundle(self,folder):
        from profiles import freeze_codebook
        folder=Path(folder);ids=['synthetic_cal'+str(i) for i in range(1000)]
        p=profile();choice=dict(stable_id=p['stable_id'],profile=p)
        selected=dict(status='CALIBRATION_SELECTED',selected=choice)
        cells=[dict(N=1024,snr_db=snr,ready_for_actual_link=True,probability_refinement_complete=True,
                    independent_optima={f:copy.deepcopy(selected) for f in ('B0','B1','B2','B3','B4')},
                    matched_to_B3={f:dict(status='NOT_APPLICABLE') for f in ('B1','B2','B3')},
                    strong_baseline=choice,frozen_distinct_validation_candidate=dict(status='FROZEN',selected=choice)) for snr in e.SNRS]
        quality=dict(population='calibration',source_ids=ids,development_read=False,
                     rows=[dict(source_id=i,state_id='gray',receiver='gray',dinov2_vitl14_cosine=.1) for i in ids])
        bler=dict(status='REFINEMENT_COMPLETE',synthetic=False,rows=[dict(phy_key='phy0',snr_db=s,n_blocks=20000,n_correct=20000) for s in e.SNRS])
        policies=dict(status='CALIBRATION_POLICIES_SELECTED',synthetic=False,stage='final',source_count=1000,
            development_read=False,source_ids=ids,cells=cells,quality_input_sha256=identity(quality),bler_input_sha256=identity(bler),
            refinement_requirements=[dict(phy_key='phy0',snr_db=s) for s in e.SNRS])
        table=folder/'quality_per_source.csv';csv_write(table,quality['rows'])
        reg=dict(source_count=1000,source_ids=ids,development_read=False,renderer_identity=dict(frozen_identity={'synthetic_model':'fixture'}))
        write(folder/'registration.json',reg)
        completion=dict(status='SOURCE_QUALITY_COMPLETE',population='calibration',stage='final1000',sources=1000,
            source_ids=ids,development_read=False,metric='dinov2_vitl14_cosine',registration_sha256=sha(folder/'registration.json'),outputs={str(table):sha(table)})
        objects=[policies,quality,completion,bler,freeze_codebook([p])];paths=[folder/name for name in ('policies.json','quality.json','completion.json','bler.json','codebook.json')]
        for path,value in zip(paths,objects):write(path,value)
        return paths,objects

    def test_completed_tables_and_exact_policy_codebook_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            paths,objects=self.bundle(folder);frozen=e.admit_frozen(*paths)
            self.assertEqual(frozen['required_ms'],[]);self.assertEqual(set(frozen['labels_by_snr']),{'4','7','10','13'})
            self.assertIn(str(Path(folder)/'quality_per_source.csv'),frozen['input_bindings'])
            objects[4]['candidate_to_profile']['injected']=0;write(paths[4],objects[4])
            with self.assertRaises(RuntimeError):e.admit_frozen(*paths)

    def test_changed_quality_csv_or_premature_refinement_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            paths,objects=self.bundle(folder)
            table=Path(folder)/'quality_per_source.csv';rows=copy.deepcopy(objects[1]['rows']);rows[0]['dinov2_vitl14_cosine']=.2
            csv_write(table,rows);objects[2]['outputs'][str(table)]=sha(table);write(paths[2],objects[2])
            with self.assertRaises(RuntimeError):e.admit_frozen(*paths)
            paths,objects=self.bundle(folder);objects[3]['rows'][0].update(n_blocks=256,n_correct=256)
            objects[0]['bler_input_sha256']=identity(objects[3]);write(paths[0],objects[0]);write(paths[3],objects[3])
            with self.assertRaises(RuntimeError):e.admit_frozen(*paths)


if __name__=='__main__':unittest.main()
