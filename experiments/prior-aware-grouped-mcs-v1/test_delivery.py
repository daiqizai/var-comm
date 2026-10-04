"""Synthetic CPU admission, receiver and paired-report contracts."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import evaluate
import source_quality as q
import tx_export
import score_actual as scorer
from actual_phy import accepted_state,indices_to_bits


def profile():
    return dict(N=1024,profile_id=0,G=2,m=5,K=1,j=4,groups=[
        dict(source_bits=360,phy_key='a'),dict(source_bits=312,phy_key='b')])


def received(accepts=(True,True),wrong=False):
    p=profile();tokens=np.arange(56,dtype=np.int64)
    if wrong:tokens[0]=2048
    payload=[indices_to_bits(tokens[:30]),indices_to_bits(tokens[30:])]
    event=dict(N=1024,header_ok=True,profile_id=0,groups=[dict(index=i,source_bits=g['source_bits'],phy_key=g['phy_key'],
        crc_accept=accepts[i],accepted_payload=payload[i].tolist() if accepts[i] else None) for i,g in enumerate(p['groups'])])
    event['state']=accepted_state(p,payload,accepts);event['state_sha256']=q.identity(event['state'])
    return event,dict(entries=[p])


class ReceiverTests(unittest.TestCase):
    def test_first_failure_gray_and_second_failure_accepted_prefix_only(self):
        event,codebook=received((False,True));self.assertEqual(scorer.actual_state(event,codebook)['kind'],'gray')
        event,codebook=received((True,False));result=scorer.actual_state(event,codebook)
        self.assertEqual((result['m'],result['K'],len(result['prefix'])),(4,0,4))
        self.assertEqual(result['partial_values'],[])

    def test_crc_accepted_wrong_values_remain_wrong_values(self):
        event,codebook=received(wrong=True);result=scorer.actual_state(event,codebook)
        self.assertEqual(result['prefix'][0],[2048]);self.assertEqual(result['partial_values'],[55])

    def test_rejected_hard_bits_or_repaired_state_are_rejected(self):
        event,codebook=received((True,False));event['groups'][1]['accepted_payload']=[0]*312
        with self.assertRaises(RuntimeError):scorer.actual_state(event,codebook)
        event,codebook=received(wrong=True);event['state']['prefix'][0][0]=0
        with self.assertRaises(RuntimeError):scorer.actual_state(event,codebook)

    def test_header_rejection_cannot_keep_a_body(self):
        event,codebook=received();event['header_ok']=False
        with self.assertRaises(RuntimeError):scorer.actual_state(event,codebook)
        event['groups']=[];event['state']=dict(kind='gray',prefix=[],partial_values=[],m=0,K=0,accepted_groups=0)
        event['state_sha256']=q.identity(event['state'])
        self.assertEqual(scorer.actual_state(event,codebook)['kind'],'gray')


class TXTests(unittest.TestCase):
    def test_exported_archive_accepts_only_required_tokens_and_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.npz';values=np.arange(680,dtype=np.int64)
            np.savez(path,tokens=values,entropy_order_m4=np.arange(25)[::-1])
            actual,orders=tx_export.validate_archive(path,[4],tokens=values)
            self.assertEqual(orders[4][0],24);np.testing.assert_array_equal(actual,values)
            with self.assertRaises(RuntimeError):tx_export.validate_archive(path,[])
            np.savez(path,tokens=values,label=np.array(2))
            with self.assertRaises(RuntimeError):tx_export.validate_archive(path,[])

    def test_screen_receipt_is_rejected_before_development_access(self):
        called=[];native=SimpleNamespace(loaded={'identity':{'a':1}},data=lambda _:called.append('data'))
        admission=dict(visual_identity={'a':1},required_ms=[],policies=dict(status='SCREEN_ONLY_NOT_DEPLOYABLE'))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):tx_export.export(native,admission,tmp,'unused','unused')
        self.assertEqual(called,[])

    def test_whole_only_export_is_label_free_and_manifest_matches_evaluator(self):
        class Tensor:
            def cpu(self):return self
            def numpy(self):return np.arange(680,dtype=np.int64)
        records=[dict(image_id='dev'+str(i),preprocessing_id='p',class_index=i) for i in range(100)]
        identity={'models':{'synthetic':'fixture'}}
        native=SimpleNamespace(loaded={'identity':identity},flags={},driver_bindings={},receiver=SimpleNamespace(ENTROPY_DECIMALS=6),
            data=lambda role:dict(records=records,T=[Tensor()]*100,bindings={}) if role=='development' else None,frozen=lambda:None)
        policies=dict(status='CALIBRATION_POLICIES_SELECTED',source_count=1000,stage='final',synthetic=False,
                      development_read=False,source_ids=['cal'+str(i) for i in range(1000)])
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);pp=tmp/'p.json';cb=tmp/'c.json';q.write(pp,policies);q.write(cb,dict(entries=[]))
            admitted=dict(visual_identity=identity,required_ms=[],policies=policies,codebook=dict(entries=[]),input_bindings={},
                          selected_policies_sha256=q.sha(pp),codebook_sha256=q.identity([]))
            result=tx_export.export(native,admitted,tmp/'out',pp,cb)
            self.assertEqual(len(evaluate.validate_tx_manifest(result,admitted)),100)
            self.assertEqual(result['required_ms'],[]);self.assertFalse(result['labels_given_to_receiver'])
            with np.load(result['sources'][0]['archive']) as archive:self.assertEqual(archive.files,['tokens'])
            before=q.sha(tmp/'out/manifest.json')
            tx_export.export(native,admitted,tmp/'out',pp,cb)
            self.assertEqual(q.sha(tmp/'out/manifest.json'),before)


class PairedTests(unittest.TestCase):
    def test_scorer_method_context_is_consumable_by_extension_gate(self):
        import optimize
        from test_optimize import gate_fixture
        rows,ids,policies,validations=gate_fixture()
        for row in rows:
            family=row.pop('family');row.pop('synthetic');row.pop('actual_bit_chain_executed')
            row.update(scorer.method_context(dict(synthetic=False,actual_bit_chain_executed=True),
                                             dict(family=family,stable_id=row['stable_id'])))
        self.assertTrue(optimize.development_extension_gate(rows,ids,policies,validations)['N2048_extension_allowed'])
        with self.assertRaises(RuntimeError):
            scorer.method_context(dict(synthetic=True,actual_bit_chain_executed=True),dict(family='B3',stable_id='B3'))

    def rows(self):
        return [dict(N=1024,snr_db=snr,source_index=i,noise_seed=seed,method=method,
                     dinov2_vitl14_cosine=.25+(method=='B3')*.1+i*.0001)
                for snr in scorer.SNRS for method in ('B0','B3','validation_candidate') for i in range(100) for seed in scorer.SEEDS]

    def test_neighbor_is_paired_and_repeated_noise_is_clustered_by_source(self):
        with patch.object(scorer,'METRICS',('dinov2_vitl14_cosine',)):
            summary,paired=scorer.analyze(self.rows())
        self.assertEqual(len(summary),12);self.assertEqual(len(paired),8)
        neighbor=[r for r in paired if r['method_B']=='validation_candidate']
        self.assertEqual(len(neighbor),4)
        for row in neighbor:
            self.assertAlmostEqual(row['mean'],.1);self.assertAlmostEqual(row['ci_low'],.1)
            self.assertEqual((row['sources'],row['frames']),(100,300))

    def test_missing_noise_frame_cannot_enter_summary(self):
        rows=self.rows();rows.pop()
        with patch.object(scorer,'METRICS',('dinov2_vitl14_cosine',)):
            with self.assertRaises(RuntimeError):scorer.analyze(rows)


if __name__=='__main__':unittest.main()
