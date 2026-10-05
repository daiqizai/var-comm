from pathlib import Path
from unittest.mock import patch
import copy
import hashlib
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import register_header_reuse as r


def checkpoint(snr):
    batches=[];start=0
    while start<20000:
        n=min(128,20000-start)
        batches.append(dict(start=start,n_blocks=n,n_correct=n,n_reject=0,n_undetected=0,
            event_bits_sha256='a'*64,energy_sum=136.*n,energy_sum_squares=136.**2*n,
            energy_min=136.,energy_max=136.))
        start+=n
    value=dict(synthetic=False,binding='shardzero',point=dict(phy_key=r.HEADER_KEY,snr_db=snr),batches=batches,
               counts=dict(n_blocks=20000,n_correct=20000,n_reject=0,n_undetected=0))
    value['payload_sha256']=r.digest(value)
    return value


def row(snr,cp):
    return dict(phy_key=r.HEADER_KEY,snr_db=snr,source_id='*',kind='header',applicability='source_independent_verified',
        **cp['counts'],p_correct=1.,p_reject=0.,p_undetected=0.,
        p_correct_ci=[r.wilson(20000,20000)[0],.9999999999999999],
        p_reject_ci=r.wilson(0,20000),p_undetected_ci=r.wilson(0,20000),block_errors=0,
        zero_observed_errors_is_not_true_BLER_zero=True,actual_energy_mean=136.,actual_energy_sd=0.,
        observed_average_energy_per_complex_symbol=2.,physical_configuration=dict(phy_key=r.HEADER_KEY,
        kind='header',source_bits=12,crc_bits=16,tail_bits=6,symbols=68,transmitted_bits=136,q=2,
        modulation='QPSK',profile_id_range=[0,4095]),point_checkpoint_sha256=cp['payload_sha256'])


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf-8')


class Tests(unittest.TestCase):
    def test_exact_counts_and_contiguous_actual_batches(self):
        cp=checkpoint(13);rr=row(13,cp)
        r.validate_header_row(rr,13)
        proof=r.validate_checkpoint(cp,13,rr)
        self.assertEqual(proof['verified_packet_count'],20000)
        for change in ('synthetic','counts','gap'):
            altered=copy.deepcopy(cp)
            if change=='synthetic':altered['synthetic']=True
            elif change=='counts':altered['counts']['n_correct']-=1
            else:altered['batches'][1]['start']+=1
            altered['payload_sha256']=r.digest({k:v for k,v in altered.items() if k!='payload_sha256'})
            changed_row=dict(rr,point_checkpoint_sha256=altered['payload_sha256'])
            with self.assertRaises(RuntimeError):r.validate_checkpoint(altered,13,changed_row)
    def test_table_probability_roundoff_is_preserved_but_real_difference_rejected(self):
        cp=checkpoint(19);rr=row(19,cp)
        r.validate_header_row(rr,19)
        self.assertEqual(rr['p_correct_ci'][1],.9999999999999999)
        altered=copy.deepcopy(rr);altered['p_undetected_ci'][1]=0
        with self.assertRaises(RuntimeError):r.validate_header_row(altered,19)
    def test_reuse_contract_does_not_claim_U_zero_or_16dB_measurements(self):
        cp=checkpoint(13);rr=row(13,cp)
        value=r.point_contract(13,rr,{})
        self.assertFalse(value['new_finite_codebook_U_exact_reuse'])
        self.assertIsNone(value['new_finite_codebook_U_probability'])
        self.assertGreater(value['new_finite_codebook_U_upper_envelope'],0)
        self.assertEqual(value['p_header_correct_point'],1)
        self.assertLess(value['p_header_correct_wilson95'][0],1)
    def test_single_new_receipt_is_idempotent_but_not_overwritable(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'HEADER_PROBABILITY_REUSE.json'
            self.assertEqual(r.seal_new(path,{'status':'TEST'}),'CREATED')
            before=path.read_bytes()
            self.assertEqual(r.seal_new(path,{'status':'TEST'}),'VERIFIED_EXISTING')
            with self.assertRaises(RuntimeError):r.seal_new(path,{'status':'CHANGED'})
            self.assertEqual(before,path.read_bytes())
    def test_end_to_end_read_only_binding_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bindings={}
            for relative in r.SOURCES:
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(relative,encoding='utf-8')
                bindings[str(path)]=r.sha(path)
            reg=root/'initial_registration.json';save(reg,dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',source_bindings=bindings))
            base=root/'outputs/PRIOR-AWARE-UEP-20261004-V1'
            points={};rows=[];cpbind={}
            for snr in (13,19):
                cp=checkpoint(snr);rows.append(row(snr,cp));dirname=str(snr)
                for name in ('checkpoint.json','refined.json'):
                    path=base/'phy_shards/0/points'/dirname/name;save(path,cp);cpbind[str(path)]=r.sha(path)
                points[snr]=(dirname,r.sha(path))
            merge={'shards':[{'shard_index':0,'registration_sha256':'shardzero'}]}
            table=base/'bler_refined_02.json';save(table,dict(status='REFINEMENT_COMPLETE',synthetic=False,
                version='UEP-ACTUAL-LDPC-HEADER-BLER-V1',merge_registration=merge,registration_sha256=r.digest(merge),rows=rows,checkpoint_bindings=cpbind))
            originals={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch.object(r,'TABLE_SHA',r.sha(table)),patch.object(r,'CPP_SHA',r.sha(root/r.SOURCES[2])),patch.object(r,'POINTS',points):
                value=r.build(root,reg,lambda ref,relative:(root/relative).read_bytes())
                self.assertEqual(value['new_packet_decodes'],0)
                self.assertEqual(value['diagnostic_16dB']['status'],'NOT_MEASURED')
                self.assertEqual(len(value['points']),2)
                self.assertEqual(len(value['input_bindings']),6)
                with self.assertRaisesRegex(RuntimeError,'source differs'):
                    r.build(root,reg,lambda ref,relative:b'changed')
            self.assertEqual(originals,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})


if __name__=='__main__':unittest.main()
