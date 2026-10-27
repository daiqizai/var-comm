from pathlib import Path
import sys,unittest,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import asha
from c_predclass_protocol import container,generation_seed
from c_predclass_frame import FAMILY,NULL,source_projection
from plan import received_input,build_plan


CONTRACT='a'*64
def frame(key,index=2,cl=5,noise=4101,prefix=3,bits=(0,1,1,0)):
    a=np.array(bits,np.uint8);payload=container(cl,a).tolist()
    rx=dict(header=dict(header_ok=True),body=dict(crc_accepted=True,parser_accepted=True,payload=payload),rx_profile=dict(family=FAMILY,m=prefix,K=0),status='ACCEPTED')
    value=dict(event=dict(key=key,source_index=index,source_id='synthetic_%d'%index,snr_db=4,candidate_id='p256_q2_r1-2',noise_seed=noise),actual_RX=rx,source_projection=source_projection(rx))
    return dict(descriptor=dict(path='/synthetic/'+key+'.json',sha256='b'*64,bytes=42),value=value)
def cached(row):
    p=row['value']['source_projection'];inp=received_input(p)
    return dict(contract_sha256=CONTRACT,input=inp,output=dict(key=inp['key'],prefix=inp['prefix'],received_class=p['received_class'],payload_sha256=inp['payload_sha256'],generation_seed=generation_seed(np.array(inp['bits'],np.uint8),inp['prefix'],p['received_class']),pixel_sha256='c'*64,image=dict(path='/synthetic/closed.npz',sha256='d'*64,bytes=64),source_truth_used=False,TX_CDF_used=False))


class Admission(unittest.TestCase):
    def test_all_logical_noise_events_retain_one_unique_input(self):
        rows=[frame(str(i),noise=4101+i)for i in range(3)]
        p=build_plan(rows,[],CONTRACT,3)
        self.assertEqual(p['counts']['logical_events'],3);self.assertEqual(p['counts']['new_unique'],1)
    def test_exact_old_output_reused(self):
        rows=[frame(str(i),noise=4101+i)for i in range(3)]
        p=build_plan(rows,[cached(rows[0])],CONTRACT,3)
        self.assertEqual(p['counts']['reused_logical'],3);self.assertEqual(p['counts']['new_unique'],0)
    def test_class_prefix_or_bits_changes_prevent_reuse(self):
        base=frame('a');rows=[base,frame('b',cl=6),frame('c',prefix=4),frame('d',bits=(1,0,0,1))]
        p=build_plan(rows,[cached(base)],CONTRACT,4)
        self.assertEqual(p['counts']['new_unique'],3);self.assertEqual(p['counts']['reused_unique'],1)
    def test_contract_mismatch_refuses_cache(self):
        f=frame('a');c=cached(f);c['contract_sha256']='d'*64
        with self.assertRaises(AssertionError):build_plan([f],[c],CONTRACT,1)
    def test_cache_class_or_generation_mismatch_refused(self):
        f=frame('a')
        for field in ('received_class','generation_seed'):
            c=cached(f);c['output'][field]+=1
            with self.assertRaises(AssertionError):build_plan([f],[c],CONTRACT,1)
    def test_header_reject_gray_stays_in_denominator(self):
        f=frame('a');v=f['value'];v['actual_RX']['header']['header_ok']=False;v['actual_RX']['status']='HEADER_REJECT';v['source_projection']=source_projection(v['actual_RX'])
        p=build_plan([f],[],CONTRACT,1)
        self.assertEqual(p['counts']['gray_events'],1);self.assertEqual(p['counts']['new_unique'],0)
    def test_reserved_paid_class_gray(self):
        f=frame('a');v=f['value'];v['actual_RX']['body']['payload'][:10]=[1]*10;v['source_projection']=source_projection(v['actual_RX'])
        p=build_plan([f],[],CONTRACT,1)
        self.assertEqual(p['logical'][0]['reason'],'CLASS_CONTAINER_REJECT')
    def test_actual_accepted_different_from_TX_not_corrected(self):
        f=frame('a',cl=9,bits=(1,1,0,1));v=f['value'];v.update(TX_expected_class=3,TX_expected_prefix=24,post_output_profile_and_payload_exact=False)
        p=build_plan([f],[],CONTRACT,1);i=p['shards'][0][0]
        self.assertEqual(i,received_input(v['source_projection']))
        self.assertNotIn('source_id',i);self.assertNotIn('TX_expected_class',i)
    def test_deterministic_shards_original_source_identity(self):
        rows=[frame(str(i),index=900-i,cl=i)for i in range(5)]
        a=build_plan(rows,[],CONTRACT,5,2);b=build_plan(list(reversed(rows)),[],CONTRACT,5,2)
        self.assertEqual(a,b);self.assertEqual([len(v)for v in a['shards']],[2,2,1])
        self.assertEqual([v['source_index']for v in a['logical']],[896,897,898,899,900])
    def test_duplicate_event_or_missing_denominator_refused(self):
        f=frame('a')
        with self.assertRaises(AssertionError):build_plan([f,f],[],CONTRACT,2)
        with self.assertRaises(AssertionError):build_plan([f],[],CONTRACT,2)
    def test_stale_projection_refused(self):
        f=frame('a');f['value']['source_projection']['received_class']=99
        with self.assertRaises(AssertionError):build_plan([f],[],CONTRACT,1)
    def test_model_modules_absent(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
