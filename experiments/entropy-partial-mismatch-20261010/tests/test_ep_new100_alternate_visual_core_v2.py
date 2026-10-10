import contextlib,copy,hashlib,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ep_new100_alternate_visual_core_v2 as v


class AlternateCoreTests(unittest.TestCase):
    def test_post_RX_reference_read_and_CDF_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'tokens.npz';tokens=np.arange(680,dtype=np.int64);np.savez(path,tokens=tokens)
            witness=dict(m=4,K=0,source_assets=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),source_metadata={})
            evidence=dict(kind='tokens',source_decode_called=True,render_called=True,transmitted_CDF_used=False,comparison_truth_used=False,
                received_tokens=tokens[:30].tolist(),actual_RX_CDF_trace=[dict(scale=i,cdf_sha256=str(i)) for i in range(4)])
            order=[]
            def write(p,value):order.append(('write',value['status']));Path(p).write_text(json.dumps(value))
            def read(pin):order.append(('expected_CDF_read',));return dict(CDF_trace=[dict(scale=i,cdf_sha256=str(i)) for i in range(10)])
            proof=v.assert_after_RX(witness,evidence,read,Path,'image',write,td)
            self.assertEqual(order[0],('write','ACTUAL_RX_COMPLETED_BEFORE_EXPECTED_TOKEN_READ'))
            self.assertEqual(order[1][0],'expected_CDF_read')
            self.assertEqual(json.loads(Path(proof['path']).read_text())['status'],'NORMAL_FRAME_RELOCATION_WITNESS_PASS')
            evidence['actual_RX_CDF_trace'][0]['cdf_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'token/CDF mismatch'):
                v.assert_after_RX(witness,evidence,read,Path,'image',write,td)
            self.assertEqual(order[-1][1],'NORMAL_FRAME_RELOCATION_WITNESS_MISMATCH')

    def test_source_parse_failure_reads_no_expected_tokens(self):
        with tempfile.TemporaryDirectory() as td,patch.object(np,'load',side_effect=AssertionError('expected source read too early')):
            e=dict(source_decode_called=True,kind='gray',render_called=False)
            with self.assertRaisesRegex(ValueError,'incompatible'):
                v.assert_after_RX({},e,lambda p:self.fail('metadata oracle'),Path,'gray',lambda p,x:None,td)

    def test_primed_normal_frame_counts_once_without_second_decode(self):
        counts=dict(caps=v.CAPS,completed=dict(model_load=1,encoder=0,source_tx=0,source_rx=1,var_render=1,prior_scale=14,decoder_forward=1),unresolved=0)
        counts['reserved']=dict(counts['completed'])
        evidence=dict(kind='tokens',source_decode_called=True,source_status='ARITHMETIC_SOURCE_DECODED')
        recon=dict(evidence=evidence,image_archive={},image_sha256='same',actual_source_input_key='key')
        rows=[dict(frame_index=i,provider_group='whole',source_index=0,physical_frame={'path':'actual'}) for i in range(3600)]
        r=dict(logical_rows=rows,logical_events=[{}]*3600,historical_image_reuse_admitted=0,caps=v.CAPS,raw_binding={},providers={},visual_identity={})
        status=dict(actual_RX_status='ACCEPT',fixed_gray=False)
        seen=[]
        g=types.SimpleNamespace(write=lambda p,result:seen.append(result),inside=Path,image_sha=lambda a:'image')
        with tempfile.TemporaryDirectory() as td,patch.object(v.original.confirmation,'complete_grid'),\
             patch.object(v.original.raw,'parser',return_value=(None,None)),patch.object(v.original,'validate_packet',return_value={}),\
             patch.object(v.original,'input_and_status',side_effect=lambda *a:('key',None,dict(status))),\
             patch.object(v.original,'reconstruct',side_effect=AssertionError('normal frame decoded twice')),\
             patch.object(v.original.pilot,'descriptor',side_effect=lambda p:dict(path=str(p))),\
             patch.object(v,'prime',return_value=({'key':recon},'key')) as prime:
            result=v.images(r,None,None,None,types.SimpleNamespace(summary=lambda:counts),lambda:None,Path(td)/'out',g,lambda p:{})
        prime.assert_called_once();self.assertEqual(result['reuse'],dict(actual_new_inputs=1,exact_same_input=3599))
        self.assertEqual(len(seen),3600);self.assertEqual(result['extra_qualification_calls'],0)

    def test_prime_passes_actual_RX_only_then_checks(self):
        events=[];row=dict(frame_index=0,provider_group='whole',source_index=3,physical_frame={'path':'actual'})
        r=dict(relocation_witness=dict(frame_index=0),logical_rows=[row],providers={},visual_identity={})
        image=np.zeros((3,256,256),np.float32);backend=types.SimpleNamespace(source_index=None)
        def reconstruction(*args):events.append('actual_RX_and_render');self.assertIs(args[0],rx);return image,{'actual':True}
        rx={'only':'received'}
        with tempfile.TemporaryDirectory() as td,patch.object(v.original,'validate_packet',return_value=rx),\
             patch.object(v.original,'input_and_status',return_value=('key',None,{})),\
             patch.object(v.original,'reconstruct',side_effect=reconstruction),\
             patch.object(v,'assert_after_RX',side_effect=lambda *a:events.append('external_expected_check') or {}):
            (Path(td)/'images').mkdir();g=types.SimpleNamespace(inside=Path,image_sha=lambda x:'image',write=lambda *a:None)
            cache,key=v.prime(r,backend,None,None,None,lambda:None,td,g,lambda p:{})
        self.assertEqual(events,['actual_RX_and_render','external_expected_check']);self.assertEqual(key,'key');self.assertEqual(backend.source_index,3)

    def test_failed_prime_does_not_start_any_later_decode(self):
        rows=[dict(frame_index=i,provider_group='whole',source_index=0,physical_frame={}) for i in range(3600)]
        r=dict(logical_rows=rows,logical_events=[{}]*3600,historical_image_reuse_admitted=0,caps=v.CAPS,raw_binding={})
        with tempfile.TemporaryDirectory() as td,patch.object(v.original.confirmation,'complete_grid'),\
             patch.object(v.original.raw,'parser',return_value=(None,None)),\
             patch.object(v,'prime',side_effect=ValueError('paid first CDF mismatch')) as prime,\
             patch.object(v.original,'reconstruct',side_effect=AssertionError('second scientific call')) as later:
            with self.assertRaisesRegex(ValueError,'paid first CDF mismatch'):
                v.images(r,None,None,None,None,lambda:None,Path(td)/'out',None,None)
        prime.assert_called_once();later.assert_not_called()


if __name__=='__main__':unittest.main()
