"""Small, synthetic metadata checks. Does not render or access production data."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import report as r


def tables():
    ids=['s'+str(i) for i in range(100)]
    frames=[];sources=[];means=[];paired=[]
    for method in r.METHODS:
        for sid in ids:
            for snr in (4,10):
                for noise in (6201,6202,6203):
                    frames.append(dict(method=method,source_id=sid,snr_db=str(snr),noise_seed=str(noise),
                                       **{m[0]:'0.5' for m in r.METRICS}))
                for metric,_,_,_ in r.METRICS:
                    sources.append(dict(method=method,source_id=sid,snr_db=str(snr),metric=metric,
                                        source_mean='0.5',noise_count='3'))
        for snr in (4,10):
            for metric,_,_,direction in r.METRICS:
                means.append(dict(method=method,snr_db=str(snr),metric=metric,direction=direction,
                                  source_count='100',mean='0.5',ci_low='0.4',ci_high='0.6'))
    for snr in (4,10):
        for metric,_,_,direction in r.METRICS:
            paired.append(dict(comparison='NeST_adaptive_minus_ARPC_adaptive',snr_db=str(snr),
                metric=metric,direction=direction,source_count='100',mean='-0.05',ci_low='-0.2',ci_high='0.1',
                interval_crosses_or_touches_zero='True'))
    return frames,sources,means,paired,ids,r.METHODS,(4,10)


class MetadataTests(unittest.TestCase):
    def test_transport_fields_and_false_accept_are_not_correct_delivery(self):
        frames=[];pixels=[];fallback=[];choices=[]
        geometry=dict(total_complex_symbols=2048,header_symbols_per_fragment=68,
            length_bits_per_fragment=13,crc_bits_per_fragment=16,reassembly_bits_per_fragment=96,
            object_bits=1600,fragment_bytes=100,q=2,nominal_rate='1/2',n=1912,k=956)
        for codec in (0,1):
            for snr in (4,10):
                policy=f'p{codec}_{snr}';choices.append(dict(codec=codec,snr_db=snr,policy_id=policy))
                for i in range(100):
                    fb=dict(source_id=f's{i}',codec=codec,snr_db=snr,policy_id=policy,
                            actual_endpoint=4,object_bits=1001,capacity_bits=1600)
                    fallback.append(fb)
                    for noise in (6201,6202,6203):
                        key=f'{codec}_{snr}_{i}_{noise}'
                        wrong=codec==0 and snr==4 and i==0 and noise==6201
                        f=dict(key=key,codec=codec,snr_db=snr,policy_id=policy,source_id=f's{i}',source_index=i,
                               noise_seed=noise,layout_id='g',target=7,sent_endpoint=4,object_bits=1001,
                               source_no_fit=False,N=2048,nominal_Es=2,source_input_key=key,
                               actual_received_codec=codec,actual_received_endpoint=4,accepted=True,
                               exact_post_exit=not wrong,accepted_wrong=wrong,failure_reason='',
                               header_statuses=['HEADER_OK']*2,fragment_statuses=['BODY_OK']*2)
                        frames.append(f)
                        pixels.append(dict(f,receiver_status='SOURCE_DECODED',accepted_wrong_post_exit=wrong,image_float_sha256='a'*64))
        out=r.enrich_frames(frames,pixels,fallback,{'g':geometry},choices)
        self.assertEqual(out[0]['outcome'],'ACCEPTED_WRONG_SOURCE')
        self.assertFalse(out[0]['correct_delivery'])
        self.assertEqual(out[0]['source_byte_padding_bits'],7)
        self.assertEqual((out[0]['total_length_bits'],out[0]['total_crc_bits'],out[0]['total_reassembly_bits']),(26,32,192))
        summary,_=r.aggregate_transport(out)
        self.assertEqual(sum(x['correct_delivery'] for x in summary),1199)

    def test_live_or_cross_host_pid_is_not_silently_accepted(self):
        with patch.object(r.os,'uname',lambda:SimpleNamespace(nodename='local-synthetic'),create=True):
            with self.assertRaises(ValueError):r.gone(dict(host='not-this-host',pid=1,start_ticks=1))

    def test_complete_grid_to_400_preserves_saved_values(self):
        out=r.validate_metric_tables(*tables())
        self.assertEqual(len(out),400)
        self.assertEqual(out[0]['dreamsim'],'0.5')

    def test_missing_failure_frame_fails(self):
        args=list(tables());args[0].pop()
        with self.assertRaises(ValueError):r.validate_metric_tables(*args)

    def test_duplicate_noise_fails(self):
        args=list(tables());args[0][1]=dict(args[0][0])
        with self.assertRaises(ValueError):r.validate_metric_tables(*args)

    def test_old_noise_fails(self):
        args=list(tables());args[0][0]['noise_seed']='4101'
        with self.assertRaises(ValueError):r.validate_metric_tables(*args)

    def test_negative_and_cross_zero_retained(self):
        args=tables();r.validate_metric_tables(*args)
        self.assertEqual(r.format_ci(args[3][0]),'-0.0500 [-0.2000, 0.1000]')

    def test_zero_flag_mismatch_fails(self):
        args=tables();args[3][0]['interval_crosses_or_touches_zero']='False'
        with self.assertRaises(ValueError):r.validate_metric_tables(*args)

    def test_nan_score_fails(self):
        args=tables();args[0][0]['dists']='nan'
        with self.assertRaises(ValueError):r.validate_metric_tables(*args)

    def test_boolean_exit_is_not_actual_zero(self):
        with self.assertRaises(ValueError):r.successful(dict(returncode=False))
        r.successful(dict(returncode=0,timed_out=False,terminated=False,killed=False,metadata_error=None))

    def test_timeout_even_exit_zero_fails(self):
        with self.assertRaises(ValueError):r.successful(dict(returncode=0,timed_out=True))

    def test_latex_has_two_backslashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'table.tex';r.latex_table(p,tables()[2],r.METHODS,'Synthetic only')
            rows=[x for x in p.read_text().splitlines() if 'dB &' in x]
            self.assertEqual(len(rows),28)
            self.assertTrue(all(x.endswith('\\\\') for x in rows))

    def test_selection_consumes_all_nine_without_reselecting(self):
        means=[dict(codec=c,snr_db=s,policy_id=p,calibration_mean_DINO=.5,
                    source_mean_count=1000,noise_per_source=3) for c,s,p in r.B2_CELLS]
        selected=[dict(**{k:x[k] for k in ('codec','snr_db','policy_id','calibration_mean_DINO')},tie_break='Saved rule')
                  for x in [means[0],means[2],means[7],means[8]]]
        s=dict(schema='N2048_B2_DINO_FOUR_POLICIES_V1',status='FOUR_POLICIES_FROZEN',metric='dinov2_vitl14_cosine',
               calibration_bootstrap_calls=0,no_additional_candidates=True,candidate_means=means,rows=selected)
        self.assertEqual(r.validate_selection(s),(means,selected))
        s['candidate_means']=means[:-1]
        with self.assertRaises(ValueError):r.validate_selection(s)

    def test_report_has_no_scientific_import_or_sampling_api(self):
        tree=ast.parse(Path(r.__file__).read_text(encoding='utf8'))
        imports=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):imports += [x.name.split('.')[0] for x in n.names]
            if isinstance(n,ast.ImportFrom):imports.append(n.module.split('.')[0])
            if isinstance(n,ast.Call):
                name=ast.unparse(n.func)
                self.assertFalse(any(x in name for x in ('default_rng','bootstrap','load_model','transmit','random.')))
        self.assertFalse(set(imports)&{'torch','transformers','timm','diffusers','scipy','numpy','shared','plan','native','statistical','worker'})


if __name__=='__main__':
    unittest.main()
