"""Read-only reporting interface checks. No model/channel/bootstrap calls."""
import argparse,csv,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import t1_final_report as report
import t1_readonly_frame_join as join
import t4_summarize as t4
import t2_pilot as h

class Checks(unittest.TestCase):
    def test_actual_t4_long_format_all_only(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);freeze={'path':'sealed_policy','sha256':'same'};req=p/'request.json';h.save(req,dict(frozen_policy=freeze))
            rows=[]
            for method in ('RAW64_WHOLE','RAW64_PARTIAL',*report.FAMILIES):
                for s in report.SNRS:
                    for condition in ('all','non_gray_output','gray_fallback'):
                        for metric in ('TX_seconds','RX_seconds','software_e2e_excluding_channel_seconds'):
                            rows.append(dict(method=method,snr_db=s,condition=condition,metric=metric,sample_count=48 if condition=='all'else 2,
                                source_count=16 if condition=='all'else 1,mean=.25 if condition=='all'else 99.,median=.2,p95=.4))
            csvpath=p/'timing_summary.csv';report.csv_write(csvpath,rows)
            done=p/'completion.json';h.save(done,dict(status='T4_FIXED16_ONLINE_TIMING_COMPLETE',total_frames=1152,measured_frames=576,
                source_count=16,no_online_tokens_or_output_cache=True,packet_budget={'unresolved_frames':0},request_path=str(req),request_sha256=h.sha(req),
                outputs={str(csvpath):h.sha(csvpath)}))
            result=report.read_timing(SimpleNamespace(timing_csv=str(csvpath),timing_completion=str(done)),freeze)
            self.assertEqual(len(result),12);self.assertEqual(1000*float(result[report.RAW_REFS[1],19]['TX_seconds']['mean']),250.)
            with self.assertRaises(FileNotFoundError):
                report.read_timing(SimpleNamespace(timing_csv=str(csvpath),timing_completion=str(done)),{'different':True})

    def test_timing_projection_proves_original_rank1_policy(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);rows=[];policies={f:{}for f in report.FAMILIES}
            for f in report.FAMILIES:
                for s in report.SNRS:
                    c=dict(candidate_id='m9_q6_r5-6',target_m=9,q=6,nominal_rate='5/6');policies[f][str(s)]=c
                    rows.append(dict(family=f,snr_db=s,rank=1,**c))
            rankings=p/'calibration_rankings.csv';report.csv_write(rankings,rows)
            calibration=p/'calibration.json';h.save(calibration,dict(status='T1_FULL_CALIBRATION_COMPLETE',source_count=1000,noise_count=3,holdout_used=False,outputs={str(rankings):h.sha(rankings)}))
            cp=h.desc(calibration);original=p/'original.json';h.save(original,dict(policies=policies,input_bindings={cp['path']:cp['sha256']}));freeze=h.desc(original)
            projection=p/'projection.json';h.save(projection,dict(original_policy_freeze=freeze,calibration_completion=cp,rows=rows))
            req=p/'request.json';h.save(req,dict(frozen_policy=h.desc(projection),calibration_completion=cp));csvpath=p/'timing_summary.csv'
            timing=[dict(method=m,snr_db=s,condition='all',metric=k,sample_count=48,source_count=16,mean=.2,median=.1,p95=.15)
                for m in('RAW64_WHOLE','RAW64_PARTIAL',*report.FAMILIES)for s in report.SNRS
                for k in('TX_seconds','RX_seconds','software_e2e_excluding_channel_seconds')]
            report.csv_write(csvpath,timing);done=p/'completion.json'
            value=dict(status='T4_FIXED16_ONLINE_TIMING_COMPLETE',total_frames=1152,measured_frames=576,source_count=16,
                no_online_tokens_or_output_cache=True,packet_budget={'unresolved_frames':0},request_path=str(req),request_sha256=h.sha(req),outputs={str(csvpath):h.sha(csvpath)})
            h.save(done,value);args=SimpleNamespace(timing_csv=str(csvpath),timing_completion=str(done))
            self.assertEqual(len(report.read_timing(args,freeze)),12)
            data=h.read(projection);data['rows'][0]['target_m']=8;projection.write_text(json.dumps(data),encoding='utf-8')
            req.write_text(json.dumps(dict(frozen_policy=h.desc(projection),calibration_completion=cp)),encoding='utf-8')
            value['request_sha256']=h.sha(req);done.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'rank1'):report.read_timing(args,freeze)

    def test_source_lengths_fallback_and_m9_real_fraction(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);records=[]
            policy={'policies':{f:{str(s):dict(target_m=9,source_capacity=4751)for s in report.SNRS}for f in report.FAMILIES}}
            for i,m9 in enumerate((4400,5000)):
                sr=dict(schema='T1_SOURCE_ASSET_V1',source_role='calibration',streams={f:{str(m):dict(payload_bits=m9 if m==9 else 100*m)for m in range(4,10)}for f in report.FAMILIES})
                cp=p/f'{i}.json';h.save(cp,sr);records.append(dict(source_index=i,source_id='synthetic'+str(i),checkpoint=str(cp),checkpoint_sha256=h.sha(cp)))
            mf=p/'manifest.json';h.save(mf,dict(records=records,source_count=2));done=p/'completion.json';h.save(done,dict(outputs={str(mf):h.sha(mf)}))
            rows,_=report.source_table(str(done),'calibration',policy);m9=[x for x in rows if x['complete_scale_m']==9]
            self.assertEqual([x['actual_selected_m_SNR19']for x in m9],[9,9,8,8]);self.assertEqual([x['fits_frozen_MCS_SNR19']for x in m9],[True,True,False,False])
            with patch.object(report,'FAMILIES',[report.FAMILIES[0]]):
                # Fill the second population only to test its explicit denominator.
                summary=report.m9_summary(rows+[dict(x,population='holdout')for x in rows])
            self.assertTrue(all(x['m9_fits_fraction_all_sources']==.5 and x['actual_selected_m9_count']==1 for x in summary))

    def test_raw_drop_uses_crc_only_and_counts_partial_errors(self):
        score=dict(source_id='synthetic',**{m:.5 for m in report.METRICS});tokens=np.zeros(680,np.int64)
        item=dict(source_id='synthetic',family=report.RAW_DROP[1],same_actual_receive_evidence={},gray_requested=False,
            TX_truth_used_for_decision=False,output_rule='ORIGINAL_KEEP_OUTPUT_UNCHANGED',original_image={'image_sha256':'original'})
        fact=dict(source_id='synthetic',header_ok=True,body_crc_accept=True,gray=False,m=1,K=2,N=1024,modulation='QPSK',source_status='KEEP_ACTUAL_HARD_TOKENS',
            receiver_state=dict(kind='tokens',order='raster',m=1,K=2,prefix=[[0]],partial_values=[0,3]),
            transmission=dict(N=1024,E=2048.,header_uses=68,body_uses=956,idle_uses=0,groups=[dict(k=60,n=1912,source_bits=36,crc_bits=16,tail_bits=0)]))
        with patch.object(join,'receive_evidence',lambda x:{}):
            row=join.raw_diagnostic_row(item,fact,tokens,score)
            self.assertEqual(row['original_KEEP_token_error_count'],1);self.assertEqual(row['original_KEEP_token_error_denominator'],3)
            self.assertTrue(row['CRC_undetected_token_error']);self.assertFalse(row['gray']);self.assertEqual(row['output_image_identity'],'original')
            fact['body_crc_accept']=False;item.update(gray_requested=True,output_rule='CONSTANT_0P5_RGB')
            row=join.raw_diagnostic_row(item,fact,tokens,score);self.assertTrue(row['gray']);self.assertFalse(row['CRC_undetected_token_error'])
            self.assertEqual(row['failure_state'],'BODY_CRC_REJECT_DIAGNOSTIC_GRAY');self.assertFalse(row['diagnostic_truth_used_for_output'])

    def test_t4_join_keeps_registered_6201_seeds_and_drop_states(self):
        rows=[]
        for family in (report.FAMILIES[1],report.RAW_DROP[1]):
            for i in range(500):
                for seed in report.SEEDS:
                    row=dict(family=family,snr_db=10,source_index=i,source_id=str(i),noise_seed=seed,N=1024,E_frame=2048.,rho=1.,
                        header_ok=True,body_crc_accept=True,body_parser_accept=True,source_canonical_accept=True,**{m:0. for m in report.METRICS})
                    if family in report.RAW_DROP:row['failure_state']='CRC_ACCEPTED_WITH_TOKEN_ERRORS_KEEP_UNCHANGED'
                    rows.append(row)
        done=dict(status='T1_POSTHOC_COMMON500_FOUR_METRICS_COMPLETE',source_count=500,holdout_used_for_selection=False,frame_count=len(rows))
        with patch.object(t4,'bound_file',lambda *a:None),patch.object(join,'joined_rows',lambda *a:rows):
            result,missing=t4.entropy_rows('synthetic',done)
        self.assertFalse(missing);self.assertEqual(set(r['noise_seed']for r in result),{6201,6202,6203})
        self.assertEqual({r['failure_state']for r in result},{'SOURCE_DECODED','CRC_ACCEPTED_WITH_TOKEN_ERRORS_KEEP_UNCHANGED'})

    def test_actual_paired_19db_sign_and_zero_not_hidden(self):
        root=Path(__file__).resolve().parents[3]
        rows=report.csv_read(root/'.research/wcl_evidence_closure_20261009/t1_actual_statistics/paired.csv')
        selected=[r for r in rows if r['method']=='EC_VAR_WHOLE_SNR_19'and r['reference']=='RAW64_PARTIAL_VAR_COMPLETION_SNR_19']
        self.assertEqual(len(selected),4)
        for r in selected:
            direction=report.interval_direction(r['metric'],float(r['ci_low']),float(r['ci_high']))
            if r['metric']=='convnext_top1_source_prediction':self.assertEqual(float(r['mean']),0.);self.assertEqual(direction,'interval includes zero')
            else:self.assertEqual(direction,'interval favors method')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    h.save(a.out,dict(status='PASS'if result.wasSuccessful()else'FAIL',tests_run=result.testsRun,new_model_calls=0,new_channel_calls=0,new_bootstrap_calls=0,
        source_sha256={n:h.sha(Path(__file__).with_name(n))for n in('t1_final_report.py','t1_readonly_frame_join.py','t4_summarize.py','test_t1_t4_reporting.py')}))
    raise SystemExit(0 if result.wasSuccessful()else 1)
