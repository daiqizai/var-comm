"""CPU-only scientific report checks; never create model reconstructions."""
from pathlib import Path
import csv
import gzip
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import report_builder as r


def rows(method):
    return [dict(N=1024,snr_db=snr,source_index=i,source_id='d'+str(i),preprocessing_id='same',
        reference_sha256='same-reference-'+str(i),noise_seed=seed,method=method,
        dinov2_vitl14_cosine=.3 if method=='B3' else .2)
        for snr in r.SNRS for i in range(100) for seed in r.SEEDS]


class Tables(unittest.TestCase):
    def test_unmeasured_p_point_stays_blank_and_incomplete_point_is_not_paired(self):
        p=[row for row in rows('P1024') if row['snr_db']!=10]
        p.pop()
        with patch.object(r,'HEADLINE',('dinov2_vitl14_cosine',)):
            paired,summary,coverage=r.pair_p(rows('B3'),p)
        self.assertEqual({row['snr_db'] for row in paired},{4,7})
        self.assertEqual({row['snr_db'] for row in summary},{4,7})
        self.assertEqual([row['snr_db'] for row in coverage if not row['complete_paired_population']],[10,13])
        self.assertTrue(all(abs(row['mean']-.1)<1e-12 for row in paired))

    def test_p_reference_mismatch_is_not_hidden_by_matching_index(self):
        p=rows('P1024');p[0]['source_id']='other-source'
        with patch.object(r,'HEADLINE',('dinov2_vitl14_cosine',)):
            with self.assertRaises(RuntimeError):r.pair_p(rows('B3'),p)

    def test_absent_method_or_metric_has_explicit_blank(self):
        self.assertEqual(r.fmt(None),'—');self.assertEqual(r.fmt(''),'—')
        self.assertEqual(r.fmt(.25),'0.2500')


class Panels(unittest.TestCase):
    def test_measured_pixels_are_pasted_at_original_resolution(self):
        axis=np.arange(256,dtype=np.float32)/255
        image=np.stack([*np.meshgrid(axis,axis,indexing='xy'),np.full((256,256),.5,np.float32)])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'panel.png'
            receipt=r.make_panel([dict(source_index=0,images={'Original':image,'B3':image})],
                                  ['Original','B3','P1024'],path,'Engineering fixture')
            self.assertEqual(len(receipt['cells']),2);self.assertTrue(receipt['native_resolution'])
            with Image.open(path) as im:array=np.asarray(im)
            for cell in receipt['cells']:
                x0,y0,x1,y1=cell['box'];self.assertEqual((x1-x0,y1-y0),(256,256))
                np.testing.assert_array_equal(array[y0:y1,x0:x1],r.display_rgb(image))
            self.assertLess(path.stat().st_size,10_000_000)


class PublicationFormat(unittest.TestCase):
    def test_full_table_published_as_lossless_plain_csv_parts_with_index(self):
        import release_tables
        copy_light=release_tables.copy_light
        rows=[['source','note'],*[[str(i),'comma, newline\n保留字段'+str(i)] for i in range(30)]]
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'scored';source.mkdir();result=Path(tmp)/'published';result.mkdir()
            compressed=source/'metrics_per_frame.csv.gz'
            with gzip.open(compressed,'wt',encoding='utf-8',newline='') as f:csv.writer(f).writerows(rows)
            small=('convnext_summary.csv','convnext_transitions.csv','development_source_quality.csv')
            for name in small:(source/name).write_bytes(b'field,value\nkept,1\n')
            with patch.object(release_tables,'copy_light',side_effect=lambda src,dst:copy_light(src,dst,limit=160)) as publish:
                r.publish_source_tables(source,result)
            publish.assert_called_once_with(compressed,result/'metrics_per_frame.csv')
            index=r.q.read(result/'metrics_per_frame.index.json');reassembled=[]
            self.assertTrue(index['lossless_records']);self.assertEqual(index['rows'],len(rows)-1)
            self.assertGreater(len(index['parts']),1)
            for part in index['parts']:
                path=result/part['path'];self.assertLessEqual(path.stat().st_size,160)
                self.assertEqual(r.q.sha(path),part['sha256'])
                with path.open(encoding='utf-8',newline='') as f:
                    reader=csv.reader(f);self.assertEqual(next(reader),rows[0]);reassembled.extend(reader)
            self.assertEqual(reassembled,rows[1:])
            self.assertFalse(any(p.suffix in ('.gz','.zip') for p in result.rglob('*')))
            for name in small:self.assertEqual((source/name).read_bytes(),(result/name).read_bytes())


class CalibrationScope(unittest.TestCase):
    def test_screen_scope_binds_final_shortlist_and_rejects_tampering(self):
        from artifact_bridge import combined_cost_rule
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scored=root/'actual_scores';profiles=[{'stable_id':'fixture'}]
            pilot=root/'quality_pilot/benchmark.json'
            r.q.write(pilot,dict(estimated_full1000_source_Q_hours=50.))
            decision=combined_cost_rule(50.,0.,1.,1.);decision['quality_pilot_sha256']=r.q.sha(pilot)
            r.q.write(root/'stage_a_cost_decision.json',decision)
            shortlist=root/'shortlist_profiles.json';screen=root/'screen300_policies.json'
            r.q.write(shortlist,profiles)
            r.q.write(screen,dict(source_count=300,stage='screen',development_read=False,synthetic=False))
            r.q.write(root/'shortlist_profiles.json.receipt.json',dict(status='FIXED300_SHORTLIST_FROZEN',source_count=300,
                development_read=False,shortlist_sha256=r.q.sha(shortlist),screen_sha256=r.q.sha(screen),candidates=1))
            policies=dict(source_count=1000,development_read=False,profile_input_sha256=r.q.identity(profiles))
            scope,bindings,text=r.calibration_scope(scored,policies)
            self.assertEqual(scope['final_candidate_count'],1);self.assertFalse(scope['full_grid_optimum_claimed'])
            self.assertEqual(len(bindings),5);self.assertIn('shortlist',text);self.assertIn('不能',text)
            policies['profile_input_sha256']='changed'
            with self.assertRaises(RuntimeError):r.calibration_scope(scored,policies)
            r.q.write(pilot,dict(estimated_full1000_source_Q_hours=1.))
            with self.assertRaises(RuntimeError):r.calibration_scope(scored,policies)

    def test_unscreened_cost_and_absent_receipt_do_not_invent_global_optimality(self):
        from artifact_bridge import combined_cost_rule
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);policies=dict(source_count=1000,development_read=False)
            scope,bindings,text=r.calibration_scope(root/'actual_scores',policies)
            self.assertFalse(scope['full_grid_optimum_claimed']);self.assertFalse(bindings)
            pilot=root/'quality_pilot/benchmark.json';r.q.write(pilot,dict(estimated_full1000_source_Q_hours=1.))
            decision=combined_cost_rule(1.,0.,1.,1.);decision['quality_pilot_sha256']=r.q.sha(pilot)
            r.q.write(root/'stage_a_cost_decision.json',decision)
            scope,bindings,text=r.calibration_scope(root/'actual_scores',policies)
            self.assertFalse(scope['screen300']);self.assertIn('未触发300',text)


if __name__=='__main__':unittest.main()
