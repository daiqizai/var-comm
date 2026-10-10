"""Fake metadata regression for legitimate same-pixel archive aliases only."""
import copy
from pathlib import Path
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_new100_statistics_v2 as owner
import test_h800_ep_new100_statistics_v1 as old_tests


def source_local_pairs():
    store,pins,r,expected,module,pin=old_tests.closure_fixture();c=owner.core.confirmation
    original=copy.deepcopy(store['pair']);worker=store[store[pins['completion']['path']]['worker_completion']['path']]
    result=worker['results'];counts=result['counts']['reserved']
    for name in ('image_scores','dinov2_vitl14_reconstruction','convnext_reconstruction','lpips_pair'):counts[name]=100
    counts['lpips_alexnet_backbone_forward']=200;result['actual_unique_pairs']=100;result['reused_pairs']=3500
    for row in store['metric_rows']:
        i=row['source_index'];key='pair-'+str(i)
        if key not in store:
            pair=copy.deepcopy(original);pair['reference_pixel_sha256']=f'{i+1:064x}'
            pair['metric_pair_key']=c.digest(dict(reference_sha256=pair['reference_pixel_sha256'],reconstruction_sha256=pair['reconstruction_pixel_sha256'],metric_identity=pair['metric_identity']))
            store[key]=pair
        row['metric_pair_result']=pin(key);row['metric_pair_key']=store[key]['metric_pair_key']
        row['reuse']='ACTUAL_NEW_PAIR' if row['frame_index']%36==0 else 'SAME_RUN_IDENTICAL_PIXEL_PAIR'
    return store,pins,r,expected,module,pin


class AliasTests(unittest.TestCase):
    def closure(self,alter=None):
        store,pins,r,expected,module,pin=source_local_pairs()
        if alter:alter(store)
        with ExitStack() as stack:
            stack.enter_context(patch.object(owner,'metric_implementation',return_value=module))
            stack.enter_context(patch.object(owner,'read',side_effect=lambda p:store[p['path']]))
            stack.enter_context(patch.object(owner,'pin',side_effect=pin))
            stack.enter_context(patch.object(owner,'METRIC_OWNER_SHA','f'*64))
            stack.enter_context(patch.object(owner.io,'sha',side_effect=lambda p:r['tool_bindings'][Path(p).name]))
            stack.enter_context(patch.object(owner.core.confirmation,'frame_grid',return_value=expected))
            return owner.metric_closure(pins)

    def test_same_source_and_same_pixels_different_later_archive_accepted(self):
        def alias(s):
            s['recon/2']['reconstruction']=copy.deepcopy(s['recon/2']['reconstruction'])
            s['recon/2']['reconstruction']['image_archive']['path']='another_actual_closed_NPZ'
        rows,events,inputs=self.closure(alias)
        self.assertEqual((len(rows),len(events)),(3600,3600))
        self.assertEqual(rows[2]['reuse'],'SAME_RUN_IDENTICAL_PIXEL_PAIR')

    def test_first_archive_mismatch_pixel_mismatch_unknown_first_and_cross_source_rejected(self):
        def first_archive(s):
            s['recon/0']['reconstruction']=copy.deepcopy(s['recon/0']['reconstruction'])
            s['recon/0']['reconstruction']['image_archive']['path']='not_actually_scored'
        def pixels(s):
            s['recon/2']['reconstruction']=copy.deepcopy(s['recon/2']['reconstruction']);s['recon/2']['reconstruction']['image_sha256']='z'*64
        def unknown(s):s['metric_rows'][0]['reuse']='SAME_RUN_IDENTICAL_PIXEL_PAIR'
        def cross(s):
            for k in ('metric_pair_key','metric_pair_result'):s['metric_rows'][36][k]=s['metric_rows'][0][k]
            s['metric_rows'][36]['reuse']='SAME_RUN_IDENTICAL_PIXEL_PAIR'
        for alter in (first_archive,pixels,unknown,cross):
            with self.subTest(alter=alter.__name__),self.assertRaises(ValueError):self.closure(alter)

    def test_v2_fake_export_and_paid_failure_no_replay_remain(self):
        original=old_tests.ExecutionTests()
        original_fake=original.fake_run
        def fake(*args,**kwargs):
            stack,arguments,out=original_fake(*args,**kwargs)
            # Existing fake registration is already patched inside fake_run.
            patched=owner.registration.return_value
            patched[0]['previous_failed_preparation']={'previous_draw_matrix_calls':0}
            return stack,arguments,out
        with patch.object(old_tests,'owner',owner),patch.object(original,'fake_run',side_effect=fake):
            original.test_fake_export_complete48_24_and_no_replay()
            original.test_failed_interval_preserves_unresolved_paid_call_and_blocks_replay()


if __name__=='__main__':unittest.main()
