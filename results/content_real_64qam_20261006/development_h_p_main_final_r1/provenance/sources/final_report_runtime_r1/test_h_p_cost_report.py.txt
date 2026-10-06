import copy
import unittest

import h_p_cost_report as r


def fixture():
    catalog = []
    for slot in range(18):
        role = r.ROLES[0] if slot < 8 else r.ROLES[1] if slot < 10 else r.ROLES[2]
        pos = slot if slot < 8 else slot - 10
        snr = (13, 19)[slot - 8] if role == r.ROLES[1] else (13 if pos < 4 else 19)
        arm = 'H64-RAW-PARTIAL' if role == r.ROLES[1] else ('H16-R', 'H16-A', 'H64-R', 'H64-A')[pos % 4]
        catalog.append(dict(point_id=f'H18_SLOT_{slot:02d}', development_slot=slot, snr_db=snr,
            role=role, arm=arm, candidate_id=f'candidate{slot}'))
    points = {x['point_id']: dict(branch='H', snr_db=x['snr_db'], noise_seeds=r.NOISE['H'],
        metric_identity={m: 'identity:' + m for m in r.METRICS}) for x in catalog}
    points.update({f'P1024_SNR_{s}': dict(branch='P', snr_db=s, noise_seeds=r.NOISE['P'],
        metric_identity={m: 'identity:' + m for m in r.METRICS}) for s in (13, 19)})
    interval = dict(mean='0.12345678901234567', ci_low='-0.00012345678901234', ci_high='0.23456789012345678',
        source_count=100, noise_count=3, frame_count=300, bootstrap_replicates=10000, bootstrap_seed=2026100605)
    pairs = [dict(method=x['point_id'], reference=f"P1024_SNR_{x['snr_db']}") for x in catalog]
    hp = dict(completion=dict(status='H_P_FIXED_SOURCE_COMPARISON_COMPLETE_MAIN_PENDING', source_count=100,
        H_frames=5400, P_frames=600, comparison_count=18, exact_original_pixel_pairs=100, noise_seeds=r.NOISE,
        original_rows_modified=False, policy_selection=False, MAIN_complete=False, H_success_evaluated=False),
        point_identity=points, comparison_plan=dict(comparisons=pairs, H_catalog=copy.deepcopy(catalog),
            status='FROZEN_H18_TO_SAME_SNR_P18_PAIRS_BEFORE_SCORE_ROWS'),
        common_metric_admission=dict(status='FROZEN_H_P_COMMON_METRIC_ADMISSION', used_for_selection=False,
            metrics={m: dict(status='ADMITTED' if m in r.METRICS else 'MISSING_IN_P') for m in r.METRICS + r.MISSING}),
        summary=[dict(point_id=p, branch=x['branch'], snr_db=x['snr_db'], noise_seeds=x['noise_seeds'],
            metric=m, metric_identity=x['metric_identity'][m], **interval) for p, x in points.items() for m in r.METRICS],
        paired=[dict(**pair, snr_db=points[pair['method']]['snr_db'], metric=m, metric_identity='identity:' + m,
            delta_definition='method minus reference', method_noise_seeds=r.NOISE['H'], reference_noise_seeds=r.NOISE['P'],
            frame_level_noise_pairing_claimed=False, **interval) for pair in pairs for m in r.METRICS])
    h = dict(completion=dict(status='H18_DEVELOPMENT_DESCRIPTIVE_AGGREGATION_COMPLETE', source_count=100,
        frame_count=5400, points=18, comparison_count=26, policy_selection=False), point_catalog=catalog)
    for name, den in (('tx_source_scale', 100), ('rx_received_scale', 300), ('receiver_status', 300)):
        h[name] = [dict(x, count=den, denominator=den) for x in catalog]
    t = dict(completion=dict(status='H_FIXED16_ONLINE_COMPONENTS_COMPLETE', source_count=16,
        component_case_count=288, H_policy_snr_points=18, source_indices=r.FIXED16, noise_seed=6201,
        warmup_repetitions=1, measured_repetitions=3, exact_TX_parity=True, exact_RX_parity=True,
        new_packet_decodes=0, new_noise_draws=0, new_quality_samples=0, new_metric_calls=0, budget_writes=0,
        exclusive_PHY_latency_measured=False, PHY_encode_measured=False, end_to_end_latency_measured=False),
        component_summary=dict(status='H_FIXED16_COMPONENT_SUMMARY', source_count=16, component_case_count=288,
            uncached=True, warmups_excluded=True, historical_PHY_can_be_added_to_GPU_totals=False,
            PHY_encode=None, exclusive_PHY_decode=None, end_to_end_latency=None,
            components=[dict(development_slot=s, side=d, component=n, source_count=16, repetitions_per_source=3,
                mean_seconds=0.42, median_source_seconds=0.39, warmups_excluded=True, inclusive=True)
                for s in range(18) for d,n in (('TX','TX_source_total'), ('RX','RX_source_and_image_total'))]),
        historical_phy_event_windows=dict(status='HISTORICAL_PHY_EVENT_WINDOWS_READ_ONLY', rows=[],
            new_packet_decodes=0, standalone_PHY_measured=False, compatible_for_end_to_end_sum=False))
    return hp, h, t


class Report(unittest.TestCase):
    def test_projection_keeps_exact_intervals_and_all_points(self):
        hp, h, t = fixture(); original = copy.deepcopy((hp,h,t)); out = r.project(hp,h,t)
        self.assertEqual(out['summary'], hp['summary']); self.assertEqual(out['paired'], hp['paired'])
        self.assertEqual((hp,h,t), original); self.assertEqual(len(out['paired']),324)
        out['paired'][0]['mean']='changed'; self.assertNotEqual(out['paired'],hp['paired'])

    def test_partial_cannot_be_omitted_or_relabelled_whole(self):
        for mode in ('remove', 'relabel'):
            hp,h,t=fixture()
            if mode=='remove':h['point_catalog'].pop(8)
            else:h['point_catalog'][8]['role']=r.ROLES[0]
            with self.assertRaises(ValueError):r.project(hp,h,t)

    def test_no_new_pairs_or_other_snr(self):
        hp,h,t=fixture();hp['paired'][0]['reference']='H18_SLOT_08'
        with self.assertRaises(ValueError):r.project(hp,h,t)
        hp,h,t=fixture();hp['comparison_plan']['comparisons'][0]['reference']='P1024_SNR_19'
        with self.assertRaises(ValueError):r.project(hp,h,t)

    def test_original_seeds_bootstrap_and_identity(self):
        for field,value in [('reference_noise_seeds',[6201,6202,6203]),('bootstrap_seed',2026100504),
                            ('metric_identity','wrong'),('frame_level_noise_pairing_claimed',True)]:
            hp,h,t=fixture();hp['paired'][0][field]=value
            with self.assertRaises(ValueError):r.project(hp,h,t)

    def test_no_missing_metric_imputation(self):
        hp,h,t=fixture();hp['common_metric_admission']['metrics']['mse']['status']='ADMITTED'
        with self.assertRaises(ValueError):r.project(hp,h,t)
        hp,h,t=fixture();hp['summary'].pop()
        with self.assertRaises(ValueError):r.project(hp,h,t)

    def test_fallback_counts_keep_sources_and_failures(self):
        hp,h,t=fixture();h['rx_received_scale'][0]['count']=299
        with self.assertRaises(ValueError):r.project(hp,h,t)
        hp,h,t=fixture();h['tx_source_scale'][0]['denominator']=300
        with self.assertRaises(ValueError):r.project(hp,h,t)

    def test_costs_never_sum_inclusive_or_fabricate_missing(self):
        hp,h,t=fixture();out=r.project(hp,h,t)
        self.assertEqual(out['costs']['components'],t['component_summary']['components'])
        self.assertIsNone(out['costs']['end_to_end_latency']);self.assertEqual(out['costs']['P_online_cost'],'NOT_MEASURED')
        t['component_summary']['PHY_encode']=0
        with self.assertRaises(ValueError):r.project(hp,h,t)

    def test_timing_pending_and_report_caveats(self):
        hp,h,t=fixture();out=r.project(hp,h);text=r.markdown(out)
        self.assertEqual(out['costs']['status'],'PENDING_NORMAL_TIMING_CLOSURE')
        for phrase in ('MAIN 尚未完成','部分尺度 raw64','2001–2003','float64','float32','每点 100 源','每点 300 帧','暂不填数'):
            self.assertIn(phrase,text)
        self.assertFalse(out['normal_closure_verified_by_this_module'])
        self.assertEqual(sum(line.startswith('| 13 | H18_') or line.startswith('| 19 | H18_') for line in text.splitlines()),18)

    def test_no_completion_or_final_success_substitution(self):
        for field,value in [('status','RUNNING'),('H_success_evaluated',True),('MAIN_complete',True)]:
            hp,h,t=fixture();hp['completion'][field]=value
            with self.assertRaises(ValueError):r.project(hp,h,t)

    def test_policy_identity_bound_across_preexisting_tables(self):
        hp,h,t=fixture();hp['comparison_plan']['H_catalog'][0]['candidate_id']='another calibration winner'
        with self.assertRaises(ValueError):r.project(hp,h,t)


if __name__ == '__main__':
    unittest.main()
